#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MongoDB数据库操作工具类
用于YOLOv5图像检测项目的数据存储和查询
"""

import hashlib
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from pymongo import MongoClient
from bson import ObjectId
import logging
from PIL import Image
from dotenv import load_dotenv

# 根据FLASK_ENV环境变量加载对应的配置文件
env = os.getenv('FLASK_ENV', 'development')
if env == 'production':
    load_dotenv('.env.production')
else:
    load_dotenv('.env')
print(f"已加载 {env} 环境配置")

logger = logging.getLogger(__name__)

class DetectionDatabase:
    def __init__(self, connection_string=None, db_name=None):
        """
        初始化检测数据库操作类
        
        Args:
            connection_string: MongoDB连接字符串（可选，优先使用环境变量）
            db_name: 数据库名称（可选，优先使用环境变量）
        """
        # 从环境变量获取数据库配置
        username = os.getenv('MONGODB_USERNAME')
        password = os.getenv('MONGODB_PASSWORD')
        host = os.getenv('MONGODB_HOST', 'localhost')
        port = os.getenv('MONGODB_PORT', '27017')
        database = os.getenv('MONGODB_DATABASE', 'yolov5_detection')
        
        # 构建连接字符串
        if username and password:
            # 检查是否为本地连接
            if host in ['localhost', '127.0.0.1', '::1']:
                # 本地连接，尝试无认证连接
                self.connection_string = f"mongodb://{host}:{port}/"
                logger.info("检测到本地MongoDB连接，使用无认证模式")
            else:
                # 远程连接，使用认证
                self.connection_string = f"mongodb://{username}:{password}@{host}:{port}/{database}?authSource={database}"
                logger.info("检测到远程MongoDB连接，使用认证模式")
        else:
            self.connection_string = connection_string or f"mongodb://{host}:{port}/"
        
        self.db_name = db_name or database
        self.client = None
        self.db = None
        self._connect()
    
    def _connect(self):
        """连接到数据库"""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.db_name]
            # 测试连接
            self.client.admin.command('ping')
            logger.info(f"数据库连接成功: {self.db_name}")
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            raise
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_image_info(self, file_path: str) -> Dict[str, Any]:
        """获取图片的基本信息"""
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                format_name = img.format.lower() if img.format else 'unknown'
            
            file_size = os.path.getsize(file_path)
            return {
                'width': width,
                'height': height,
                'format': format_name,
                'size': file_size
            }
        except Exception as e:
            logger.error(f"获取图片信息失败: {str(e)}")
            return {
                'width': 0,
                'height': 0,
                'format': 'unknown',
                'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
    
    def save_image_info(self, image_data: bytes, filename: str, original_filename: str, 
                       image_type: str = "original", content_type: str = "image/jpeg") -> ObjectId:
        """
        保存图片信息到数据库
        
        Args:
            image_data: 图片二进制数据
            filename: 存储的文件名
            original_filename: 原始文件名
            image_type: 图片类型 (original/detected)
            content_type: 图片内容类型
        
        Returns:
            ObjectId: 插入的文档ID
        """
        try:
            # 计算文件大小和哈希值
            file_size = len(image_data)
            file_hash = hashlib.md5(image_data).hexdigest()
            
            # 检查是否已存在相同哈希的文件
            existing = self.db.image_info.find_one({"file_hash": file_hash})
            if existing and image_type == "original":
                logger.info(f"文件已存在，返回现有记录ID: {existing['_id']}")
                return existing['_id']
            
            # 获取图片尺寸和格式
            from io import BytesIO
            try:
                with Image.open(BytesIO(image_data)) as img:
                    image_width, image_height = img.size
                    image_format = img.format.lower() if img.format else 'unknown'
            except Exception as e:
                logger.error(f"获取图片信息失败: {str(e)}")
                image_width, image_height = 0, 0
                image_format = 'unknown'
            
            image_doc = {
                "filename": filename,
                "original_filename": original_filename,
                "image_data": image_data,
                "content_type": content_type,
                "file_size": file_size,
                "file_hash": file_hash,
                "image_width": image_width,
                "image_height": image_height,
                "image_format": image_format,
                "upload_time": datetime.now(),
                "image_type": image_type
            }
            
            result = self.db.image_info.insert_one(image_doc)
            logger.info(f"图片信息保存成功: {result.inserted_id}")
            return result.inserted_id
        
        except Exception as e:
            logger.error(f"保存图片信息失败: {str(e)}")
            raise
    
    def save_detection_record(self, detection_id: str, original_image_id: ObjectId, 
                            detected_image_id: ObjectId, model_name: str, 
                            detection_count: int, processing_time: float, 
                            user_ip: str = None, user_agent: str = None) -> ObjectId:
        """
        保存检测记录
        
        Args:
            detection_id: 检测唯一标识符
            original_image_id: 原始图片ID
            detected_image_id: 检测后图片ID
            model_name: 使用的模型名称
            detection_count: 检测到的目标数量
            processing_time: 处理时间
            user_ip: 用户IP
            user_agent: 用户代理
        
        Returns:
            ObjectId: 插入的文档ID
        """
        try:
            record_doc = {
                "detection_id": detection_id,
                "original_image_id": original_image_id,
                "detected_image_id": detected_image_id,
                "model_name": model_name,
                "detection_count": detection_count,
                "processing_time": processing_time,
                "status": "success",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "user_ip": user_ip,
                "user_agent": user_agent
            }
            
            result = self.db.detection_records.insert_one(record_doc)
            logger.info(f"检测记录保存成功: {result.inserted_id}")
            return result.inserted_id
        
        except Exception as e:
            logger.error(f"保存检测记录失败: {str(e)}")
            raise
    
    def save_detection_results(self, detection_id: str, results: List[Dict[str, Any]]) -> List[ObjectId]:
        """
        保存检测结果详情
        
        Args:
            detection_id: 检测ID
            results: 检测结果列表
        
        Returns:
            List[ObjectId]: 插入的文档ID列表
        """
        try:
            result_docs = []
            for result in results:
                bbox = result.get('bbox', [])
                if len(bbox) >= 4:
                    # 计算检测区域面积
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    
                    result_doc = {
                        "detection_id": detection_id,
                        "class_id": result.get('class', 0),
                        "class_name": result.get('class_name', 'unknown'),
                        "confidence": result.get('confidence', 0.0),
                        "bbox": {
                            "x1": float(bbox[0]),
                            "y1": float(bbox[1]),
                            "x2": float(bbox[2]),
                            "y2": float(bbox[3])
                        },
                        "area": area,
                        "created_at": datetime.now()
                    }
                    result_docs.append(result_doc)
            
            if result_docs:
                insert_result = self.db.detection_results.insert_many(result_docs)
                logger.info(f"检测结果保存成功: {len(insert_result.inserted_ids)} 条记录")
                return insert_result.inserted_ids
            else:
                logger.info("没有有效的检测结果需要保存")
                return []
        
        except Exception as e:
            logger.error(f"保存检测结果失败: {str(e)}")
            raise
    
    def get_detection_history(self, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """
        获取检测历史记录
        
        Args:
            limit: 返回记录数量限制
            skip: 跳过记录数量
        
        Returns:
            List[Dict]: 检测历史记录列表
        """
        try:
            pipeline = [
                {
                    "$lookup": {
                        "from": "image_info",
                        "localField": "original_image_id",
                        "foreignField": "_id",
                        "as": "original_image"
                    }
                },
                {
                    "$lookup": {
                        "from": "image_info",
                        "localField": "detected_image_id",
                        "foreignField": "_id",
                        "as": "detected_image"
                    }
                },
                {
                    "$sort": {"created_at": -1}
                },
                {
                    "$skip": skip
                },
                {
                    "$limit": limit
                }
            ]
            
            results = list(self.db.detection_records.aggregate(pipeline))
            logger.info(f"获取检测历史记录成功: {len(results)} 条记录")
            return results
        
        except Exception as e:
            logger.error(f"获取检测历史记录失败: {str(e)}")
            return []
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        获取检测统计信息
        
        Returns:
            Dict: 统计信息
        """
        try:
            # 总检测次数
            total_detections = self.db.detection_records.count_documents({})
            
            # 成功检测次数
            successful_detections = self.db.detection_records.count_documents({"status": "success"})
            
            # 总检测目标数量
            total_objects = self.db.detection_results.count_documents({})
            
            # 按模型统计
            model_stats = list(self.db.detection_records.aggregate([
                {"$group": {
                    "_id": "$model_name",
                    "count": {"$sum": 1},
                    "avg_processing_time": {"$avg": "$processing_time"}
                }}
            ]))
            
            # 按类别统计
            class_stats = list(self.db.detection_results.aggregate([
                {"$group": {
                    "_id": "$class_name",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"}
                }}
            ]))
            
            stats = {
                "total_detections": total_detections,
                "successful_detections": successful_detections,
                "total_objects_detected": total_objects,
                "success_rate": successful_detections / total_detections if total_detections > 0 else 0,
                "model_statistics": model_stats,
                "class_statistics": class_stats
            }
            
            logger.info("获取统计信息成功")
            return stats
        
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {}
    
    def search_detections(self, model_name: str = None, class_name: str = None, 
                         min_confidence: float = None, start_date: datetime = None, 
                         end_date: datetime = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        搜索检测记录
        
        Args:
            model_name: 模型名称
            class_name: 类别名称
            min_confidence: 最小置信度
            start_date: 开始日期
            end_date: 结束日期
            limit: 返回记录数量限制
        
        Returns:
            List[Dict]: 搜索结果
        """
        try:
            match_conditions = {}
            
            if model_name:
                match_conditions["model_name"] = model_name
            
            if start_date or end_date:
                date_condition = {}
                if start_date:
                    date_condition["$gte"] = start_date
                if end_date:
                    date_condition["$lte"] = end_date
                match_conditions["created_at"] = date_condition
            
            pipeline = [
                {"$match": match_conditions},
                {
                    "$lookup": {
                        "from": "detection_results",
                        "localField": "detection_id",
                        "foreignField": "detection_id",
                        "as": "results"
                    }
                }
            ]
            
            # 添加类别和置信度过滤
            if class_name or min_confidence:
                result_match = {}
                if class_name:
                    result_match["results.class_name"] = class_name
                if min_confidence:
                    result_match["results.confidence"] = {"$gte": min_confidence}
                pipeline.append({"$match": result_match})
            
            pipeline.extend([
                {"$sort": {"created_at": -1}},
                {"$limit": limit}
            ])
            
            results = list(self.db.detection_records.aggregate(pipeline))
            logger.info(f"搜索检测记录成功: {len(results)} 条记录")
            return results
        
        except Exception as e:
            logger.error(f"搜索检测记录失败: {str(e)}")
            return []
    
    def get_model_info_from_db(self, model_name: str) -> Optional[Dict[str, Any]]:
        """从数据库获取模型信息"""
        if self.db is None:
            return None
            
        try:
            model_info = self.db.model_info.find_one({"model_name": model_name})
            if model_info:
                # 转换ObjectId为字符串
                model_info['_id'] = str(model_info['_id'])
                logger.info(f"从数据库获取模型信息成功: {model_name}")
                return model_info
            else:
                logger.warning(f"数据库中未找到模型信息: {model_name}")
                return None
        except Exception as e:
            logger.error(f"从数据库获取模型信息失败: {str(e)}")
            return None
    
    def get_all_models_from_db(self) -> List[Dict[str, Any]]:
        """从数据库获取所有模型信息"""
        if self.db is None:
            return []
            
        try:
            models = list(self.db.model_info.find({}))
            # 转换ObjectId为字符串
            for model in models:
                model['_id'] = str(model['_id'])
            logger.info(f"从数据库获取到 {len(models)} 个模型信息")
            return models
        except Exception as e:
            logger.error(f"从数据库获取模型列表失败: {str(e)}")
            return []
    
    def register_model_to_db(self, model_info: Dict[str, Any]) -> Optional[ObjectId]:
        """注册模型信息到数据库"""
        if self.db is None:
            return None
            
        try:
            # 检查模型是否已存在
            existing = self.db.model_info.find_one({"model_name": model_info["model_name"]})
            if existing:
                # 更新现有模型信息
                model_info["updated_at"] = datetime.now()
                result = self.db.model_info.update_one(
                    {"model_name": model_info["model_name"]},
                    {"$set": model_info}
                )
                logger.info(f"模型信息更新成功: {model_info['model_name']}")
                return existing['_id']
            else:
                # 插入新模型信息
                model_info["created_at"] = datetime.now()
                model_info["updated_at"] = datetime.now()
                result = self.db.model_info.insert_one(model_info)
                logger.info(f"模型信息注册成功: {result.inserted_id}")
                return result.inserted_id
        except Exception as e:
            logger.error(f"注册模型信息失败: {str(e)}")
            return None
    
    def close(self):
        """关闭数据库连接"""
        if self.client:
            self.client.close()
            logger.info("数据库连接已关闭")

# 全局数据库实例
_db_instance = None

def get_database() -> DetectionDatabase:
    """获取数据库实例（单例模式）"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DetectionDatabase()
    return _db_instance

def close_database():
    """关闭数据库连接"""
    global _db_instance
    if _db_instance:
        _db_instance.close()
        _db_instance = None