#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MongoDB数据库初始化脚本
用于YOLOv5图像检测项目的数据存储
"""

import os
from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
import logging
from dotenv import load_dotenv

# 根据FLASK_ENV环境变量加载对应的配置文件
env = os.getenv('FLASK_ENV', 'development')
if env == 'production':
    load_dotenv('.env.production')
else:
    load_dotenv('.env')
print(f"已加载 {env} 环境配置")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, connection_string=None, db_name=None):
        """
        初始化数据库管理器
        
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
            # 尝试使用目标数据库作为认证数据库
            self.connection_string = f"mongodb://{username}:{password}@{host}:{port}/{database}?authSource={database}"
        else:
            self.connection_string = connection_string or f"mongodb://{host}:{port}/"
        
        self.db_name = db_name or database
        self.client = None
        self.db = None
    
    def connect(self):
        """连接到MongoDB数据库"""
        try:
            print(f"正在连接到MongoDB数据库: {self.connection_string}")
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.db_name]
            # 测试连接
            self.client.admin.command('ping')
            logger.info(f"成功连接到MongoDB数据库: {self.db_name}")
            return True
        except Exception as e:
            logger.error(f"连接数据库失败: {str(e)}")
            return False
    
    def create_collections(self):
        """创建所需的集合"""
        try:
            # 1. 检测记录集合 - 存储每次检测的基本信息
            detection_records = self.db.detection_records
            
            # 2. 图片信息集合 - 存储图片的详细信息
            image_info = self.db.image_info
            
            # 3. 检测结果集合 - 存储具体的检测结果
            detection_results = self.db.detection_results
            
            # 4. 模型信息集合 - 存储使用的模型信息
            model_info = self.db.model_info
            
            logger.info("集合创建完成")
            return True
        except Exception as e:
            logger.error(f"创建集合失败: {str(e)}")
            return False
    
    def create_indexes(self):
        """创建索引以提高查询性能"""
        try:
            # 检测记录集合索引
            self.db.detection_records.create_index([("created_at", DESCENDING)])
            self.db.detection_records.create_index([("model_name", ASCENDING)])
            self.db.detection_records.create_index([("status", ASCENDING)])
            
            # 图片信息集合索引
            self.db.image_info.create_index([("file_hash", ASCENDING)], unique=True)
            self.db.image_info.create_index([("upload_time", DESCENDING)])
            self.db.image_info.create_index([("file_size", ASCENDING)])
            
            # 检测结果集合索引
            self.db.detection_results.create_index([("detection_id", ASCENDING)])
            self.db.detection_results.create_index([("class_name", ASCENDING)])
            self.db.detection_results.create_index([("confidence", DESCENDING)])
            
            # 模型信息集合索引
            self.db.model_info.create_index([("model_name", ASCENDING)], unique=True)
            
            logger.info("索引创建完成")
            return True
        except Exception as e:
            logger.error(f"创建索引失败: {str(e)}")
            return False
    
    def insert_sample_data(self):
        """插入示例数据"""
        try:
            # 插入示例模型信息
            sample_model = {
                "model_name": "fires.pt",
                "model_path": "model/fires.pt",
                "classes": ["fire"],
                "input_size": [640, 640],
                "created_at": datetime.now(),
                "description": "火灾检测模型",
                "version": "1.0"
            }
            
            self.db.model_info.insert_one(sample_model)
            logger.info("示例数据插入完成")
            return True
        except Exception as e:
            logger.error(f"插入示例数据失败: {str(e)}")
            return False
    
    def get_collection_schemas(self):
        """返回集合的数据结构说明"""
        schemas = {
            "detection_records": {
                "description": "检测记录主表",
                "fields": {
                    "_id": "ObjectId - 主键",
                    "detection_id": "String - 检测唯一标识符",
                    "original_image_id": "ObjectId - 原始图片ID(关联image_info)",
                    "detected_image_id": "ObjectId - 检测后图片ID(关联image_info)",
                    "model_name": "String - 使用的模型名称",
                    "detection_count": "Integer - 检测到的目标数量",
                    "processing_time": "Float - 处理时间(秒)",
                    "status": "String - 检测状态(success/failed/processing)",
                    "created_at": "DateTime - 创建时间",
                    "updated_at": "DateTime - 更新时间",
                    "user_ip": "String - 用户IP地址",
                    "user_agent": "String - 用户代理信息"
                }
            },
            "image_info": {
                "description": "图片信息表",
                "fields": {
                    "_id": "ObjectId - 主键",
                    "filename": "String - 文件名",
                    "original_filename": "String - 原始文件名",
                    "image_data": "Binary - 图片二进制数据",
                    "file_size": "Integer - 文件大小(字节)",
                    "file_hash": "String - 文件MD5哈希值",
                    "image_width": "Integer - 图片宽度",
                    "image_height": "Integer - 图片高度",
                    "image_format": "String - 图片格式(jpg/png等)",
                    "upload_time": "DateTime - 上传时间",
                    "image_type": "String - 图片类型(original/detected)",
                    "content_type": "String - MIME类型(image/jpeg, image/png等)"
                }
            },
            "detection_results": {
                "description": "检测结果详情表",
                "fields": {
                    "_id": "ObjectId - 主键",
                    "detection_id": "String - 检测ID(关联detection_records)",
                    "class_id": "Integer - 类别ID",
                    "class_name": "String - 类别名称",
                    "confidence": "Float - 置信度(0-1)",
                    "bbox": {
                        "x1": "Float - 边界框左上角X坐标",
                        "y1": "Float - 边界框左上角Y坐标",
                        "x2": "Float - 边界框右下角X坐标",
                        "y2": "Float - 边界框右下角Y坐标"
                    },
                    "area": "Float - 检测区域面积",
                    "created_at": "DateTime - 创建时间"
                }
            },
            "model_info": {
                "description": "模型信息表",
                "fields": {
                    "_id": "ObjectId - 主键",
                    "model_name": "String - 模型名称",
                    "model_path": "String - 模型文件路径",
                    "classes": "Array - 支持的类别列表",
                    "input_size": "Array - 输入尺寸[width, height]",
                    "model_size": "Integer - 模型文件大小(字节)",
                    "accuracy": "Float - 模型准确率",
                    "version": "String - 模型版本",
                    "description": "String - 模型描述",
                    "created_at": "DateTime - 创建时间",
                    "updated_at": "DateTime - 更新时间"
                }
            }
        }
        return schemas
    
    def initialize_database(self):
        """初始化整个数据库"""
        logger.info("开始初始化数据库...")
        
        if not self.connect():
            return False
        
        if not self.create_collections():
            return False
        
        if not self.create_indexes():
            return False
        
        if not self.insert_sample_data():
            return False
        
        logger.info("数据库初始化完成!")
        return True
    
    def close_connection(self):
        """关闭数据库连接"""
        if self.client:
            self.client.close()
            logger.info("数据库连接已关闭")

def main():
    """主函数"""
    # 创建数据库管理器实例
    db_manager = DatabaseManager()
    
    try:
        # 初始化数据库
        if db_manager.initialize_database():
            print("\n=== 数据库初始化成功 ===")
            print(f"数据库名称: {db_manager.db_name}")
            print("\n=== 集合结构说明 ===")
            schemas = db_manager.get_collection_schemas()
            for collection_name, schema in schemas.items():
                print(f"\n{collection_name}: {schema['description']}")
                for field, desc in schema['fields'].items():
                    if isinstance(desc, dict):
                        print(f"  {field}:")
                        for sub_field, sub_desc in desc.items():
                            print(f"    {sub_field}: {sub_desc}")
                    else:
                        print(f"  {field}: {desc}")
        else:
            print("数据库初始化失败")
    
    except Exception as e:
        logger.error(f"执行过程中出错: {str(e)}")
    
    finally:
        db_manager.close_connection()

if __name__ == "__main__":
    main()