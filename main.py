import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv
from database_utils import get_database

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 根据FLASK_ENV环境变量加载对应的配置文件
env = os.getenv('FLASK_ENV', 'development')
if env == 'production':
    load_dotenv('.env.production')
else:
    load_dotenv('.env')
logger.info(f"已加载 {env} 环境配置")

# 设置项目根目录
FILE = Path(__file__).resolve()
ROOT = FILE.parent

# YOLOv5路径设置
YOLOV5_PATH = ROOT / 'yolov5-master'
if str(YOLOV5_PATH) not in sys.path:
    sys.path.append(str(YOLOV5_PATH))

import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from waitress import serve
from PIL import Image
import time
from werkzeug.utils import secure_filename

# 检查YOLOv5路径是否存在
if not YOLOV5_PATH.exists():
    logger.error(f"YOLOv5目录不存在：{YOLOV5_PATH}")
    sys.exit(1)

try:
    # 确保YOLOv5模块可以被正确导入
    sys.path.insert(0, str(YOLOV5_PATH))
    from models.experimental import attempt_load
    from utils.general import non_max_suppression
    from utils.plots import Annotator, colors
    logger.info("YOLOv5模块导入成功")
except ImportError as e:
    logger.error(f"导入YOLOv5模块时出错：{e}")
    logger.error("请确保YOLOv5已正确安装")
    sys.exit(1)

app = Flask(__name__)

# 配置文件存储路径
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 类别名称
CLASS_NAMES = ['person']  # 根据您的模型类别进行修改

# 加载YOLOv5模型
model = None
current_model_name = None

def get_available_models():
    """获取model目录下所有可用的模型文件"""
    model_dir = ROOT / 'model'
    model_files = [f.name for f in model_dir.glob('*.pt')]
    return model_files

def load_model(model_name='fires.pt'):
    """加载指定的YOLOv5模型"""
    global model, current_model_name
    model_path = ROOT / 'model' / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")
    try:
        logger.info(f"正在加载模型：{model_path}")
        model = attempt_load(model_path, device='cpu')
        model.eval()
        current_model_name = model_name
        
        # 获取模型支持的类别
        if hasattr(model, 'names'):
            class_names = model.names
        else:
            class_names = CLASS_NAMES
        
        # 获取数据库实例并尝试注册/更新模型信息
        db = get_database()
        if db.db is not None:
            # 获取模型文件大小
            model_size = os.path.getsize(model_path)
            
            # 准备模型信息
            model_info = {
                "model_name": model_name,
                "model_path": str(model_path),
                "classes": list(class_names.values()) if isinstance(class_names, dict) else list(class_names),
                "input_size": [640, 640],  # YOLOv5默认输入尺寸
                "model_size": model_size,
                "accuracy": None,  # 可以后续通过验证集计算
                "version": "1.0",
                "description": f"YOLOv5模型 - {model_name}"
            }
            
            # 注册模型信息到数据库
            db.register_model_to_db(model_info)
        
        logger.info(f"模型加载成功，支持的类别：{class_names}")
        return model, class_names
    except Exception as e:
        logger.error(f"加载模型时出错：{str(e)}")
        raise

def get_model(model_name=None):
    """获取或加载模型实例"""
    global model, current_model_name
    if model is None or (model_name is not None and model_name != current_model_name):
        model, class_names = load_model(model_name if model_name is not None else 'fires.pt')
        return model, class_names
    return model, model.names if hasattr(model, 'names') else CLASS_NAMES

def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_detections(image_data, detections, original_filename):
    """在图像上绘制检测结果，返回处理后的图片二进制数据"""
    try:
        from io import BytesIO
        
        # 从二进制数据读取图像
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解码图像数据")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotator = Annotator(image)
        
        # 为每个检测结果绘制边界框和标签
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']  # 使用class_name而不是class index
            
            # 转换坐标为整数
            x1, y1, x2, y2 = map(int, bbox)
            
            # 使用固定的颜色
            color = (0, 255, 0)  # 使用绿色作为边界框颜色
            
            # 绘制边界框和标签
            label = f'{class_name} {conf:.2f}'  # 使用class_name
            annotator.box_label([x1, y1, x2, y2], label, color=color)
        
        # 转换回BGR格式并编码为JPEG
        result_image = cv2.cvtColor(annotator.result(), cv2.COLOR_RGB2BGR)
        
        # 编码为JPEG格式的二进制数据
        success, encoded_image = cv2.imencode('.jpg', result_image)
        if not success:
            raise ValueError("无法编码检测结果图像")
        
        detected_image_data = encoded_image.tobytes()
        detected_filename = f"detected_{original_filename}"
        
        logger.info(f"检测结果图像处理完成，大小：{len(detected_image_data)} 字节")
        return detected_image_data, detected_filename
    except Exception as e:
        logger.error(f"绘制检测结果时出错：{str(e)}")
        raise

def test_model():
    """测试模型是否正常工作"""
    try:
        # 加载模型
        model, class_names = get_model()
        logger.info(f"模型加载成功！支持的类别：{class_names}")
        
        # 创建测试输入
        dummy_input = torch.zeros((1, 3, 640, 640))
        
        # 测试推理
        with torch.no_grad():
            result = model(dummy_input)  # 只使用模型部分，不使用类别名称
        logger.info("模型推理测试成功！")
        return True
    except Exception as e:
        logger.error(f"模型测试失败：{str(e)}")
        return False

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/image/<image_id>')
def get_image(image_id):
    """从数据库获取图片数据"""
    try:
        from bson import ObjectId
        from flask import Response
        
        db = get_database()
        if db.db is None:
            return jsonify({'error': '数据库连接不可用'}), 503
        
        # 从数据库获取图片信息
        image_doc = db.db.image_info.find_one({"_id": ObjectId(image_id)})
        if not image_doc:
            return jsonify({'error': '图片不存在'}), 404
        
        # 返回图片数据
        return Response(
            image_doc['image_data'],
            mimetype=image_doc.get('content_type', 'image/jpeg'),
            headers={
                'Content-Disposition': f'inline; filename="{image_doc["filename"]}"',
                'Cache-Control': 'public, max-age=3600'
            }
        )
    except Exception as e:
        logger.error(f"获取图片失败：{str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    try:
        return jsonify({
            'status': 'online',
            'current_model': current_model_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'service': 'YOLOv5 Detection API'
        })
    except Exception as e:
        logger.error(f"健康检查失败：{str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/models', methods=['GET'])
def get_models():
    """获取可用模型列表"""
    try:
        # 首先尝试从数据库获取模型信息
        db = get_database()
        db_models = db.get_all_models_from_db() if db.db is not None else []
        
        # 如果数据库中有模型信息，优先返回数据库中的信息
        if db_models:
            return jsonify({
                'success': True,
                'models': db_models,
                'source': 'database'
            })
        else:
            # 如果数据库中没有模型信息，从文件系统获取
            file_models = get_available_models()
            return jsonify({
                'success': True,
                'models': [{'model_name': model} for model in file_models],
                'source': 'filesystem'
            })
    except Exception as e:
        logger.error(f"获取模型列表时出错：{str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """获取当前加载的模型信息"""
    try:
        model_name = request.args.get('model', 'fires.pt')
        
        # 首先尝试从数据库获取详细的模型信息
        db = get_database()
        db_model_info = db.get_model_info_from_db(model_name) if db.db is not None else None
        
        if db_model_info:
            # 如果数据库中有详细信息，返回数据库中的信息
            logger.info(f"从数据库获取模型信息成功：{model_name}")
            return jsonify({
                'success': True,
                'model_info': db_model_info,
                'source': 'database'
            })
        else:
            # 如果数据库中没有信息，加载模型并返回基本信息
            model, class_names = get_model(model_name)
            
            # 确保class_names是列表格式
            if isinstance(class_names, dict):
                # 如果是字典格式，转换为列表
                classes_list = [class_names[i] for i in range(len(class_names))]
            elif isinstance(class_names, (list, tuple)):
                classes_list = list(class_names)
            else:
                classes_list = [str(class_names)]
                
            logger.info(f"获取模型信息成功：{model_name}, 类别：{classes_list}")
            
            return jsonify({
                'success': True,
                'model_name': model_name,
                'classes': classes_list,
                'status': '已加载',
                'source': 'runtime'
            })
    except Exception as e:
        logger.error(f"获取模型信息时出错：{str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/models/register', methods=['POST'])
def register_model():
    """注册模型信息到数据库"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请提供模型信息'}), 400
        
        # 验证必需字段
        required_fields = ['model_name', 'model_path']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'缺少必需字段: {field}'}), 400
        
        # 检查模型文件是否存在
        model_path = Path(data['model_path'])
        if not model_path.exists():
            return jsonify({'error': f'模型文件不存在: {data["model_path"]}'}), 400
        
        db = get_database()
        if db.db is None:
            return jsonify({'error': '数据库连接不可用'}), 503
        
        # 准备模型信息，设置默认值
        model_info = {
            'model_name': data['model_name'],
            'model_path': data['model_path'],
            'classes': data.get('classes', []),
            'input_size': data.get('input_size', [640, 640]),
            'model_size': data.get('model_size', os.path.getsize(model_path)),
            'accuracy': data.get('accuracy'),
            'version': data.get('version', '1.0'),
            'description': data.get('description', f'YOLOv5模型 - {data["model_name"]}')
        }
        
        # 注册模型到数据库
        result_id = db.register_model_to_db(model_info)
        
        if result_id:
            return jsonify({
                'success': True,
                'message': '模型注册成功',
                'model_id': str(result_id)
            })
        else:
            return jsonify({'error': '模型注册失败'}), 500
            
    except Exception as e:
        logger.error(f"注册模型时出错：{str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detection-history', methods=['GET'])
def get_detection_history():
    """获取检测历史记录"""
    try:
        limit = int(request.args.get('limit', 50))
        skip = int(request.args.get('skip', 0))
        
        db = get_database()
        if db.db is None:
            return jsonify({'error': '数据库连接不可用'}), 503
            
        # 获取检测历史记录
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
        
        results = list(db.db.detection_records.aggregate(pipeline))
        
        # 转换ObjectId为字符串
        for result in results:
            result['_id'] = str(result['_id'])
            if result.get('original_image_id'):
                result['original_image_id'] = str(result['original_image_id'])
            if result.get('detected_image_id'):
                result['detected_image_id'] = str(result['detected_image_id'])
            
            # 处理关联的图片信息
            for img in result.get('original_image', []):
                img['_id'] = str(img['_id'])
            for img in result.get('detected_image', []):
                img['_id'] = str(img['_id'])
        
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"获取检测历史记录失败：{str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detection-statistics', methods=['GET'])
def get_detection_statistics():
    """获取检测统计信息"""
    try:
        db = get_database()
        if db.db is None:
            return jsonify({'error': '数据库连接不可用'}), 503
            
        # 总检测次数
        total_detections = db.db.detection_records.count_documents({})
        
        # 成功检测次数
        successful_detections = db.db.detection_records.count_documents({"status": "success"})
        
        # 总检测目标数量
        total_objects = db.db.detection_results.count_documents({})
        
        # 按模型统计
        model_stats = list(db.db.detection_records.aggregate([
            {"$group": {
                "_id": "$model_name",
                "count": {"$sum": 1},
                "avg_processing_time": {"$avg": "$processing_time"}
            }}
        ]))
        
        # 按类别统计
        class_stats = list(db.db.detection_results.aggregate([
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
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        logger.error(f"获取统计信息失败：{str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect():
    """处理图像检测请求"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    model_name = request.form.get('model', 'fires.pt')  # 获取选择的模型名称，默认为fires.pt
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型'}), 400

    # 记录开始时间
    start_time = time.time()
    
    # 获取数据库实例
    db = get_database()
    
    # 获取用户信息
    user_ip = request.remote_addr
    user_agent = request.headers.get('User-Agent')
    
    try:
        # 读取上传文件的二进制数据
        if file.filename:
            filename = f"{int(time.time())}_{secure_filename(file.filename)}"
            original_image_data = file.read()
            
            # 确定内容类型
            content_type = file.content_type or 'image/jpeg'
            if not content_type.startswith('image/'):
                # 根据文件扩展名推断内容类型
                ext = file.filename.lower().split('.')[-1]
                content_type_map = {
                    'jpg': 'image/jpeg',
                    'jpeg': 'image/jpeg',
                    'png': 'image/png',
                    'gif': 'image/gif'
                }
                content_type = content_type_map.get(ext, 'image/jpeg')
            
            logger.info(f"原始图片数据大小: {len(original_image_data)} 字节")
            
            # 保存原始图片信息到数据库
            original_image_id = db.save_image_info(
                image_data=original_image_data,
                filename=filename,
                original_filename=file.filename,
                image_type="original",
                content_type=content_type
            )

            # 获取指定模型实例和类别名称
            model, class_names = get_model(model_name)
            
            # 尝试从数据库获取更详细的模型信息
            db_model_info = db.get_model_info_from_db(model_name)
            if db_model_info and db_model_info.get('classes'):
                # 如果数据库中有类别信息，使用数据库中的类别
                db_classes = db_model_info['classes']
                if isinstance(db_classes, list) and len(db_classes) > 0:
                    # 将数据库中的类别列表转换为字典格式（索引->类别名）
                    class_names = {i: cls for i, cls in enumerate(db_classes)}
                    logger.info(f"使用数据库中的模型类别信息：{db_classes}")
            
            logger.info(f"使用模型 {model_name} 进行检测，支持的类别：{class_names}")

            # 进行检测
            from io import BytesIO
            img = Image.open(BytesIO(original_image_data))
            img = np.array(img)
            
            # 处理图像尺寸
            img = cv2.resize(img, (640, 640))
            
            # 转换为 tensor
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # HWC to CHW
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img.unsqueeze(0)  # 添加batch维度

            # 推理
            with torch.no_grad():
                pred = model(img)  # 只使用模型部分
                if isinstance(pred, tuple):
                    pred = pred[0]  # 如果返回元组，取第一个元素
                pred = non_max_suppression(pred, 0.25, 0.45)
            
            # 处理检测结果
            result_data = []
            if len(pred) > 0 and len(pred[0]) > 0:
                for *xyxy, conf, cls in pred[0]:
                    cls_idx = int(cls)
                    result_data.append({
                        'class': cls_idx,
                        'class_name': class_names[cls_idx],
                        'confidence': float(conf),
                        'bbox': [float(x) for x in xyxy]
                    })

            # 在图像上绘制检测结果
            detected_image_data, detected_filename = draw_detections(original_image_data, result_data, file.filename)
            logger.info(f"检测后图片文件名: {detected_filename}")
            
            # 保存检测后图片信息到数据库
            detected_image_id = db.save_image_info(
                image_data=detected_image_data,
                filename=detected_filename,
                original_filename=f"detected_{file.filename}",
                image_type="detected",
                content_type="image/jpeg"
            )
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 生成检测ID
            detection_id = f"det_{int(time.time())}_{hash(filename) % 10000}"
            
            # 保存检测记录到数据库
            detection_record_id = db.save_detection_record(
                detection_id=detection_id,
                original_image_id=original_image_id,
                detected_image_id=detected_image_id,
                model_name=model_name,
                detection_count=len(result_data),
                processing_time=processing_time,
                user_ip=user_ip,
                user_agent=user_agent
            )
            
            # 保存检测结果详情到数据库
            if result_data:
                db.save_detection_results(detection_id, result_data)
            
            # 构建图片URL
            base_url = request.url_root.rstrip('/')
            original_url = f'{base_url}/image/{original_image_id}' if original_image_id else None
            detected_url = f'{base_url}/image/{detected_image_id}' if detected_image_id else None
            
            response_data = {
                'success': True,
                'detection_id': detection_id,
                'detections': result_data,
                'original_image': original_url,
                'detected_image': detected_url,
                'original_image_id': str(original_image_id) if original_image_id else None,
                'detected_image_id': str(detected_image_id) if detected_image_id else None,
                'detection_count': len(result_data),
                'processing_time': round(processing_time, 3),
                'classes': class_names
            }
            logger.info(f"检测完成：找到 {len(result_data)} 个目标，处理时间：{processing_time:.3f}秒")
            return jsonify(response_data)
        else:
            return jsonify({'error': '文件名无效'}), 400

    except Exception as e:
        logger.error(f"处理过程中出错：{str(e)}")
        return jsonify({'error': str(e)}), 500

def auto_register_models():
    """自动扫描并注册model目录中的模型文件到数据库"""
    try:
        db = get_database()
        if db.db is None:
            logger.warning("数据库连接不可用，跳过自动注册模型")
            return
        
        model_dir = ROOT / 'model'
        if not model_dir.exists():
            logger.warning(f"模型目录不存在：{model_dir}")
            return
        
        model_files = list(model_dir.glob('*.pt'))
        logger.info(f"发现 {len(model_files)} 个模型文件")
        
        for model_file in model_files:
            model_name = model_file.name
            
            # 检查数据库中是否已存在该模型
            existing_model = db.get_model_info_from_db(model_name)
            if existing_model:
                logger.info(f"模型 {model_name} 已存在于数据库中，跳过注册")
                continue
            
            try:
                # 尝试加载模型以获取类别信息
                temp_model = attempt_load(str(model_file), device='cpu')
                if hasattr(temp_model, 'names'):
                    classes = list(temp_model.names.values()) if isinstance(temp_model.names, dict) else list(temp_model.names)
                else:
                    classes = CLASS_NAMES
                
                # 准备模型信息
                model_info = {
                    "model_name": model_name,
                    "model_path": str(model_file),
                    "classes": classes,
                    "input_size": [640, 640],
                    "model_size": model_file.stat().st_size,
                    "accuracy": None,
                    "version": "1.0",
                    "description": f"自动注册的YOLOv5模型 - {model_name}"
                }
                
                # 注册到数据库
                result_id = db.register_model_to_db(model_info)
                if result_id:
                    logger.info(f"成功注册模型：{model_name}")
                else:
                    logger.warning(f"注册模型失败：{model_name}")
                    
                # 清理临时模型对象
                del temp_model
                
            except Exception as e:
                logger.error(f"处理模型 {model_name} 时出错：{str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"自动注册模型时出错：{str(e)}")

def cleanup_on_exit():
    """应用退出时的清理工作"""
    try:
        db = get_database()
        if db:
            db.close()
            logger.info("应用退出，数据库连接已关闭")
    except Exception as e:
        logger.error(f"清理数据库连接时出错：{str(e)}")

if __name__ == '__main__':
    try:
        # 初始化数据库连接
        logger.info("正在初始化数据库连接...")
        db = get_database()
        if db.db is not None:
            logger.info("数据库连接成功")
            # 自动注册模型到数据库
            logger.info("正在自动注册模型到数据库...")
            auto_register_models()
        else:
            logger.warning("数据库连接失败，应用将在无数据库模式下运行")
        
        # 测试模型
        if not test_model():
            logger.error("模型测试失败，服务不会启动")
            sys.exit(1)
        
        # 从环境变量获取Flask配置
        port = int(os.getenv('FLASK_PORT', 5000))
        host = os.getenv('FLASK_HOST', '127.0.0.1')
        
        # 检查端口是否被占用
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
        except socket.error:
            logger.warning(f"端口 {port} 已被占用，尝试使用端口 5001")
            port = 5001
        finally:
            sock.close()
        
        # 注册退出时的清理函数
        import atexit
        atexit.register(cleanup_on_exit)

        logger.info(f"启动生产服务器在 http://{host}:{port}")
        serve(app, host=host, port=port, threads=4)
            
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务器...")
        cleanup_on_exit()
    except Exception as e:
        logger.error(f"启动失败：{str(e)}")
        cleanup_on_exit()
        sys.exit(1)
