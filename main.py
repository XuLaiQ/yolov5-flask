import os
import sys
from pathlib import Path
import logging

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
from flask import Flask, request, jsonify, render_template, send_from_directory
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
UPLOAD_FOLDER = Path('static/uploads')  # 原始图片存储路径
DETECTED_FOLDER = Path('static/detected')  # 检测后图片存储路径
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['DETECTED_FOLDER'] = str(DETECTED_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 确保上传目录存在
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DETECTED_FOLDER.mkdir(parents=True, exist_ok=True)

# 类别名称
CLASS_NAMES = ['fire']  # 根据您的模型类别进行修改

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

def draw_detections(image_path, detections):
    """在图像上绘制检测结果"""
    try:
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像：{image_path}")
            
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
        
        # 生成检测后的图片文件名
        filename = Path(image_path).name
        detected_filename = f"detected_{filename}"
        output_path = DETECTED_FOLDER / detected_filename
        
        # 保存标注后的图像
        result_image = cv2.cvtColor(annotator.result(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), result_image)
        logger.info(f"检测结果已保存到：{output_path}")
        return detected_filename
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

@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件服务"""
    return send_from_directory('static', filename)

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
        models = get_available_models()
        return jsonify({'models': models})
    except Exception as e:
        logger.error(f"获取模型列表时出错：{str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """获取当前加载的模型信息"""
    try:
        model_name = request.args.get('model', 'fires.pt')
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
            'status': '已加载'
        })
    except Exception as e:
        logger.error(f"获取模型信息时出错：{str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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

    try:
        # 保存上传的文件
        if file.filename:
            filename = f"{int(time.time())}_{secure_filename(file.filename)}"
            filepath = UPLOAD_FOLDER / filename
            file.save(str(filepath))
            logger.info(f"原始图片保存路径: {filepath}")

            # 获取指定模型实例和类别名称
            model, class_names = get_model(model_name)
            logger.info(f"使用模型 {model_name} 进行检测，支持的类别：{class_names}")

            # 进行检测
            img = Image.open(filepath)
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
            detected_filename = draw_detections(filepath, result_data)
            logger.info(f"检测后图片文件名: {detected_filename}")
            
            # 构建返回的URL
            original_url = f'/static/uploads/{filename}'
            detected_url = f'/static/detected/{detected_filename}'
            
            response_data = {
                'success': True,
                'detections': result_data,
                'original_image': original_url,
                'detected_image': detected_url,
                'detection_count': len(result_data),
                'classes': class_names
            }
            logger.info(f"检测完成：找到 {len(result_data)} 个目标")
            return jsonify(response_data)
        else:
            return jsonify({'error': '文件名无效'}), 400

    except Exception as e:
        logger.error(f"处理过程中出错：{str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # 测试模型
        if not test_model():
            logger.error("模型测试失败，服务不会启动")
            sys.exit(1)
        
        port = 5000
        host = '127.0.0.1'
        # port = int(os.environ.get('PORT', 5000))
        # host = os.environ.get('HOST', '0.0.0.0')  # 修改为0.0.0.0以支持外网访问
        
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
        
        # 根据环境选择服务器
        # if os.environ.get('FLASK_ENV') == 'development':
        #     # 开发环境使用 Flask 开发服务器
        #     logger.info(f"启动开发服务器在 http://{host}:{port}")
        #     app.run(debug=True, host=host, port=port, threaded=True)
        # else:
        #     # 生产环境使用 Waitress 服务器
        #     logger.info(f"启动生产服务器在 http://{host}:{port}")
        #     serve(app, host=host, port=port, threads=4)
        # 使用生产环境服务器
        logger.info(f"启动生产服务器在 http://{host}:{port}")
        serve(app, host=host, port=port, threads=4)
            
    except Exception as e:
        logger.error(f"启动失败：{str(e)}")
        sys.exit(1)
