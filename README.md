# YOLOv5 火灾检测系统

这是一个基于YOLOv5和Flask的火灾检测Web应用系统。该系统能够通过上传图片来检测图片中是否存在火灾情况。

## 项目结构

```
yolov5-flask/
├── app.log                 # 应用日志文件
├── main.py                 # 主应用程序
├── model/                  # 模型目录
│   └── fires.pt           # 训练好的火灾检测模型
├── requirements.txt        # 项目依赖
├── static/                 # 静态资源目录
│   ├── detected/          # 检测后的图片存储目录
│   └── uploads/           # 上传图片存储目录
├── templates/             # 模板目录
│   └── index.html        # 主页面模板
└── yolov5-master/        # YOLOv5源代码
```

## 技术栈

- Python 3.x
- Flask 3.0.2
- PyTorch
- YOLOv5
- OpenCV
- Pillow
- Waitress (生产环境服务器)

## 实现流程

### 1. 系统初始化

#### 1.1 配置日志系统
```python
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)
```

#### 1.2 设置项目路径
```python
# 设置项目根目录
FILE = Path(__file__).resolve()
ROOT = FILE.parent

# YOLOv5路径设置
YOLOV5_PATH = ROOT / 'yolov5-master'
if str(YOLOV5_PATH) not in sys.path:
    sys.path.append(str(YOLOV5_PATH))
```

#### 1.3 初始化Flask应用
```python
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
```

### 2. 模型管理

#### 2.1 模型加载
```python
def load_model():
    """加载YOLOv5模型"""
    global model
    model_path = ROOT / 'model' / 'fires.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")
    try:
        logger.info(f"正在加载模型：{model_path}")
        model = attempt_load(model_path, device='cpu')
        model.eval()
        logger.info("模型加载成功")
        return model
    except Exception as e:
        logger.error(f"加载模型时出错：{str(e)}")
        raise

def get_model():
    """获取或加载模型实例"""
    global model
    if model is None:
        model = load_model()
    return model
```

#### 2.2 模型测试
```python
def test_model():
    """测试模型是否正常工作"""
    try:
        # 加载模型
        model = get_model()
        logger.info("模型加载成功！")
        
        # 创建测试输入
        dummy_input = torch.zeros((1, 3, 640, 640))
        
        # 测试推理
        with torch.no_grad():
            result = model(dummy_input)
        logger.info("模型推理测试成功！")
        return True
    except Exception as e:
        logger.error(f"模型测试失败：{str(e)}")
        return False
```

### 3. 图像处理流程

#### 3.1 图像上传验证
```python
def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/detect', methods=['POST'])
def detect():
    """处理图像检测请求"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型'}), 400
```

#### 3.2 图像预处理
```python
# 保存上传的文件
filename = f"{int(time.time())}_{secure_filename(file.filename)}"
filepath = UPLOAD_FOLDER / filename
file.save(str(filepath))

# 图像预处理
img = Image.open(filepath)
img = np.array(img)
img = cv2.resize(img, (640, 640))
img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # HWC to CHW
img /= 255.0  # 0 - 255 to 0.0 - 1.0
if len(img.shape) == 3:
    img = img.unsqueeze(0)  # 添加batch维度
```

#### 3.3 目标检测
```python
# 推理
with torch.no_grad():
    pred = current_model(img)
    if isinstance(pred, tuple):
        pred = pred[0]  # 如果返回元组，取第一个元素
    pred = non_max_suppression(pred, 0.25, 0.45)

# 处理检测结果
result_data = []
if len(pred) > 0 and len(pred[0]) > 0:
    for *xyxy, conf, cls in pred[0]:
        result_data.append({
            'class': int(cls),
            'confidence': float(conf),
            'bbox': [float(x) for x in xyxy]
        })
```

#### 3.4 结果可视化
```python
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
            cls = det['class']
            
            # 转换坐标为整数
            x1, y1, x2, y2 = map(int, bbox)
            
            # 使用固定的颜色
            color = (0, 255, 0)  # 使用绿色作为边界框颜色
            
            # 绘制边界框和标签
            label = f'{CLASS_NAMES[cls]} {conf:.2f}'
            annotator.box_label([x1, y1, x2, y2], label, color=color)
        
        # 生成检测后的图片文件名
        filename = Path(image_path).name
        detected_filename = f"detected_{filename}"
        output_path = DETECTED_FOLDER / detected_filename
        
        # 保存标注后的图像
        result_image = cv2.cvtColor(annotator.result(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), result_image)
        return detected_filename
    except Exception as e:
        logger.error(f"绘制检测结果时出错：{str(e)}")
        raise
```

### 4. Web服务

#### 4.1 路由设计
```python
@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件服务"""
    return send_from_directory('static', filename)

@app.route('/detect', methods=['POST'])
def detect():
    """处理图像检测请求"""
    # ... 检测代码 ...
```

#### 4.2 服务器配置
```python
if __name__ == '__main__':
    try:
        # 测试模型
        if not test_model():
            logger.error("模型测试失败，服务不会启动")
            sys.exit(1)
        
        port = 5000
        host = '127.0.0.1'
        
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
        if os.environ.get('FLASK_ENV') == 'development':
            # 开发环境使用 Flask 开发服务器
            logger.info(f"启动开发服务器在 http://{host}:{port}")
            app.run(debug=True, host=host, port=port, threaded=True)
        else:
            # 生产环境使用 Waitress 服务器
            logger.info(f"启动生产服务器在 http://{host}:{port}")
            serve(app, host=host, port=port, threads=4)
            
    except Exception as e:
        logger.error(f"启动失败：{str(e)}")
        sys.exit(1)
```

## 使用说明

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 启动服务器：
```bash
python main.py
```

3. 访问应用：
   - 开发环境：http://127.0.0.1:5000
   - 如果5000端口被占用，将自动使用5001端口

4. 使用流程：
   - 打开网页
   - 点击上传图片
   - 等待检测结果
   - 查看检测后的图片和详细信息

## 注意事项

1. 确保模型文件 `fires.pt` 已放置在 `model` 目录下
2. 上传图片大小不要超过16MB
3. 仅支持png、jpg、jpeg格式的图片
4. 生产环境部署时建议使用反向代理（如Nginx）

## 系统优化

1. 性能优化
   - 使用Waitress多线程处理请求
   - 图片尺寸标准化
   - 模型单例模式避免重复加载

2. 安全性
   - 文件类型验证
   - 文件大小限制
   - 安全的文件名处理

3. 可靠性
   - 完善的错误处理
   - 详细的日志记录
   - 服务器自动容错 