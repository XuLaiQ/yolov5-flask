# YOLOv5 目标检测系统

这是一个基于YOLOv5和Flask的智能目标检测Web应用系统。该系统支持多模型切换、实时图像检测、数据库存储和检测历史管理等功能，主要用于火灾检测，同时支持扩展到其他目标检测任务。

## 项目结构

```
yolov5-flask/
├── .gitignore              # Git忽略文件配置
├── README.md               # 项目说明文档
├── main.py                 # 主应用程序
├── app.log                 # 应用日志文件
├── requirements.txt        # 项目依赖
├── database_init.py        # 数据库初始化脚本
├── database_schema.sql     # 数据库结构设计
├── database_utils.py       # 数据库工具函数
├── gunicorn_config.py      # Gunicorn配置文件
├── model/                  # 模型目录
│   └── fires.pt           # 训练好的火灾检测模型
├── templates/             # 前端模板目录
│   └── index.html        # 主页面模板
└── yolov5-master/        # YOLOv5源代码
    ├── models/           # YOLOv5模型定义
    └── utils/            # YOLOv5工具函数
```

## 技术栈

### 后端技术
- **Python 3.x** - 主要开发语言
- **Flask 3.0.2** - Web框架
- **PyTorch 2.2.0** - 深度学习框架
- **YOLOv5** - 目标检测模型
- **Waitress 3.0.0** - WSGI服务器

### 数据库
- **MongoDB** - 主数据库（图片存储、检测记录）
- **PyMongo 4.0+** - MongoDB Python驱动

### 图像处理
- **OpenCV** - 图像处理库
- **Pillow** - 图像操作库
- **NumPy** - 数值计算库

### 前端技术
- **HTML5/CSS3** - 页面结构和样式
- **JavaScript (ES6+)** - 前端交互逻辑
- **Bootstrap风格** - 响应式UI设计

### 开发工具
- **python-dotenv** - 环境变量管理
- **Werkzeug** - WSGI工具库

## 核心功能特性

### 🎯 智能检测功能
- **多模型支持**: 支持动态切换不同的YOLOv5模型
- **实时检测**: 上传图片后实时进行目标检测
- **高精度识别**: 基于YOLOv5的高精度目标检测算法
- **置信度显示**: 显示每个检测目标的置信度分数
- **边界框标注**: 自动在检测结果上绘制边界框和标签

### 💾 数据管理功能
- **图片存储**: 原始图片和检测结果图片存储在MongoDB中
- **检测记录**: 完整的检测历史记录和统计信息
- **模型管理**: 自动注册和管理多个检测模型
- **数据统计**: 检测次数、成功率、处理时间等统计分析

### 🌐 Web界面功能
- **响应式设计**: 支持桌面和移动设备访问
- **拖拽上传**: 支持拖拽文件上传和点击上传
- **实时预览**: 原始图片和检测结果的对比显示
- **模型切换**: 前端动态选择和切换检测模型
- **加载动画**: 优雅的加载状态提示

### 🔧 系统管理功能
- **健康检查**: 系统状态监控API
- **日志记录**: 详细的操作日志和错误追踪
- **环境配置**: 通过环境变量灵活配置系统参数
- **自动容错**: 端口占用检测和自动切换

### 📊 API接口功能
- **RESTful API**: 标准的REST API设计
- **图片检测API**: `/detect` - 图片上传和检测
- **模型信息API**: `/models`, `/model-info` - 模型管理
- **历史记录API**: `/detection-history` - 检测历史查询
- **统计信息API**: `/detection-statistics` - 数据统计分析
- **图片服务API**: `/image/<id>` - 图片数据获取

## 系统架构设计

### 1. 整体架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   前端界面      │    │   Flask后端     │    │   MongoDB数据库 │
│                 │    │                 │    │                 │
│ • HTML/CSS/JS   │◄──►│ • REST API      │◄──►│ • 图片存储      │
│ • 响应式设计    │    │ • 业务逻辑      │    │ • 检测记录      │
│ • 文件上传      │    │ • 模型管理      │    │ • 统计数据      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   YOLOv5模型    │
                       │                 │
                       │ • 目标检测      │
                       │ • 图像处理      │
                       │ • 结果标注      │
                       └─────────────────┘
```

### 2. 核心组件

#### 2.1 数据库设计
- **DetectionDatabase类**: 统一的数据库操作接口
- **图片存储**: 二进制数据直接存储在MongoDB中
- **检测记录**: 完整的检测历史和元数据
- **模型管理**: 自动注册和版本管理
- **统计分析**: 实时数据统计和性能监控

#### 2.2 模型管理系统
- **动态加载**: 支持运行时切换不同模型
- **自动注册**: 启动时自动扫描并注册模型文件
- **版本管理**: 模型版本信息和元数据管理
- **性能监控**: 模型推理时间和准确率统计

#### 2.3 图像处理流水线
- **预处理**: 图像尺寸标准化和格式转换
- **推理**: YOLOv5模型目标检测
- **后处理**: 非极大值抑制和结果过滤
- **可视化**: 边界框绘制和标签标注

### 3. API接口设计

#### 3.1 核心检测接口
```
POST /detect
- 功能: 图片上传和目标检测
- 参数: file (图片文件), model (模型名称)
- 返回: 检测结果、图片URL、统计信息
```

#### 3.2 模型管理接口
```
GET /models              # 获取可用模型列表
GET /model-info          # 获取模型详细信息
POST /models/register    # 注册新模型到数据库
```

#### 3.3 数据查询接口
```
GET /detection-history      # 获取检测历史记录
GET /detection-statistics   # 获取统计信息
GET /image/<id>            # 获取图片数据
GET /health                # 系统健康检查
```

## 环境配置

### 1. 环境变量配置

创建 `.env` 文件配置数据库连接：

```env
# MongoDB配置
MONGODB_USERNAME=your_username
MONGODB_PASSWORD=your_password
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE=yolov5_detection

# Flask配置
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
FLASK_ENV=production
```

### 2. 数据库初始化

```bash
# 启动MongoDB服务
# Windows: net start MongoDB
# Linux/Mac: sudo systemctl start mongod

# 运行数据库初始化脚本
python database_init.py
```

## 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd yolov5-flask

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型准备

```bash
# 创建模型目录
mkdir model

# 下载或放置YOLOv5模型文件到model目录
# 例如：fires.pt（火灾检测模型）
```

### 3. 启动服务

```bash
# 启动应用
python main.py

# 访问应用
# 浏览器打开: http://127.0.0.1:5000
```

## 使用指南

### 1. Web界面操作

#### 1.1 模型选择
- 在页面顶部选择要使用的检测模型
- 系统会显示模型状态和支持的检测类别
- 支持动态切换不同的模型

#### 1.2 图像上传
- **拖拽上传**：将图片文件拖拽到上传区域
- **点击上传**：点击上传区域选择文件
- **支持格式**：PNG、JPG、JPEG、GIF
- **文件大小**：建议不超过10MB

#### 1.3 检测结果
- **原始图像**：显示上传的原始图片
- **检测结果**：显示标注后的图片，包含边界框和标签
- **检测统计**：显示检测到的目标数量和详细信息
- **置信度显示**：每个检测目标都会显示置信度分数

### 2. API接口使用

#### 2.1 图像检测API
```bash
# 使用curl进行检测
curl -X POST -F "file=@image.jpg" -F "model=fires.pt" http://127.0.0.1:5000/detect
```

#### 2.2 模型信息API
```bash
# 获取可用模型列表
curl http://127.0.0.1:5000/models

# 获取当前模型信息
curl http://127.0.0.1:5000/model-info
```

#### 2.3 检测历史API
```bash
# 获取检测历史
curl http://127.0.0.1:5000/detection-history?page=1&limit=10

# 获取检测统计
curl http://127.0.0.1:5000/detection-statistics
```

## 注意事项

### 系统要求
- **Python版本**：3.8+
- **内存要求**：建议至少8GB RAM
- **存储空间**：至少2GB可用空间
- **网络连接**：首次运行需要下载依赖

### 模型要求
- **格式支持**：YOLOv5 PyTorch模型（.pt格式）
- **版本兼容**：确保模型与当前YOLOv5版本兼容
- **GPU加速**：支持CUDA的NVIDIA GPU可显著提升性能

### 使用限制
- **文件大小**：单个图片建议不超过10MB
- **并发处理**：默认支持4个并发请求
- **存储清理**：建议定期清理上传和检测结果文件

## 系统优化建议

### 性能优化
- **GPU加速**：安装CUDA版本的PyTorch
- **模型优化**：使用TensorRT或ONNX优化推理速度
- **缓存策略**：使用Redis缓存模型信息和检测结果
- **负载均衡**：生产环境建议使用Nginx + Gunicorn

### 安全加固
- **文件验证**：严格验证上传文件类型和内容
- **访问控制**：添加API访问频率限制
- **数据加密**：生产环境使用HTTPS
- **定期更新**：及时更新依赖包和安全补丁

### 扩展功能
- **多模型并行**：支持同时运行多个检测模型
- **批量处理**：实现图片批量上传和检测
- **实时检测**：集成摄像头实时检测功能
- **数据分析**：添加检测数据统计和可视化
- **用户系统**：实现用户注册、登录和权限管理

## 常见问题

### Q: 模型加载失败怎么办？
A: 检查模型文件是否存在于 `model/` 目录，确保模型格式为 `.pt` 且与YOLOv5版本兼容。

### Q: 检测速度很慢怎么优化？
A: 建议安装CUDA版本的PyTorch，使用GPU加速。也可以考虑使用更小的模型或降低输入图像分辨率。

### Q: 数据库连接失败？
A: 确保MongoDB服务已启动，检查 `.env` 文件中的数据库配置是否正确。

### Q: 如何添加新的检测模型？
A: 将新模型文件放入 `model/` 目录，重启应用后系统会自动注册新模型。

## 技术支持

如果您在使用过程中遇到问题，可以通过以下方式获取帮助：

1. **查看日志**：检查应用运行日志获取错误信息
2. **检查配置**：确认环境变量和数据库配置正确
3. **更新依赖**：尝试更新到最新版本的依赖包
4. **重启服务**：重启应用和数据库服务

## 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- [YOLOv5](https://github.com/ultralytics/yolov5) - 核心检测算法
- [Flask](https://flask.palletsprojects.com/) - Web框架
- [MongoDB](https://www.mongodb.com/) - 数据库支持
- [PyTorch](https://pytorch.org/) - 深度学习框架