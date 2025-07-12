# YOLOv5 目标检测 API 文档

## 基础信息

- 基础URL: `http://{host}:{port}`
- 默认端口: 5000
- 所有响应均为JSON格式
- 图片上传大小限制: 16MB
- 支持的图片格式: PNG, JPG, JPEG

## API 端点

### 1. 主页

#### 请求
- 方法: `GET`
- 路径: `/`
- 描述: 返回Web界面主页

### 2. 健康检查

#### 请求
- 方法: `GET`
- 路径: `/health`
- 描述: 检查服务运行状态

#### 响应
```json
{
    "status": "online",
    "current_model": "fires.pt",
    "timestamp": "2024-01-01 12:00:00",
    "service": "YOLOv5 Detection API"
}
```

### 3. 获取可用模型列表

#### 请求
- 方法: `GET`
- 路径: `/models`
- 描述: 获取所有可用的模型列表

#### 响应
```json
{
    "success": true,
    "models": [
        {
            "model_name": "fires.pt",
            "classes": ["fire", "smoke"],
            "input_size": [640, 640],
            "model_size": 12345678,
            "version": "1.0"
        }
    ],
    "source": "database"
}
```

### 4. 获取模型信息

#### 请求
- 方法: `GET`
- 路径: `/model-info`
- 参数:
  - `model`: 模型名称（可选，默认为"fires.pt"）

#### 响应
```json
{
    "success": true,
    "model_info": {
        "model_name": "fires.pt",
        "classes": ["fire", "smoke"],
        "status": "已加载",
        "input_size": [640, 640],
        "model_size": 12345678,
        "version": "1.0",
        "description": "火灾检测模型"
    }
}
```

### 5. 注册模型

#### 请求
- 方法: `POST`
- 路径: `/models/register`
- Content-Type: `application/json`
- 请求体:
```json
{
    "model_name": "fires.pt",
    "model_path": "/path/to/model/fires.pt",
    "classes": ["fire", "smoke"],
    "input_size": [640, 640],
    "version": "1.0",
    "description": "火灾检测模型"
}
```

#### 响应
```json
{
    "success": true,
    "message": "模型注册成功",
    "model_id": "12345678"
}
```

### 6. 目标检测

#### 请求
- 方法: `POST`
- 路径: `/detect`
- Content-Type: `multipart/form-data`
- 参数:
  - `file`: 图片文件
  - `model`: 模型名称（可选，默认为"fires.pt"）

#### 响应
```json
{
    "success": true,
    "detection_id": "det_1234567890",
    "detections": [
        {
            "class": 0,
            "class_name": "fire",
            "confidence": 0.95,
            "bbox": [100, 200, 300, 400]
        }
    ],
    "original_image": "http://host:port/image/12345",
    "detected_image": "http://host:port/image/12346",
    "detection_count": 1,
    "processing_time": 0.234,
    "classes": ["fire", "smoke"]
}
```

### 7. 获取图片

#### 请求
- 方法: `GET`
- 路径: `/image/{image_id}`
- 描述: 获取指定ID的图片

#### 响应
- Content-Type: image/jpeg 或 image/png
- 二进制图片数据

### 8. 获取检测历史

#### 请求
- 方法: `GET`
- 路径: `/detection-history`
- 参数:
  - `limit`: 返回记录数量（可选，默认50）
  - `skip`: 跳过记录数量（可选，默认0）

#### 响应
```json
{
    "success": true,
    "data": [
        {
            "detection_id": "det_1234567890",
            "model_name": "fires.pt",
            "detection_count": 1,
            "processing_time": 0.234,
            "created_at": "2024-01-01 12:00:00",
            "original_image_id": "12345",
            "detected_image_id": "12346"
        }
    ],
    "count": 1
}
```

### 9. 获取检测统计信息

#### 请求
- 方法: `GET`
- 路径: `/detection-statistics`

#### 响应
```json
{
    "success": true,
    "statistics": {
        "total_detections": 100,
        "successful_detections": 98,
        "total_objects_detected": 150,
        "success_rate": 0.98,
        "model_statistics": [
            {
                "_id": "fires.pt",
                "count": 100,
                "avg_processing_time": 0.245
            }
        ],
        "class_statistics": [
            {
                "_id": "fire",
                "count": 80,
                "avg_confidence": 0.92
            }
        ]
    }
}
```

## 错误响应

所有API在发生错误时都会返回以下格式的响应：

```json
{
    "error": "错误描述信息"
}
```

常见HTTP状态码：
- 200: 请求成功
- 400: 请求参数错误
- 404: 资源不存在
- 500: 服务器内部错误
- 503: 服务不可用（如数据库连接失败）

## 注意事项

1. 图片上传大小限制为16MB
2. 仅支持PNG、JPG、JPEG格式的图片
3. 所有时间戳均为UTC+8时区
4. 检测结果中的边界框坐标格式为[x1, y1, x2, y2]
5. 置信度范围为0-1之间的浮点数 