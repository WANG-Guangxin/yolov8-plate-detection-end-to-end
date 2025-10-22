> 本项目完全有 AI 制作

# 基于 YOLOv8 的中文车牌字符检测与识别服务

## 启动服务

### 1. 使用默认配置启动

```bash
# 启动服务（使用默认配置）
python3 api_server.py
```

默认配置：
- 模型文件: `char_detection/yolov8_chars2/weights/best.pt`
- 字符集文件: `char_detection_data_balanced/charset.json`
- 服务地址: `http://0.0.0.0:7389`

### 2. 自定义配置启动

```bash
python3 api_server.py --model <模型路径> --charset <字符集路径> --port <端口> --host <主机地址>
```

参数说明：
- `--model`: YOLOv8 模型文件路径
- `--charset`: 字符集 JSON 文件路径
- `--host`: 服务器主机地址（默认: 0.0.0.0）
- `--port`: 服务器端口（默认: 7389）
- `--debug`: 启用调试模式

### 3. 启动成功示例

```
============================================================
车牌字符识别 Flask API 服务
============================================================
模型文件: char_detection/yolov8_chars2/weights/best.pt
字符集文件: char_detection_data_balanced/charset.json
服务地址: http://0.0.0.0:7389
============================================================
API端点:
  POST http://0.0.0.0:7389/api/detect - 车牌字符检测
  GET  http://0.0.0.0:7389/api/health - 健康检查
  GET  http://0.0.0.0:7389/api/info - API信息
============================================================
```

## API 说明

### 端点列表

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/detect` | POST | 车牌字符检测和识别 |
| `/api/health` | GET | 服务健康检查 |
| `/api/info` | GET | API 信息和版本 |

### 1. 车牌字符检测 `/api/detect`

#### 请求方式 1：文件上传

```bash
curl -X POST -F "image=@车牌图片.jpg" http://localhost:7389/api/detect
```

#### 请求方式 2：Base64 编码

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."}' \
     http://localhost:7389/api/detect
```

#### 可选参数

- `confidence`: 置信度阈值（0.0-1.0，默认 0.5）

```bash
# 设置置信度阈值为 0.3
curl -X POST -F "image=@车牌图片.jpg" -F "confidence=0.3" http://localhost:7389/api/detect
```

#### 响应示例

成功响应：
```json
{
  "success": true,
  "plate_number": "皖DU2017",
  "message": "识别成功"
}
```

失败响应：
```json
{
  "success": false,
  "error": "图片读取失败",
  "message": "无法解析图片数据"
}
```

### 2. 健康检查 `/api/health`

```bash
curl http://localhost:7389/api/health
```

响应示例：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "车牌字符识别API服务正常运行"
}
```

### 3. API 信息 `/api/info`

```bash
curl http://localhost:7389/api/info
```

响应示例：
```json
{
  "name": "车牌字符识别API",
  "version": "1.0.0",
  "description": "基于YOLOv8的中文车牌字符检测与识别服务",
  "endpoints": {
    "/api/detect": "POST - 车牌字符检测",
    "/api/health": "GET - 健康检查",
    "/api/info": "GET - API信息"
  },
  "supported_formats": ["png", "jpg", "jpeg", "bmp", "tiff"],
  "max_file_size": "16MB"
}
```

## 支持的图片格式

- PNG
- JPG/JPEG
- BMP
- TIFF

最大文件大小：16MB

## 常见问题

### 1. 模型文件不存在

确保模型文件路径正确：
```bash
ls -la char_detection/yolov8_chars2/weights/best.pt
```

### 2. 字符集文件不存在

确保字符集文件路径正确：
```bash
ls -la char_detection_data_balanced/charset.json
```

### 3. 端口被占用

可以更换端口：
```bash
python3 api_server.py --port 8080
```

### 4. 依赖包缺失

重新安装依赖：
```bash
pip install -r requirements.txt
```

## 部署到其他机器

### 1. 导出环境（在开发机器上）

```bash
# 导出 conda 环境
conda env export > environment.yml

# 或导出 pip 依赖
pip freeze > requirements.txt
```

### 2. 在目标机器上部署

```bash
# 拷贝项目文件到目标机器
# 包括：代码文件、模型文件、环境配置文件

# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate yolo-plate-detection

# 启动服务
python3 api_server.py
```

## 开发和调试

### 启用调试模式

```bash
python3 api_server.py --debug
```

调试模式特性：
- 代码修改后自动重启
- 详细的错误信息
- Flask 调试工具

### 查看日志

服务运行时会在终端输出详细日志，包括：
- 模型加载状态
- 请求处理过程
- 错误信息

## 性能优化

1. **GPU 加速**：如果有 GPU，YOLOv8 会自动使用 GPU 加速
2. **批量处理**：对于大量图片，建议使用批量 API
3. **置信度调节**：根据实际需求调整置信度阈值
