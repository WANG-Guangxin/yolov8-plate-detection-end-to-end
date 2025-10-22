#!/usr/bin/env python3
"""
Flask API for License Plate Character Recognition
车牌字符识别 Flask API 服务 - 简化版
"""

import os
import cv2
import json
import base64
import numpy as np
from flask import Flask, request
from werkzeug.utils import secure_filename
import tempfile
from pathlib import Path
from ultralytics import YOLO
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 配置
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 创建上传目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class PlateCharacterAPI:
    def __init__(self, model_path, charset_path=None):
        """初始化车牌字符识别API"""
        logger.info(f"加载模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 加载字符集
        self.class_to_char = {}
        if charset_path and os.path.exists(charset_path):
            with open(charset_path, 'r', encoding='utf-8') as f:
                charset_info = json.load(f)
            self.class_to_char = {int(k): v for k, v in charset_info['class_to_char'].items()}
            logger.info(f"加载字符集: {len(self.class_to_char)} 个字符")
        else:
            logger.warning("未找到字符集文件，将使用类别ID")
        
        logger.info("车牌字符识别API初始化完成")
    
    def detect_characters(self, image, conf_threshold=0.5):
        """检测图片中的车牌字符"""
        try:
            # 进行检测
            results = self.model(image, conf=conf_threshold)
            
            # 处理检测结果
            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # 转换为字符
                        char = self.class_to_char.get(class_id, f"class_{class_id}")
                        
                        detection = {
                            "confidence": conf,
                            "character": char,
                            "center_x": float((x1 + x2) / 2)
                        }
                        detections.append(detection)
            
            # 按 x 坐标排序（从左到右）
            detections.sort(key=lambda x: x["center_x"])
            
            # 组合车牌号码
            plate_number = ""
            for detection in detections:
                plate_number += detection["character"]
            
            logger.info(f"检测成功: {plate_number} ({len(detections)} 个字符)")
            return {
                "success": True,
                "plate_number": plate_number,
                "message": "识别成功"
            }
            
        except Exception as e:
            logger.error(f"检测失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "字符检测失败"
            }

# 全局检测器实例
detector = None

def json_response(data, status_code=200):
    """返回UTF-8编码的JSON响应"""
    return app.response_class(
        response=json.dumps(data, ensure_ascii=False, indent=2),
        status=status_code,
        mimetype='application/json; charset=utf-8'
    )

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def decode_base64_image(base64_string):
    """解码base64图片"""
    try:
        # 移除data:image/jpeg;base64,前缀（如果有的话）
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # 解码base64
        img_data = base64.b64decode(base64_string)
        
        # 转换为numpy数组
        nparr = np.frombuffer(img_data, np.uint8)
        
        # 解码为OpenCV图像
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        logger.error(f"Base64图片解码失败: {str(e)}")
        return None

@app.route('/api/detect', methods=['POST'])
def detect_plate():
    """车牌字符检测API"""
    if detector is None:
        return json_response({
            "success": False,
            "error": "模型未加载",
            "message": "服务器模型未正确初始化"
        }, 500)
    
    try:
        image = None
        
        # 方式1：文件上传
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return json_response({
                    "success": False,
                    "error": "未选择文件",
                    "message": "请选择要上传的图片文件"
                }, 400)
            
            if file and allowed_file(file.filename):
                # 保存临时文件
                filename = secure_filename(file.filename)
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(temp_path)
                
                # 读取图片
                image = cv2.imread(temp_path)
                
                # 删除临时文件
                try:
                    os.remove(temp_path)
                except:
                    pass
            else:
                return json_response({
                    "success": False,
                    "error": "不支持的文件格式",
                    "message": f"支持的格式: {', '.join(ALLOWED_EXTENSIONS)}"
                }, 400)
        
        # 方式2：Base64编码
        elif request.is_json:
            data = request.get_json()
            if 'image_base64' in data:
                image = decode_base64_image(data['image_base64'])
            else:
                return json_response({
                    "success": False,
                    "error": "缺少图片数据",
                    "message": "请提供'image_base64'字段"
                }, 400)
        
        else:
            return json_response({
                "success": False,
                "error": "无效的请求格式",
                "message": "请使用文件上传或Base64编码格式"
            }, 400)
        
        # 检查图片是否成功读取
        if image is None:
            return json_response({
                "success": False,
                "error": "图片读取失败",
                "message": "无法解析图片数据"
            }, 400)
        
        # 获取置信度阈值
        conf_threshold = float(request.form.get('confidence', 0.5)) if not request.is_json else \
                        float(request.get_json().get('confidence', 0.5))
        
        # 进行字符检测
        result = detector.detect_characters(image, conf_threshold)
        
        if result["success"]:
            return json_response(result, 200)
        else:
            return json_response(result, 500)
            
    except Exception as e:
        logger.error(f"API请求处理失败: {str(e)}")
        return json_response({
            "success": False,
            "error": str(e),
            "message": "服务器内部错误"
        }, 500)

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查API"""
    return json_response({
        "status": "healthy",
        "model_loaded": detector is not None,
        "message": "车牌字符识别API服务正常运行"
    })

@app.route('/api/info', methods=['GET'])
def get_info():
    """获取API信息"""
    return json_response({
        "name": "车牌字符识别API",
        "version": "1.0.0",
        "description": "基于YOLOv8的中文车牌字符检测与识别服务",
        "endpoints": {
            "/api/detect": "POST - 车牌字符检测",
            "/api/health": "GET - 健康检查",
            "/api/info": "GET - API信息"
        },
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size": "16MB"
    })

def initialize_detector(model_path, charset_path=None):
    """初始化检测器"""
    global detector
    try:
        detector = PlateCharacterAPI(model_path, charset_path)
        logger.info("检测器初始化成功")
        return True
    except Exception as e:
        logger.error(f"检测器初始化失败: {str(e)}")
        return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='车牌字符识别 Flask API 服务')
    parser.add_argument('--model', type=str, default='char_detection/yolov8_chars2/weights/best.pt',
                        help='YOLOv8模型文件路径')
    parser.add_argument('--charset', type=str, default='char_detection_data_balanced/charset.json',
                        help='字符集JSON文件路径')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='服务器主机地址')
    parser.add_argument('--port', type=int, default=7389,
                        help='服务器端口')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误：模型文件不存在: {args.model}")
        exit(1)
    
    # 初始化检测器
    if not initialize_detector(args.model, args.charset):
        print("错误：检测器初始化失败")
        exit(1)
    
    print("=" * 60)
    print("车牌字符识别 Flask API 服务")
    print("=" * 60)
    print(f"模型文件: {args.model}")
    print(f"字符集文件: {args.charset}")
    print(f"服务地址: http://{args.host}:{args.port}")
    print("=" * 60)
    print("API端点:")
    print(f"  POST http://{args.host}:{args.port}/api/detect - 车牌字符检测")
    print(f"  GET  http://{args.host}:{args.port}/api/health - 健康检查")
    print(f"  GET  http://{args.host}:{args.port}/api/info - API信息")
    print("=" * 60)
    
    # 启动Flask应用
    app.run(host=args.host, port=args.port, debug=args.debug)
