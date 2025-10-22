#!/usr/bin/env python3
"""
YOLOv8 车牌字符检测推理脚本

从图片中检测车牌字符并组合成完整的车牌号码
"""

import os
import cv2
import argparse
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO
import time


class PlateCharacterDetector:
    def __init__(self, model_path, charset_path=None):
        """
        初始化字符检测器
        
        Args:
            model_path: YOLOv8 字符检测模型路径
            charset_path: 字符集文件路径
        """
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 加载字符集
        self.class_to_char = {}
        if charset_path and os.path.exists(charset_path):
            with open(charset_path, 'r', encoding='utf-8') as f:
                charset_info = json.load(f)
            self.class_to_char = {int(k): v for k, v in charset_info['class_to_char'].items()}
            print(f"加载字符集: {len(self.class_to_char)} 个字符")
        else:
            print("警告: 未找到字符集文件，将使用类别ID")
        
        print("字符检测器初始化完成")
    
    def detect_and_recognize(self, image_path, conf_threshold=0.5, output_dir=None):
        """
        检测并识别车牌字符
        
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            output_dir: 输出目录
            
        Returns:
            result: 检测结果字典
        """
        print(f"处理图片: {Path(image_path).name}")
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None
        
        img_height, img_width = image.shape[:2]
        
        # 进行检测
        results = self.model(image, conf=conf_threshold)
        
        # 处理检测结果
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # 转换为字符
                    char = self.class_to_char.get(class_id, f"class_{class_id}")
                    
                    detection = {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class_id": class_id,
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
        
        result = {
            "image_path": image_path,
            "image_size": [img_width, img_height],
            "detections": detections,
            "plate_number": plate_number,
            "total_chars": len(detections)
        }
        
        print(f"检测到 {len(detections)} 个字符")
        print(f"车牌号码: {plate_number}")
        
        # 绘制检测结果
        if output_dir:
            result_image = self.draw_detections(image, detections, plate_number)
            
            # 保存结果图片
            os.makedirs(output_dir, exist_ok=True)
            output_image_path = os.path.join(output_dir, f"result_{Path(image_path).name}")
            cv2.imwrite(output_image_path, result_image)
            result["output_image"] = output_image_path
            
            # 保存 JSON 结果
            json_path = os.path.join(output_dir, f"result_{Path(image_path).stem}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            result["json_path"] = json_path
        
        return result
    
    def draw_detections(self, image, detections, plate_number):
        """
        在图片上绘制检测结果
        """
        result_image = image.copy()
        
        # 为每个字符绘制边界框和标签
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            char = detection["character"]
            conf = detection["confidence"]
            
            # 绘制边界框
            cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 绘制字符标签
            label = f"{char} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(result_image, (int(x1), int(y1) - label_size[1] - 10), 
                         (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
            cv2.putText(result_image, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # 绘制字符顺序
            cv2.putText(result_image, str(i + 1), (int(x1) + 5, int(y1) + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # 在图片顶部绘制完整车牌号码
        if plate_number:
            plate_label = f"Plate: {plate_number}"
            label_size = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            cv2.rectangle(result_image, (10, 10), (20 + label_size[0], 50), (0, 0, 255), -1)
            cv2.putText(result_image, plate_label, (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return result_image
    
    def process_batch(self, input_dir, output_dir, conf_threshold=0.5):
        """
        批量处理图片
        """
        print(f"批量处理目录: {input_dir}")
        
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print("未找到图片文件")
            return
        
        print(f"找到 {len(image_files)} 个图片文件")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        batch_results = []
        successful_recognition = 0
        total_chars_detected = 0
        
        start_time = time.time()
        
        # 逐个处理图片
        for i, image_path in enumerate(image_files):
            print(f"\\n[{i+1}/{len(image_files)}] 处理: {image_path.name}")
            
            try:
                result = self.detect_and_recognize(
                    str(image_path), conf_threshold, output_dir
                )
                
                if result:
                    batch_results.append(result)
                    total_chars_detected += result["total_chars"]
                    
                    # 统计成功识别的车牌（通常为7个字符）
                    if result["total_chars"] >= 6:  # 至少6个字符认为是成功的
                        successful_recognition += 1
                        
            except Exception as e:
                print(f"处理图片时出错: {e}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 保存批量结果汇总
        summary = {
            "total_images": len(image_files),
            "successful_recognition": successful_recognition,
            "recognition_rate": successful_recognition / len(image_files) if len(image_files) > 0 else 0,
            "total_chars_detected": total_chars_detected,
            "avg_chars_per_image": total_chars_detected / len(image_files) if len(image_files) > 0 else 0,
            "processing_time": processing_time,
            "avg_time_per_image": processing_time / len(image_files),
            "results": batch_results
        }
        
        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\\n批量处理完成!")
        print(f"处理时间: {processing_time:.2f} 秒")
        print(f"平均每张: {processing_time/len(image_files):.2f} 秒")
        print(f"成功识别: {successful_recognition}/{len(image_files)} ({successful_recognition/len(image_files)*100:.1f}%)")
        print(f"平均字符数: {total_chars_detected/len(image_files):.1f}")
        print(f"结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 车牌字符检测推理")
    
    # 必需参数
    parser.add_argument("--model", type=str, required=True,
                        help="YOLOv8 字符检测模型文件路径 (.pt)")
    
    # 检测模式
    parser.add_argument("--mode", type=str, default='image',
                        choices=['image', 'batch'],
                        help="检测模式 (默认: image)")
    
    # 输入输出参数
    parser.add_argument("--input", type=str, required=True,
                        help="输入路径 (图片/目录)")
    parser.add_argument("--output", type=str, default="./char_detection_results",
                        help="输出路径 (默认: ./char_detection_results)")
    
    # 检测参数
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值 (默认: 0.5)")
    parser.add_argument("--charset", type=str, default=None,
                        help="字符集文件路径 (.json)")
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误：模型文件不存在: {args.model}")
        return
    
    # 检查输入路径
    if not os.path.exists(args.input):
        print(f"错误：输入路径不存在: {args.input}")
        return
    
    # 自动查找字符集文件
    if not args.charset:
        model_dir = Path(args.model).parent.parent
        charset_candidates = [
            model_dir / "charset.json",
            Path("./charset.json"),
            Path("./char_detection/datasets/charset.json")
        ]
        
        for candidate in charset_candidates:
            if candidate.exists():
                args.charset = str(candidate)
                break
    
    print("=" * 60)
    print("YOLOv8 车牌字符检测推理系统")
    print("=" * 60)
    print(f"模型文件: {args.model}")
    print(f"字符集文件: {args.charset}")
    print(f"处理模式: {args.mode}")
    print(f"置信度阈值: {args.conf}")
    print("=" * 60)
    
    try:
        # 初始化检测器
        detector = PlateCharacterDetector(args.model, args.charset)
        
        # 根据模式处理
        if args.mode == 'image':
            result = detector.detect_and_recognize(
                args.input, args.conf, args.output
            )
            
            if result:
                print(f"\\n检测结果保存到: {args.output}")
        
        elif args.mode == 'batch':
            detector.process_batch(
                args.input, args.output, args.conf
            )
        
        print("\\n处理完成！")
        
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
