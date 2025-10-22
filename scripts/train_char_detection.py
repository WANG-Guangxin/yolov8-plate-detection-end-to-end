#!/usr/bin/env python3
"""
YOLOv8 车牌字符检测模型训练脚本
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import json


def train_char_detection_model(
    config_path,
    model_size='n',
    epochs=100,
    batch_size=16,
    img_size=640,
    device='auto',
    workers=8,
    project_name='char_detection',
    name='yolov8_chars'
):
    """
    训练 YOLOv8 字符检测模型
    """
    
    # 检查配置文件
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 读取数据集信息
    config_dir = Path(config_path).parent
    charset_path = config_dir / "charset.json"
    
    if charset_path.exists():
        with open(charset_path, 'r', encoding='utf-8') as f:
            charset_info = json.load(f)
        num_classes = charset_info['num_classes']
        print(f"字符集大小: {num_classes}")
        print(f"支持字符: {' '.join(charset_info['all_chars'][:20])}...")
    else:
        print("警告: 未找到字符集信息文件")
    
    # 创建模型
    model_name = f"yolov8{model_size}.pt"
    print(f"创建 YOLOv8{model_size.upper()} 模型...")
    
    # 加载预训练模型
    model = YOLO(model_name)
    
    # 检查设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    print(f"训练参数:")
    print(f"  - 模型尺寸: YOLOv8{model_size.upper()}")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 图像尺寸: {img_size}")
    print(f"  - 工作进程: {workers}")
    
    # 开始训练
    try:
        results = model.train(
            data=config_path,           # 数据集配置文件
            epochs=epochs,              # 训练轮数
            imgsz=img_size,            # 输入图像尺寸
            batch=batch_size,          # 批次大小
            device=device,             # 训练设备
            workers=workers,           # 数据加载器进程数
            project=project_name,      # 项目目录名
            name=name,                 # 实验名称
            save=True,                 # 保存模型
            save_period=10,            # 每10轮保存一次
            val=True,                  # 启用验证
            plots=True,                # 生成训练图表
            verbose=True,              # 详细输出
            # 优化器设置
            optimizer='AdamW',         # 使用 AdamW 优化器
            lr0=0.01,                  # 初始学习率
            lrf=0.1,                   # 最终学习率
            momentum=0.937,            # SGD momentum/Adam beta1
            weight_decay=0.0005,       # 权重衰减
            warmup_epochs=3.0,         # 预热轮数
            warmup_momentum=0.8,       # 预热动量
            warmup_bias_lr=0.1,        # 预热偏置学习率
            # 数据增强（字符检测需要更谨慎的增强）
            mixup=0.0,                 # 关闭 mixup
            copy_paste=0.0,            # 关闭 copy-paste
            degrees=10.0,              # 旋转角度范围
            translate=0.1,             # 平移范围
            scale=0.2,                 # 缩放范围
            shear=5.0,                 # 剪切角度
            perspective=0.0001,        # 透视变换
            flipud=0.0,                # 垂直翻转概率（车牌不应该垂直翻转）
            fliplr=0.0,                # 水平翻转概率（车牌不应该水平翻转）
            mosaic=0.5,                # Mosaic 增强概率
            # 其他设置
            patience=50,               # 早停耐心值
            resume=False,              # 不从断点恢复
        )
        
        print("\\n训练完成！")
        print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
        print(f"最新模型保存在: {results.save_dir}/weights/last.pt")
        
        # 验证最佳模型
        print("\\n开始验证最佳模型...")
        best_model = YOLO(f"{results.save_dir}/weights/best.pt")
        val_results = best_model.val()
        
        print(f"验证结果:")
        print(f"  - mAP50: {val_results.box.map50:.4f}")
        print(f"  - mAP50-95: {val_results.box.map:.4f}")
        print(f"  - Precision: {val_results.box.mp:.4f}")
        print(f"  - Recall: {val_results.box.mr:.4f}")
        
        # 保存训练信息
        training_info = {
            "model_size": model_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "device": str(device),
            "best_model_path": f"{results.save_dir}/weights/best.pt",
            "validation_results": {
                "mAP50": float(val_results.box.map50),
                "mAP50_95": float(val_results.box.map),
                "precision": float(val_results.box.mp),
                "recall": float(val_results.box.mr)
            }
        }
        
        info_path = f"{results.save_dir}/training_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)
        
        print(f"训练信息保存在: {info_path}")
        
        return results
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 车牌字符检测模型训练")
    
    # 必需参数
    parser.add_argument("--config", type=str, required=True,
                        help="数据集配置文件路径 (.yaml)")
    
    # 模型参数
    parser.add_argument("--model", type=str, default='m', 
                        choices=['n', 's', 'm', 'l', 'x'],
                        help="模型尺寸 (默认: m)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数 (默认: 50)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="批次大小 (默认: 16)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="输入图像尺寸 (默认: 640)")
    parser.add_argument("--device", type=str, default='auto',
                        help="训练设备 (默认: auto)")
    parser.add_argument("--workers", type=int, default=8,
                        help="数据加载器进程数 (默认: 8)")
    
    # 项目参数
    parser.add_argument("--project", type=str, default='char_detection',
                        help="项目名称 (默认: char_detection)")
    parser.add_argument("--name", type=str, default='yolov8_chars',
                        help="实验名称 (默认: yolov8_chars)")
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误：配置文件不存在: {args.config}")
        return
    
    print("=" * 60)
    print("YOLOv8 车牌字符检测模型训练")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"模型尺寸: YOLOv8{args.model.upper()}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print("=" * 60)
    
    try:
        # 开始训练
        results = train_char_detection_model(
            config_path=args.config,
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            workers=args.workers,
            project_name=args.project,
            name=args.name
        )
        
        print("\\n字符检测模型训练完成！")
        
    except Exception as e:
        print(f"训练失败: {e}")


if __name__ == "__main__":
    main()
