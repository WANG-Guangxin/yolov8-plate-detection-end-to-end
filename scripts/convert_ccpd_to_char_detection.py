#!/usr/bin/env python3
"""
CCPD 数据集转换为 YOLOv8 字符检测格式

将 CCPD 车牌检测数据转换为字符级检测数据
每张图片包含7个字符的边界框和类别标签
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import json


# CCPD 官方字符集定义
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]

ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']

ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# 创建统一的字符集（去除重复的'O'占位符）
ALL_CHARS = []
CHAR_TO_CLASS = {}
CLASS_TO_CHAR = {}

def init_charset():
    """初始化字符集和映射关系"""
    global ALL_CHARS, CHAR_TO_CLASS, CLASS_TO_CHAR
    
    # 收集所有有效字符（排除'O'占位符）
    all_chars_set = set()
    
    # 添加省份字符（排除'O'）
    for char in PROVINCES:
        if char != 'O':
            all_chars_set.add(char)
    
    # 添加字母（排除'O'）
    for char in ALPHABETS:
        if char != 'O':
            all_chars_set.add(char)
    
    # 添加数字
    for char in ADS:
        if char != 'O' and char.isdigit():
            all_chars_set.add(char)
    
    # 转换为排序的列表
    ALL_CHARS = sorted(list(all_chars_set))
    
    # 创建映射关系
    CHAR_TO_CLASS = {char: idx for idx, char in enumerate(ALL_CHARS)}
    CLASS_TO_CHAR = {idx: char for idx, char in enumerate(ALL_CHARS)}
    
    print(f"字符集大小: {len(ALL_CHARS)}")
    print(f"支持的字符: {' '.join(ALL_CHARS[:20])}...")
    
    return len(ALL_CHARS)


def parse_ccpd_filename(filename):
    """
    解析 CCPD 文件名，提取车牌信息
    
    返回：
    - vertices: 车牌四个顶点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    - plate_chars: 车牌7个字符列表
    """
    try:
        basename = filename.split('.')[0]
        parts = basename.split('-')
        
        if len(parts) < 6:
            return None, None
        
        # 解析四个顶点坐标（第4部分）
        vertices_part = parts[3]
        vertex_coords = vertices_part.split('_')
        
        if len(vertex_coords) < 4:
            return None, None
        
        vertices = []
        for coord_str in vertex_coords:
            x, y = map(int, coord_str.split('&'))
            vertices.append((x, y))
        
        # 解析字符索引（第5部分）
        char_indices = parts[4].split('_')
        
        if len(char_indices) < 7:
            return None, None
        
        # 转换为字符
        plate_chars = []
        for i, idx_str in enumerate(char_indices[:7]):
            idx = int(idx_str)
            
            if i == 0:  # 省份
                if 0 <= idx < len(PROVINCES) and PROVINCES[idx] != 'O':
                    plate_chars.append(PROVINCES[idx])
                else:
                    return None, None
            elif i == 1:  # 第二位字母
                if 0 <= idx < len(ALPHABETS) and ALPHABETS[idx] != 'O':
                    plate_chars.append(ALPHABETS[idx])
                else:
                    return None, None
            else:  # 后5位字母或数字
                if 0 <= idx < len(ADS) and ADS[idx] != 'O':
                    plate_chars.append(ADS[idx])
                else:
                    return None, None
        
        return vertices, plate_chars
        
    except Exception as e:
        print(f"解析文件名失败 {filename}: {e}")
        return None, None


def calculate_char_boxes(vertices, plate_chars):
    """
    根据车牌四个顶点计算每个字符的边界框
    
    Args:
        vertices: 车牌四个顶点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        plate_chars: 7个字符列表
        
    Returns:
        char_boxes: [(x1,y1,x2,y2, class_id), ...] 每个字符的边界框和类别
    """
    try:
        # CCPD 顶点顺序：右下 -> 左下 -> 左上 -> 右上
        bottom_right, bottom_left, top_left, top_right = vertices
        
        # 计算车牌的宽度和高度向量
        width_vector = np.array(bottom_right) - np.array(bottom_left)
        height_vector = np.array(top_left) - np.array(bottom_left)
        
        char_boxes = []
        
        # 中国车牌字符分布：通常7个字符等宽分布
        for i, char in enumerate(plate_chars):
            # 计算字符在车牌中的相对位置
            char_ratio_start = i / 7.0
            char_ratio_end = (i + 1) / 7.0
            
            # 计算字符的四个角点
            # 左下角
            left_bottom = np.array(bottom_left) + width_vector * char_ratio_start
            # 右下角
            right_bottom = np.array(bottom_left) + width_vector * char_ratio_end
            # 左上角
            left_top = left_bottom + height_vector
            # 右上角
            right_top = right_bottom + height_vector
            
            # 计算最小外接矩形
            all_x = [left_bottom[0], right_bottom[0], left_top[0], right_top[0]]
            all_y = [left_bottom[1], right_bottom[1], left_top[1], right_top[1]]
            
            x1, x2 = min(all_x), max(all_x)
            y1, y2 = min(all_y), max(all_y)
            
            # 获取字符类别ID
            if char in CHAR_TO_CLASS:
                class_id = CHAR_TO_CLASS[char]
                char_boxes.append((int(x1), int(y1), int(x2), int(y2), class_id))
            else:
                print(f"未知字符: {char}")
                return None
        
        return char_boxes
        
    except Exception as e:
        print(f"计算字符边界框失败: {e}")
        return None


def convert_to_yolo_format(char_boxes, img_width, img_height):
    """
    将字符边界框转换为 YOLO 格式
    """
    yolo_annotations = []
    
    for x1, y1, x2, y2, class_id in char_boxes:
        # 边界检查
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        # 确保 x1 < x2, y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 计算中心点和宽高
        center_x = (x1 + x2) / 2.0 / img_width
        center_y = (y1 + y2) / 2.0 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # 验证范围
        if (0 <= center_x <= 1 and 0 <= center_y <= 1 and 
            0 < width <= 1 and 0 < height <= 1):
            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations


def process_ccpd_to_char_detection(ccpd_dir, output_dir, train_ratio=0.8, max_samples=None):
    """
    处理 CCPD 数据集，转换为字符检测格式
    """
    print("开始转换 CCPD 数据集为字符检测格式...")
    
    # 初始化字符集
    num_classes = init_charset()
    
    # 创建输出目录结构
    output_path = Path(output_dir)
    train_images_dir = output_path / "images" / "train"
    val_images_dir = output_path / "images" / "val"
    train_labels_dir = output_path / "labels" / "train"
    val_labels_dir = output_path / "labels" / "val"
    
    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    ccpd_path = Path(ccpd_dir)
    image_files = list(ccpd_path.glob("*.jpg")) + list(ccpd_path.glob("*.png"))
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 过滤有效文件
    valid_files = []
    for img_file in tqdm(image_files, desc="验证文件"):
        vertices, plate_chars = parse_ccpd_filename(img_file.name)
        if vertices and plate_chars and len(plate_chars) == 7:
            valid_files.append(img_file)
    
    print(f"有效文件: {len(valid_files)}")
    
    if len(valid_files) == 0:
        print("错误：没有找到有效的 CCPD 格式文件！")
        return
    
    # 划分训练集和验证集
    train_files, val_files = train_test_split(
        valid_files, train_size=train_ratio, random_state=42
    )
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    
    # 处理训练集
    stats_train = process_split(train_files, train_images_dir, train_labels_dir, "训练集")
    
    # 处理验证集
    stats_val = process_split(val_files, val_images_dir, val_labels_dir, "验证集")
    
    # 保存数据集配置
    dataset_config = {
        "path": str(output_path),
        "train": "images/train",
        "val": "images/val",
        "nc": num_classes,
        "names": {i: char for i, char in enumerate(ALL_CHARS)}
    }
    
    config_path = output_path / "dataset.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        import yaml
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
    
    # 保存字符集信息
    charset_info = {
        "all_chars": ALL_CHARS,
        "char_to_class": CHAR_TO_CLASS,
        "class_to_char": CLASS_TO_CHAR,
        "num_classes": num_classes,
        "provinces": [char for char in PROVINCES if char != 'O'],
        "alphabets": [char for char in ALPHABETS if char != 'O'],
        "digits": [char for char in ADS if char.isdigit()]
    }
    
    charset_path = output_path / "charset.json"
    with open(charset_path, 'w', encoding='utf-8') as f:
        json.dump(charset_info, f, ensure_ascii=False, indent=2)
    
    print(f"\\n数据集转换完成!")
    print(f"训练集统计: 成功 {stats_train['success']}, 失败 {stats_train['failed']}")
    print(f"验证集统计: 成功 {stats_val['success']}, 失败 {stats_val['failed']}")
    print(f"配置文件: {config_path}")
    print(f"字符集文件: {charset_path}")


def process_split(files, images_dir, labels_dir, split_name):
    """
    处理数据集的一个分割
    """
    successful = 0
    failed = 0
    total_chars = 0
    
    for img_file in tqdm(files, desc=f"处理{split_name}"):
        try:
            # 读取图像
            img = cv2.imread(str(img_file))
            if img is None:
                failed += 1
                continue
            
            img_height, img_width = img.shape[:2]
            
            # 解析车牌信息
            vertices, plate_chars = parse_ccpd_filename(img_file.name)
            if not vertices or not plate_chars:
                failed += 1
                continue
            
            # 计算字符边界框
            char_boxes = calculate_char_boxes(vertices, plate_chars)
            if not char_boxes:
                failed += 1
                continue
            
            # 转换为 YOLO 格式
            yolo_annotations = convert_to_yolo_format(char_boxes, img_width, img_height)
            if not yolo_annotations:
                failed += 1
                continue
            
            # 复制图像文件
            dest_img_path = images_dir / img_file.name
            shutil.copy2(img_file, dest_img_path)
            
            # 保存标注文件
            label_filename = img_file.stem + ".txt"
            label_path = labels_dir / label_filename
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            successful += 1
            total_chars += len(yolo_annotations)
            
        except Exception as e:
            print(f"处理文件 {img_file.name} 时出错: {e}")
            failed += 1
    
    print(f"{split_name}处理完成: 成功 {successful}, 失败 {failed}, 总字符数 {total_chars}")
    return {"success": successful, "failed": failed, "total_chars": total_chars}


def main():
    parser = argparse.ArgumentParser(description="CCPD 数据集转换为字符检测格式")
    parser.add_argument("--ccpd_dir", type=str, required=True,
                        help="CCPD 数据集目录路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录路径")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例 (默认: 0.8)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数量，用于测试 (默认: None)")
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.ccpd_dir):
        print(f"错误：CCPD 数据集目录不存在: {args.ccpd_dir}")
        return
    
    print(f"CCPD 数据集目录: {args.ccpd_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练集比例: {args.train_ratio}")
    
    # 安装依赖
    try:
        import yaml
    except ImportError:
        print("安装 PyYAML...")
        os.system("pip install PyYAML")
        import yaml
    
    # 开始转换
    process_ccpd_to_char_detection(args.ccpd_dir, args.output_dir, args.train_ratio, args.max_samples)


if __name__ == "__main__":
    main()
