"""
01 号脚本 · 数据集结构检查（只读，不改任何文件）

检查四件事
----------
1. 图片和标签是否一一对应（有没有缺标签的图、有没有多余的标签）
2. 每个标签行的字段数是不是 5（标准 YOLO 检测框格式）
3. 类别编号是否在 [0, 类别总数) 范围内
4. 归一化坐标是否都在 [0, 1] 内、宽高是否大于 0

2026-10-14 三处改动
------------------
① 划分从 ["train", "valid"] 扩成 train / valid / test（原来漏了 test）
② 路径、类别数不再写死，统一从 dataset_config 拿（它再去读 data.yaml）
③ 修掉一个小毛病：原来的 total_boxes 会把类别越界等不合格的行也当成框计数，
   现在只有通过全部校验的行才计数
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import (
    IMAGE_EXTENSIONS,
    SOURCE_YAML,
    SPLITS,
    load_names,
    read_label_lines,
    split_dirs,
)

# 类别名和类别数都从 data.yaml 读，不再写死 NUM_CLASSES = 12
CLASS_NAMES = load_names(SOURCE_YAML)
NUM_CLASSES = len(CLASS_NAMES)

print(f"读取配置：{SOURCE_YAML}")
print(f"类别数：{NUM_CLASSES} -> {', '.join(CLASS_NAMES)}")

for split in SPLITS:
    images_dir, labels_dir = split_dirs(split, SOURCE_YAML)

    if images_dir is None or not images_dir.is_dir():
        print(f"\n[{split}] 缺少 images 目录：{images_dir}")
        continue

    images = {
        p.relative_to(images_dir).with_suffix("").as_posix()
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels = {
        p.relative_to(labels_dir).with_suffix("").as_posix()
        for p in labels_dir.rglob("*.txt")
    }

    missing_labels = sorted(images - labels)
    missing_images = sorted(labels - images)

    empty_labels = []       # 内容为空的标签文件
    field_count_issues = []  # 字段数不是 5 的行（多为多边形标注，不算坏数据）
    value_issues = []        # 数值或类别真正有问题的行
    total_boxes = 0

    for label_path in sorted(labels_dir.rglob("*.txt")):
        name = label_path.relative_to(labels_dir).as_posix()
        lines = read_label_lines(label_path)

        if not any(line.strip() for line in lines):
            empty_labels.append(name)
            continue

        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                field_count_issues.append(
                    f"{name}:{line_no} 字段数量不是 5（实际 {len(parts)}）"
                )
                continue

            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                value_issues.append(f"{name}:{line_no} 存在非数字字段")
                continue

            qualified = True

            if not 0 <= cls < NUM_CLASSES:
                value_issues.append(f"{name}:{line_no} 类别编号越界：{cls}")
                qualified = False

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                value_issues.append(f"{name}:{line_no} 坐标或宽高超出 0 到 1")
                qualified = False

            if w <= 0 or h <= 0:
                value_issues.append(f"{name}:{line_no} 宽高必须大于 0")
                qualified = False

            # 只有全部校验通过的才算一个合格框
            if qualified:
                total_boxes += 1

    print(f"\n===== {split} =====")
    print(f"图片数：{len(images)}")
    print(f"标签文件数：{len(labels)}")
    print(f"合格标注框数：{total_boxes}")
    print(f"图片缺少标签：{len(missing_labels)}")
    print(f"标签缺少图片：{len(missing_images)}")
    print(f"空标签文件：{len(empty_labels)}")
    print(f"字段数不等于 5 的行：{len(field_count_issues)}")
    print(f"数值或类别异常的行：{len(value_issues)}")

    if missing_labels:
        print("缺少标签示例：", missing_labels[:5])
    if missing_images:
        print("缺少图片示例：", missing_images[:5])
    if empty_labels:
        print("空标签示例：", empty_labels[:5])
    if field_count_issues:
        print("字段数异常示例：", field_count_issues[:5])
    if value_issues:
        print("数值异常示例：", value_issues[:10])
