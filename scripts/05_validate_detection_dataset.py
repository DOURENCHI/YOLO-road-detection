"""
05 号脚本 · 转换后数据集的验收（只读）

这是全流程的"验收关"：对 04 号脚本生成的新数据集做一次比 01 号更严格的检查，
同时输出**每个类别的检测框数量** —— 这组数字直接暴露了数据集的一个隐藏问题
（类别不均衡）。

六道校验（只有全过的行才计入"有效框"）
------------------------------------
① 字段数必须是 5
② 类别编号必须在 [0, 类别总数) 内
③ 数值必须都能解析
④ 坐标必须在 [0, 1] 内
⑤ 宽高必须大于 0
⑥ 框不能超出图像边界

注意：和 01 号脚本不同，这里的计数放在所有校验之后，
所以 total_boxes 是严格意义上的"合法框数"。

2026-10-14 三处改动
------------------
① 去掉写死的 NUM_CLASSES = 12，改成从 data.yaml 读
② 修掉第 116 行的残留写法 CLASS_NAMES[cls] if False else cls（那是调试留下的），
   现在直接打印真实类别名
③ 数据集路径不再用 Path.cwd() 拼，改由 dataset_config 统一给出
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import (
    IMAGE_EXTENSIONS,
    SPLITS,
    TARGET_DATASET,
    TARGET_YAML,
    load_names,
    read_label_lines,
    split_dirs,
)

CLASS_NAMES = load_names(TARGET_YAML)
NUM_CLASSES = len(CLASS_NAMES)

print(f"数据集：{TARGET_DATASET}")
print(f"读取配置：{TARGET_YAML}")
print(f"类别数：{NUM_CLASSES} -> {', '.join(CLASS_NAMES)}")

overall_errors = 0

for split in SPLITS:
    image_dir, label_dir = split_dirs(split, TARGET_YAML)

    if image_dir is None or not image_dir.is_dir():
        continue

    images = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels = list(label_dir.rglob("*.txt")) if label_dir.is_dir() else []

    image_stems = {
        p.relative_to(image_dir).with_suffix("").as_posix()
        for p in images
    }
    label_stems = {
        p.relative_to(label_dir).with_suffix("").as_posix()
        for p in labels
    }

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    empty_labels = 0
    total_boxes = 0
    class_counts = Counter()
    errors = []

    for label_path in labels:
        relative_name = label_path.relative_to(label_dir).as_posix()
        lines = read_label_lines(label_path)
        valid_lines = [line for line in lines if line.strip()]

        if not valid_lines:
            empty_labels += 1
            continue

        for line_no, line in enumerate(valid_lines, 1):
            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"{relative_name}:{line_no} 字段数为 {len(parts)}，应为 5"
                )
                continue

            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(
                    f"{relative_name}:{line_no} 存在无法解析的数值"
                )
                continue

            if not 0 <= cls < NUM_CLASSES:
                errors.append(
                    f"{relative_name}:{line_no} 类别编号非法：{cls}"
                )
                continue

            if not all(0 <= v <= 1 for v in [x, y, w, h]):
                errors.append(
                    f"{relative_name}:{line_no} 数值超出 [0,1]"
                )
                continue

            if w <= 0 or h <= 0:
                errors.append(
                    f"{relative_name}:{line_no} 宽或高不大于 0"
                )
                continue

            if (
                x - w / 2 < -1e-6
                or x + w / 2 > 1 + 1e-6
                or y - h / 2 < -1e-6
                or y + h / 2 > 1 + 1e-6
            ):
                errors.append(
                    f"{relative_name}:{line_no} 检测框超出图像边界"
                )
                continue

            total_boxes += 1
            class_counts[cls] += 1

    overall_errors += (
        len(missing_labels) + len(orphan_labels) + len(errors)
    )

    print(f"\n===== {split} =====")
    print(f"图片数：{len(images)}")
    print(f"标签文件数：{len(labels)}")
    print(f"图片缺少标签：{len(missing_labels)}")
    print(f"标签没有对应图片：{len(orphan_labels)}")
    print(f"空标签文件：{empty_labels}")
    print(f"有效检测框数：{total_boxes}")
    print(f"格式或数值异常：{len(errors)}")

    print("各类别检测框数量：")
    for cls in range(NUM_CLASSES):
        name = CLASS_NAMES[cls]
        print(f"  类别 {cls:>2} {name:<14}: {class_counts[cls]}")

    if missing_labels:
        print("缺少标签示例：", list(missing_labels)[:5])
    if orphan_labels:
        print("多余标签示例：", list(orphan_labels)[:5])
    if errors:
        print("异常示例：")
        for error in errors[:10]:
            print(" ", error)

print("\n===== 总结 =====")
print(f"发现的结构或标注问题：{overall_errors}")
