"""
02 号脚本 · 标注格式分类统计（只读）

回答一个问题：这个数据集里到底混了几种标注格式？

01 号脚本只会告诉你"有 N 行不是 5 个字段"，但没说清它们长什么样。
这个脚本把每一行按字段数归类统计，一眼就能看出格式构成：

    标准检测框：类别 + 中心x + 中心y + 宽 + 高        -> 5 个字段
    多边形格式：类别 + x1 y1 x2 y2 ...               -> 5 个以上字段，坐标成对出现
    坐标数量为奇数 / 字段不足                          -> 真的有问题

2026-10-14 两处改动
------------------
① 划分扩成 train / valid / test（原来只有 train 和 valid）
② 路径不再写死，从 dataset_config 读
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import SOURCE_YAML, SPLITS, read_label_lines, split_dirs

print(f"读取配置：{SOURCE_YAML}")

for split in SPLITS:
    _images_dir, labels_dir = split_dirs(split, SOURCE_YAML)

    if labels_dir is None or not labels_dir.is_dir():
        print(f"\n[{split}] 缺少 labels 目录：{labels_dir}")
        continue

    counts = Counter()
    examples = []

    for path in sorted(labels_dir.rglob("*.txt")):
        for line_no, line in enumerate(read_label_lines(path), 1):
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) == 5:
                counts["标准检测框"] += 1

            elif len(parts) > 5:
                coordinate_count = len(parts) - 1

                if coordinate_count % 2 == 0:
                    counts["多边形格式候选"] += 1
                else:
                    counts["坐标数量为奇数"] += 1
                    if len(examples) < 10:
                        examples.append(f"{path.name}:{line_no}: {line}")

            else:
                counts["字段不足"] += 1
                if len(examples) < 10:
                    examples.append(f"{path.name}:{line_no}: {line}")

    print(f"\n===== {split} =====")
    for name, count in counts.items():
        print(f"{name}: {count}")

    print("异常示例：")
    for item in examples:
        print(item)
