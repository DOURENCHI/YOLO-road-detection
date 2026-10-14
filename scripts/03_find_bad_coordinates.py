"""
03 号脚本 · 多边形标注专项排查（只读）

02 号脚本统计出"有 N 行是多边形格式"，但"格式不一样"不等于"数据是坏的"。
这个脚本专门回答：这些多边形标注本身，数值上合法吗？

三项检查
--------
1. 除类别外的所有字段能否转成数字（能抓住"数字粘连"那种情况）
2. 坐标数量是不是偶数（x, y 成对）
3. 所有坐标是否都落在 [0, 1] 内

为什么这一步不能省
------------------
下一步 04 号脚本要把多边形压成矩形框。如果多边形本身就是坏的（坐标超范围、
数字粘连），那转出来的框一定是错的，而且从结果上看不出错。
所以要**先证明"多边形是好的，只是格式不同"，才敢放心转换**。

2026-10-14 两处改动
------------------
① 划分扩成 train / valid / test（原来只有 train 和 valid）
② 路径不再写死，从 dataset_config 读
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import SOURCE_YAML, SPLITS, read_label_lines, split_dirs

print(f"读取配置：{SOURCE_YAML}")

for split in SPLITS:
    _images_dir, labels_dir = split_dirs(split, SOURCE_YAML)

    if labels_dir is None or not labels_dir.is_dir():
        print(f"\n[{split}] 缺少 labels 目录：{labels_dir}")
        continue

    bad_lines = []

    for path in sorted(labels_dir.rglob("*.txt")):
        for line_no, line in enumerate(read_label_lines(path), 1):
            parts = line.split()

            # 只排查多边形行（字段数 > 5）；标准检测框由 01 号脚本管
            if len(parts) <= 5:
                continue

            try:
                cls = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                bad_lines.append((path.name, line_no, "存在无法解析的字段", line))
                continue

            if len(coords) % 2 != 0:
                bad_lines.append((path.name, line_no, "坐标数量不是偶数", line))
                continue

            if not all(0 <= v <= 1 for v in coords):
                bad_lines.append((path.name, line_no, "坐标超出 0 到 1", line))

    print(f"\n===== {split} =====")
    print(f"异常多边形行数：{len(bad_lines)}")

    for filename, line_no, reason, line in bad_lines[:20]:
        print(f"{filename}:{line_no} [{reason}]")
        print(line)
