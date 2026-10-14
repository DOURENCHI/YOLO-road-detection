"""
标签可视化抽检：把 YOLO 标签里的框画回原图，用人眼核对。

为什么必须有这一步
------------------
01~05 号脚本只能证明"格式合法"——字段数是 5、数值在 0~1、类别编号没越界。
它们证明不了"每个框都准确地包住了正确目标"。类别标错、框拖偏、整张漏标，
这类脚本一个都发现不了。

只有把框画回原图上看一眼，才能回答"标注到底对不对"。

用法
----
    python scripts/visualize_labels.py                  # 三个划分各抽 8 张
    python scripts/visualize_labels.py --n 30           # 各抽 30 张
    python scripts/visualize_labels.py --split valid    # 只看验证集
    python scripts/visualize_labels.py --seed 42        # 换一批抽样

输出（都放在 results/label_check/ 下）
-------------------------------------
    <split>_sample.png        每个划分一张抽检拼图
    empty_label_check.png     空标签对应的原图，单独放大核对
    summary.txt               本次抽检的文字记录（可截图放进报告）
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

# 让脚本无论从哪里启动，都能 import 到同目录下的 dataset_config
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")  # 不弹窗口，直接把图存成文件

import matplotlib.pyplot as plt
from matplotlib import colormaps, font_manager
from matplotlib.patches import Rectangle
from PIL import Image

from dataset_config import (
    IMAGE_EXTENSIONS,
    RESULT_DIR,
    SPLITS,
    TARGET_YAML,
    load_names,
    read_label_lines,
    split_dirs,
)

# 中文标题需要指定字体，否则会显示成方框。
# 找不到字体也不影响画框，只是标题里的中文会变成方块。
_CHINESE_FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
if _CHINESE_FONT.exists():
    font_manager.fontManager.addfont(str(_CHINESE_FONT))
    plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

COLORS = colormaps["tab20"]


# ------------------------------------------------------------------ 读数据


def load_split(split: str, yaml_path: Path):
    """
    读出一个划分的全部图片、标签和框。

    返回 (items, empty_labels, images_dir)：
      items        [(图片路径, 标签路径, [(类别, x, y, w, h), ...]), ...]
      empty_labels 内容为空的标签文件列表（这批图最需要人眼确认）
      images_dir   图片目录；该划分没配置时返回 None
    """
    images_dir, labels_dir = split_dirs(split, yaml_path)

    if images_dir is None or not images_dir.is_dir():
        return [], [], None

    items = []
    empty_labels = []

    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative = image_path.relative_to(images_dir)
        label_path = (labels_dir / relative).with_suffix(".txt")

        boxes = []

        if label_path.is_file():
            lines = read_label_lines(label_path)

            # 空标签文件：0 字节，或者只有空白 —— 单独记下来
            if not any(line.strip() for line in lines):
                empty_labels.append(label_path)

            for line in lines:
                parts = line.split()

                # 只画标准的 5 字段检测框；
                # 多边形那类行已经在 04 号脚本里被转成矩形了，这里不该再出现
                if len(parts) != 5:
                    continue

                try:
                    boxes.append((int(parts[0]), *map(float, parts[1:])))
                except ValueError:
                    continue

        items.append((image_path, label_path, boxes))

    return items, empty_labels, images_dir


# ------------------------------------------------------------------ 画图


def draw_one(axes, image_path, boxes, names, title):
    """把一张图的框画出来。X-Y 用的是归一化坐标，乘回像素尺寸即可。"""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size

        axes.imshow(image)
        axes.set_title(title, fontsize=9.5)
        axes.axis("off")

        for index, (cls, x, y, w, h) in enumerate(boxes, start=1):
            box_w, box_h = w * width, h * height
            left, top = x * width - box_w / 2, y * height - box_h / 2

            color = COLORS(cls % 20)

            axes.add_patch(
                Rectangle(
                    (left, top), box_w, box_h,
                    fill=False, edgecolor=color, linewidth=1.3,
                )
            )

            name = names[cls] if 0 <= cls < len(names) else f"cls{cls}"
            axes.text(
                left, max(top - 4, 0), f"{name} {index}",
                fontsize=6.5, color="white", va="bottom",
                bbox={
                    "facecolor": color,
                    "edgecolor": "none",
                    "pad": 0.7,
                    "alpha": 0.85,
                },
            )


def make_montage(items, names, split, out_path, columns):
    """把若干张图拼成一张网格图存下来。"""
    rows = max(1, -(-len(items) // columns))  # 向上取整

    figure, axes_grid = plt.subplots(
        rows, columns, figsize=(4.6 * columns, 3.5 * rows)
    )
    figure.suptitle(
        f"{split} 抽检：共 {len(items)} 张（框按类别上色，标签右上角数字是框序号）",
        fontsize=13,
    )

    axes_list = list(axes_grid.flat)

    for axes, (image_path, _label_path, boxes) in zip(axes_list, items):
        title = f"{image_path.name[:38]}\n{len(boxes)} 个框"
        try:
            draw_one(axes, image_path, boxes, names, title)
        except Exception as error:  # 单张图坏了不该让整批停下
            axes.axis("off")
            axes.set_title(f"{image_path.name[:38]}\n读取失败：{error}", fontsize=8)

    for axes in axes_list[len(items):]:
        axes.axis("off")

    figure.tight_layout(rect=(0, 0, 1, 0.955))
    figure.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def make_empty_detail(empty_items, names, out_path):
    """把所有空标签对应的原图放大排一张图，专门用来核对'这图到底有没有目标'。"""
    rows = max(1, -(-len(empty_items) // 2))
    figure, axes_grid = plt.subplots(rows, 2, figsize=(13, 5.4 * rows))
    figure.suptitle(
        "空标签核对：这些图的标签文件是空的。请逐张判断——确实没有目标，还是漏标了？",
        fontsize=13,
    )

    axes_list = list(axes_grid.flat)

    for axes, (image_path, _label_path, boxes) in zip(axes_list, empty_items):
        axes.imshow(Image.open(image_path).convert("RGB"))
        axes.set_title(f"{image_path.name}\n标签：0 框", fontsize=10, color="#b91c1c")
        axes.axis("off")

    for axes in axes_list[len(empty_items):]:
        axes.axis("off")

    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(figure)


# ------------------------------------------------------------------ 主流程


def parse_args():
    parser = argparse.ArgumentParser(
        description="把 YOLO 标签画回原图，人工抽检标注质量"
    )
    parser.add_argument(
        "--yaml", default=str(TARGET_YAML),
        help="数据集配置文件，默认用 road-traffic-detect/data.yaml",
    )
    parser.add_argument(
        "--split", default="all", choices=[*SPLITS, "all"],
        help="只抽某一个划分，默认三个划分都抽",
    )
    parser.add_argument(
        "--n", type=int, default=8,
        help="每个划分抽多少张，默认 8",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="随机种子，默认 0；改成别的数字就换一批图",
    )
    parser.add_argument(
        "--columns", type=int, default=4,
        help="拼图的列数，默认 4",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    yaml_path = Path(args.yaml)
    names = load_names(yaml_path)
    splits = list(SPLITS) if args.split == "all" else [args.split]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    report = [
        "标签可视化抽检记录",
        f"配置文件：{yaml_path}",
        f"类别数：{len(names)}",
        f"类别名：{', '.join(names)}",
        f"随机种子：{args.seed}    每划分抽样数：{args.n}",
        "",
    ]

    all_empty_items = []

    for split in splits:
        items, empty_labels, images_dir = load_split(split, yaml_path)

        print(f"\n===== {split} =====")

        if images_dir is None:
            print("该划分在 data.yaml 里没有配置，跳过")
            report.append(f"===== {split} ===== 未配置，跳过")
            continue

        if not items:
            print(f"目录里没有图片：{images_dir}")
            report.append(f"===== {split} ===== 没有图片")
            continue

        rng = random.Random(args.seed)
        empty_set = set(empty_labels)

        # 空标签图优先抽 —— 它们是最需要人眼确认的一批
        forced = [it for it in items if it[1] in empty_set]
        rest = [it for it in items if it[1] not in empty_set]
        rng.shuffle(rest)

        picked = forced + rest[: max(0, args.n - len(forced))]

        box_counter = Counter()
        for _image, _label, boxes in picked:
            for cls, *_ in boxes:
                box_counter[cls] += 1

        total_boxes = sum(box_counter.values())
        total_boxes_all = sum(len(b) for _i, _l, b in items)
        empty_in_pick = len([1 for it in picked if it[1] in empty_set])

        out_path = RESULT_DIR / f"{split}_sample.png"
        make_montage(picked, names, split, out_path, args.columns)

        print(f"图片总数：{len(items)}")
        print(f"标签文件为空的：{len(empty_labels)}")
        print(f"本次抽检：{len(picked)} 张（其中空标签 {empty_in_pick} 张，已强制包含）")
        print(f"抽到的框：{total_boxes} 个（该划分共 {total_boxes_all} 个）")
        print("抽检图中出现最多的类别：")
        for cls, count in box_counter.most_common(5):
            name = names[cls] if 0 <= cls < len(names) else f"cls{cls}"
            print(f"  类别 {cls:>2} {name:<14} {count} 个框")
        print(f"已保存：{out_path}")

        report.extend(
            [
                f"===== {split} =====",
                f"图片总数：{len(items)}",
                f"标签文件为空的：{len(empty_labels)}",
                f"本次抽检张数：{len(picked)}",
                f"抽到的框数：{total_boxes}（该划分共 {total_boxes_all}）",
                "抽检图中的类别分布：" + ", ".join(
                    f"{names[c] if 0 <= c < len(names) else c}={n}"
                    for c, n in box_counter.most_common()
                ),
                f"拼图文件：{out_path.name}",
                "",
            ]
        )

        all_empty_items.extend(it for it in picked if it[1] in empty_set)

    if all_empty_items:
        empty_out = RESULT_DIR / "empty_label_check.png"
        make_empty_detail(all_empty_items, names, empty_out)
        print(f"\n空标签核对图已保存：{empty_out}")
        report.append(f"空标签核对图：{empty_out.name}")
    else:
        print("\n三个划分里没有空标签文件。")
        report.append("三个划分里没有空标签文件。")

    summary_path = RESULT_DIR / "summary.txt"
    summary_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\n抽检记录：{summary_path}")


if __name__ == "__main__":
    main()
