"""
04 号脚本 · 标注格式统一（会写文件：生成 datasets/road-traffic-detect）

把混合格式的原始标注，统一成 YOLO 目标检测要的矩形框格式：

    多边形 类别 x1 y1 x2 y2 ...   ->   矩形框 类别 中心x 中心y 宽 高

做法是取多边形所有顶点的最小外接矩形。同时把图片复制过去，
最后生成一份干净的 data.yaml。

为什么敢动手做这个转换
--------------------
因为 03 号脚本已经证明：那 800 多条多边形的数值本身是合法的，
只是格式和目标检测不一样。所以转换是安全的，不是"猜"。

2026-10-14 改动
--------------
类别名不再写死在脚本里，改成读源数据集 data.yaml 的 names 字段。
换一份数据集时，只要 data.yaml 里 names 变了，这个脚本自动跟着变。

一个仍然存在的坑（暂未改，先记在这里）
------------------------------------
脚本生成 data.yaml 时写的 path 是 "."。Ultralytics 解析相对 path 时是按
**进程当前工作目录**算的，不是按 yaml 文件所在目录算的。所以如果你从项目根
目录启动训练，它会去找 项目根/valid/images —— 找不到就报
"missing path"。这就是当初训练报错的根因（详见解析文档 4.3 节）。
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import (
    PROJECT_ROOT,
    SOURCE_DATASET,
    SOURCE_YAML,
    TARGET_DATASET,
    load_names,
)

# 项目根目录、源目录、目标目录都由 dataset_config 统一给出，
# 不再用 Path.cwd() —— 那样一旦在别的目录启动脚本，路径就全错了。
SOURCE = SOURCE_DATASET
DEST = TARGET_DATASET
REPORT = PROJECT_ROOT / "docs" / "label_conversion_report.txt"

# 类别名从源数据集的 data.yaml 读，不写死
CLASS_NAMES = load_names(SOURCE_YAML)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def convert_line(line, file_name, line_no):
    """将一行 YOLO 检测框或多边形标注统一为检测框格式。"""
    parts = line.split()

    if not parts:
        return None, None  # 忽略空行

    try:
        cls = int(parts[0])
        values = [float(x) for x in parts[1:]]
    except ValueError:
        return None, f"{file_name}:{line_no} 存在无法解析的数值"

    if not 0 <= cls < len(CLASS_NAMES):
        return None, f"{file_name}:{line_no} 类别编号超出范围：{cls}"

    # 标准 YOLO 检测框：类别 + x中心 + y中心 + 宽 + 高
    if len(parts) == 5:
        x, y, w, h = values

        if not all(0 <= v <= 1 for v in values):
            return None, f"{file_name}:{line_no} 检测框数值超出 [0,1]"

        if w <= 0 or h <= 0:
            return None, f"{file_name}:{line_no} 检测框宽高必须大于 0"

        if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6:
            return None, f"{file_name}:{line_no} 检测框超出图像左右边界"

        if y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
            return None, f"{file_name}:{line_no} 检测框超出图像上下边界"

        return f"{cls} {x:.8f} {y:.8f} {w:.8f} {h:.8f}", None

    # 多边形：类别 + x1 y1 x2 y2 ...
    if len(parts) > 5:
        if len(values) < 6 or len(values) % 2 != 0:
            return None, f"{file_name}:{line_no} 多边形坐标数量不合法"

        if not all(0 <= v <= 1 for v in values):
            return None, f"{file_name}:{line_no} 多边形坐标超出 [0,1]"

        xs = values[0::2]
        ys = values[1::2]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        w = x_max - x_min
        h = y_max - y_min

        if w <= 0 or h <= 0:
            return None, f"{file_name}:{line_no} 多边形外接框宽高为 0"

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        result = (
            f"{cls} {x_center:.8f} {y_center:.8f} "
            f"{w:.8f} {h:.8f}"
        )
        return result, None

    return None, f"{file_name}:{line_no} 字段数量不足"


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"找不到原始数据集：{SOURCE}\n"
            "请确认 datasets 目录结构完整。"
        )

    if DEST.exists():
        raise FileExistsError(
            f"目标目录已存在：{DEST}\n"
            "为避免覆盖数据，脚本已停止。"
            "请先检查已有目录，再决定如何处理。"
        )

    report_lines = [
        "YOLO 检测数据集标注转换报告",
        f"原始目录：{SOURCE}",
        f"输出目录：{DEST}",
        f"类别来源：{SOURCE_YAML}",
        "",
    ]

    total_boxes = 0
    total_polygons = 0
    total_errors = 0
    yaml_splits = []

    try:
        for split in ["train", "valid", "test"]:
            src_images = SOURCE / split / "images"
            src_labels = SOURCE / split / "labels"

            if not src_images.exists():
                continue

            dst_images = DEST / split / "images"
            dst_labels = DEST / split / "labels"

            dst_images.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_images, dst_images)
            dst_labels.mkdir(parents=True, exist_ok=True)

            image_paths = [
                p for p in dst_images.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]

            if split == "train":
                yaml_splits.append(f"train: {split}/images")
            elif split == "valid":
                yaml_splits.append(f"val: {split}/images")
            elif split == "test":
                yaml_splits.append(f"test: {split}/images")

            split_boxes = 0
            split_polygons = 0
            split_errors = 0

            # 以图像文件名为基准，确保每张图都有对应的标签文件。
            for image_path in image_paths:
                relative_path = image_path.relative_to(dst_images)
                label_path = dst_labels / relative_path.with_suffix(".txt")
                label_path.parent.mkdir(parents=True, exist_ok=True)

                source_label = src_labels / relative_path.with_suffix(".txt")
                converted_lines = []

                if source_label.exists():
                    original_lines = source_label.read_text(
                        encoding="utf-8-sig"
                    ).splitlines()

                    for line_no, line in enumerate(original_lines, 1):
                        if not line.strip():
                            continue

                        result, error = convert_line(
                            line, str(source_label.relative_to(SOURCE)), line_no
                        )

                        if error:
                            split_errors += 1
                            report_lines.append(f"异常：{error}")
                            report_lines.append(f"原始内容：{line}")
                            continue

                        if result is not None:
                            converted_lines.append(result)

                            if len(line.split()) == 5:
                                split_boxes += 1
                            else:
                                split_polygons += 1
                else:
                    report_lines.append(
                        f"注意：图像没有原始标签文件："
                        f"{source_label.relative_to(SOURCE)}"
                    )

                label_path.write_text(
                    "\n".join(converted_lines) +
                    ("\n" if converted_lines else ""),
                    encoding="utf-8"
                )

            total_boxes += split_boxes
            total_polygons += split_polygons
            total_errors += split_errors

            report_lines.extend([
                "",
                f"===== {split} =====",
                f"图像数：{len(image_paths)}",
                f"保留的原始检测框：{split_boxes}",
                f"转换的多边形：{split_polygons}",
                f"异常标注行：{split_errors}",
            ])

        # 生成独立的数据集配置文件。
        yaml_lines = [
            "path: .",
            *yaml_splits,
            "",
            f"nc: {len(CLASS_NAMES)}",
            "names:",
        ]

        for name in CLASS_NAMES:
            yaml_lines.append(f"  - {name}")

        (DEST / "data.yaml").write_text(
            "\n".join(yaml_lines) + "\n",
            encoding="utf-8"
        )

        report_lines.extend([
            "",
            "===== 总计 =====",
            f"原始检测框行：{total_boxes}",
            f"转换的多边形行：{total_polygons}",
            f"异常标注行：{total_errors}",
            "",
            "说明：异常行没有写入新标签，须结合原图人工核查。",
        ])

        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("\n".join(report_lines), encoding="utf-8")

        print("转换完成。")
        print(f"新数据集：{DEST}")
        print(f"配置文件：{DEST / 'data.yaml'}")
        print(f"异常报告：{REPORT}")
        print(f"保留的检测框行：{total_boxes}")
        print(f"转换的多边形行：{total_polygons}")
        print(f"异常标注行：{total_errors}")

    except Exception:
        # 出错时不删除目标目录，方便检查现场。
        raise


if __name__ == "__main__":
    main()
