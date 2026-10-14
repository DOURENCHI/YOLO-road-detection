
from pathlib import Path
from collections import Counter

ROOT = Path.cwd()
DATASET = ROOT / "datasets" / "road-traffic-detect"
NUM_CLASSES = 12
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

overall_errors = 0

for split in ["train", "valid", "test"]:
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    if not image_dir.exists():
        continue

    images = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels = list(label_dir.rglob("*.txt")) if label_dir.exists() else []

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
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
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
        print(f"  类别 {cls} ({CLASS_NAMES[cls] if False else cls}): "
              f"{class_counts[cls]}")

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

