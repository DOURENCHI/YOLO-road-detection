from pathlib import Path

ROOT = Path("datasets/road-traffic")
SPLITS = ["train", "valid"]
NUM_CLASSES = 12

for split in SPLITS:
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        print(f"\n[{split}] 缺少 images 或 labels 目录")
        continue

    images = {
        p.stem for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    }
    labels = {p.stem for p in label_dir.glob("*.txt")}

    missing_labels = sorted(images - labels)
    missing_images = sorted(labels - images)

    empty_labels = []
    invalid_lines = []
    total_boxes = 0

    for label_path in label_dir.glob("*.txt"):
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()

        if not lines:
            empty_labels.append(label_path.name)
            continue

        for line_no, line in enumerate(lines, 1):
            parts = line.split()

            if len(parts) != 5:
                invalid_lines.append(
                    f"{label_path.name}:{line_no} 字段数量不是5"
                )
                continue

            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                invalid_lines.append(
                    f"{label_path.name}:{line_no} 存在非数字字段"
                )
                continue

            if not (0 <= cls < NUM_CLASSES):
                invalid_lines.append(
                    f"{label_path.name}:{line_no} 类别编号越界"
                )
            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                invalid_lines.append(
                    f"{label_path.name}:{line_no} 坐标或宽高超出0到1"
                )
            if w <= 0 or h <= 0:
                invalid_lines.append(
                    f"{label_path.name}:{line_no} 宽高必须大于0"
                )

            total_boxes += 1

    print(f"\n===== {split} =====")
    print(f"图片数：{len(images)}")
    print(f"标签文件数：{len(labels)}")
    print(f"标注框数量：{total_boxes}")
    print(f"图片缺少标签：{len(missing_labels)}")
    print(f"标签缺少图片：{len(missing_images)}")
    print(f"空标签文件：{len(empty_labels)}")
    print(f"格式或数值异常：{len(invalid_lines)}")

    if missing_labels:
        print("缺少标签示例：", missing_labels[:5])
    if missing_images:
        print("缺少图片示例：", missing_images[:5])
    if invalid_lines:
        print("异常示例：", invalid_lines[:10])
