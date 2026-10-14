from pathlib import Path
from collections import Counter

ROOT = Path("datasets/road-traffic")

for split in ["train", "valid"]:
    label_dir = ROOT / split / "labels"
    counts = Counter()
    examples = []

    for path in label_dir.glob("*.txt"):
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
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
                        examples.append(
                            f"{path.name}:{line_no}: {line}"
                        )
            else:
                counts["字段不足"] += 1
                if len(examples) < 10:
                    examples.append(
                        f"{path.name}:{line_no}: {line}"
                    )

    print(f"\n===== {split} =====")
    for name, count in counts.items():
        print(f"{name}: {count}")

    print("异常示例：")
    for item in examples:
        print(item)