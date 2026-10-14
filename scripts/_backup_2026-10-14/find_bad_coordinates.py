from pathlib import Path

ROOT = Path("datasets/road-traffic")

for split in ["train", "valid"]:
    label_dir = ROOT / split / "labels"
    bad_lines = []

    for path in sorted(label_dir.glob("*.txt")):
        lines = path.read_text(encoding="utf-8-sig").splitlines()

        for line_no, line in enumerate(lines, 1):
            parts = line.split()

            if len(parts) <= 5:
                continue

            # 多边形候选：检查每个字段能否转换为数字
            try:
                cls = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                bad_lines.append(
                    (path.name, line_no, "存在无法解析的字段", line)
                )
                continue

            if len(coords) % 2 != 0:
                bad_lines.append(
                    (path.name, line_no, "坐标数量不是偶数", line)
                )
                continue

            if not all(0 <= v <= 1 for v in coords):
                bad_lines.append(
                    (path.name, line_no, "坐标超出0到1", line)
                )

    print(f"\n===== {split} =====")
    print(f"异常多边形行数：{len(bad_lines)}")

    for filename, line_no, reason, line in bad_lines[:20]:
        print(f"{filename}:{line_no} [{reason}]")
        print(line)