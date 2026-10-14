"""
07 号脚本：生成「演示拼图」——一眼就能看出模型哪里行、哪里不行。

为什么需要这个
--------------
09 号脚本给出的是数字（mAP、P、R）。数字精确，但不直观，
而且面试时你没法拿一张表格讲两分钟。
这张图把**好消息和坏消息放在同一张画布上**：

    绿色虚线 = 人工标注（Ground Truth）
    红色实线 = 模型预测（Prediction）

    绿色框里没有红框  -> 漏检（该报没报）
    红色框里没有绿框  -> 误检（不该报报了）
    两个框套在一起    -> 检出正确

自动分三档挑图
--------------
    「检测完整」  标注的框一个不漏、一个不多
    「典型漏检」  漏掉的比误报的多
    「典型误检」  误报的比漏掉的多

挑选规则写在输出 JSON 里，所以这张图不是"我挑了几张好看的"，
而是按明确标准筛出来的，谁都能复算。

输出
----
    results/demo/demo_grid.png     演示拼图（可以直接当简历配图）
    results/demo/demo_grid.json    每张图是谁、按什么规则选中的
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import (  # noqa: E402
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    SPLITS,
    TARGET_YAML,
    load_names,
    split_dirs,
)

DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "runs" / "road_baseline" / "weights" / "best.pt"
RESULT_DIR = PROJECT_ROOT / "results" / "demo"

#: 判定"检出正确"的 IoU 门槛。和 COCO / YOLO 评估时用的一致。
MATCH_IOU = 0.5

#: Windows 自带的中文字体。matplotlib 默认字体画不出汉字，必须显式注册。
CN_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),    # 微软雅黑
    Path(r"C:\Windows\Fonts\simhei.ttf"),  # 黑体
    Path(r"C:\Windows\Fonts\simsun.ttc"),  # 宋体
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按检测质量自动挑图，生成人工标注 vs 模型预测的对比拼图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", default="valid", choices=list(SPLITS),
                        help="从哪个划分挑图。默认 valid——把 test 留给 09 号的正式评估")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="权重文件")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入边长")
    parser.add_argument("--device", default="0", help="显卡编号，或 'cpu'")
    parser.add_argument("--limit", type=int, default=0, help="只在前 N 张里挑，0 表示全部")
    parser.add_argument("--per-group", type=int, default=2, help="每一档挑几张")
    parser.add_argument("--name", default="demo_grid", help="输出文件名（不含扩展名）")
    return parser.parse_args()


def setup_chinese_font():
    """注册中文字体，否则图里的中文会变成一堆方框。"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    for path in CN_FONT_CANDIDATES:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(path)).get_name()]
            break
    rcParams["axes.unicode_minus"] = False


def read_ground_truth(label_path: Path, width: int, height: int):
    """把 YOLO 标签文件读成 [(class_id, x1, y1, x2, y2)]，坐标是像素。"""
    boxes = []
    if not label_path.is_file():
        return boxes

    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        boxes.append((
            class_id,
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        ))
    return boxes


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(pred_boxes, gt_boxes, threshold: float = MATCH_IOU):
    """
    贪心匹配：每次找出当前 IoU 最大的一对（且要 >= 阈值），配掉后从两边移除。

    返回 (配对成功的下标对, 没配上的预测下标, 没配上的标注下标)。
    """
    pairs, used_pred, used_gt = [], set(), set()

    candidates = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            score = iou(pb, gb)
            if score >= threshold:
                candidates.append((score, pi, gi))
    candidates.sort(reverse=True)

    for _, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        pairs.append((pi, gi))

    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in used_pred]
    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]
    return pairs, unmatched_pred, unmatched_gt


def main() -> int:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        print("找不到权重文件：%s" % weights, file=sys.stderr)
        return 1

    images_dir, labels_dir = split_dirs(args.split, TARGET_YAML)
    if images_dir is None or not images_dir.is_dir():
        print("找不到 %s 划分：%s" % (args.split, images_dir), file=sys.stderr)
        return 1

    names = load_names(TARGET_YAML)
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit:
        files = files[: args.limit]

    from ultralytics import YOLO

    model = YOLO(str(weights))
    print("在 %s 划分的 %d 张图上跑推理……" % (args.split, len(files)))
    results = model.predict(
        source=[str(p) for p in files],
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        verbose=False,
    )

    # ---------------- 逐图算 TP / FP / FN ----------------
    records = []
    for result in results:
        image_path = Path(result.path)
        height, width = result.orig_shape

        pred_boxes = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.tolist()
            classes = result.boxes.cls.tolist()
            confs = result.boxes.conf.tolist()
            for box, cls_id, conf in zip(xyxy, classes, confs):
                pred_boxes.append({"box": tuple(box), "cls": int(cls_id), "conf": float(conf)})

        gt_raw = read_ground_truth(labels_dir / (image_path.stem + ".txt"), width, height)
        gt_boxes = [{"box": (b[1], b[2], b[3], b[4]), "cls": b[0]} for b in gt_raw]

        pairs, unmatched_pred, unmatched_gt = match(
            [p["box"] for p in pred_boxes], [g["box"] for g in gt_boxes]
        )

        records.append({
            "path": image_path,
            "labels_path": labels_dir / (image_path.stem + ".txt"),
            "width": width,
            "height": height,
            "predictions": pred_boxes,
            "ground_truth": gt_boxes,
            "tp": len(pairs),
            "fp": len(unmatched_pred),
            "fn": len(unmatched_gt),
            "matched_pred_index": [p for p, _ in pairs],
        })

    if not records:
        print("一张图都没跑到。", file=sys.stderr)
        return 1

    # ---------------- 按规则分档 ----------------
    perfect = [r for r in records if r["fn"] == 0 and r["fp"] == 0 and r["tp"] > 0]
    perfect.sort(key=lambda r: r["tp"], reverse=True)          # 检出越多越靠前

    misses = [r for r in records if r["fn"] > 0]
    misses.sort(key=lambda r: (r["fn"] - r["fp"], r["fn"]), reverse=True)

    false_pos = [r for r in records if r["fp"] > 0]
    false_pos.sort(key=lambda r: (r["fp"] - r["fn"], r["fp"]), reverse=True)

    picked = []
    for record in perfect[: args.per_group]:
        picked.append(("检测完整", record))
    for record in misses[: args.per_group]:
        picked.append(("典型漏检", record))
    for record in false_pos[: args.per_group]:
        picked.append(("典型误检", record))

    print("挑选结果：")
    for group, record in picked:
        print("  %-8s TP=%-2d FP=%-2d FN=%-2d  %s" % (
            group, record["tp"], record["fp"], record["fn"], record["path"].name[:52]))

    # ---------------- 画图 ----------------
    setup_chinese_font()
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    columns = min(3, len(picked))
    rows = (len(picked) + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 5.2 * rows))
    if rows * columns == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in row]

    group_color = {"检测完整": "#16a34a", "典型漏检": "#f59e0b", "典型误检": "#dc2626"}

    for index, (group, record) in enumerate(picked):
        ax = axes[index]
        image = Image.open(record["path"]).convert("RGB")
        ax.imshow(image)

        # 画框策略：正常检出的框**不打标签**，只有出问题的地方才写字。
        # 一屏几十个框，每个都标类别的话整张图就糊了，重点反而看不见。
        pairs, unmatched_pred, unmatched_gt = match(
            [p["box"] for p in record["predictions"]],
            [g["box"] for g in record["ground_truth"]],
        )

        # ① 配对成功的：标注和预测指的是同一个目标，画一个绿框就够
        for pi, _ in pairs:
            x1, y1, x2, y2 = record["predictions"][pi]["box"]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor="#16a34a", linewidth=1.8))

        # ② 漏检：有标注、没预测
        for gi in unmatched_gt:
            gt = record["ground_truth"][gi]
            x1, y1, x2, y2 = gt["box"]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor="#f59e0b", linewidth=2.2, linestyle="--"))
            ax.text(x1, max(9, y1 - 3), "漏检 %s" % names[gt["cls"]], fontsize=7.0,
                    color="#7c2d12", va="bottom", ha="left",
                    bbox=dict(facecolor="#fde68a", edgecolor="none", pad=1.4, alpha=0.95))

        # ③ 误检：有预测、没标注
        for pi in unmatched_pred:
            pred = record["predictions"][pi]
            x1, y1, x2, y2 = pred["box"]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor="#dc2626", linewidth=2.0))
            ax.text(x2, max(9, y1 - 3), "误检 %s %.2f" % (names[pred["cls"]], pred["conf"]),
                    fontsize=7.0, color="#7f1d1d", va="bottom", ha="right",
                    bbox=dict(facecolor="#fecaca", edgecolor="none", pad=1.4, alpha=0.95))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("%s · TP %d / 误检 %d / 漏检 %d\n%s" % (
            group, record["tp"], record["fp"], record["fn"], record["path"].name[:34]),
            fontsize=8.5, color=group_color[group], pad=5)

    for ax in axes[len(picked):]:
        ax.axis("off")

    legend = (
        "绿框 = 检出正确　｜　黄框 = 漏检（有标注、模型没报）　｜　红框 = 误检（模型报了、没有标注）\n"
        "数据：%s 划分 %d 张 ｜ 阈值 conf=%.2f ｜ 判定检出正确的 IoU>=%.1f"
        % (args.split, len(records), args.conf, MATCH_IOU)
    )
    fig.suptitle("人工标注 vs 模型预测：模型在哪里行、在哪里不行\n" + legend,
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.945))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = RESULT_DIR / ("%s.png" % args.name)
    fig.savefig(figure_path, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    payload = {
        "划分": args.split,
        "权重": str(weights),
        "参数": {"conf": args.conf, "imgsz": args.imgsz, "device": str(args.device)},
        "判定IoU": MATCH_IOU,
        "评估图数": len(records),
        "整体": {
            "TP": sum(r["tp"] for r in records),
            "FP": sum(r["fp"] for r in records),
            "FN": sum(r["fn"] for r in records),
        },
        "挑选规则": {
            "检测完整": "漏检为 0、误检为 0，按检出数降序取前 %d" % args.per_group,
            "典型漏检": "按 (漏检数 - 误检数) 降序取前 %d" % args.per_group,
            "典型误检": "按 (误检数 - 漏检数) 降序取前 %d" % args.per_group,
        },
        "选中的图": [
            {"档位": group, "文件": record["path"].name,
             "TP": record["tp"], "FP": record["fp"], "FN": record["fn"]}
            for group, record in picked
        ],
    }
    json_path = RESULT_DIR / ("%s.json" % args.name)
    json_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    print("-" * 72)
    print("整体：TP %d / 误检 %d / 漏检 %d（%d 张图）" % (
        payload["整体"]["TP"], payload["整体"]["FP"], payload["整体"]["FN"], len(records)))
    print("拼图写到：%s" % figure_path.relative_to(PROJECT_ROOT))
    print("挑选记录：%s" % json_path.relative_to(PROJECT_ROOT))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
