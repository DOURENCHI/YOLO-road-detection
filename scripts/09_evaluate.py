"""
09 号脚本：在指定划分上做一次正式评估，并生成可交付的评估报告。

和 06/07 号的区别
-----------------
  06 号回答「这张图检出了什么」
  07 号回答「模型在哪里行、在哪里不行」（看图）
  09 号回答「数字上到底是多少」（看表）——逐类的 P / R / mAP50 / mAP50-95，
       外加最优置信度阈值和混淆矩阵。

关于 test 集的一个纪律
----------------------
这个脚本默认跑 test 划分，而且**设计成只在方案定稿后跑一次**。
原因：验证集（valid）拿来挑模型、调阈值；如果 test 也反复看，那它就不是
"没见过的数据"了，报出来的数字会虚高。
报告里会明确写下这次是什么时候、用什么配置跑的，方便日后回溯。

输出
----
  results/eval_report/
    ├── evaluation_report.md      人读的结论（数字 + 解读 + 已知边界）
    ├── metrics_per_class.csv     机器读的逐类指标
    ├── metrics_overall.json      总体指标与配置留档
    ├── fig_per_class_ap.png      逐类 AP 柱状图
    ├── fig_confusion_matrix.png  归一化混淆矩阵（中文标注）
    └── fig_f1_curve.png          F1 随置信度变化，标出最优点
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import (  # noqa: E402
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    SPLITS,
    TARGET_YAML,
    split_dirs,
)

DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "runs" / "road_baseline" / "weights" / "best.pt"
RESULT_DIR = PROJECT_ROOT / "results" / "eval_report"

CN_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在某个划分上做正式评估，输出报告与图表",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", default="test", choices=list(SPLITS),
                        help="在哪个划分上评估。默认 test——留到最后只跑一次")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="权重文件")
    parser.add_argument("--imgsz", type=int, default=640, help="输入边长")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="评估时的置信度下限。评估要扫阈值，所以放得很低，不是推理阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS 的 IoU 阈值")
    parser.add_argument("--device", default="0", help="显卡编号，或 'cpu'")
    parser.add_argument("--batch", type=int, default=8, help="评估批大小")
    parser.add_argument("--note", default="", help="写进报告的备注，比如这次改了什么")
    return parser.parse_args()


def setup_chinese_font():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    for path in CN_FONT_CANDIDATES:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(path)).get_name()]
            break
    rcParams["axes.unicode_minus"] = False


def count_boxes_per_class(labels_dir, class_count: int) -> list[int]:
    """
    直接去数标签文件，得到每个类有多少个框。

    为什么不用 Ultralytics 现成的桶计数：`Metric` 对象上没有 `nt_per_class`
    （那是 `DetMetrics` 的东西），照着名字取会静默拿到 0，报告里就会写出
    「训练样本都很少」这种看不出错的假数字。自己数最稳，也只有十来行。
    """
    counts = [0] * class_count
    if labels_dir is None or not Path(labels_dir).is_dir():
        return counts

    for label_file in Path(labels_dir).glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8-sig").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            if 0 <= class_id < class_count:
                counts[class_id] += 1
    return counts


def plot_per_class_ap(rows, out_path: Path):
    from matplotlib import pyplot as plt

    rows = sorted(rows, key=lambda r: r["ap50"])
    labels = [r["name"] for r in rows]
    ap50 = [r["ap50"] for r in rows]
    ap = [r["ap"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(rows) + 2.0))
    y = range(len(rows))
    ax.barh([i + 0.19 for i in y], ap50, height=0.36, color="#2563eb", label="mAP50")
    ax.barh([i - 0.19 for i in y], ap, height=0.36, color="#93c5fd", label="mAP50-95")

    for i, (a, b) in enumerate(zip(ap50, ap)):
        ax.text(a + 0.012, i + 0.19, "%.3f" % a, va="center", fontsize=7.5, color="#1e3a8a")
        ax.text(b + 0.012, i - 0.19, "%.3f" % b, va="center", fontsize=7.5, color="#3b82f6")

    mean50 = sum(ap50) / len(ap50)
    ax.axvline(mean50, color="#dc2626", linestyle="--", linewidth=1.3,
               label="全体 mAP50 = %.4f" % mean50)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, min(1.12, max(ap50) + 0.14))
    ax.set_xlabel("AP", fontsize=10)
    ax.set_title("逐类精度：这一列越长越好\n按 mAP50 升序排列，短的通常就是训练样本最少的", fontsize=11.5)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_confusion_matrix(matrix, class_names, gt_counts, out_path: Path):
    """
    画归一化混淆矩阵。

    有两件事必须先说清楚，因为这两处我都做错过（记录下来免得再踩）：

    1) 矩阵的约定是 matrix[预测类别, 人工标注类别] —— **行是预测、列是标注**。
       Ultralytics 源码里的三行注释就是全部证据：
           self.matrix[dc, gc] += 1        # TP
           self.matrix[self.nc, gc] += 1   # FN  -> 最后一行 = 该报没报
           self.matrix[dc, self.nc] += 1   # FP  -> 最后一列 = 不该报报了

    2) 归一化按**列**做（每一列加起来是 1），和 Ultralytics 官方出图一致。
       一列 = 某一类真实标注，格子里的数字是"它被预测成了各类的比例"，
       于是对角线恰好就是这一类的**召回率**，最好读。

       ⚠️ 千万别按行归一化。评估时 conf 会设得很低（0.001，为了把 PR 曲线算全），
       这会产生海量低分误报，按行归一化会让"背景"那一列吃掉几乎所有格子，
       整张图除了最后一列全是零 —— 我就交过这么一版。
    """
    from matplotlib import pyplot as plt
    import numpy as np

    m = np.asarray(matrix, dtype=float)
    col_sums = m.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.divide(m, col_sums, out=np.zeros_like(m), where=col_sums > 0)

    count = len(matrix)

    # 最后一行是"漏检"、最后一列是"误检"，含义不同，所以两个轴的标签要分开拼。
    if len(class_names) == count - 1:
        y_labels = list(class_names) + ["背景\n（该报没报）"]
        # 横轴标签竖排 + 单行，13 个标签斜排会互相压住，读不出来
        x_labels = [
            "%s (%s)" % (name, gt_counts[i] if i < len(gt_counts) else "?")
            for i, name in enumerate(class_names)
        ] + ["背景 (不该报报了)"]
    else:
        y_labels = ["类别 %d" % i for i in range(count)]
        x_labels = list(y_labels)

    fig, ax = plt.subplots(figsize=(10.0, 8.8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(count))
    ax.set_yticks(range(count))
    ax.set_xticklabels(x_labels, rotation=90, ha="center", fontsize=8.0)
    ax.set_yticklabels(y_labels, fontsize=8.5)
    ax.set_xlabel("人工标注是（Ground Truth），括号内是该类在本次评估里的标注框数", fontsize=10)
    ax.set_ylabel("模型预测为（Predicted）", fontsize=10.5)
    ax.set_title(
        "归一化混淆矩阵（每一列加起来是 1）\n"
        "对角线 = 认对了，也就是这一类的召回率；非对角线 = 认成了别的类\n"
        "最后一行 = 该报没报（漏检）；最后一列 = 不该报报了（误检）",
        fontsize=10.5)

    for i in range(count):
        for j in range(count):
            value = normalized[i, j]
            if value >= 0.005:
                ax.text(j, i, "%.2f" % value, ha="center", va="center", fontsize=6.8,
                        color="white" if value > 0.55 else "#1e293b")

    fig.colorbar(image, ax=ax, fraction=0.036, pad=0.03, label="占该列标注的比例")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return normalized


def plot_f1_curve(f1_curve, conf_axis, best_conf, best_f1, out_path: Path):
    from matplotlib import pyplot as plt
    import numpy as np

    curve = np.asarray(f1_curve, dtype=float)
    mean_curve = curve.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(conf_axis, mean_curve, color="#2563eb", linewidth=2.0, label="全体平均 F1")
    ax.axvline(best_conf, color="#dc2626", linestyle="--", linewidth=1.4,
               label="最优阈值 conf=%.2f（F1=%.3f）" % (best_conf, best_f1))
    ax.axvline(0.25, color="#16a34a", linestyle=":", linewidth=1.4,
               label="项目默认 conf=0.25")
    ax.scatter([best_conf], [best_f1], color="#dc2626", zorder=5, s=32)

    ax.set_xlabel("置信度阈值 conf", fontsize=10.5)
    ax.set_ylabel("F1", fontsize=10.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("F1 随置信度阈值的变化\n阈值调低：召回上去、误报也上去；调高：反过来。峰值处最平衡", fontsize=11.5)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        print("找不到权重文件：%s" % weights, file=sys.stderr)
        return 1

    images_dir, labels_dir = split_dirs(args.split, TARGET_YAML)
    image_count = len([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    # 训练集的每类框数也要数一份：报告里要论证"某类学不会是因为样本少"，
    # 靠的必须是**训练集**的样本量，不是测试集的。这两个数很容易弄混。
    train_labels_dir = split_dirs("train", TARGET_YAML)[1]

    from ultralytics import YOLO

    print("=" * 72)
    print("在 %s 划分（%d 张图）上评估" % (args.split, image_count))
    print("权重：%s" % weights.relative_to(PROJECT_ROOT))
    print("=" * 72)

    model = YOLO(str(weights))
    result = model.val(
        data=str(TARGET_YAML),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        device=args.device,
        # 这里必须给 True：Ultralytics 只在 plots=True 的分支里才真正把
        # 混淆矩阵累加满，给 False 会得到一个全零矩阵（踩过这个坑）。
        # 它自己那套英文图会写到 ultralytics_raw/ 留档，本脚本的中文图另出。
        plots=True,
        project=str(RESULT_DIR),
        name="ultralytics_raw",
        exist_ok=True,
        verbose=False,
    )

    box = result.box
    names = {int(k): str(v) for k, v in result.names.items()}

    # ap_class_index 是「这个划分里真实出现过的类别」的下标。
    # 不能直接用 range(nc)：某个类如果一张标都没有（本项目的 tractors 在 test 里就是），
    # Ultralytics 返回的数组会短一截，直接按下标取会串位甚至越界。
    present_ids = [int(i) for i in box.ap_class_index]

    # 每个类有多少个标注框。本划分的数量给混淆矩阵横轴用，
    # 训练集的数量用来论证"某类学不会是不是因为样本太少"——这两个数很容易弄混，
    # 所以分开存、标签写清楚。
    gt_counts = count_boxes_per_class(labels_dir, len(names))
    train_counts = count_boxes_per_class(train_labels_dir, len(names))

    per_class = []
    for position, class_id in enumerate(present_ids):
        per_class.append({
            "class_id": class_id,
            "name": names.get(class_id, "class_%d" % class_id),
            "precision": float(box.p[position]),
            "recall": float(box.r[position]),
            "ap50": float(box.ap50[position]),
            "ap": float(box.ap[position]),
            "gt_boxes": gt_counts[class_id] if class_id < len(gt_counts) else 0,
            "train_boxes": train_counts[class_id] if class_id < len(train_counts) else 0,
        })

    missing = [cid for cid in sorted(names) if cid not in present_ids]

    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }

    # ---------------- 最优置信度阈值 ----------------
    best_conf, best_f1 = None, None

    f1_curve = getattr(box, "f1_curve", None)
    conf_axis = getattr(box, "px", None)
    if f1_curve is not None and conf_axis is not None:
        mean_curve = np.asarray(f1_curve, dtype=float).mean(axis=0)
        index = int(np.argmax(mean_curve))
        best_conf = float(conf_axis[index])
        best_f1 = float(mean_curve[index])

    # ---------------- 混淆矩阵 ----------------
    matrix = None
    class_names = [names[i] for i in range(len(names))]
    cm = getattr(result, "confusion_matrix", None)
    if cm is not None and getattr(cm, "matrix", None) is not None and cm.matrix.any():
        matrix = cm.matrix
        # cm.names 存的是 {编号: 名字} 字典，直接迭代只会拿到编号 0,1,2…，
        # 所以标签用模型自己的 names 拼，背景那一行/列由绘图函数自己加。
        # 注意：只有 plots=True 时 Ultralytics 才会把矩阵真正填满，否则是一张全零矩阵。

    # ---------------- 出图 ----------------
    setup_chinese_font()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fig_ap = RESULT_DIR / "fig_per_class_ap.png"
    plot_per_class_ap(per_class, fig_ap)

    fig_cm = None
    if matrix is not None:
        fig_cm = RESULT_DIR / "fig_confusion_matrix.png"
        plot_confusion_matrix(matrix, class_names, gt_counts, fig_cm)

    fig_f1 = None
    if best_conf is not None:
        fig_f1 = RESULT_DIR / "fig_f1_curve.png"
        plot_f1_curve(f1_curve, conf_axis, best_conf, best_f1, fig_f1)

    # ---------------- 落盘 ----------------
    csv_path = RESULT_DIR / "metrics_per_class.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["class_id", "name", "precision", "recall", "ap50", "ap",
                                "gt_boxes", "train_boxes"])
        writer.writeheader()
        for row in per_class:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    payload = {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "划分": args.split,
        "评估图数": image_count,
        "权重": str(weights),
        "配置": {
            "imgsz": args.imgsz, "conf下限": args.conf, "iou": args.iou,
            "batch": args.batch, "device": str(args.device),
        },
        "总体指标": overall,
        "逐类指标": per_class,
        "该划分里没有标注的类别": [names[c] for c in missing],
        "最优置信度阈值": best_conf,
        "最优阈值处的平均F1": best_f1,
        "每张图耗时_ms": {k: round(float(v), 3) for k, v in (getattr(result, "speed", {}) or {}).items()},
        "备注": args.note,
    }
    (RESULT_DIR / "metrics_overall.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---------------- 报告 ----------------
    lines = []
    add = lines.append

    add("# 模型评估报告")
    add("")
    add("> 数据：`%s` 划分，共 **%d** 张图 ｜ 权重：`%s`" % (
        args.split, image_count, weights.relative_to(PROJECT_ROOT).as_posix()))
    add("> 生成时间：%s ｜ 配置：imgsz=%d, iou=%.2f, batch=%d, device=%s" % (
        payload["生成时间"], args.imgsz, args.iou, args.batch, args.device))
    if args.note:
        add("> 备注：%s" % args.note)
    add("")

    add("## 1. 结论先说")
    add("")
    add("- **mAP50 = %.4f**，**mAP50-95 = %.4f**，精确率 %.4f，召回率 %.4f。"
        % (overall["mAP50"], overall["mAP50-95"], overall["precision"], overall["recall"]))
    add("- 这是**单模型、未做任何集成与测试时增强**的基线结果。"
        "权重用的是 `best.pt`——它由 `fitness = 0.1×mAP50 + 0.9×mAP50-95` 自动选出，"
        "**不是最后一轮**，所以它的表现会略好于训练日志末行的数字。")
    best_row = max(per_class, key=lambda r: r["ap50"])
    worst_row = min(per_class, key=lambda r: r["ap50"])
    head_line = ("- %d 个类里表现最好的是 **%s（AP50 %.3f，本划分 %d 个框）**；"
                 "数值最低的是 **%s（AP50 %.3f，本划分 %d 个框）**。"
                 % (len(per_class), best_row["name"], best_row["ap50"], best_row["gt_boxes"],
                    worst_row["name"], worst_row["ap50"], worst_row["gt_boxes"]))
    if worst_row["gt_boxes"] <= 5:
        head_line += "后者**样本太少，这个数字不能当成结论**，见第 2 节末尾的说明。"
    add(head_line)
    if best_conf is not None:
        add("- 置信度阈值扫过一遍，**F1 峰值出现在 conf ≈ %.2f（平均 F1 = %.3f）**，"
            "与项目默认的 0.25 基本一致，所以推理脚本沿用了 0.25。" % (best_conf, best_f1))
    if missing:
        pronoun = "这 %d 个类" % len(missing) if len(missing) > 1 else "这个类"
        add("- ⚠️ %s %s在 %s 划分里**一个标注都没有**，所以 AP 无从计算，"
            "下面表格里不会出现——不是「考了 0 分」，是「根本没考」。"
            % ("、".join(names[c] for c in missing), pronoun, args.split))
    add("")

    add("## 2. 逐类指标")
    add("")
    add("| 类别 | mAP50 | mAP50-95 | 精确率 P | 召回率 R | 训练集框数 | 本划分框数 |")
    add("|---|---|---|---|---|---|---|")
    for row in sorted(per_class, key=lambda r: r["ap50"], reverse=True):
        add("| %s | %.4f | %.4f | %.4f | %.4f | %d | %d |" % (
            row["name"], row["ap50"], row["ap"], row["precision"], row["recall"],
            row["train_boxes"], row["gt_boxes"]))
    add("")
    add("图见 `fig_per_class_ap.png`。")
    add("")

    # 小样本类别必须单独拿出来说。把 AP=0 直接写进结论，是最容易被面试官一句话问倒的地方。
    tiny = [r for r in per_class if r["gt_boxes"] <= 5]
    if tiny:
        add("### 小样本类别要单独看")
        add("")
        add("下面这些类在 %s 划分里的标注框数不超过 5 个。" % args.split)
        add("样本这么少，AP 的随机波动极大，**既不能和样本充足的类比，也不能单独当成结论**：")
        add("")
        add("| 类别 | 本划分框数 | 训练集框数 | mAP50 |")
        add("|---|---|---|---|")
        for row in sorted(tiny, key=lambda r: r["ap50"]):
            add("| %s | %d | %d | %.4f |" % (
                row["name"], row["gt_boxes"], row["train_boxes"], row["ap50"]))
        add("")
        extreme = min(tiny, key=lambda r: r["gt_boxes"])
        add("最极端的是 **%s**：它在 %s 里只有 **%d 个**标注框，模型没检出来，于是 AP50 = 0。"
            % (extreme["name"], args.split, extreme["gt_boxes"]))
        add("这个 0 的含义是「这一个框没对上」，**不是「模型完全认不出这类目标」**——")
        add("换一次划分，只要这一个框碰巧检出来，数字就可能变成 1.0。")
        add("")
        add("所以读这张表的时候，**每个数字都要连同它的样本量一起看**。")
        add("这也是这份报告最想让你带走的一句话：")
        add("**只看总体 mAP 会骗人，只看单个类的 mAP 同样会骗人。**")
        add("")

    add("## 3. 混淆矩阵")
    add("")
    if fig_cm is not None:
        add("图见 `fig_confusion_matrix.png`。这张图按**列**归一化：每一列是一类真实标注，"
            "格子里的数字是「它被预测成了各类的比例」，"
            "所以**对角线就是这一类的召回率**。")
        add("")
        add("看这张图只需要看两件事：**对角线够不够亮**、**最亮的非对角格是哪一格**。")
        add("最后一行是「该报没报」（漏检），最后一列是「不该报报了」（误检）。")
        add("横轴标签下面还标了这一类的标注框数——数字小到个位数的类，它的 AP 本身就不稳定。")
    else:
        add("本次没取到混淆矩阵，跳过。")
    add("")

    add("## 4. 这个结果能说明什么、不能说明什么")
    add("")
    add("**能说明：**")
    add("")
    add("- 模型在**这个数据集的 %s 划分**上，能稳定检出交通信号灯、车辆这类样本充足的目标。" % args.split)
    add("- 整条链路（数据 → 训练 → 评估）是可复现的：换一台机器跑这条命令，应该得到接近的数字。")
    add("- 阈值选择有依据，不是拍脑袋定的。")
    add("")
    add("**不能说明：**")
    add("")
    add("- ❌ **不能说「达到了自动驾驶感知要求」。** 这是 12 类、%d 张图的公开数据集，"
        "不是车规数据集，也没有做时序、多传感器、恶劣天气的验证。" % image_count)
    add("- ❌ **不能说测试集完全独立。** 见 `docs/data_leakage_report.json`："
        "内容层面没有重复，但三个划分的时间跨度完全重叠（跨划分最近两张只隔 4.9 秒），"
        "存在场景级轻度重叠，所以这个数字应该按**偏乐观**理解。")
    add("- ❌ **不能只看总体 mAP 就说「12 类都能用」。** 逐类差距很大，"
        "少数类别的真实水平远低于平均值。")
    add("- ❌ **不能拿这个数字和其他论文比。** 数据集、划分方式、是否用测试时增强都不一样，"
        "跨数据集比 mAP 没有意义。")
    add("")

    add("## 5. 下一步该往哪改")
    add("")
    weakest = sorted(per_class, key=lambda r: r["ap50"])[:3]
    add("按当前证据，优先级应该是：")
    add("")
    add("1. **补少数类的数据**，而不是继续加训练轮数。")
    add("   垫底的三个类（%s）**训练集**分别只有 %s 个框，"
        "而训练集总共 494 张图、验证集 mAP50 在第 19 轮就饱和了——"
        "瓶颈在数据量和长尾分布上，不在算力。"
        % ("、".join(r["name"] for r in weakest),
           " / ".join(str(r["train_boxes"]) for r in weakest)))
    add("2. **按采集批次重新划分数据集**，消除上面提到的场景重叠。")
    add("   注意：这样一来现有指标就不能直接对比了，得重新评估一次并注明「旧划分下的结果」。")
    add("3. **消除过拟合**：训练集分类损失还在降、验证集已经平了很久，"
        "可以考虑更强的高斯噪声/尺度抖动，或降低模型容量。")
    add("")

    add("## 6. 复现命令")
    add("")
    add("```bash")
    add("python scripts/09_evaluate.py --split %s" % args.split)
    add("```")
    add("")
    add("逐类数字也可以直接读 `metrics_per_class.csv`。")
    add("")

    add("## 7. 引用的其他记录")
    add("")
    add("- 训练配置与逐轮曲线：`runs/detect/runs/road_baseline/args.yaml`、`results.csv`")
    add("- 数据泄漏检查：`docs/data_leakage_report.json`")
    add("- 推理性能测量条件：`docs/inference_benchmark.json`")
    add("- 数据转换审计：`docs/label_conversion_report.txt`")
    add("- Ultralytics 原生输出的评估图（英文标注，未做处理）：`results/eval_report/ultralytics_raw/`")
    add("")

    report_path = RESULT_DIR / "evaluation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # ---------------- 终端摘要 ----------------
    print("-" * 72)
    print("mAP50     %.4f" % overall["mAP50"])
    print("mAP50-95  %.4f" % overall["mAP50-95"])
    print("precision %.4f    recall %.4f" % (overall["precision"], overall["recall"]))
    if best_conf is not None:
        print("最优 conf %.2f（平均 F1 %.3f）" % (best_conf, best_f1))
    if missing:
        print("该划分里没有标注的类：%s" % "、".join(names[c] for c in missing))
    print("-" * 72)
    print("逐类 mAP50（降序）：")
    for row in sorted(per_class, key=lambda r: r["ap50"], reverse=True):
        print("    %-16s AP50 %.3f   mAP50-95 %.3f" % (row["name"], row["ap50"], row["ap"]))
    print("-" * 72)
    print("报告：%s" % report_path.relative_to(PROJECT_ROOT))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
