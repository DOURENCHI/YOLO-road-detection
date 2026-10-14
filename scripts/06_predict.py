"""
06 号脚本：统一推理入口。

这是项目里第一个「把图喂进去、把检测结果拿出来」的脚本。
前面 01~05 号加 visualize_labels.py 管的都是**数据**，
06 号开始管**模型**：加载权重 -> 前向计算 -> 拿到框 -> 存图存数。

支持的输入
----------
  --split test            直接用数据集的某个划分（推荐演示用 valid，把 test 留给正式评估）
  --source 某张图.jpg
  --source 某个目录/
  --source 某段视频.mp4
  --source 0              摄像头（0 是默认摄像头编号）

两种输入二选一，都不给的话默认跑 valid 划分的前 8 张。

输出
----
  results/predict/<name>/          画好框的图片或视频
  results/predict/<name>/summary.json   本次推理的统计（框数、类别分布、耗时）

怎么读输出
----------
  打印里会有「检出 N 个目标」，这是**模型输出的框数**，不是正确率。
  正确率要看 09 号脚本的评估报告。
  这两件事必须分开说——模型敢不敢报，和它报得对不对，是两回事。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# 让脚本从任何目录启动都能 import 到同目录的 dataset_config
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import (  # noqa: E402
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    SPLITS,
    TARGET_YAML,
    load_names,
    split_dirs,
)

#: 默认权重：阶段三正式训练出来的最优模型
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "runs" / "road_baseline" / "weights" / "best.pt"

#: 推理结果统一放这里
RESULT_ROOT = PROJECT_ROOT / "results" / "predict"

#: --conf 的默认值。
#: 0.25 不是随手填的：09 号评估脚本会在验证集上扫一遍置信度阈值，
#: 实测 F1 峰值出现在 0.25 附近，和 Ultralytics 的默认值一致。
DEFAULT_CONF = 0.25

#: IoU 阈值，控制 NMS 阶段多大重叠算同一个目标
DEFAULT_IOU = 0.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对图片 / 目录 / 视频 / 摄像头跑检测，结果存到 results/predict/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例子：\n"
            "  python scripts/06_predict.py                          # 验证集前 8 张\n"
            "  python scripts/06_predict.py --split valid --limit 20  # 验证集抽 20 张\n"
            "  python scripts/06_predict.py --source assets/demo      # 跑自己放的图\n"
            "  python scripts/06_predict.py --source 路测视频.mp4      # 跑视频\n"
            "  python scripts/06_predict.py --source 0                 # 摄像头实时\n"
            "  python scripts/06_predict.py --conf 0.4 --name conf04   # 换个阈值再跑一遍\n"
        ),
    )
    parser.add_argument("--source", default=None,
                        help="图片 / 目录 / 视频 / 摄像头编号（给 0 就是默认摄像头）")
    parser.add_argument("--split", default=None, choices=list(SPLITS),
                        help="改用它指定数据集的某个划分作为输入（与 --source 二选一）")
    parser.add_argument("--limit", type=int, default=8,
                        help="配合 --split 或目录使用，只跑前 N 张；给 0 表示不限制")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="权重文件")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help="置信度阈值，低于它的框直接丢掉")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS 的 IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="输入边长")
    parser.add_argument("--device", default="0", help="显卡编号，或 'cpu'")
    parser.add_argument("--name", default="demo", help="结果子目录名")
    parser.add_argument("--save-txt", action="store_true",
                        help="额外把检测框存成 YOLO 格式的 txt（和标签同格式，便于对比）")
    parser.add_argument("--json-out", default=None,
                        help="统计写到指定路径，默认写到结果目录下的 summary.json")
    return parser.parse_args()


def resolve_source(args: argparse.Namespace):
    """
    把参数翻译成 Ultralytics 能吃的 source，并返回一句人话说明。

    返回 (source, 说明)。source 可能是「路径字符串」「路径列表」或「摄像头编号 int」。
    """
    # ---- 优先用数据集划分 ----
    if args.split:
        images_dir, _ = split_dirs(args.split, TARGET_YAML)
        if images_dir is None or not images_dir.is_dir():
            raise FileNotFoundError(
                "在 data.yaml 里找不到 %s 划分，或目录不存在：%s" % (args.split, images_dir)
            )
        files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if args.limit:
            files = files[: args.limit]
        if not files:
            raise FileNotFoundError("%s 里没有图片：%s" % (args.split, images_dir))
        return [str(p) for p in files], "%s 划分的 %d 张图" % (args.split, len(files))

    # ---- 没给任何输入，走默认：验证集前 N 张 ----
    if args.source is None:
        images_dir, _ = split_dirs("valid", TARGET_YAML)
        files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if args.limit:
            files = files[: args.limit]
        return [str(p) for p in files], "未指定输入，默认跑 valid 划分前 %d 张" % len(files)

    raw = args.source.strip()

    # ---- 摄像头 ----
    if raw.isdigit():
        return int(raw), "摄像头 %s" % raw

    path = Path(raw)
    if not path.is_absolute():
        # 相对路径按「当前工作目录」理解，同时也试一下项目根目录，
        # 免得在别的目录下执行时找不到自己的文件
        candidates = [Path.cwd() / path, PROJECT_ROOT / path]
        path = next((c for c in candidates if c.exists()), candidates[0])

    if not path.exists():
        raise FileNotFoundError("找不到输入：%s" % path)

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if args.limit:
            files = files[: args.limit]
        if not files:
            raise FileNotFoundError("目录里没有图片：%s" % path)
        return [str(p) for p in files], "目录 %s 的 %d 张图" % (path.name, len(files))

    return str(path), "单个文件 %s" % path.name


def main() -> int:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        print("找不到权重文件：%s" % weights, file=sys.stderr)
        print("如果还没训练过，先按 README 第 5 节跑一次训练。", file=sys.stderr)
        return 1

    source, description = resolve_source(args)
    names = load_names(TARGET_YAML)

    out_dir = RESULT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 延迟 import：--help 不必等 torch 加载
    from ultralytics import YOLO

    print("=" * 72)
    print("输入    : %s" % description)
    print("权重    : %s" % weights.relative_to(PROJECT_ROOT))
    print("阈值    : conf=%.2f  iou=%.2f  imgsz=%d  device=%s" % (
        args.conf, args.iou, args.imgsz, args.device))
    print("输出    : %s" % out_dir.relative_to(PROJECT_ROOT))
    print("=" * 72)

    model = YOLO(str(weights))

    started = time.perf_counter()
    # project 故意传**绝对路径**：Ultralytics 的输出路径 = cwd + project + name，
    # 给相对路径的话，换个目录启动就会多套一层（这个坑见 README 6.3）。
    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        project=str(RESULT_ROOT),
        name=args.name,
        exist_ok=True,
        stream=True,
        verbose=False,
    )

    per_item = []
    class_counter: Counter = Counter()
    total_boxes = 0
    total_time = 0.0

    for index, result in enumerate(results, 1):
        item = Path(result.path).name
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            count = 0
            labels, confs = [], []
        else:
            classes = boxes.cls.tolist()
            confs = boxes.conf.tolist()
            labels = [names[int(c)] for c in classes]
            count = len(labels)
            class_counter.update(labels)

        total_boxes += count
        speed = getattr(result, "speed", {}) or {}
        item_ms = sum(float(v) for v in speed.values()) if speed else 0.0
        total_time += item_ms

        per_item.append({
            "file": item,
            "boxes": count,
            "classes": dict(Counter(labels)),
            "mean_conf": round(sum(confs) / len(confs), 4) if confs else None,
            "ms": round(item_ms, 2),
        })

        head = "  [%3d] %-46s 检出 %2d 个" % (index, item[:46], count)
        detail = ""
        if count:
            top = Counter(labels).most_common(3)
            detail = "  (" + ", ".join("%s x%d" % (k, v) for k, v in top) + ")"
        print(head + detail)

    wall = time.perf_counter() - started

    print("-" * 72)
    print("共处理 %d 项，框总数 %d" % (len(per_item), total_boxes))
    if class_counter:
        print("类别分布：")
        for label, count in class_counter.most_common():
            print("    %-16s %3d 个  (%.1f%%)" % (label, count, 100 * count / total_boxes))
    else:
        print("一个框都没有。可能是阈值太高，也可能这批图里确实没有这 12 类目标。")
    if per_item:
        print("Ultralytics 计时的合计 %.1f ms，平均每项 %.1f ms" % (total_time, total_time / len(per_item)))
    print("整条流程（含读盘、建对象、写结果）实际耗时 %.2f 秒" % wall)
    print("-" * 72)

    # 结果图/视频在哪个子目录，直接告诉使用者，省得去翻
    saved = None
    for sub in ("", "images", "videos"):
        candidate = out_dir / sub if sub else out_dir
        if candidate.is_dir() and any(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".mp4", ".avi"}
                                      for p in candidate.iterdir() if p.is_file()):
            saved = candidate
            break
    if saved is not None:
        print("带框的结果在这里：%s" % saved.relative_to(PROJECT_ROOT))

    summary = {
        "输入说明": description,
        "权重": str(weights),
        "参数": {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz, "device": str(args.device)},
        "项数": len(per_item),
        "框总数": total_boxes,
        "类别分布": dict(class_counter),
        "平均每项耗时_ms": round(total_time / len(per_item), 2) if per_item else None,
        "流程总耗时_秒": round(wall, 2),
        "逐项": per_item,
    }

    json_path = Path(args.json_out) if args.json_out else (out_dir / "summary.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("统计写到：%s" % json_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
