"""
08 号脚本：视频多目标跟踪（检测 + 跨帧身份关联）。

和 06 号脚本的区别
------------------
  06 号：一帧一帧独立地检测，第 1 帧里的车和第 2 帧里的车是两个互不相干的框。
  08 号：在检测的基础上加了「跟踪器」，给同一个目标分配一个**固定的 ID**，
         让它在连续帧之间保持同一个身份。这就是 MOT（Multi-Object Tracking）。

顺手还带来了一个能力：**去重计数**。
逐帧检测时，同一辆车在 100 帧里会被数成 100 次；
跟踪之后它只有一个 ID，可以回答"这段视频里一共出现了多少辆车"。

跟踪器用的是 Ultralytics 内置的 ByteTrack（只用检测框 + 卡尔曼滤波预测位置 + IoU 关联），
没有用任何外观特征。这意味着两件事：
  - 好处：快，不额外耗算力
  - 代价：目标长时间被遮挡后重新出现，很可能被分配成新 ID（俗称 ID switch）

⚠️ 关于本项目的数据集，有一件事必须说清楚：
datasets/road-traffic 的 814 张图**不是连续帧**。实测相邻两张的时间戳最小间隔 1.09 秒，
普遍在 4.8~5.5 秒。它是按固定间隔抽帧的快照集，不是视频。
所以**不要拿这个数据集去跑跟踪评估**——目标在 5 秒里早就跑出画面了，
运动模型完全失效，ID 会疯狂新建，得到的数字没有意义。
想演示跟踪，用 --source 指一段**真实的道路视频**（手机拍 20 秒就够）。

支持的输入
----------
  --source 某段视频.mp4        真实视频（推荐）
  --source 0                   摄像头实时
  --source 某个目录/            把目录里的图片当成一个序列（⚠️ 仅用于验证链路能跑通，
                               若图片之间间隔很大，ID 无意义，脚本会提示）

输出
----
  results/track/<name>/track.mp4        带框和 ID 的视频
  results/track/<name>/track_summary.json  轨迹统计
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import (  # noqa: E402
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    TARGET_YAML,
    load_names,
)

DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "runs" / "road_baseline" / "weights" / "best.pt"
RESULT_ROOT = PROJECT_ROOT / "results" / "track"

#: 图片序列合成视频时的默认帧率。
#: 故意压到 2 fps：本项目数据集相邻图间隔约 5 秒，放太快看着像跳帧。
DEFAULT_SEQ_FPS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对视频 / 摄像头 / 图片序列做多目标跟踪，结果存到 results/track/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例子：\n"
            "  python scripts/08_track.py --source assets/videos/road.mp4\n"
            "  python scripts/08_track.py --source 0 --max-frames 300      # 摄像头跑 300 帧\n"
            "  python scripts/08_track.py --tracker botsort.yaml           # 换跟踪器对比\n"
        ),
    )
    parser.add_argument("--source", required=True,
                        help="视频文件 / 摄像头编号（如 0）/ 图片目录")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="权重文件")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        help="跟踪器配置，可选 bytetrack.yaml / botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS 的 IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入边长")
    parser.add_argument("--device", default="0", help="显卡编号，或 'cpu'")
    parser.add_argument("--name", default="demo", help="结果子目录名")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0 表示不限")
    parser.add_argument("--fps", type=float, default=DEFAULT_SEQ_FPS,
                        help="仅对图片目录生效：合成视频的帧率")
    parser.add_argument("--json-out", default=None, help="统计写到哪里")
    return parser.parse_args()


class FrameReader:
    """
    把三种输入统一成「一帧一帧地吐 BGR 图像」。

    视频和摄像头都用 cv2.VideoCapture；
    图片目录自己维护一个文件列表，按文件名（本项目就是时间戳）排序当序列。
    """

    def __init__(self, source: str, fps: float):
        import cv2  # 延迟 import

        self.cv2 = cv2
        self.is_sequence = False
        self.files: list[Path] = []
        self.cap = None
        self.fps = fps

        raw = source.strip()
        if raw.isdigit():
            self.cap = cv2.VideoCapture(int(raw))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.description = "摄像头 %s" % raw
            return

        path = Path(raw)
        if not path.is_absolute():
            candidates = [Path.cwd() / path, PROJECT_ROOT / path]
            path = next((c for c in candidates if c.exists()), candidates[0])

        if not path.exists():
            raise FileNotFoundError("找不到输入：%s" % path)

        if path.is_dir():
            self.is_sequence = True
            self.files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            if not self.files:
                raise FileNotFoundError("目录里没有图片：%s" % path)
            self.description = "图片序列 %s（%d 张）" % (path.name, len(self.files))
            return

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError("打不开视频文件：%s" % path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.description = "视频 %s" % path.name

    def frames(self):
        """产出 (帧号, BGR 图)。帧号从 1 开始。"""
        if self.is_sequence:
            for index, file in enumerate(self.files, 1):
                image = self.cv2.imread(str(file))
                if image is None:
                    print("  跳过读不出来的图：%s" % file.name)
                    continue
                yield index, image
            return

        index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            index += 1
            yield index, frame

    def width_height(self):
        if self.cap is not None:
            return (int(self.cap.get(self.cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT)))
        image = self.cv2.imread(str(self.files[0]))
        return (image.shape[1], image.shape[0])

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def warn_if_sequence(self):
        """图片序列不适合做跟踪评估时，明确提醒，避免把结果读歪。"""
        if not self.is_sequence:
            return
        print("⚠️  输入是图片序列，不是视频。")
        print("   如果这些图不是连续帧，跟踪器的运动模型会失效，ID 会频繁新建——")
        print("   此时这次运行只能证明「链路跑通了」，不能用来评价跟踪质量。")


def make_writer(cv2, out_path: Path, fps: float, size):
    """优先写 mp4；编码器不可用时退回写 avi，并告诉调用方实际扩展名。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, max(fps, 1.0), size)
    if writer.isOpened():
        return writer, out_path

    writer.release()
    fallback = out_path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"XVID"), max(fps, 1.0), size)
    if writer.isOpened():
        print("   mp4 编码器不可用，改写 avi")
        return writer, fallback
    raise RuntimeError("视频写入器打不开，可能是 OpenCV 缺少视频编码支持")


def main() -> int:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        print("找不到权重文件：%s" % weights, file=sys.stderr)
        return 1

    reader = FrameReader(args.source, args.fps)
    names = load_names(TARGET_YAML)

    out_dir = RESULT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("输入    : %s" % reader.description)
    print("权重    : %s" % weights.relative_to(PROJECT_ROOT))
    print("跟踪器  : %s   conf=%.2f  imgsz=%d  device=%s" % (
        args.tracker, args.conf, args.imgsz, args.device))
    print("输出    : %s" % out_dir.relative_to(PROJECT_ROOT))
    print("=" * 72)
    reader.warn_if_sequence()

    from ultralytics import YOLO

    model = YOLO(str(weights))

    width, height = reader.width_height()
    if width <= 0 or height <= 0:
        width, height = 640, 640

    writer, actual_video = make_writer(reader.cv2, out_dir / "track.mp4", reader.fps, (width, height))

    # 轨迹台账：ID -> 这条轨迹的信息
    tracks: dict[int, dict] = defaultdict(lambda: {"classes": Counter(), "first": None, "last": None, "frames": 0})
    boxes_per_frame: list[int] = []
    frame_counter = 0
    started = time.perf_counter()

    try:
        for frame_index, frame in reader.frames():
            if args.max_frames and frame_index > args.max_frames:
                break

            # persist=True 是关键：不加它，每帧都会重新初始化跟踪器，ID 永远从 1 开始
            result = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]

            frame_counter += 1
            boxes = result.boxes
            ids = boxes.id.tolist() if (boxes is not None and boxes.id is not None) else []
            classes = boxes.cls.tolist() if boxes is not None else []
            boxes_per_frame.append(len(classes))

            for track_id, class_id in zip(ids, classes):
                tid = int(track_id)
                record = tracks[tid]
                record["classes"][names[int(class_id)]] += 1
                record["frames"] += 1
                if record["first"] is None:
                    record["first"] = frame_index
                record["last"] = frame_index

            writer.write(result.plot())

            if frame_index % 25 == 0 or frame_index == 1:
                print("  第 %4d 帧：本帧 %2d 个目标，累计轨迹 %3d 条" % (
                    frame_index, len(classes), len(tracks)))

    finally:
        writer.release()
        reader.release()

    elapsed = time.perf_counter() - started

    # ---------------- 统计 ----------------
    total_boxes = sum(boxes_per_frame)
    by_class_ids: Counter = Counter()
    lifetimes: list[int] = []
    for info in tracks.values():
        main_class = info["classes"].most_common(1)[0][0]
        by_class_ids[main_class] += 1
        lifetimes.append(info["frames"])

    one_frame_ids = sum(1 for n in lifetimes if n == 1)

    print("-" * 72)
    print("处理帧数      : %d" % frame_counter)
    print("帧均目标数    : %.2f（合计 %d 个框）" % (total_boxes / frame_counter if frame_counter else 0, total_boxes))
    print("轨迹（ID）总数: %d" % len(tracks))
    if lifetimes:
        lifetimes_sorted = sorted(lifetimes)
        print("轨迹寿命      : 中位 %d 帧，最短 %d，最长 %d" % (
            lifetimes_sorted[len(lifetimes_sorted) // 2], lifetimes_sorted[0], lifetimes_sorted[-1]))
        print("只活了 1 帧的 : %d 条（占 %.1f%%）—— 这个比例高，通常说明关联不稳或场景变化太快" % (
            one_frame_ids, 100 * one_frame_ids / len(tracks)))
    if by_class_ids:
        print("按类别统计的轨迹数：")
        for label, count in by_class_ids.most_common():
            print("    %-16s %3d 条" % (label, count))
    print("平均 %.1f FPS（含检测 + 跟踪 + 写视频）" % (
        frame_counter / elapsed if elapsed else 0))
    print("-" * 72)
    print("带 ID 的视频：%s" % actual_video.relative_to(PROJECT_ROOT))
    print("⚠️  「轨迹总数」不等于「画面里出现的真实目标数」。")
    print("   目标被遮挡后重现会被分配新 ID，所以真实数量通常**少于**这个数字。")

    summary = {
        "输入说明": reader.description,
        "权重": str(weights),
        "跟踪器": args.tracker,
        "参数": {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz, "device": str(args.device)},
        "处理帧数": frame_counter,
        "框总数": total_boxes,
        "帧均目标数": round(total_boxes / frame_counter, 2) if frame_counter else None,
        "轨迹总数": len(tracks),
        "只出现1帧的轨迹数": one_frame_ids,
        "轨迹寿命_中位帧": sorted(lifetimes)[len(lifetimes) // 2] if lifetimes else None,
        "按类别的轨迹数": dict(by_class_ids),
        "输出视频": str(actual_video),
        "总耗时_秒": round(elapsed, 2),
        "说明": "轨迹总数不等于真实目标数；遮挡后重现会分配新 ID。",
    }

    json_path = Path(args.json_out) if args.json_out else (out_dir / "track_summary.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("统计写到：%s" % json_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
