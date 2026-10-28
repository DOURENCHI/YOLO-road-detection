"""
生成用于「视频流 + 长时间稳定性」测试的合成视频。

为什么不直接下一个真实道路视频
----------------------------
试过 Ultralytics 官方示例（decelera_landscape_min.mov），内容是「Decelera Mayakoba」
活动演讲现场（沙滩 + 讲者），不是道路场景，检测器在上面几乎全是误检。
其他免费源（Pexels / Pixabay）取直链需要 API key，公开数据集（BDD100K）要签署授权。
所以改用项目自己的 valid 图片合成——**画面是真实道路场景**，而且帧数、分辨率可控。

对稳定性测试来说，帧间是否「连续运动」不影响结论：
    要测的是 FPS、p99 延迟、内存增长，这些只跟「连续处理 N 帧」有关，
    跟画面内容连不连续无关。合成视频反而更好——每帧都不一样，比循环 11 帧
    更能暴露解码/缓存问题。

每帧左上角烧了帧号，方便核对解码有没有跳帧或重复。

用法
    python scripts/17_make_test_video.py                    # 1280x720 / 1000 帧
    python scripts/17_make_test_video.py --frames 300 --size 640 640
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dataset_config import split_dirs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1000, help="总帧数")
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720], metavar=("W", "H"))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default="assets/videos/_stability_test.mp4")
    ap.add_argument("--split", default="valid", help="图片来源划分（用 valid，不碰 test）")
    args = ap.parse_args()

    folder, _ = split_dirs(args.split)
    images = sorted(Path(folder).glob("*.jpg"))
    if not images:
        print("找不到图片：%s" % folder, file=sys.stderr)
        return 2

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = args.size
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
    if not writer.isOpened():
        print("VideoWriter 打不开（编码器不可用？）", file=sys.stderr)
        return 2

    print("图片 %d 张 → 循环填充成 %d 帧，%dx%d @ %.0f fps"
          % (len(images), args.frames, w, h, args.fps))

    written = 0
    while written < args.frames:
        for path in images:
            if written >= args.frames:
                break
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            # 烧帧号，方便核对解码有没有跳帧/重复
            cv2.putText(frame, "frame %04d" % written, (18, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(frame, "frame %04d" % written, (18, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
            writer.write(frame)
            written += 1
    writer.release()

    mb = out_path.stat().st_size / 1048576
    print("产出 %s（%.1f MB，%d 帧，%.1f 秒）"
          % (out_path.relative_to(PROJECT_ROOT), mb, written, args.frames / args.fps))

    # 回读校验：确认帧数、分辨率、能否逐帧解码
    cap = cv2.VideoCapture(str(out_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    read = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        read += 1
    cap.release()
    print("回读校验：报告 %d 帧 / 实际解出 %d 帧 / %dx%d" % (n, read, fw, fh))
    return 0 if read >= args.frames * 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
