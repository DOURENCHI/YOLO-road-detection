"""
在 valid 划分上对比四种格式的精度，回答「FP16 到底掉了多少分」。

为什么用 valid 而不是 test
    test 只允许评一次（2026-10-27 得 0.680），再用一次这个数就不再是「没被调过」的结果。
    valid 可以反复用，训练时每轮都在它上面选 best.pt，所以它衡量的是「和训练时同一个尺子」。
    比较两种精度之间的相对差异，用 valid 完全够。

四档
    best.pt       原始权重（训练时的参照）
    best.onnx     ORT FP32
    engine_fp32   TensorRT FP32
    engine_fp16   TensorRT FP16  ← 真正要看的

注意 batch=1：engine 是静态 batch=1 构建的，喂 batch>1 会直接报错。

用法
    python scripts/16_eval_valid_split.py
    python scripts/16_eval_valid_split.py --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = PROJECT_ROOT / "runs" / "detect" / "runs" / "road_baseline" / "weights"
DATA_YAML = PROJECT_ROOT / "datasets" / "road-traffic-detect" / "data.yaml"
REPORT = PROJECT_ROOT / "docs" / "deployment" / "valid_precision_compare.json"
TRT_HOME = Path(r"D:\TensorRT\TensorRT-10.8.0.43")

CANDIDATES = [
    ("pt", WEIGHT_DIR / "best.pt", "原始 PyTorch 权重（训练时的尺子）"),
    ("onnx", PROJECT_ROOT / "weights" / "best.onnx", "ONNX Runtime FP32"),
    ("engine_fp32", PROJECT_ROOT / "weights" / "best_fp32.engine", "TensorRT FP32"),
    ("engine_fp16", PROJECT_ROOT / "weights" / "best_fp16.engine", "TensorRT FP16"),
]


def setup_dll_paths() -> None:
    for d in (TRT_HOME / "lib", TRT_HOME / "bin",
              Path(r"D:\Miniconda\envs\yolo_road\Lib\site-packages\torch\lib")):
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    setup_dll_paths()
    from ultralytics import YOLO

    if not DATA_YAML.is_file():
        raise SystemExit("找不到 %s" % DATA_YAML)

    print("valid 划分上对比四种格式（batch=1、imgsz=%d）" % args.imgsz)
    print("注：test 只评过一次（mAP50 0.680），这里不碰它。\n")

    rows, detail = [], {}
    for tag, path, desc in CANDIDATES:
        if not path.is_file():
            print("跳过 %-12s 缺少 %s" % (tag, path.name))
            continue
        print("── %-12s %s" % (tag, desc))
        t0 = time.perf_counter()
        model = YOLO(str(path))
        m = model.val(data=str(DATA_YAML), split="val", imgsz=args.imgsz,
                      batch=1, conf=0.001, iou=0.7, verbose=False, plots=False)
        elapsed = time.perf_counter() - t0
        row = {
            "标签": tag, "说明": desc, "文件": str(path.name),
            "mAP50": round(float(m.box.map50), 4),
            "mAP50_95": round(float(m.box.map), 4),
            "精确率": round(float(m.box.mp), 4),
            "召回率": round(float(m.box.mr), 4),
            "耗时_s": round(elapsed, 1),
        }
        rows.append(row)
        detail[tag] = row
        print("   mAP50 %.4f　mAP50-95 %.4f　P %.4f　R %.4f　(%.0f s)"
              % (row["mAP50"], row["mAP50_95"], row["精确率"], row["召回率"], elapsed))

    if not rows:
        raise SystemExit("没有任何模型可评")

    base = rows[0]
    print("\n" + "=" * 62)
    print("%-13s %8s %10s %8s" % ("格式", "mAP50", "mAP50-95", "ΔmAP50"))
    for r in rows:
        d = r["mAP50"] - base["mAP50"]
        print("%-13s %8.4f %10.4f %+8.4f" % (r["标签"], r["mAP50"], r["mAP50_95"], d))

    report = {
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "划分": "val（data.yaml 里键名是 val，目录叫 valid）",
        "说明": "test 只评一次（mAP50 0.680），本表不涉及 test",
        "参数": {"imgsz": args.imgsz, "batch": 1, "conf": 0.001, "iou": 0.7},
        "结果": rows,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n报告：%s" % REPORT.relative_to(PROJECT_ROOT))

    if "engine_fp16" in detail and "engine_fp32" in detail:
        d = detail["engine_fp16"]["mAP50"] - detail["engine_fp32"]["mAP50"]
        print("FP16 相对 FP32：mAP50 %+.4f（%.2f%%）" % (d, 100 * d / detail["engine_fp32"]["mAP50"]))


if __name__ == "__main__":
    main()
