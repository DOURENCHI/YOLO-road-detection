"""
在 valid 划分上对比五种格式的精度，回答「FP16 / INT8 到底掉了多少分」。

为什么用 valid 而不是 test
    test 只允许评一次（2026-10-27 得 0.680），再用一次这个数就不再是「没被调过」的结果。
    valid 可以反复用，训练时每轮都在它上面选 best.pt，所以它衡量的是「和训练时同一个尺子」。
    比较两种精度之间的相对差异，用 valid 完全够。

五档
    best.pt       原始权重（训练时的参照）
    best.onnx     ORT FP32
    engine_fp32   TensorRT FP32   ← 判断另两档的基准（同一条推理路径）
    engine_fp16   TensorRT FP16
    engine_int8   TensorRT INT8（PTQ，校准集 = train）

判断规则
    FP16 / INT8 的好坏都跟「同路径的 engine_fp32」比，不跟 pt 比。
    pt → onnx 本身就换了后处理路径，掉的那部分不该算在量化头上。

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
    ("engine_int8", PROJECT_ROOT / "weights" / "best_int8.engine",
     "TensorRT INT8（PTQ，校准集 train）"),
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

    print("valid 划分上对比五种格式（batch=1、imgsz=%d）" % args.imgsz)
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
            # 逐类 mAP50-95：量化最典型的失效方式是「某一类整体崩」，
            # 只看总 mAP 会把「一类崩 + 其余正常」平均成一个温和的下降
            "逐类_mAP50_95": {model.names[i]: round(float(v), 4)
                              for i, v in enumerate(m.box.maps)},
        }
        rows.append(row)
        detail[tag] = row
        print("   mAP50 %.4f　mAP50-95 %.4f　P %.4f　R %.4f　(%.0f s)"
              % (row["mAP50"], row["mAP50_95"], row["精确率"], row["召回率"], elapsed))
        worst = sorted(row["逐类_mAP50_95"].items(), key=lambda kv: kv[1])[:3]
        print("   最差三类：" + "　".join("%s %.3f" % kv for kv in worst))

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

    # 量化档位的基准是「同路径 FP32 引擎」，不是 pt
    ref = detail.get("engine_fp32")
    if ref:
        for tag, label in (("engine_fp16", "FP16"), ("engine_int8", "INT8")):
            if tag not in detail:
                continue
            d = detail[tag]["mAP50"] - ref["mAP50"]
            print("%-4s 相对 FP32：mAP50 %+.4f（%+.2f%%）　mAP50-95 %+.4f"
                  % (label, d, 100 * d / ref["mAP50"],
                     detail[tag]["mAP50_95"] - ref["mAP50_95"]))
            verdict = "可接受" if d > -0.01 else ("需权衡" if d > -0.03 else "不建议")
            print("     判断：%s（>-0.01 可接受，-0.03~-0.01 需权衡，<-0.03 不建议）" % verdict)


if __name__ == "__main__":
    main()
