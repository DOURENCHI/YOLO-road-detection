"""
构建 INT8 量化引擎（TensorRT PTQ），并把「校准集是什么」写进报告。

为什么单独一个脚本
    量化和 FP32/FP16 不是同一类动作：它不是"换个计算精度"，而是"把权重和激活的
    数值范围压成 256 级整数"，需要一批真实图片统计每层激活的动态范围（校准）。
    所以除了产物，还多出两个必须留痕的东西：**校准集是谁**、**校准缓存文件**。
    没有这两条，半年后没人说得清这个 engine 是用什么数据量化出来的。

三个刻意的选择
    1) 校准用 **train**，不用 val/test
       框架默认 split="val"。但 valid 是本次量化效果的评估集，拿它校准等于让
       INT8 提前看过要考的题，分数会偏乐观。train 与 valid/test 完全不相交。
    2) 用 PTQ（训练后量化），不做 QAT
       QAT 要重新微调，耗时与风险都高一个量级；先看 PTQ 掉多少分，再决定要不要上 QAT。
    3) 报告里记 sha256 + 校准张数 + TensorRT 版本 + 显卡算力
       INT8 引擎绑定的东西比 FP16 更多（TRT 版本、SM 架构、校准缓存），
       换机器必须重建，不能拷。

用法
    python scripts/18_export_engine_int8.py
    python scripts/18_export_engine_int8.py --split train --fraction 1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "runs" / "road_baseline" / "weights" / "best.pt"
ENGINE_DIR = PROJECT_ROOT / "weights"
DATA_YAML = PROJECT_ROOT / "datasets" / "road-traffic-detect" / "data.yaml"
REPORT = PROJECT_ROOT / "docs" / "deployment" / "engine_int8_export_report.json"
TRT_HOME = Path(r"D:\TensorRT\TensorRT-10.8.0.43")


def setup_dll_paths() -> list[str]:
    """与 14 号脚本一致：TRT 10 的 DLL 在 lib/，不在 bin/。"""
    added = []
    for d in (TRT_HOME / "lib", TRT_HOME / "bin",
              Path(r"D:\Miniconda\envs\yolo_road\Lib\site-packages\torch\lib")):
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            added.append(str(d))
    return added


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_image_count(split: str) -> int:
    """数校准集有多少张图 —— 报告里必须有这个数，否则校准规模不可复现。

    必须显式传 DATA_YAML：split_dirs 默认读的是**源**数据集（road-traffic），
    而训练与导出用的是转换后的 road-traffic-detect。两者图片数相同，但路径不同，
    报告里写错路径就等于写错了校准来源。
    """
    from dataset_config import split_dirs
    folder, _ = split_dirs(split, DATA_YAML)
    if folder is None:
        raise SystemExit("%s 里没有 %s 划分" % (DATA_YAML.name, split))
    return len(list(Path(folder).glob("*.jpg")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val"],
                    help="校准集，默认 train（不能与评估集重叠）")
    ap.add_argument("--fraction", type=float, default=1.0, help="只取校准集的这个比例")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workspace", type=int, default=4, help="构建显存上限 GB")
    args = ap.parse_args()

    added = setup_dll_paths()
    print("DLL 搜索路径：")
    for d in added:
        print("   " + d)

    try:
        import tensorrt as trt
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("import 失败：%s\n确认 TensorRT python 包与 DLL 路径。" % e)

    print("\nTensorRT %s" % trt.__version__)
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print("GPU %s  cc %d.%d" % (name, props.major, props.minor))

    if not WEIGHTS.is_file():
        raise SystemExit("找不到权重：%s" % WEIGHTS)
    if not DATA_YAML.is_file():
        raise SystemExit("找不到数据配置：%s" % DATA_YAML)

    n_calib = split_image_count(args.split)
    print("\n校准集：%s　%d 张（fraction %.2f → 约 %d 张参与统计）"
          % (args.split, n_calib, args.fraction, int(n_calib * args.fraction)))
    print("注意：valid 与 test 不参与校准，保证量化效果评估干净。")

    target = ENGINE_DIR / "best_int8.engine"
    print("\n── 构建 INT8 engine（PTQ，MinMax 校准）…")
    t0 = time.perf_counter()
    model = YOLO(str(WEIGHTS))
    out = model.export(
        format="engine",
        imgsz=640,
        quantize=8,             # 这个版本用 quantize 控制精度档位，不是老的 int8=
        data=str(DATA_YAML),
        split=args.split,       # 框架默认 val，这里显式改掉
        fraction=args.fraction,
        batch=args.batch,
        workspace=args.workspace,
        device=0,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    produced = Path(out)
    if not produced.is_file():
        raise SystemExit("导出未产出文件：%s" % produced)
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(target))

    size_mb = target.stat().st_size / 1048576
    print("   %s  %.1f MB  构建 %.1f s" % (target.name, size_mb, elapsed))

    # 顺带记一份 calibrator 可能写出的缓存，便于复现（有就记，没有不报错）
    caches = []
    for pat in ("*.calib", "*.cache", "*calibration*"):
        for p in (PROJECT_ROOT / "weights").glob(pat):
            caches.append({"file": str(p.relative_to(PROJECT_ROOT)),
                           "size_kb": round(p.stat().st_size / 1024, 1),
                           "sha256": sha256_of(p)})

    report = {
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "源权重": {"path": str(WEIGHTS.relative_to(PROJECT_ROOT)), "sha256": sha256_of(WEIGHTS)},
        "量化方式": "PTQ（训练后量化）+ TensorRT MinMax 校准器（日志原文 TRT-100800-MinMaxCalibration，由 ultralytics 调用）",
        "校准集": {
            "split": args.split,
            "路径": str(DATA_YAML.relative_to(PROJECT_ROOT)),
            "图片数": n_calib,
            "fraction": args.fraction,
            "实际参与张数_约": int(n_calib * args.fraction),
            "为什么用 train": "valid 是量化效果的评估集，用它校准会让分数偏乐观；"
                              "train 与 valid/test 完全不相交",
        },
        "环境": {
            "tensorrt": trt.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": name,
            "compute_capability": "%d.%d" % (props.major, props.minor),
            "tensorrt_home": str(TRT_HOME),
        },
        "产物": {
            "文件": str(target.relative_to(PROJECT_ROOT)),
            "size_mb": round(size_mb, 2),
            "构建耗时_s": round(elapsed, 1),
            "sha256": sha256_of(target),
            "参数": {"imgsz": 640, "batch": args.batch,
                     "workspace_GB": args.workspace, "quantize": 8},
        },
        "校准缓存": caches or "未生成独立缓存文件（框架内联处理）",
        "限制": "engine 绑定 TensorRT %s + %s + 校准缓存，换显卡或换 TRT 版本必须重建"
                % (trt.__version__, name),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n报告：%s" % REPORT.relative_to(PROJECT_ROOT))
    print("下一步：python scripts/19_verify_int8.py    # 四层验证 + 延迟对照")
    print("       python scripts/16_eval_valid_split.py # valid 上 mAP 对照（已含 int8）")


if __name__ == "__main__":
    sys.exit(main())
