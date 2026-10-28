"""
导出 TensorRT engine（FP32 与 FP16 各一份），并把构建参数与环境留痕。

为什么单独写脚本而不是敲一行 yolo export
    构建 engine 是个「一次性、耗时、不可复现」的动作：
      · 产物绑 TensorRT 版本 + GPU 架构/SM 数 + 输入尺寸，换机器就作废
      · 构建要几分钟，事后想解释「这份引擎是怎么来的」只能靠记录
      · FP32 与 FP16 会互相覆盖同名文件，必须导出即改名
    所以这里把版本、GPU、耗时、大小、sha256 全部落到 JSON。

用法
    python scripts/14_export_engine.py                # 导出 fp32 + fp16
    python scripts/14_export_engine.py --only fp16
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
WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "runs" / "road_baseline" / "weights" / "best.pt"
ENGINE_DIR = PROJECT_ROOT / "weights"
REPORT = PROJECT_ROOT / "docs" / "deployment" / "engine_export_report.json"

TRT_HOME = Path(r"D:\TensorRT\TensorRT-10.8.0.43")


def setup_dll_paths() -> list[str]:
    """TensorRT 与 cuDNN/CUDA 的 DLL 都要能被找到。

    TensorRT 的 Windows 包不是 pip 安装，`nvinfer_10.dll` 不会进 PATH；
    cuDNN 与 cuBLAS 则由 torch 自带（torch/lib）。缺任何一个都会在
    `import tensorrt` 或构建引擎时直接崩，而且报错信息很不明确。
    """
    added = []
    # TRT 10 的 Windows 布局：DLL 在 lib/（不是 bin/，bin 里只有 trtexec.exe）。
    # 这一点很容易踩：加错目录后 import tensorrt 直接崩，报错却只说是 DLL 加载失败。
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


def export_one(half: bool) -> dict:
    """导出单个精度。ultralytics 会把结果写成 best.engine，导出后立刻改名。"""
    from ultralytics import YOLO

    tag = "fp16" if half else "fp32"
    target = ENGINE_DIR / f"best_{tag}.engine"

    print("\n── 构建 %s engine（%s）…" % (tag.upper(), "内部 FP16 计算" if half else "全程 FP32"))
    t0 = time.perf_counter()
    model = YOLO(str(WEIGHTS))
    out = model.export(
        format="engine",
        imgsz=640,
        half=half,
        device=0,
        batch=1,
        workspace=4,        # GB，构建时的显存上限
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    produced = Path(out)
    if not produced.is_file():
        raise SystemExit("导出未产出文件：%s" % produced)
    shutil.move(str(produced), str(target))

    size_mb = target.stat().st_size / 1048576
    print("   %s  %.1f MB  构建 %.1f s" % (target.name, size_mb, elapsed))
    return {
        "精度": tag,
        "文件": str(target.relative_to(PROJECT_ROOT)),
        "size_mb": round(size_mb, 2),
        "构建耗时_s": round(elapsed, 1),
        "sha256": sha256_of(target),
        "参数": {"imgsz": 640, "batch": 1, "workspace_GB": 4, "half": half, "device": 0},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["fp32", "fp16"], help="只导出其中一种")
    args = ap.parse_args()

    added = setup_dll_paths()
    print("DLL 搜索路径：")
    for d in added:
        print("   " + d)

    try:
        import tensorrt as trt
        import torch
    except ImportError as e:
        raise SystemExit("import 失败：%s\n先确认 TensorRT 的 python 包已装、DLL 路径已加。" % e)

    print("\nTensorRT %s" % trt.__version__)
    print("torch %s (cuda %s)" % (torch.__version__, torch.version.cuda))
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print("GPU %s  cc %d.%d  sm_%d%d  %d MiB" % (name, props.major, props.minor,
                                                props.major, props.minor,
                                                props.total_memory // 1024 // 1024))

    if not WEIGHTS.is_file():
        raise SystemExit("找不到权重：%s" % WEIGHTS)
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    if args.only != "fp16":
        results.append(export_one(half=False))
    if args.only != "fp32":
        results.append(export_one(half=True))

    report = {
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "源权重": {"path": str(WEIGHTS.relative_to(PROJECT_ROOT)), "sha256": sha256_of(WEIGHTS)},
        "环境": {
            "tensorrt": trt.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": name,
            "compute_capability": "%d.%d" % (props.major, props.minor),
            "tensorrt_home": str(TRT_HOME),
        },
        "产物": results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n报告：%s" % REPORT.relative_to(PROJECT_ROOT))
    print("提示：engine 绑定 TensorRT %s + %s，换显卡或换版本必须重建。" % (trt.__version__, name))


if __name__ == "__main__":
    sys.exit(main())
