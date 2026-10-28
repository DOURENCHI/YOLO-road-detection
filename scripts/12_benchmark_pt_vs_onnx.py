"""
12 号脚本：PyTorch 与 ONNX Runtime 的同条件性能对比。

为什么不能直接拿两边的「官方数字」比
------------------------------------
和 benchmark_inference.py 是同一个道理：换条件测出来的数不可比。所以本脚本对两边用
**完全相同的三件事**：

    · 同一批图片（valid 排序后前 N 张，固定不变）
    · 同样的预热次数与重复次数
    · 同样的计时边界：纯推理 / 端到端 分开算

并且在 Session 创建前做一件容易被忽略的事：**把 torch 自带的 CUDA 运行库目录加进 DLL 搜索路径**。
onnxruntime-gpu 依赖 cublasLt / cuDNN，这些 DLL 由 torch 提供；不注册目录的话 ORT 会静默回落到 CPU，
程序照样能跑、结果照样对，只是慢十倍——这是最容易误判的一条。

用法
----
    python scripts/12_benchmark_pt_vs_onnx.py                 # 默认 n=200，warmup=30
    python scripts/12_benchmark_pt_vs_onnx.py --runs 50 --warmup 10   # 快速看趋势
    python scripts/12_benchmark_pt_vs_onnx.py --providers cpu          # 强制 CPU 对照
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import PROJECT_ROOT, split_dirs  # noqa: E402

WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "runs" / "road_baseline" / "weights" / "best.pt"
ONNX_PATH = PROJECT_ROOT / "weights" / "best.onnx"
REPORT_DIR = PROJECT_ROOT / "docs" / "deployment"


def register_cuda_dlls() -> str | None:
    """把 torch 的 lib 目录注册为 DLL 搜索路径。Windows 上 ORT 找 cuBLAS/cuDNN 全靠这一步。"""
    try:
        import torch
        lib = Path(torch.__file__).parent / "lib"
        if lib.is_dir() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(lib))
            return str(lib)
    except Exception as exc:                      # noqa: BLE001
        return "注册失败：%s" % exc
    return None


def make_session(path: Path, want_gpu: bool):
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if want_gpu else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), providers=providers)
    return sess, sess.get_providers()


def stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    n = len(ordered)
    return {
        "n": n,
        "mean": round(statistics.fmean(samples), 3),
        "median": round(statistics.median(samples), 3),
        "min": round(ordered[0], 3),
        "p95": round(ordered[min(n - 1, int(round(0.95 * (n - 1))))], 3),
        "max": round(ordered[-1], 3),
        "std": round(statistics.pstdev(samples), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTorch vs ONNX Runtime 同条件对比")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--onnx", type=Path, default=ONNX_PATH)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--num-images", type=int, default=40, help="从 valid 取多少张作为固定输入")
    parser.add_argument("--providers", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    import numpy as np
    import torch
    from ultralytics import YOLO

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dll_dir = register_cuda_dlls()

    # ---- 固定输入：ort 用张量，端到端用图片路径 ----
    folder, _ = split_dirs("valid")
    images = sorted(Path(folder).glob("*.*"))[: args.num_images]
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        print("valid 里找不到图片", file=sys.stderr)
        return 2

    rng = np.random.default_rng(0)
    x = rng.random((1, 3, 640, 640), dtype=np.float32)

    report = {
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "条件": {
            "硬件": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "输入尺寸": "640x640，batch=1",
            "预热次数": args.warmup,
            "重复次数": args.runs,
            "固定图片数": len(images),
            "DLL目录": dll_dir,
            "计时方式": "perf_counter + cuda.synchronize（端到端）；ORT 仅 run()（纯推理）",
        },
    }

    # ================= PyTorch =================
    pt = YOLO(str(args.weights))
    pt_model = pt.model.eval().cuda()
    if hasattr(pt_model, "fuse"):
        pt_model = pt_model.fuse()
    xt = torch.from_numpy(x).cuda()

    with torch.no_grad():
        for _ in range(args.warmup):
            pt_model(xt)
    torch.cuda.synchronize()

    pt_forward = []
    with torch.no_grad():
        for _ in range(args.runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pt_model(xt)
            torch.cuda.synchronize()
            pt_forward.append((time.perf_counter() - t0) * 1000)

    # 端到端：走 Ultralytics 完整流水（读图、预处理、推理、后处理）
    src = str(images[0])
    for _ in range(min(args.warmup, 10)):
        pt.predict(src, imgsz=640, device=0, verbose=False)
    torch.cuda.synchronize()

    pt_e2e = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = pt.predict(src, imgsz=640, device=0, verbose=False)
        torch.cuda.synchronize()
        pt_e2e.append((time.perf_counter() - t0) * 1000)
    pt_speed = r[0].speed

    report["pytorch"] = {
        "纯前向_ms": stats(pt_forward),
        "端到端_ms": stats(pt_e2e),
        "框架计时_ms": {k: round(v, 3) for k, v in pt_speed.items()},
    }

    # ================= ONNX Runtime =================
    sess, actual = make_session(args.onnx, want_gpu=(args.providers == "cuda"))
    in_name = sess.get_inputs()[0].name

    for _ in range(args.warmup):
        sess.run(None, {in_name: x})
    torch.cuda.synchronize()

    ort_pure = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sess.run(None, {in_name: x})
        torch.cuda.synchronize()
        ort_pure.append((time.perf_counter() - t0) * 1000)

    on = YOLO(str(args.onnx))
    for _ in range(min(args.warmup, 10)):
        on.predict(src, imgsz=640, device=0, verbose=False)
    torch.cuda.synchronize()

    ort_e2e = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r2 = on.predict(src, imgsz=640, device=0, verbose=False)
        torch.cuda.synchronize()
        ort_e2e.append((time.perf_counter() - t0) * 1000)

    report["onnxruntime"] = {
        "版本": __import__("onnxruntime").__version__,
        "实际生效的EP": actual,
        "纯推理_ms": stats(ort_pure),
        "端到端_ms": stats(ort_e2e),
        "框架计时_ms": {k: round(v, 3) for k, v in r2[0].speed.items()},
    }

    # ================= 对比结论 =================
    ratio = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    report["对比"] = {
        "纯推理_ONNX除以PT": ratio(report["onnxruntime"]["纯推理_ms"]["median"],
                                   report["pytorch"]["纯前向_ms"]["median"]),
        "端到端_ONNX除以PT": ratio(report["onnxruntime"]["端到端_ms"]["median"],
                                   report["pytorch"]["端到端_ms"]["median"]),
        "注意": "比值 <1 表示 ONNX 更快。同一台机器同一次运行内比较才有意义。",
    }

    out = REPORT_DIR / "benchmark_pt_vs_onnx.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=" * 70)
    print("实际生效的 EP：%s" % actual)
    print("%-28s %10s %10s %10s" % ("项目", "中位数", "p95", "均值"))
    for label, d in (("PyTorch 纯前向", report["pytorch"]["纯前向_ms"]),
                     ("PyTorch 端到端", report["pytorch"]["端到端_ms"]),
                     ("ORT 纯推理", report["onnxruntime"]["纯推理_ms"]),
                     ("ORT 端到端", report["onnxruntime"]["端到端_ms"])):
        print("%-28s %10.2f %10.2f %10.2f" % (label, d["median"], d["p95"], d["mean"]))
    print("-" * 70)
    print("纯推理比值 %.3f　端到端比值 %.3f　（<1 = ONNX 更快）"
          % (report["对比"]["纯推理_ONNX除以PT"], report["对比"]["端到端_ONNX除以PT"]))
    print("报告：%s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
