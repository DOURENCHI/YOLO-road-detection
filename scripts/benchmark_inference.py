"""
推理性能测试：按 5 项口径测全，输出可以直接写进报告的数字。

为什么必须记全 5 项口径
----------------------
"延迟 23.7 ms" 这句话本身几乎没有信息量。同一份权重、同一张图，换个测法就能差一倍以上：

    · 纯 GPU 前向    用 torch.cuda.Event 卡在 GPU 上，只算模型前向
    · 推理阶段       Ultralytics 自带的 result.speed["inference"]（它的流水计时）
    · 完整端到端     perf_counter + cuda.synchronize，含读图解码、letterbox、前后处理、NMS

所以报告里每个速度数字都要同时写明这 5 项：
    [硬件型号] [输入尺寸] [预热次数] [计时方式（含不含前后处理）] [精度模式]
少任何一项，别人就没法判断这个数字能不能拿来比。

用法
----
    python scripts/benchmark_inference.py                        # best.pt，FP32 与 FP16 各测一遍
    python scripts/benchmark_inference.py --runs 500 --warmup 50
    python scripts/benchmark_inference.py --half fp32            # 只测 FP32
    python scripts/benchmark_inference.py --imgsz 320            # 换输入尺寸
    python scripts/benchmark_inference.py --json-out docs/bench.json

版本差异提醒
------------
本机装的 Ultralytics 8.4.176 里，`half=` 这个参数已经**废弃**，每次调用都会打印
"WARNING 'half' is deprecated and will be removed in the future. Use 'quantize' instead."。
新写法是 `quantize`：16 = FP16，32 = FP32。本脚本用的是新写法 —— 这类"教程里写的写法
在当前版本已经过时"的情况很常见，跑之前先看一眼有没有 warning 是个好习惯。

注意：脚本只读文件、只往 stdout 和可选的 --json-out 写东西，不碰数据集和训练产物。
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import (
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    SPLITS,
    TARGET_YAML,
    split_dirs,
)

DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "runs" / "road_baseline" / "weights" / "best.pt"
DEFAULT_DEVICE = 0


def precision_arg(half):
    """
    返回本版 Ultralytics 的精度参数值。

    注意（实测踩到的版本差异）：这个版本里 `half=` 已经废弃，
    每次调用都会打印 "WARNING 'half' is deprecated ... Use 'quantize' instead."。
    新写法是 `quantize`：16 = FP16，32 = FP32（'fp16' / 'fp32' 这类别名也会被规范化成 16 / 32）。
    旧参数目前还能用（内部会被转成 quantize），但迟早会移除。
    """
    return 16 if half else 32


# ------------------------------------------------------------------ 工具


def stat(samples):
    """把一串毫秒样本汇总成均值 / 中位数 / p95 / 标准差。"""
    ordered = sorted(samples)
    return {
        "n": len(samples),
        "mean": round(statistics.mean(samples), 3),
        "median": round(statistics.median(samples), 3),
        "min": round(min(samples), 3),
        "p95": round(ordered[int(0.95 * len(ordered)) - 1], 3),
        "max": round(max(samples), 3),
        "std": round(statistics.stdev(samples), 3),
    }


def driver_version():
    """尽量拿到显卡驱动版本；拿不到就返回 unknown，不影响测试。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        text = out.stdout.strip().splitlines()
        return text[0].strip() if text else "unknown"
    except Exception:
        return "unknown"


def timed(function, warmup, runs, sync):
    """
    计时：预热若干次后，逐次跑 runs 遍。

    两个关键点：
      1. CUDA 调用是异步的 —— 计时前后各加一次同步，否则测到的是"下发指令花了多久"
      2. 必须先预热 —— 第一次推理要加载 kernel、分配显存，比稳定值大几个数量级

    sync 由调用方传入：用 GPU 时是 torch.cuda.synchronize，用 CPU 时是空函数。
    """
    for _ in range(warmup):
        function()
    sync()

    samples = []
    for _ in range(runs):
        sync()
        start = time.perf_counter()
        function()
        sync()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def pick_images(yaml_path, split, count):
    """从某个划分里按文件名顺序取 count 张图，保证可复现。"""
    images_dir, _labels_dir = split_dirs(split, yaml_path)

    if images_dir is None or not images_dir.is_dir():
        raise FileNotFoundError(f"找不到 {split} 的图片目录：{images_dir}")

    images = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise FileNotFoundError(f"{images_dir} 里没有图片")

    if len(images) < count:
        raise ValueError(f"{split} 只有 {len(images)} 张图，凑不齐 {count} 张")

    return [str(p) for p in images[:count]]


# ------------------------------------------------------------------ 主流程


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO 推理性能测试（5 项口径齐全）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="权重文件")
    parser.add_argument("--yaml", default=str(TARGET_YAML), help="数据集配置，用来找测试图片")
    parser.add_argument("--split", default="valid", choices=list(SPLITS), help="从哪个划分取图")
    parser.add_argument("--imgsz", type=int, default=640, help="输入边长")
    parser.add_argument("--batch", type=int, default=8, help="批量测试的批大小")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="显卡编号，或 'cpu'")
    parser.add_argument("--warmup", type=int, default=30, help="预热次数")
    parser.add_argument("--runs", type=int, default=200, help="单张测试的计时次数")
    parser.add_argument("--batch-runs", type=int, default=50, help="批量测试的计时次数")
    parser.add_argument("--half", default="both", choices=["both", "fp32", "fp16"],
                        help="精度模式")
    parser.add_argument("--json-out", default=None, help="把结果同时写一份 JSON 到这里")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(
            f"找不到权重文件：{weights}\n"
            "先确认训练产物在哪，或显式指定 --weights。"
        )
    if str(args.device) != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() 为 False —— 装的很可能是 CPU 版 PyTorch。\n"
            "换 CPU 测可以加 --device cpu，但速度没有参考价值。"
        )

    torch.backends.cudnn.benchmark = True   # 让 cuDNN 自己挑最快的算法

    on_gpu = str(args.device) != "cpu"
    sync = torch.cuda.synchronize if on_gpu else (lambda: None)

    model = YOLO(str(weights))
    single = pick_images(Path(args.yaml), args.split, 1)
    batch_images = pick_images(Path(args.yaml), args.split, args.batch)

    modes = [False, True] if args.half == "both" else [args.half == "fp16"]

    # ---------------- 环境与 5 项口径 ----------------
    env = {
        "weights": str(weights),
        "weights_MB": round(weights.stat().st_size / 1024 ** 2, 2),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "warmup": args.warmup,
        "runs_single": args.runs,
        "runs_batch": args.batch_runs,
        "split": args.split,
    }

    if not on_gpu:
        env["hardware"] = "CPU"
    else:
        device_id = int(str(args.device).split(":")[-1])
        prop = torch.cuda.get_device_properties(device_id)
        env.update({
            "hardware": torch.cuda.get_device_name(device_id),
            "vram_MiB": round(prop.total_memory / 1024 ** 2),
            "compute_capability": f"sm_{prop.major}{prop.minor}",
            "driver": driver_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        })

    try:
        from ultralytics import __version__ as ul_version
        env["ultralytics"] = ul_version
    except Exception:
        pass
    env["python"] = sys.version.split()[0]

    print("=" * 78)
    print("推理性能测试 · 5 项口径")
    print("=" * 78)
    print("① 硬件型号 : %s（%s MB，%s，驱动 %s）"
          % (env.get("hardware"), env.get("vram_MiB", "-"),
             env.get("compute_capability", "-"), env.get("driver", "-")))
    print("             软件栈：PyTorch %s · CUDA %s · cuDNN %s · Ultralytics %s · Python %s"
          % (env.get("torch"), env.get("cuda_build"), env.get("cudnn"),
             env.get("ultralytics"), env["python"]))
    print("② 输入尺寸 : %d×%d（letterbox 等比缩放 + 空白填充）" % (args.imgsz, args.imgsz))
    print("③ 预热次数 : 单张 %d 次；批量 %d 次（预热后才开始计时）"
          % (args.warmup, max(2, args.warmup // 5)))
    print("④ 计时方式 : 三档分开报 —— torch.cuda.Event 纯前向 / result.speed 推理阶段 / "
          "perf_counter+同步 完整端到端")
    print("             含读图解码、letterbox、前后处理与 NMS；单张 n=%d、批量 n=%d"
          % (args.runs, args.batch_runs))
    print("⑤ 精度模式 : %s" % ("FP32 与 FP16 各一遍" if args.half == "both"
                                else ("FP16" if args.half == "fp16" else "FP32")))
    print("   权重     : %s（%.2f MB）" % (weights.name, env["weights_MB"]))
    print()

    result = {"口径": env, "结果": {}}

    # ---------------- 冷启动 ----------------
    if on_gpu:
        start = time.perf_counter()
        model.predict(single, imgsz=args.imgsz, device=args.device, verbose=False)
        torch.cuda.synchronize()
        cold = round((time.perf_counter() - start) * 1000, 1)
        result["冷启动_ms"] = cold
        print("冷启动（本进程第一次 predict，含显存分配与 kernel 首次加载）：%.1f ms" % cold)
        print()

    # ---------------- 端到端 ----------------
    print("【端到端】含读图 + 前处理 + 推理 + 后处理")
    print("  %-26s %8s %8s %8s %8s %10s" % ("配置", "均值ms", "中位ms", "p95", "标准差", "吞吐"))
    for half in modes:
        tag = "FP16" if half else "FP32"
        e = stat(timed(
            lambda h=half: model.predict(single, imgsz=args.imgsz,
                                         quantize=precision_arg(h),
                                         device=args.device, verbose=False),
            args.warmup, args.runs, sync))
        print("  %-26s %8.1f %8.1f %8.1f %8.2f %8.0f FPS"
              % (tag + " · 单张", e["mean"], e["median"], e["p95"], e["std"], 1000 / e["mean"]))
        result["结果"]["端到端_单张_" + tag] = e

    for half in modes:
        tag = "FP16" if half else "FP32"
        b = stat(timed(
            lambda h=half: model.predict(batch_images, imgsz=args.imgsz,
                                         quantize=precision_arg(h),
                                         device=args.device, verbose=False),
            max(2, args.warmup // 5), args.batch_runs, sync))
        b["per_image_ms"] = round(b["mean"] / args.batch, 3)
        b["throughput_fps"] = round(1000 / b["per_image_ms"], 1)
        print("  %-26s %8.2f %8.2f %8.2f %8.2f %8.0f FPS"
              % ("%s · 批量%d（每张）" % (tag, args.batch), b["per_image_ms"],
                 b["median"] / args.batch, b["p95"] / args.batch,
                 b["std"] / args.batch, b["throughput_fps"]))
        result["结果"]["端到端_批量%d_%s" % (args.batch, tag)] = b
    print()

    # ---------------- 耗时构成 ----------------
    print("【耗时构成】Ultralytics 自带的 result.speed 拆解")
    print("  %-10s %10s %10s %10s %10s" % ("精度", "前处理", "模型推理", "后处理", "合计"))
    for half in modes:
        tag = "FP16" if half else "FP32"
        quant = precision_arg(half)
        model.predict(single, imgsz=args.imgsz, quantize=quant,
                      device=args.device, verbose=False)
        buckets = {"preprocess": [], "inference": [], "postprocess": []}
        for _ in range(min(50, args.runs)):
            r = model.predict(single, imgsz=args.imgsz, quantize=quant,
                              device=args.device, verbose=False)[0]
            for key in buckets:
                buckets[key].append(r.speed[key])
        parts = {k: round(statistics.mean(v), 3) for k, v in buckets.items()}
        parts["total"] = round(sum(parts.values()), 3)
        print("  %-10s %10.1f %10.1f %10.1f %10.1f"
              % (tag, parts["preprocess"], parts["inference"],
                 parts["postprocess"], parts["total"]))
        result["结果"]["耗时构成_" + tag] = parts
    print()

    # ---------------- 纯 GPU 前向 ----------------
    print("【纯 GPU 前向】torch.cuda.Event 计时，排除全部 CPU 侧工作")
    if not on_gpu:
        print("  跳过（--device cpu）")
    else:
        net = model.model.eval().to("cuda")
        probe = torch.rand(1, 3, args.imgsz, args.imgsz, device="cuda")
        try:
            for half in modes:
                tag = "FP16" if half else "FP32"
                # 模型和输入必须同精度，否则报 Input type 与 weight type 不一致
                net = net.half() if half else net.float()
                x = probe.half() if half else probe.float()
                with torch.no_grad():
                    for _ in range(args.warmup + 20):
                        net(x)
                    torch.cuda.synchronize()
                    samples = []
                    for _ in range(args.runs):
                        s = torch.cuda.Event(enable_timing=True)
                        e = torch.cuda.Event(enable_timing=True)
                        s.record()
                        net(x)
                        e.record()
                        torch.cuda.synchronize()
                        samples.append(s.elapsed_time(e))
                f = stat(samples)
                print("  %-10s 均值 %7.2f  中位数 %7.2f  p95 %7.2f  标准差 %6.2f"
                      % (tag, f["mean"], f["median"], f["p95"], f["std"]))
                result["结果"]["纯前向_" + tag] = f
        finally:
            net.float()   # 恢复浮点，避免影响后续调用
    print()

    # ---------------- 模型规模（两个口径） ----------------
    from ultralytics.utils.torch_utils import get_flops

    raw_params = int(sum(p.numel() for p in model.model.parameters()))
    print("【模型规模】注意有两个口径，写报告时必须说明用哪个")
    print("  未融合：%d 参数（%.2f M）· %.2f GFLOPs @%d"
          % (raw_params, raw_params / 1e6, get_flops(model.model, imgsz=args.imgsz), args.imgsz))
    model.model.fuse(verbose=False)
    fused_params = int(sum(p.numel() for p in model.model.parameters()))
    print("  融合后：%d 参数（%.2f M）· %.2f GFLOPs @%d —— Ultralytics 的 model.info() 用的是这个"
          % (fused_params, fused_params / 1e6,
             get_flops(model.model, imgsz=args.imgsz), args.imgsz))
    result["模型规模"] = {
        "params_unfused": raw_params,
        "params_fused": fused_params,
        "gflops_unfused": float(get_flops(model.model, imgsz=args.imgsz)),
    }
    print()

    print("=" * 78)
    print("提醒：以上数字只在上面这 5 项条件成立时才可比。")
    print("      尤其是 —— 笔记本 GPU 的数字不能当成车载端侧性能，")
    print("      单张离线推理也不等于视频流管线的吞吐。")
    print("=" * 78)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        print("结果已写入：%s" % out)


if __name__ == "__main__":
    main()
