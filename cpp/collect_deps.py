#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 build/yolo_cpp.exe 打成一个能脱离开发环境运行的分发包。

为什么需要它：手工拷 DLL 一定会漏。本次实测就漏了两处——
  (1) dist 里的 exe 是加 --ep 之前的旧版（387 KB vs 443 KB），
      于是"分发包里跑 TRT"实际跑的是 CUDA EP（5.25 ms），不是 TRT（3.22 ms）；
  (2) 加了视频模式后 exe 依赖 libopencv_videoio，而 dist 里没有这个文件，
      直接报 0xC0000135（STATUS_DLL_NOT_FOUND）。
这个脚本从 exe 出发递归解析导入表，把这类"编译时 link、运行时才要"的 DLL 一并找出来。

用法：
    python cpp/collect_deps.py                # 打 CPU 版到 cpp/dist
    python cpp/collect_deps.py --gpu          # 打 GPU 版到 cpp/dist_gpu
    python cpp/collect_deps.py --out X        # 自定义输出目录
    python cpp/collect_deps.py --check --gpu  # 只体检现有包，不复制
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # 项目根
CPP = ROOT / "cpp"
BUILD_EXE = CPP / "build" / "yolo_cpp.exe"
OBJDUMP = r"D:\msys64\usr\bin\objdump"

# DLL 搜索路径，顺序即优先级（靠前的先被采用）
SEARCH = [
    CPP / "third_party" / "onnxruntime" / "lib",        # onnxruntime.dll
    Path(r"D:\msys64\ucrt64\bin"),                      # MinGW 运行时 + OpenCV + FFmpeg + GStreamer
    Path(r"D:\TensorRT\TensorRT-10.8.0.43\lib"),        # nvinfer 等
    Path(r"D:\Miniconda\envs\yolo_road\Lib\site-packages\onnxruntime\capi"),
    Path(r"D:\Miniconda\envs\yolo_road\Lib\site-packages\torch\lib"),
]

# GPU 版额外需要、但**不在导入表里**的 DLL：
# ORT 和 TensorRT 都是运行时 LoadLibrary 加载的，静态分析发现不了。
GPU_EXTRA = [
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_tensorrt.dll",
    "nvinfer_10.dll",
    "nvinfer_plugin_10.dll",
    "nvonnxparser_10.dll",
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cufft64_11.dll",
    "nvrtc64_120_0.dll",
    "nvJitLink_120_0.dll",
    # cudnn64_9.dll 只是加载器（0.3 MB），真正提供 cudnnCreate 的是下面这些子库。
    # 少了任何一个都会报 "Cannot load symbol cudnnCreate"，然后 0xC0000409 直接崩。
    "cudnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
]

# 系统自带的，不打包（宁可漏判，也不要拷系统 DLL 出去）
SYSTEM_PREFIX = (
    "kernel32", "kernelbase", "user32", "advapi32", "shell32", "shlwapi", "ole32",
    "oleaut32", "gdi32", "gdiplus", "ws2_32", "wsock32", "bcrypt", "bcryptprimitives",
    "crypt32", "ncrypt", "psapi", "dbghelp", "rpcrt4", "secur32", "sspicli", "iphlpapi",
    "userenv", "ntdll", "ucrtbase", "msvcrt", "winmm", "comdlg32", "combase", "version",
    "imm32", "dwmapi", "setupapi", "cfgmgr32", "powrprof", "mpr", "winspool", "oleacc",
    "uxtheme", "d3d9", "d3d11", "d3d12", "dxgi", "dxcore", "opengl32", "glu32", "avrt",
    "ksuser", "dinput8", "hid", "wtsapi32", "authz", "credui", "wldap32", "dnsapi",
    "shcore", "propsys", "dwrite", "d2d1", "pdh", "wevtapi", "twinapi", "twain", "mscms",
    "icm32", "winhttp", "wininet", "urlmon", "sechost", "msvcp", "vcruntime", "concrt",
    "api-ms-win", "ext-ms-win", "wlanapi", "dhcpcsvc", "normaliz", "netapi32", "wintrust",
    # 这三个是系统 DLL，之前漏判了（objdump 能从导入表看到，但 Windows 自带）
    "avicap32", "msimg32", "usp10", "msacm32", "winusb", "hidclass",
)


def is_system(name: str) -> bool:
    low = name.lower()
    return any(low.startswith(p) for p in SYSTEM_PREFIX)


def imports_of(dll: Path) -> list[str]:
    """用 MinGW 的 objdump 读 PE 导入表。"""
    try:
        r = subprocess.run([OBJDUMP, "-p", str(dll)],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=180)
        out = r.stdout
    except Exception:
        return []
    return [l.split("DLL Name:")[-1].strip()
            for l in out.splitlines() if "DLL Name:" in l]


def index_search_paths() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for d in SEARCH:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.dll")):
            idx.setdefault(f.name.lower(), f)      # 先到先得，保持优先级
    return idx


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# 两个版本都带：ORT 的公共库，很小。缺它时 CUDA EP 的报错会变成
# "providers_shared.dll missing"，比真正的 "providers_cuda.dll missing" 更难定位。
BASE_EXTRA = ["onnxruntime_providers_shared.dll"]


def resolve(gpu: bool, idx: dict[str, Path]):
    """从 exe 出发 BFS，返回 (已解析 {名字: 路径}, 缺失集合)。"""
    queue: deque = deque()
    queue.append(BUILD_EXE)
    queue.extend(BASE_EXTRA)
    if gpu:
        queue.extend(GPU_EXTRA)

    resolved: dict[str, Path] = {}
    missing: set[str] = set()
    seen: set[str] = set()

    while queue:
        item = queue.popleft()
        if isinstance(item, Path):
            dll, name = item, item.name
        else:
            name = item
            dll = idx.get(name.lower())
            if dll is None:
                missing.add(name)
                continue

        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        resolved[name] = dll

        for dep in imports_of(dll):
            if is_system(dep) or dep.lower() in seen:
                continue
            queue.append(dep)

    return resolved, missing


def write_manifest(out: Path, resolved: dict[str, Path], gpu: bool) -> None:
    lines = [
        "YOLO 道路检测 · C++ 分发包清单",
        "",
        "模式：%s" % ("GPU（TensorRT / CUDA EP 可用）" if gpu else "CPU（仅 CPU 推理）"),
        "生成方式：python cpp/collect_deps.py%s" % (" --gpu" if gpu else ""),
        "",
        "%-46s %10s  %-64s %s" % ("文件名", "大小(MB)", "sha256 前 16 位", "来源"),
        "-" * 150,
    ]
    total = 0
    for name in sorted(resolved):
        p = resolved[name]
        size = p.stat().st_size
        total += size
        lines.append("%-46s %10.1f  %-64s %s"
                     % (name, size / 1048576, sha256(p)[:16], p.parent))
    lines += [
        "-" * 150,
        "共 %d 个文件，合计 %.1f MB" % (len(resolved), total / 1048576),
        "",
        "用法：把 exe 与本目录所有 DLL 放在同一目录，直接运行 exe。",
        "不需要设置 PATH，也不需要安装 CUDA / cuDNN / TensorRT / Python。",
        "",
        "校验方式（换机器后先跑这个）：",
        "  yolo_cpp.exe --model best.onnx --image test.jpg --ep trt --runs 20 --warmup 5",
        "  看输出里是否有「EP：TensorRT（FP16）+ CUDA 兜底」，且中位延迟应在 3 ms 量级。",
        "  若显示「TensorRT EP 不可用」或延迟 > 20 ms，说明缺库，对照本清单逐个核。",
    ]
    (out / "MANIFEST.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="打包 GPU 版（含 TensorRT / cuDNN）")
    ap.add_argument("--out", default=None, help="输出目录，默认 cpp/dist 或 cpp/dist_gpu")
    ap.add_argument("--check", action="store_true", help="只体检现有包，不复制")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (CPP / ("dist_gpu" if args.gpu else "dist"))

    if not BUILD_EXE.is_file():
        print("找不到 %s —— 先 cmake --build build" % BUILD_EXE)
        return 2

    idx = index_search_paths()
    print("搜索路径索引：%d 个 DLL" % len(idx))

    resolved, missing = resolve(args.gpu, idx)

    if args.check:
        print("\n=== 体检 %s ===" % out)
        have = {f.name.lower() for f in out.glob("*")} if out.is_dir() else set()
        lack = [n for n in resolved if n.lower() not in have]
        exe_same = (out / "yolo_cpp.exe").is_file() and \
                   sha256(out / "yolo_cpp.exe") == sha256(BUILD_EXE)
        print("应有 %d 个文件，实际 %d 个" % (len(resolved) + 1, len(have)))
        print("exe 与 build 一致：%s" % ("是" if exe_same else "否 —— 需要重新打包"))
        if lack:
            print("缺 %d 个：%s" % (len(lack), ", ".join(sorted(lack)[:12])))
        if missing:
            print("解析不到（需人工确认是否为系统库）：%s" % ", ".join(sorted(missing)[:10]))
        return 0 if (exe_same and not lack) else 1

    out.mkdir(parents=True, exist_ok=True)

    def place(src: Path, dst: Path) -> str:
        """同盘优先建硬链接（不占额外空间）；跨盘或失败则复制。"""
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"

    how = place(BUILD_EXE, out / "yolo_cpp.exe")
    n_link = 1 if how == "link" else 0
    for name, src in resolved.items():
        if place(src, out / name) == "link":
            n_link += 1
    print("  落盘方式：%d 个硬链接 / %d 个复制" % (n_link, len(resolved) + 1 - n_link))

    write_manifest(out, resolved, args.gpu)

    total = sum(p.stat().st_size for p in resolved.values())
    print("\n打包完成 → %s" % out)
    print("  exe + %d 个 DLL，合计 %.1f MB" % (len(resolved), total / 1048576))
    print("  清单：%s" % (out / "MANIFEST.txt"))
    if missing:
        print("  ⚠ 有 %d 个名字解析不到（可能是系统库）：%s"
              % (len(missing), ", ".join(sorted(missing)[:8])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
