"""
对比 yolo_cpp 的两种图像后端：stb（手写双线性）与 OpenCV（cv::resize）。

要回答的问题只有一个：**换成 OpenCV 会不会改变结果？**
拆成三层验证，逐层收窄：

    第 1 层  预处理张量   — 逐值比较 float32（这一层最能说明"实现是否等价"）
    第 2 层  检测框       — 框数、类别、IoU、坐标差、分数差
    第 3 层  输出图       — 像素级差异（受画框线宽等影响，只作参考）

为什么值得单独写个脚本：
    "两个库算得一样"是不能靠感觉判断的。OpenCV 的 INTER_LINEAR 内部用定点整数运算，
    和浮点手写版本在数学上同源，但不保证逐像素相等。差 ±1 灰度会不会让某个低分框
    越过 conf 阈值？只有跑出来才知道。

用法
    python scripts/13_compare_cpp_backends.py
    python scripts/13_compare_cpp_backends.py --image docs/deployment/_test_odd.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = PROJECT_ROOT / "cpp" / "build" / "yolo_cpp.exe"
DEFAULT_IMAGE = PROJECT_ROOT / "docs" / "deployment" / "_test_odd.jpg"
DEFAULT_ONNX = PROJECT_ROOT / "weights" / "best.onnx"
OUT_DIR = PROJECT_ROOT / "docs" / "deployment"
RESULT_DIR = PROJECT_ROOT / "results" / "cpp"


def build_env() -> dict:
    """把 torch 的 lib 目录塞进 PATH。

    C++ 程序没有 torch 帮忙找 CUDA 运行库，缺 cublasLt64_12.dll 时 ORT 会
    **静默回落到 CPU**——结果照样对，只是慢两三倍。计时数字就废了。
    """
    env = os.environ.copy()
    try:
        import torch

        lib = str(Path(torch.__file__).parent / "lib")
        env["PATH"] = lib + os.pathsep + env.get("PATH", "")
    except Exception:  # noqa: BLE001
        pass
    return env


def run_backend(exe: Path, backend: str, onnx: Path, image: Path,
                runs: int, warmup: int, env: dict) -> dict:
    """跑一次指定后端，返回它落盘的三份产物路径与 stdout。"""
    stem = f"_backend_{backend}"
    dump = OUT_DIR / f"{stem}.bin"
    jout = OUT_DIR / f"{stem}.json"
    prefix = RESULT_DIR / stem
    prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(exe), "--model", str(onnx), "--image", str(image),
           "--backend", backend, "--json", str(jout),
           "--dump-input", str(dump), "--out", str(prefix),
           "--runs", str(runs), "--warmup", str(warmup)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        raise SystemExit("后端 %s 运行失败：\n%s\n%s" % (backend, proc.stdout, proc.stderr))

    return {
        "backend": backend,
        "stdout": proc.stdout,
        "tensor": dump,
        "json": json.loads(jout.read_text(encoding="utf-8")),
        "png": Path(str(prefix) + "_boxes.png"),
    }


def load_tensor(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.float32)


def compare_tensors(a: np.ndarray, b: np.ndarray) -> dict:
    """预处理张量比较。数值范围是 0–1（已 /255），所以绝对差可以直接读。"""
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "结论": "形状不同，不可比"}
    diff = np.abs(a - b)
    nz = int((diff > 0).sum())
    return {
        "元素数": int(a.size),
        "最大绝对差": float(diff.max()),
        "平均绝对差": float(diff.mean()),
        "不等于零的元素数": nz,
        "不等于零的比例": float(nz / a.size),
        "差 > 1/255 的元素数": int((diff > 1 / 255).sum()),
        "差 > 2/255 的元素数": int((diff > 2 / 255).sum()),
    }


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def compare_boxes(ja: dict, jb: dict) -> dict:
    """把 A 的每个框在 B 里找最佳匹配（贪心，一对一）。"""
    A, B = ja["boxes"], jb["boxes"]
    used, pairs = set(), []
    for p in A:
        best, idx = -1.0, -1
        for i, q in enumerate(B):
            if i in used:
                continue
            v = iou(p[:4], q[:4])
            if v > best:
                best, idx = v, i
        if idx >= 0 and best > 0.5:
            used.add(idx)
            q = B[idx]
            pairs.append({
                "iou": best,
                "cls_a": p[5], "cls_b": q[5],
                "score_a": p[4], "score_b": q[4],
                "coord_max_diff": max(abs(p[k] - q[k]) for k in range(4)),
            })

    out = {
        "框数_a": len(A), "框数_b": len(B),
        "配对成功": len(pairs),
    }
    if pairs:
        out.update({
            "最小IoU": min(x["iou"] for x in pairs),
            "最大分数差": max(abs(x["score_a"] - x["score_b"]) for x in pairs),
            "最大坐标差_px": max(x["coord_max_diff"] for x in pairs),
            "类别全对": all(x["cls_a"] == x["cls_b"] for x in pairs),
        })
    out["完全一致"] = (len(A) == len(B) and len(pairs) == len(A)
                       and out.get("类别全对", False)
                       and out.get("最大坐标差_px", 1e9) < 0.5)
    return out


def compare_images(pa: Path, pb: Path) -> dict:
    """输出图比较。画框是同一份代码，所以差异只可能来自底图（resize）的 ±1 灰度。"""
    try:
        from PIL import Image
    except ImportError:
        return {"跳过": "未安装 Pillow"}
    if not (pa.is_file() and pb.is_file()):
        return {"跳过": "缺少输出图"}
    a = np.asarray(Image.open(pa).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(pb).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return {"跳过": "尺寸不同 %s vs %s" % (a.shape, b.shape)}
    d = np.abs(a - b)
    return {
        "像素总数": int(d.size),
        "最大差": int(d.max()),
        "不同像素比例": float((d > 0).mean()),
        "差 > 2 的像素比例": float((d > 2).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(DEFAULT_EXE))
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--keep-bin", action="store_true", help="保留中间张量文件（各约 4.7 MB）")
    args = ap.parse_args()

    exe, onnx, image = Path(args.exe), Path(args.onnx), Path(args.image)
    for p, what in ((exe, "可执行文件"), (onnx, "ONNX 模型"), (image, "测试图")):
        if not p.is_file():
            raise SystemExit("找不到%s：%s" % (what, p))

    env = build_env()
    print("对比两个图像后端：stb（手写） vs OpenCV")
    print("  图片 %s" % image)
    print("  计时 %d 次 / 预热 %d（这一环不影响正确性结论）\n" % (args.runs, args.warmup))

    runs = {}
    for backend in ("stb", "opencv"):
        print("── 运行后端 %s …" % backend)
        runs[backend] = run_backend(exe, backend, onnx, image, args.runs, args.warmup, env)
        hit = [ln for ln in runs[backend]["stdout"].splitlines() if "后端" in ln or "纯推理" in ln]
        for ln in hit:
            print("   " + ln.strip())

    a, b = runs["stb"], runs["opencv"]
    tensor = compare_tensors(load_tensor(a["tensor"]), load_tensor(b["tensor"]))
    boxes = compare_boxes(a["json"], b["json"])
    images = compare_images(a["png"], b["png"])

    print("\n第 1 层 · 预处理张量")
    print("   最大绝对差 %.3e　非零差 %.4f%%　超 1/255 的元素 %d 个"
          % (tensor["最大绝对差"], 100 * tensor["不等于零的比例"],
             tensor["差 > 1/255 的元素数"]))

    print("\n第 2 层 · 检测框")
    print("   stb %d 框　opencv %d 框　配对 %d"
          % (boxes["框数_a"], boxes["框数_b"], boxes["配对成功"]))
    if "最小IoU" in boxes:
        print("   最小 IoU %.4f　坐标最大差 %.2f px　分数最大差 %.4f　类别全对 %s"
              % (boxes["最小IoU"], boxes["最大坐标差_px"],
                 boxes["最大分数差"], boxes["类别全对"]))
    print("   → %s" % ("两种后端结果完全一致" if boxes["完全一致"] else "存在差异，见报告"))

    if "不同像素比例" in images:
        print("\n第 3 层 · 输出图")
        print("   最大像素差 %d　不同像素比例 %.4f%%"
              % (images["最大差"], 100 * images["不同像素比例"]))

    report = {
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "图片": str(image.relative_to(PROJECT_ROOT)) if image.is_relative_to(PROJECT_ROOT) else str(image),
        "参数": {"runs": args.runs, "warmup": args.warmup},
        "计时_ms": {k: {"中位数": v["json"]["median_ms"], "p95": v["json"]["p95_ms"]}
                    for k, v in runs.items()},
        "张量": tensor,
        "框": boxes,
        "图": images,
        "结论": "两种后端输出等价" if boxes["完全一致"] else "两种后端存在可测差异",
    }
    out = OUT_DIR / "cpp_backend_compare.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n报告：%s" % out.relative_to(PROJECT_ROOT))

    # 中间张量各 4.7 MB，结论已经进报告了，默认删掉；需要复查时 --keep-bin
    if not args.keep_bin:
        for k in runs:
            for f in (runs[k]["tensor"], runs[k]["json"]):
                if f.is_file():
                    f.unlink()
        print("已清理中间文件（--keep-bin 可保留）")


if __name__ == "__main__":
    sys.exit(main())
