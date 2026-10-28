"""
验证 TensorRT engine，并与 ONNX FP32 对照。

和 ONNX 那次一样分三层，原因相同：
    数值层不同 ≠ 结果层不同。FP16 会让张量差变大，但只要框不动就没问题；
    反过来，如果框动了而张量看着还行，说明后处理对不同 dtype 敏感。
分开看才知道该改哪一层。

    第 1 层  结构    — 输入输出名称、shape、dtype（FP16 引擎的 IO 是否也变 half）
    第 2 层  数值    — 原始输出张量，逐元素相对误差
    第 3 层  结果    — 检测框：框数、类别、IoU、坐标差、分数差

外加性能：ORT FP32 / TRT FP32 / TRT FP16 的纯推理延迟。

用法
    python scripts/15_verify_engine.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

WEIGHTS_ONNX = PROJECT_ROOT / "weights" / "best.onnx"
ENGINE_DIR = PROJECT_ROOT / "weights"
REPORT = PROJECT_ROOT / "docs" / "deployment" / "engine_verify_report.json"
TRT_HOME = Path(r"D:\TensorRT\TensorRT-10.8.0.43")


def setup_dll_paths() -> None:
    # TRT 10 的 Windows 布局：DLL 在 lib/
    for d in (TRT_HOME / "lib", TRT_HOME / "bin",
              Path(r"D:\Miniconda\envs\yolo_road\Lib\site-packages\torch\lib")):
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


# ---------------------------------------------------------------- TensorRT 会话

def load_plan(path: Path, logger, verbose: bool = False):
    """读 .engine 并反序列化，返回 (engine, 说明)。

    踩过的坑：**Ultralytics 导出的 .engine 不是纯 plan**。它在前面写了一段
        [4 字节小端长度][JSON 元数据] + plan
    直接拿整个文件去 deserialize 会报
        plan->header.magicTag == rt::kPLAN_MAGIC_TAG failed（序列化版本不兼容）
    这个报错完全没提「文件前面有头」，很容易误判成版本不匹配。
    所以这里先探测：能直接读就是纯 plan，否则按 4+N 跳过。
    """
    import tensorrt as trt

    raw = path.read_bytes()
    runtime = trt.Runtime(logger)

    candidates = [(0, raw, "纯 plan")]
    if len(raw) > 8:
        n = int.from_bytes(raw[:4], "little")
        if 0 < n < len(raw) - 4 and raw[4:5] == b"{":
            candidates.insert(0, (4 + n, raw[4 + n:], "Ultralytics 头（4 + %d 字节 JSON）" % n))

    for offset, buf, label in candidates:
        engine = runtime.deserialize_cuda_engine(buf)
        if engine is not None:
            if verbose:
                print("   载入 %s  ← %s，偏移 %d" % (path.name, label, offset))
            return engine, label
    return None, "失败"


class TrtSession:
    """TRT 10 的最小封装：deserialize → context → set_tensor_address → execute_async_v3。

    为什么不用 ultralytics 的封装：那样拿不到「网络原始输出」，只能拿到后处理过的框，
    就分不清差异是引擎精度造成的还是解码/NMS 造成的。
    """

    def __init__(self, engine_path: Path):
        import tensorrt as trt

        self.trt = trt
        logger = trt.Logger(trt.Logger.ERROR)
        self.engine, self.header_note = load_plan(Path(engine_path), logger)
        if self.engine is None:
            raise SystemExit("反序列化失败：%s" % engine_path)
        self.context = self.engine.create_execution_context()

        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            (self.inputs if mode == trt.TensorIOMode.INPUT else self.outputs).append(name)

    def io_info(self) -> dict:
        """fp16 引擎不一定把 IO 也变半精度，这里如实记录。"""
        info = {"输入": [], "输出": []}
        for side, names in (("输入", self.inputs), ("输出", self.outputs)):
            for n in names:
                info[side].append({
                    "name": n,
                    "shape": list(self.engine.get_tensor_shape(n)),
                    "dtype": str(self.engine.get_tensor_dtype(n)).replace("DataType.", ""),
                })
        return info

    def run(self, x: np.ndarray, warmup: int, runs: int):
        import torch

        in_name = self.inputs[0]
        out_name = self.outputs[0]
        self.context.set_input_shape(in_name, x.shape)
        out_shape = tuple(self.context.get_tensor_shape(out_name))

        torch_dtype = torch.float32
        if "HALF" in str(self.engine.get_tensor_dtype(out_name)).upper():
            torch_dtype = torch.float16

        dx = torch.from_numpy(x).cuda()
        dout = torch.empty(out_shape, dtype=torch_dtype, device="cuda")
        self.context.set_tensor_address(in_name, dx.data_ptr())
        self.context.set_tensor_address(out_name, dout.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream

        for _ in range(warmup):
            self.context.execute_async_v3(stream)
        torch.cuda.synchronize()

        samples = []
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.context.execute_async_v3(stream)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000)
        return dout.float().cpu().numpy(), samples


# ---------------------------------------------------------------- 工具

def stats(samples: list[float]) -> dict:
    a = np.asarray(samples)
    return {"中位数": round(float(np.median(a)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "均值": round(float(a.mean()), 2),
            "最小": round(float(a.min()), 2)}


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def compare_boxes(ref: list, cand: list) -> dict:
    used, pairs = set(), []
    for p in ref:
        best, idx = -1.0, -1
        for i, q in enumerate(cand):
            if i in used:
                continue
            v = iou(p[:4], q[:4])
            if v > best:
                best, idx = v, i
        if idx >= 0 and best > 0.5:
            used.add(idx)
            q = cand[idx]
            pairs.append((best, abs(p[4] - q[4]), p[5] == q[5],
                          max(abs(p[k] - q[k]) for k in range(4))))
    out = {"框数_参考": len(ref), "框数_对象": len(cand), "配对": len(pairs)}
    if pairs:
        out.update({
            "最小IoU": round(min(x[0] for x in pairs), 4),
            "最大分数差": round(max(x[1] for x in pairs), 4),
            "最大坐标差_px": round(max(x[3] for x in pairs), 2),
            "类别全对": all(x[2] for x in pairs),
        })
    out["完全一致"] = (len(ref) == len(cand) == len(pairs)
                       and out.get("类别全对", False)
                       and out.get("最大坐标差_px", 1e9) < 0.5)
    return out


def pick_images(n: int) -> list[Path]:
    from dataset_config import split_dirs
    folder, _ = split_dirs("valid")
    files = sorted(Path(folder).glob("*.jpg"))[:n]
    if not files:
        raise SystemExit("valid 划分里没有图片")
    return files


def letterbox_tensor(img_path: Path) -> np.ndarray:
    """按 Ultralytics 的规则做出 1×3×640×640 的 float32 张量（与 11 号脚本一致）。"""
    import cv2
    bgr = cv2.imread(str(img_path))
    h, w = bgr.shape[:2]
    gain = min(640 / h, 640 / w)
    nw, nh = int(round(w * gain)), int(round(h * gain))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, np.uint8)
    dw, dh = (640 - nw) // 2, (640 - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=16, help="结果层核对的图片数")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    setup_dll_paths()
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    eng32 = ENGINE_DIR / "best_fp32.engine"
    eng16 = ENGINE_DIR / "best_fp16.engine"
    for p in (WEIGHTS_ONNX, eng32, eng16):
        if not p.is_file():
            raise SystemExit("缺少 %s（先跑 scripts/14_export_engine.py）" % p)

    torch.cuda.init()
    lib = Path(torch.__file__).parent / "lib"
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(lib))

    report: dict = {"生成时间": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---------- 第 1 层：结构 ----------
    print("=" * 66)
    print("第 1 层 · 结构")
    s32, s16 = TrtSession(eng32), TrtSession(eng16)
    info32, info16 = s32.io_info(), s16.io_info()
    print("  文件头：fp32 = %s　fp16 = %s" % (s32.header_note, s16.header_note))
    for tag, info in (("fp32", info32), ("fp16", info16)):
        for side in ("输入", "输出"):
            for t in info[side]:
                print("  %-4s %s  %-10s %-22s %s"
                      % (tag, side, t["name"], t["dtype"], t["shape"]))
    report["结构"] = {"fp32": info32, "fp16": info16}

    # ---------- 第 2 层：数值 ----------
    print("\n" + "=" * 66)
    print("第 2 层 · 数值（原始输出张量）")
    images = pick_images(args.images)
    x = letterbox_tensor(images[0])

    sess_ort = ort.InferenceSession(str(WEIGHTS_ONNX),
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    y_onnx = sess_ort.run(None, {sess_ort.get_inputs()[0].name: x})[0]
    y32, _ = s32.run(x, args.warmup, 20)
    y16, _ = s16.run(x, args.warmup, 20)

    numeric = {}
    for tag, y in (("fp32", y32), ("fp16", y16)):
        d = np.abs(y_onnx - y)
        numeric[tag] = {
            "max_abs_diff": round(float(d.max()), 5),
            "mean_abs_diff": round(float(d.mean()), 6),
            "相对误差_mean除均值": round(float(d.mean() / max(1e-9, np.abs(y_onnx).mean())), 6),
            "输出量级_mean": round(float(np.abs(y_onnx).mean()), 2),
        }
        print("  %-5s 相对 ONNX FP32：最大 %.4f　平均 %.5f　相对 %.2e"
              % (tag, d.max(), d.mean(), numeric[tag]["相对误差_mean除均值"]))
    print("  参考：输出量级 mean|y| = %.1f（框坐标 0–640 像素），绝对差要按相对量看"
          % numeric["fp32"]["输出量级_mean"])
    report["数值"] = numeric

    # ---------- 第 3 层：结果 ----------
    print("\n" + "=" * 66)
    print("第 3 层 · 结果（%d 张图，conf 0.25 / iou 0.7）" % len(images))
    m_onnx = YOLO(str(WEIGHTS_ONNX))
    m32 = YOLO(str(eng32))
    m16 = YOLO(str(eng16))

    def boxes(model, p):
        r = model.predict(str(p), imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
        b = r.boxes
        return [[float(v) for v in xy] + [float(c), int(k)]
                for xy, c, k in zip(b.xyxy.cpu(), b.conf.cpu(), b.cls.cpu())]

    per_image, totals = [], {"onnx": 0, "fp32": 0, "fp16": 0}
    for p in images:
        b_onnx, b32, b16 = boxes(m_onnx, p), boxes(m32, p), boxes(m16, p)
        totals["onnx"] += len(b_onnx)
        totals["fp32"] += len(b32)
        totals["fp16"] += len(b16)
        per_image.append({
            "image": p.name,
            "n_onnx": len(b_onnx), "n_fp32": len(b32), "n_fp16": len(b16),
            "fp32_vs_onnx": compare_boxes(b_onnx, b32),
            "fp16_vs_onnx": compare_boxes(b_onnx, b16),
        })

    for tag in ("fp32", "fp16"):
        key = "%s_vs_onnx" % tag
        ok = sum(1 for r in per_image if r[key]["完全一致"])
        mn_iou = min((r[key].get("最小IoU", 0) for r in per_image if r[key]["配对"]), default=0)
        mx_xy = max((r[key].get("最大坐标差_px", 0) for r in per_image), default=0)
        mx_sc = max((r[key].get("最大分数差", 0) for r in per_image), default=0)
        cls_ok = all(r[key].get("类别全对") for r in per_image)
        same_n = all(r["n_onnx"] == r["n_" + tag] for r in per_image)
        print("  %-5s 框数与 ONNX 相同 %d/%d 页　完全一致 %d/%d 页　最小 IoU %.4f"
              % (tag, sum(1 for r in per_image if r["n_onnx"] == r["n_" + tag]),
                 len(per_image), ok, len(per_image), mn_iou))
        print("        坐标最大差 %.2f px　分数最大差 %.4f　类别全对 %s"
              % (mx_xy, mx_sc, cls_ok))
        report.setdefault("结果", {})[tag] = {
            "总数": totals[tag], "与ONNX总数相同": same_n and totals[tag] == totals["onnx"],
            "完全一致页数": ok, "总页数": len(per_image),
            "最小IoU": mn_iou, "坐标最大差_px": mx_xy,
            "分数最大差": mx_sc, "类别全对": cls_ok,
        }
    print("  框总数：ONNX %d　FP32 %d　FP16 %d" % (totals["onnx"], totals["fp32"], totals["fp16"]))
    report["结果"]["逐图"] = per_image
    report["结果"]["框总数"] = totals

    # ---------- 性能 ----------
    print("\n" + "=" * 66)
    print("第 4 层 · 性能（同图、预热 %d、n=%d、每次前后同步）" % (args.warmup, args.runs))
    perf = {}
    samples_ort = []
    name_in = sess_ort.get_inputs()[0].name
    for _ in range(args.warmup):
        sess_ort.run(None, {name_in: x})
    torch.cuda.synchronize()
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sess_ort.run(None, {name_in: x})
        torch.cuda.synchronize()
        samples_ort.append((time.perf_counter() - t0) * 1000)
    perf["onnx_fp32"] = stats(samples_ort)

    _, s32_t = s32.run(x, args.warmup, args.runs)
    _, s16_t = s16.run(x, args.warmup, args.runs)
    perf["engine_fp32"] = stats(s32_t)
    perf["engine_fp16"] = stats(s16_t)

    base = perf["onnx_fp32"]["中位数"]
    for k, v in perf.items():
        print("  %-13s 中位 %6.2f ms　p95 %6.2f　最小 %6.2f　相对 ONNX %.2f×"
              % (k, v["中位数"], v["p95"], v["最小"], base / v["中位数"]))
    report["性能_ms"] = perf

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n报告：%s" % REPORT.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
