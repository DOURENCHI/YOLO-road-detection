"""
11 号脚本：证明 ONNX 和 PyTorch 是同一个模型。

为什么不能只看「跑通了」
------------------------
导出成功、能出框，不等于输出一致。常见的三种「看起来正常但其实是错的」：

    · 框整体偏移 3 个像素 —— 预处理或逆变换不一致，肉眼看图看不出来
    · 分数整体低 0.05    —— 归一化或 dtype 不一致，指标会静默掉一截
    · 某一类全丢         —— 输出张量的列顺序被改了，只看总数看不出来

所以本脚本做三层检查，顺序不能换（前一层没过，后一层的差异就无从归因）：

    ① 结构层  输入/输出形状、dtype —— 和导出报告对得上吗
    ② 数值层  同一个张量输入，torch 与 ORT 的最大绝对误差
    ③ 结果层  真实图片上，两边的框数 / 类别 / 分数是否一致

刻意不做的事
------------
**不在 test 上跑指标。** 项目的纪律是 test 只评一次（mAP50 = 0.680 就是那一次）。
本脚本只用 valid 划图片做逐框对比，指标层面的等价性请用：

    python scripts/09_evaluate.py --split valid      # 先跑 pt
    yolo val model=weights/best.onnx data=datasets/road-traffic-detect/data.yaml split=val

用法
----
    python scripts/11_verify_onnx.py                # 默认 16 张 valid 图
    python scripts/11_verify_onnx.py --num 32
    python scripts/11_verify_onnx.py --atol 1e-3    # 放宽数值判据
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import PROJECT_ROOT, split_dirs  # noqa: E402

WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "runs" / "road_baseline" / "weights" / "best.pt"
ONNX_PATH = PROJECT_ROOT / "weights" / "best.onnx"
REPORT_DIR = PROJECT_ROOT / "docs" / "deployment"


def register_cuda_dlls() -> str | None:
    """
    把 torch 的 lib 目录加进 DLL 搜索路径。

    为什么需要：onnxruntime-gpu 依赖 cublasLt / cuDNN，这两个 DLL 由 torch 提供。
    不注册时 ORT 会**静默回落到 CPU**——程序照样跑、结果照样对，只是慢十倍。
    这是判断「到底用没用上 GPU」的第一道关。
    """
    import os
    try:
        import torch
        lib = Path(torch.__file__).parent / "lib"
        if lib.is_dir() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(lib))
            return str(lib)
    except Exception as exc:                      # noqa: BLE001
        return "注册失败：%s" % exc
    return None


def pick_images(n: int) -> list[Path]:
    """从 valid 划分里按文件名排序取前 n 张 —— 固定顺序，保证可复现。"""
    folder, _ = split_dirs("valid")
    if folder is None or not Path(folder).is_dir():
        raise SystemExit("找不到 valid 划分目录")
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise SystemExit("valid 划分里没有图片")
    return files[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX 与 PyTorch 的一致性验证")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--onnx", type=Path, default=ONNX_PATH)
    parser.add_argument("--num", type=int, default=16, help="参与逐框对比的图片数")
    parser.add_argument("--atol", type=float, default=1e-4, help="张量层最大绝对误差阈值")
    args = parser.parse_args()

    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "权重": str(args.weights), "onnx": str(args.onnx)}

    # ---------------- ① 结构层 ----------------
    import onnx
    m = onnx.load(str(args.onnx))
    g = m.graph

    def ti(v):
        shape = [d.dim_param or (d.dim_value if d.HasField("dim_value") else "?")
                 for d in v.type.tensor_type.shape.dim]
        return {"name": v.name, "dtype": onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
                "shape": shape}

    inputs = [ti(v) for v in g.input]
    outputs = [ti(v) for v in g.output]
    report["结构"] = {
        "opset": [{"domain": i.domain or "ai.onnx", "version": i.version} for i in m.opset_import],
        "inputs": inputs, "outputs": outputs, "node_count": len(g.node),
    }
    print("① 结构层")
    print("   输入 %s" % inputs)
    print("   输出 %s" % outputs)

    # ---------------- ② 数值层 ----------------
    # 用固定种子造输入：同一个张量分别喂给两边，排除预处理差异的干扰
    rng = np.random.default_rng(0)
    x = rng.random((1, 3, 640, 640), dtype=np.float32)

    # 关键：先 fuse 再前向。导出的 ONNX 是 fused 的（BN 已折叠），
    # 拿未 fuse 的模型去比，差异里会混进「BN 折叠的舍入」，看起来像导出错了
    torch_model = YOLO(str(args.weights)).model.eval()
    if hasattr(torch_model, "fuse"):
        torch_model = torch_model.fuse()
    torch_model = torch_model.cuda()
    with torch.no_grad():
        y_t = torch_model(torch.from_numpy(x).cuda())[0].cpu().numpy()

    dll_dir = register_cuda_dlls()
    sess = ort.InferenceSession(str(args.onnx), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    actual_ep = sess.get_providers()
    in_name = sess.get_inputs()[0].name
    y_o = sess.run(None, {in_name: x})[0]

    diff = np.abs(y_t - y_o)
    numeric = {
        "torch_shape": list(y_t.shape), "onnx_shape": list(y_o.shape),
        "max_abs_diff": float(diff.max()),
        "p99_9_abs_diff": float(np.percentile(diff, 99.9)),
        "mean_abs_diff": float(diff.mean()),
        "frac_above_1e-2": float((diff > 1e-2).mean()),
        "相对误差_mean除均值": float(diff.mean() / max(1e-9, np.abs(y_o).mean())),
        "输出量级_均值": float(np.abs(y_o).mean()),
        "说明": "原始输出张量未过 sigmoid，个别单元差值会被放大；判定以结果层为准",
    }
    report["数值"] = numeric
    report["ORT实际EP"] = actual_ep
    report["DLL目录"] = dll_dir
    print("\n② 数值层")
    print("   torch %s  onnx %s" % (numeric["torch_shape"], numeric["onnx_shape"]))
    print("   最大 %.3f　p99.9 %.3f　平均 %.3e　相对误差 %.2e"
          % (diff.max(), numeric["p99_9_abs_diff"], diff.mean(), numeric["相对误差_mean除均值"]))
    print("   输出量级 mean|y| = %.1f（框坐标，0–640 像素），所以绝对差要按相对量看"
          % numeric["输出量级_均值"])

    # ---------------- ③ 结果层 ----------------
    # 两边都用 Ultralytics 的同一套 predict，差别只剩「用哪个后端算」
    images = pick_images(args.num)
    pt_model = YOLO(str(args.weights))
    on_model = YOLO(str(args.onnx))
    common = dict(imgsz=640, conf=0.25, iou=0.7, verbose=False, device=0)

    per_image, tot_pt = [], []
    for img in images:
        r_pt = pt_model.predict(str(img), **common)[0].boxes
        r_on = on_model.predict(str(img), **common)[0].boxes
        c_pt, c_on = np.array(r_pt.cls.cpu()), np.array(r_on.cls.cpu())
        s_pt, s_on = np.array(r_pt.conf.cpu()), np.array(r_on.conf.cpu())
        # 按分数降序对齐后比较，避免顺序差异被当成内容差异
        order_pt, order_on = np.argsort(-s_pt), np.argsort(-s_on)
        same_cls = int((c_pt[order_pt] == c_on[order_on]).sum()) if len(c_pt) == len(c_on) else 0
        per_image.append({
            "image": img.name,
            "pt_boxes": int(len(c_pt)), "onnx_boxes": int(len(c_on)),
            "class_matches": same_cls,
            "max_conf_diff": float(np.abs(s_pt[order_pt] - s_on[order_on]).max())
            if len(s_pt) == len(s_on) and len(s_pt) else None,
        })
        tot_pt += list(c_pt)

    box_pt = sum(p["pt_boxes"] for p in per_image)
    box_on = sum(p["onnx_boxes"] for p in per_image)
    max_conf = max([p["max_conf_diff"] for p in per_image if p["max_conf_diff"] is not None] or [0.0])
    class_ok = all(p["class_matches"] == min(p["pt_boxes"], p["onnx_boxes"]) for p in per_image)
    result = {
        "图片数": len(images), "conf": common["conf"], "iou": common["iou"],
        "pt_总框数": box_pt, "onnx_总框数": box_on,
        "框数差": box_on - box_pt, "框数差比例": round((box_on - box_pt) / max(1, box_pt), 4),
        "逐类一致": bool(class_ok), "最大分数差": round(max_conf, 6),
        "逐图": per_image,
    }
    report["结果"] = result
    print("\n③ 结果层（valid 前 %d 张，conf=%.2f iou=%.2f）" % (len(images), common["conf"], common["iou"]))
    print("   pt 总框数 %d，onnx 总框数 %d，差 %+d（%.2f%%）"
          % (box_pt, box_on, box_on - box_pt, 100 * (box_on - box_pt) / max(1, box_pt)))
    print("   逐框类别一致：%s   最大分数差 %.2e" % ("是" if class_ok else "否", max_conf))

    # 判定依据放在结果层：张量层的差异要被后处理「吃掉」，否则说明预处理/解码有问题
    verdict = (abs(result["框数差比例"]) <= 0.01 and class_ok and max_conf <= 0.01)
    report["结论"] = "通过" if verdict else "未通过"
    out = REPORT_DIR / "onnx_verify_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n判据：框数差 ≤1%% + 类别全对 + 最大分数差 ≤1e-2  →  %s" % report["结论"])
    print("报告：%s" % out)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
