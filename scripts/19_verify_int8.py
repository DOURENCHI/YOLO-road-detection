"""
INT8 引擎的四层验证，以及与 FP32 / FP16 的延迟对照。

为什么 INT8 不能只看「能跑出框」
    FP16 只是把尾数从 23 位压到 10 位，误差是均匀的小扰动；INT8 是把每层的激活
    压成 256 级整数，误差取决于**校准统计出来的范围**——范围估窄了会饱和截断，
    估宽了会分辨率不足。所以 INT8 的失效方式通常是「某一类整体崩掉」或
    「小目标集体消失」，而不是整体轻微劣化。只对着几张图看框在不在，是看不出来的。

四层各查什么
    第 1 层 结构  — IO 名称/shape/dtype。注意 INT8 引擎的 IO 通常仍是 FP32，
                   量化发生在图内部，所以喂数据的方式和 FP32 完全一样。
    第 2 层 数值  — 原始输出张量 vs **FP32 引擎**（不是 vs ONNX，也不是 vs pt）。
                   同一条推理路径才有可比性；跨路径比较会把后处理差异算到量化头上。
    第 3 层 结果  — 检测框：框数、类别、IoU、坐标差。这一层才是「能不能用」。
    第 4 层 延迟  — 同一张图、同样的同步方式，三档引擎对比。

精度（mAP）不在这里算
    那是 16_eval_valid_split.py 的活（valid 上五格式对照）。本脚本只回答
    「INT8 的误差是什么性质、延迟快了多少」。分开跑是因为 mAP 评估要几分钟。

用法
    python scripts/19_verify_int8.py
    python scripts/19_verify_int8.py --images 24 --runs 300
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

ENGINE_DIR = PROJECT_ROOT / "weights"
REPORT = PROJECT_ROOT / "docs" / "deployment" / "engine_int8_verify_report.json"


def load_verify_module():
    """按文件路径导入 15 号脚本，复用它的 TrtSession / compare_boxes / letterbox。

    文件名以数字开头，不能直接 import，只能用 spec_from_file_location。
    复用的意义：两个脚本对「什么叫一致」的判据必须是同一套，
    否则 19 说一致、15 说不一致，等于没有标准。
    """
    path = PROJECT_ROOT / "scripts" / "15_verify_engine.py"
    spec = importlib.util.spec_from_file_location("verify_engine_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_layer_json(raw):
    """TRT 10 的层信息有时是「JSON 字符串里再套一层字符串」，这里逐层剥。"""
    for _ in range(3):
        if not isinstance(raw, str):
            return raw if isinstance(raw, dict) else {}
        try:
            raw = json.loads(raw)
        except Exception:
            return {}
    return raw if isinstance(raw, dict) else {}


def layer_precision_stats(engine_path: Path) -> dict:
    """统计引擎里每一层输出的数据类型，回答「INT8 到底用上了没有」。

    ⚠️ 这一步不能省：如果 INT8 引擎实际只有少数卷积走了 INT8，那「不快」
    的原因就不是「模型不适合量化」，而是「构建没量化下去」。两种结论的
    改进方向完全相反，必须用数据分清。
    """
    import tensorrt as trt

    data = engine_path.read_bytes()
    n = int.from_bytes(data[:4], "little")
    blob = data[4 + n:] if 0 < n < 100000 else data
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(blob)
    insp = engine.create_engine_inspector()

    all_dt, conv_dt, no_info = {}, {}, 0
    for i in range(engine.num_layers):
        d = _parse_layer_json(insp.get_layer_information(i, trt.LayerInformationFormat.JSON))
        if not d:
            no_info += 1
            continue
        ltype = d.get("LayerType", "?")
        for o in d.get("Outputs", []):
            v = o.get("Format/Datatype", "?")
            all_dt[v] = all_dt.get(v, 0) + 1
            if "Conv" in ltype:
                conv_dt[v] = conv_dt.get(v, 0) + 1
    conv_total = sum(conv_dt.values())
    return {
        "层数": engine.num_layers,
        "无层信息的层数": no_info,
        "全部输出dtype": all_dt,
        "卷积层dtype": conv_dt,
        "卷积INT8占比": round(conv_dt.get("Int8", 0) / conv_total, 3) if conv_total else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=16, help="结果层核对的图片数")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    v = load_verify_module()
    v.setup_dll_paths()

    import torch
    from ultralytics import YOLO

    eng32 = ENGINE_DIR / "best_fp32.engine"
    eng16 = ENGINE_DIR / "best_fp16.engine"
    eng8 = ENGINE_DIR / "best_int8.engine"
    for p in (eng32, eng16, eng8):
        if not p.is_file():
            hint = "先跑 scripts/18_export_engine_int8.py" if "int8" in p.name \
                else "先跑 scripts/14_export_engine.py"
            raise SystemExit("缺少 %s（%s）" % (p.name, hint))

    torch.cuda.init()
    lib = Path(torch.__file__).parent / "lib"
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(lib))

    report: dict = {"生成时间": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---------------- 第 1 层：结构 ----------------
    print("=" * 70)
    print("第 1 层 · 结构")
    s32, s16, s8 = v.TrtSession(eng32), v.TrtSession(eng16), v.TrtSession(eng8)
    info = {"fp32": s32.io_info(), "fp16": s16.io_info(), "int8": s8.io_info()}
    for tag, sess in (("fp32", s32), ("fp16", s16), ("int8", s8)):
        print("  [%s] 文件头：%s" % (tag, sess.header_note))
        for side in ("输入", "输出"):
            for t in info[tag][side]:
                print("       %s %-10s %-10s %s" % (side, t["name"], t["dtype"], t["shape"]))
    report["结构"] = info
    same_io = (info["fp32"]["输入"] == info["int8"]["输入"]
               and [(x["dtype"], x["shape"]) for x in info["fp32"]["输出"]]
               == [(x["dtype"], x["shape"]) for x in info["int8"]["输出"]])
    print("  → INT8 与 FP32 的 IO 是否一致：%s（量化发生在图内部，喂数据方式不变）"
          % ("是" if same_io else "否，需单独适配"))
    report["IO一致"] = bool(same_io)

    # ---------------- 第 1.5 层：量化覆盖率（决定后面怎么解释性能） ----------------
    print("\n" + "=" * 70)
    print("第 1.5 层 · 量化覆盖率（INT8 到底用上了没有）")
    cover = {}
    for tag, p in (("fp32", eng32), ("fp16", eng16), ("int8", eng8)):
        cover[tag] = layer_precision_stats(p)
        c = cover[tag]
        if c["无层信息的层数"] == c["层数"]:
            print("  %-5s 引擎未提供逐层信息（TRT 版本/构建方式限制），跳过" % tag.upper())
            continue
        print("  %-5s 卷积层 dtype %s　→ INT8 覆盖 %s"
              % (tag.upper(), c["卷积层dtype"],
                 ("%.0f%%" % (100 * c["卷积INT8占比"])) if c["卷积INT8占比"] is not None else "—"))
    report["量化覆盖率"] = cover
    if cover["int8"].get("卷积INT8占比") is not None:
        r = cover["int8"]["卷积INT8占比"]
        if r < 0.3:
            print("  ⚠ INT8 卷积占比只有 %.0f%%，「不快」可能是构建没量化下去，不是模型不适合。" % (100 * r))
        else:
            print("  → INT8 已覆盖 %.0f%% 的卷积层，量化确实生效；后面若仍不快，"
                  "要从「模型规模/访存瓶颈」解释，而不是「没量化」。" % (100 * r))

    # ---------------- 第 2 层：数值 ----------------
    print("\n" + "=" * 70)
    print("第 2 层 · 数值（原始输出张量，参考 = FP32 引擎）")
    import cv2  # noqa: F401  (letterbox_tensor 内部会用到)
    images = v.pick_images(args.images)
    x = v.letterbox_tensor(images[0])

    y32, _ = s32.run(x, args.warmup, 20)
    y16, _ = s16.run(x, args.warmup, 20)
    y8, _ = s8.run(x, args.warmup, 20)

    scale = float(np.abs(y32).mean())
    numeric = {}
    for tag, y in (("fp16", y16), ("int8", y8)):
        d = np.abs(y32 - y)
        numeric[tag] = {
            "max_abs_diff": round(float(d.max()), 4),
            "mean_abs_diff": round(float(d.mean()), 5),
            "p99_abs_diff": round(float(np.percentile(d, 99)), 4),
            "相对误差_mean除均值": round(float(d.mean() / max(1e-9, scale)), 5),
            "输出量级_mean": round(scale, 2),
        }
        print("  %-5s 最大 %.3f　p99 %.3f　平均 %.5f　相对 %.2e"
              % (tag.upper(), d.max(), numeric[tag]["p99_abs_diff"], d.mean(),
                 numeric[tag]["相对误差_mean除均值"]))
    print("  参考：输出量级 mean|y| = %.1f（框坐标 0–640 像素）" % scale)
    print("  解读：INT8 的相对误差通常比 FP16 大 1–2 个数量级，这是 8 bit 的固有代价；")
    print("        数值层永远只能当「有没有崩」的探针，能不能用要看第 3 层和第 4 层的 mAP。")
    report["数值"] = numeric

    # ---------------- 第 3 层：结果 ----------------
    print("\n" + "=" * 70)
    print("第 3 层 · 结果（%d 张图，conf 0.25 / iou 0.7）" % len(images))
    m32 = YOLO(str(eng32))
    m16 = YOLO(str(eng16))
    m8 = YOLO(str(eng8))

    def boxes(model, p):
        r = model.predict(str(p), imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
        b = r.boxes
        return [[float(t) for t in xy] + [float(c), int(k)]
                for xy, c, k in zip(b.xyxy.cpu(), b.conf.cpu(), b.cls.cpu())]

    per_image, agg = [], {"fp16": [], "int8": []}
    for p in images:
        b32 = boxes(m32, p)
        b16 = boxes(m16, p)
        b8 = boxes(m8, p)
        rec = {"图": p.name, "框数_fp32": len(b32), "框数_int8": len(b8)}
        c16 = v.compare_boxes(b32, b16)
        c8 = v.compare_boxes(b32, b8)
        agg["fp16"].append(c16)
        agg["int8"].append(c8)
        rec["fp16"] = c16
        rec["int8"] = c8
        per_image.append(rec)

    summary = {}
    for tag in ("fp16", "int8"):
        rows = agg[tag]
        paired = [r for r in rows if r["配对"] > 0]
        ious = [r["最小IoU"] for r in rows if "最小IoU" in r]
        summary[tag] = {
            "总框数_参考": sum(r["框数_参考"] for r in rows),
            "总框数_对象": sum(r["框数_对象"] for r in rows),
            "完全一致的图数": sum(1 for r in rows if r.get("完全一致")),
            "图数": len(rows),
            "最小IoU_全部": round(min(ious), 4) if ious else None,
            "类别全对的图数": sum(1 for r in rows if r.get("类别全对")),
            "平均框数差": round(sum(abs(r["框数_对象"] - r["框数_参考"]) for r in rows) / len(rows), 2),
        }
        print("  %-5s 框数 %d vs %d　完全一致 %d/%d 图　最小 IoU %s　类别全对 %d/%d 图　平均框数差 %.2f"
              % (tag.upper(), summary[tag]["总框数_对象"], summary[tag]["总框数_参考"],
                 summary[tag]["完全一致的图数"], len(rows), summary[tag]["最小IoU_全部"],
                 summary[tag]["类别全对的图数"], len(rows), summary[tag]["平均框数差"]))
    print("  判据：框数差 <5%、类别全对、最小 IoU >0.9 → 结果层可用；")
    print("        若个别类整类消失，说明该类激活范围被校准压坏了，需要加校准样本或换 calibration_algorithm。")
    report["结果"] = {"汇总": summary, "逐图": per_image}

    # ---------------- 第 4 层：延迟 ----------------
    print("\n" + "=" * 70)
    print("第 4 层 · 纯推理延迟（同一张图，warmup %d / runs %d，每次前后同步）"
          % (args.warmup, args.runs))
    latency = {}
    for tag, sess in (("fp32", s32), ("fp16", s16), ("int8", s8)):
        _, samples = sess.run(x, args.warmup, args.runs)
        st = v.stats(samples)
        latency[tag] = st
        print("  %-5s 中位 %6.2f ms　p95 %6.2f　均值 %6.2f　最小 %6.2f"
              % (tag.upper(), st["中位数"], st["p95"], st["均值"], st["最小"]))

    base = latency["fp32"]["中位数"]
    for tag in ("fp16", "int8"):
        latency[tag]["相对fp32加速比"] = round(base / latency[tag]["中位数"], 2)
        print("  %-5s 相对 FP32 加速 %.2f×" % (tag.upper(), latency[tag]["相对fp32加速比"]))
    report["延迟"] = {"参数": {"warmup": args.warmup, "runs": args.runs}, "结果": latency}

    report["口径"] = {
        "参考": "FP32 引擎（同一条推理路径），不用 ONNX/pt 作参考",
        "mAP": "不在本脚本内计算，见 docs/deployment/valid_precision_compare.json",
        "为什么": "INT8 的失效方式是「某一类整体崩」而非整体轻微劣化，必须看逐类 mAP",
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n报告：%s" % REPORT.relative_to(PROJECT_ROOT))
    print("下一步：python scripts/16_eval_valid_split.py  # valid 上五格式 mAP，看逐类有没有崩")


if __name__ == "__main__":
    sys.exit(main())
