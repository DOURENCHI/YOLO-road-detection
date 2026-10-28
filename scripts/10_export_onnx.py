"""
10 号脚本：把 best.pt 导出成 ONNX，并把「导出这件事」本身记录成可复查的证据。

为什么单独写一个脚本，而不是敲一行 yolo export
----------------------------------------------
导出只需要一行命令，但**导出结果需要被证明**：

    · 导出的到底是什么？（opset、输入输出形状、dtype、节点数）
    · 文件是不是这次导出的？（sha256 + 时间戳）
    · 用的是哪套参数？（simplify / dynamic / imgsz 会直接改变输出张量形状）
    · 软件版本是什么？（换一台机器重导，结果可能不一样）

这些信息如果只留在终端滚动记录里，三天后就查不到了。所以本脚本把
「命令 + 环境 + 产物指纹」一起写进 docs/deployment/onnx_export_report.json。

产物去哪
--------
导出默认落在 best.pt 旁边（runs/detect/runs/road_baseline/weights/best.onnx）。
本脚本会把它移动到 weights/best.onnx，给后续脚本一个稳定、好引用的路径。

用法
----
    python scripts/10_export_onnx.py                  # 默认导 FP32，不简化以外都用默认
    python scripts/10_export_onnx.py --dynamic        # 导出可变 batch/尺寸（TensorRT 前一般不开）
    python scripts/10_export_onnx.py --opset 17       # 目标运行时较旧时手动指定
    python scripts/10_export_onnx.py --keep-in-place  # 不移动产物，留在 runs/ 下

注意
----
只读权重、只写 weights/ 与 docs/deployment/，不碰数据集，不做任何评估。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import PROJECT_ROOT  # noqa: E402

#: 阶段三训练出来的最优权重
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "runs" / "road_baseline" / "weights" / "best.pt"

#: 导出产物与报告的固定位置
WEIGHTS_DIR = PROJECT_ROOT / "weights"
REPORT_DIR = PROJECT_ROOT / "docs" / "deployment"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_onnx(path: Path) -> dict:
    """读 .onnx 的元信息：opset、输入输出、节点数。这是「结构检查」的机器版。"""
    import onnx

    model = onnx.load(str(path))
    graph = model.graph

    def tensor_info(value_info):
        shape = []
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.dim_param:                       # 动态轴：名字是字符串
                shape.append(dim.dim_param)
            elif dim.HasField("dim_value"):         # 静态轴
                shape.append(dim.dim_value)
            else:
                shape.append("?")
        return {
            "name": value_info.name,
            "dtype": onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type),
            "shape": shape,
        }

    return {
        "ir_version": model.ir_version,
        "opset_import": [{"domain": imp.domain or "ai.onnx", "version": imp.version}
                         for imp in model.opset_import],
        "producer": "%s %s" % (model.producer_name, model.producer_version),
        "inputs": [tensor_info(v) for v in graph.input],
        "outputs": [tensor_info(v) for v in graph.output],
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 ONNX 并记录证据")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=None,
                        help="留空则由 Ultralytics 选择；目标运行时旧时手动指定")
    parser.add_argument("--dynamic", action="store_true",
                        help="导出可变 batch/尺寸；TensorRT 端需要额外配 profile")
    parser.add_argument("--no-simplify", action="store_true",
                        help="关闭 onnxsim/onnxslim 简化")
    parser.add_argument("--keep-in-place", action="store_true",
                        help="不把产物移到 weights/")
    args = parser.parse_args()

    if not args.weights.is_file():
        print("找不到权重：%s" % args.weights, file=sys.stderr)
        return 2

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 记录导出前的环境，方便以后复现 ----
    import onnx
    import onnxruntime as ort
    import torch
    import ultralytics

    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    export_kwargs = dict(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=not args.no_simplify,
    )

    print("=" * 66)
    print("导出参数：%s" % json.dumps({k: v for k, v in export_kwargs.items()}, ensure_ascii=False))
    print("源权重  ：%s" % args.weights)
    print("=" * 66)

    from ultralytics import YOLO

    started = time.time()
    model = YOLO(str(args.weights))
    exported = Path(model.export(**export_kwargs))
    elapsed = time.time() - started

    # ---- 把产物放到稳定位置 ----
    target = exported
    if not args.keep_in_place:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        target = WEIGHTS_DIR / exported.name
        if target.resolve() != exported.resolve():
            if target.exists():
                target.unlink()
            shutil.move(str(exported), str(target))

    info = describe_onnx(target)
    report = {
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "环境": env,
        "导出参数": export_kwargs,
        "源权重": {"path": str(args.weights), "size_mb": round(args.weights.stat().st_size / 2**20, 2),
                   "sha256": sha256_of(args.weights)},
        "产物": {"path": str(target), "size_mb": round(target.stat().st_size / 2**20, 2),
                 "sha256": sha256_of(target), "导出耗时_s": round(elapsed, 1)},
        "结构": info,
    }

    out = REPORT_DIR / "onnx_export_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 人看的摘要 ----
    print("\n导出完成：%s（%.1f MB，耗时 %.1f s）" % (target.name, target.stat().st_size / 2**20, elapsed))
    print("opset    ：%s" % ", ".join("%s %d" % (o["domain"], o["version"]) for o in info["opset_import"]))
    for item in info["inputs"]:
        print("输入     ：%-12s %-8s %s" % (item["name"], item["dtype"], item["shape"]))
    for item in info["outputs"]:
        print("输出     ：%-12s %-8s %s" % (item["name"], item["dtype"], item["shape"]))
    print("节点数   ：%d（initializer %d 个）" % (info["node_count"], info["initializer_count"]))
    print("报告     ：%s" % out)
    print("\n下一步：python scripts/11_verify_onnx.py —— 先证明输出一致，再谈速度。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
