"""
项目级数据集配置（唯一的路径与类别来源）。

设计目标
--------
换一份数据集时，只需要替换 datasets 下的 data.yaml，
不必再改动任何一个脚本里的类别数、类别名或路径。

这个模块被 01~05 号脚本和 visualize_labels.py 共同 import。
它自己不打印任何东西，只提供函数和常量。
"""

from __future__ import annotations

from pathlib import Path

import yaml

# ---------------------------------------------------------------- 项目根目录


def find_project_root(start: Path | None = None) -> Path:
    """
    从本文件所在目录向上找第一层"含 datasets/ 目录"的位置，作为项目根。

    这样写的好处：不管你在项目根目录、scripts/ 目录还是别的地方执行脚本，
    找到的项目根都一样，不会再出现"必须在项目根目录运行"的隐性要求。
    """
    here = Path(start) if start is not None else Path(__file__).resolve().parent

    for candidate in [here, *here.parents]:
        if (candidate / "datasets").is_dir():
            return candidate

    # 兜底：本文件在 scripts/ 下，就往上一级
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = find_project_root()

# ---------------------------------------------------------------- 数据集位置

#: 原始数据集：Roboflow 导出，标注里混着"矩形框"和"多边形"两种格式
SOURCE_DATASET = PROJECT_ROOT / "datasets" / "road-traffic"

#: 转换后的数据集：全部统一成 YOLO 检测框（类别 + 中心 xy + 宽高）
TARGET_DATASET = PROJECT_ROOT / "datasets" / "road-traffic-detect"

#: 两套数据集的配置文件。脚本要读类别和路径，都从这两个文件拿。
SOURCE_YAML = SOURCE_DATASET / "data.yaml"
TARGET_YAML = TARGET_DATASET / "data.yaml"

#: 三个划分的固定顺序。
#: 注意：目录名叫 valid，但 YOLO 配置里这个键必须写成 val，映射见 YAML_KEYS。
SPLITS = ("train", "valid", "test")

#: 划分名 -> data.yaml 里的键名
YAML_KEYS = {"train": "train", "valid": "val", "test": "test"}

#: 认作图片的扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

#: 可视化脚本默认把抽检结果存到这里
RESULT_DIR = PROJECT_ROOT / "results" / "label_check"


# ---------------------------------------------------------------- 读 data.yaml


def load_data_yaml(yaml_path=SOURCE_YAML) -> dict:
    """读一个 data.yaml，返回它解析出来的字典。"""
    path = Path(yaml_path)

    if not path.is_file():
        raise FileNotFoundError(f"找不到数据集配置文件：{path}")

    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if "names" not in data:
        raise ValueError(f"{path} 里没有 names 字段，无法确定类别")

    return data


def load_names(yaml_path=SOURCE_YAML) -> list[str]:
    """
    返回类别名列表，列表下标就是标签文件里的类别编号 0..nc-1。

    兼容两种写法：
      names: [a, b]          -> 列表
      names: {0: a, 1: b}    -> 字典，按 key 排序后取 value
    """
    names = load_data_yaml(yaml_path)["names"]

    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]

    return [str(name) for name in names]


def resolve_dataset_root(yaml_path=SOURCE_YAML) -> Path:
    """
    按 data.yaml 里的 path 字段算出数据集根目录。

    本模块的规则：
      - path 是绝对路径 -> 直接用它
      - path 是相对路径 -> 相对于 data.yaml 文件所在目录

    ⚠️ 这一条和 Ultralytics 自己的做法**不一致**，是有意为之，不是抄错。

    Ultralytics 在 check_det_dataset() 里写的是：
        path = Path(extract_dir or data.get("path") or Path(...).parent)
        if not path.exists() and not path.is_absolute():
            path = (DATASETS_DIR / path).resolve()
    当 path 写成 "."（Roboflow 导出的默认值）时，Path(".") 恒为真，
    所以那句 DATASETS_DIR 兜底**永远不会触发**，path 保持相对值 ".",
    最后由 (Path(".") / "train/images").resolve() 按**进程当前工作目录**展开。

    结果就是：同一份写着 path: . 的 yaml，
        用本模块读        -> 解析到 datasets/<数据集名>/
        交给 Ultralytics  -> 解析到「你当时站在哪个目录」
    两者给出的答案不同。

    所以本项目里，**真正交给训练用的 data.yaml 一律写绝对路径**（见 README 6.2）。
    本模块坚持"相对 yaml 文件"，是为了让 01~05 号脚本换个启动目录也能跑出同样结果。
    """
    yaml_path = Path(yaml_path)
    raw = load_data_yaml(yaml_path).get("path", ".") or "."
    root = Path(str(raw))

    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    return root


def split_dirs(split: str, yaml_path=SOURCE_YAML):
    """
    返回某个划分的 (images 目录, labels 目录)。

    约定：图片目录同级一定有一个 labels 目录。
    该划分在 yaml 里没有配置时返回 (None, None)，由调用方决定跳过还是报错。
    """
    yaml_path = Path(yaml_path)
    data = load_data_yaml(yaml_path)
    root = resolve_dataset_root(yaml_path)

    key = YAML_KEYS.get(split, split)
    relative = data.get(key)

    if relative is None:
        return None, None

    images_dir = (root / str(relative)).resolve()
    return images_dir, images_dir.parent / "labels"


def read_label_lines(label_path) -> list[str]:
    """读一个标签文件的所有行。用 utf-8-sig 是为了兼容带 BOM 的 Roboflow 导出文件。"""
    return Path(label_path).read_text(encoding="utf-8-sig").splitlines()
