# 代码笔记

关键函数、参数、数据结构与原理。

## 1. 配置层 `dataset_config.py`

这是全项目**唯一的路径与类别来源**，被 01~05 和 `visualize_labels.py` 共同 import。
它自己不打印任何东西，只提供函数和常量。

### `find_project_root(start=None)`

从本文件所在目录**逐级向上**找，第一个含 `datasets/` 目录的位置就是项目根。

```python
for candidate in [here, *here.parents]:
    if (candidate / "datasets").is_dir():
        return candidate
```

价值：脚本在项目根、`scripts/`、或任何其他目录启动，定位到的项目根都一样，
**消除"必须在项目根目录运行"这个隐性要求**。

### `load_names(yaml_path)`

返回类别名列表，下标即标签文件里的类别编号 `0..nc-1`。
兼容两种 `data.yaml` 写法：

| 写法 | 处理 |
|---|---|
| `names: [a, b]` | 直接用 |
| `names: {0: a, 1: b}` | 按 key 排序后取 value |

### `resolve_dataset_root(yaml_path)`

按 `data.yaml` 的 `path` 字段算数据集根目录：

- 绝对路径 → 直接用
- 相对路径 → **相对于 `data.yaml` 所在目录**

**这条规则和 Ultralytics 有意不一致**，是个关键设计。

Ultralytics 的 `check_det_dataset()` 里写的是：

```python
path = Path(extract_dir or data.get("path") or ...)
if not path.exists() and not path.is_absolute():
    path = (DATASETS_DIR / path).resolve()
```

当 `path` 写成 `"."`（Roboflow 导出默认值）时，`Path(".")` **恒为真**，
所以 `DATASETS_DIR` 那句兜底**永远不会触发**，`path` 保持相对值 `"."`，
最后由 `(Path(".") / "train/images").resolve()` 按**进程当前工作目录**展开。

结果：同一份写着 `path: .` 的 yaml，

| 谁读 | 解析到 |
|---|---|
| 本模块 | `datasets/<数据集名>/` |
| Ultralytics | "你当时站在哪个目录" |

两者答案不同。所以**真正交给训练用的 `data.yaml` 一律写绝对路径**；
本模块坚持"相对 yaml 文件"，是为了让 01~05 号脚本换启动目录也能跑出同样结果。

### 关键常量

| 常量 | 值 | 说明 |
|---|---|---|
| `SPLITS` | `("train","valid","test")` | 目录名叫 valid |
| `YAML_KEYS` | `{"valid": "val", ...}` | yaml 里必须写 `val` |
| `IMAGE_EXTENSIONS` | jpg/jpeg/png/bmp/webp | — |

## 2. 格式转换 `04_convert_labels_to_detection.py`

### `convert_line(line, file_name, line_no)`

按**字段数**分流，这是整个转换的核心判断：

| 字段数 | 类型 | 处理 |
|---|---|---|
| `5` | 标准检测框 `cls cx cy w h` | 校验后原样保留（保留 8 位小数） |
| `> 5` | 多边形 `cls x1 y1 x2 y2 ...` | 取**最小外接矩形**压成框 |
| `< 5` | 非法 | 记异常 |

多边形的压缩方式：

```python
xs = values[0::2]          # 所有 x
ys = values[1::2]          # 所有 y
x_min, x_max = min(xs), max(xs)
w = x_max - x_min
h = y_max - y_min
x_center = (x_min + x_max) / 2
```

**多重校验**（任一不通过就记异常、不写入新标签）：

| 校验 | 目的 |
|---|---|
| 类别在 `0 <= cls < len(names)` | 防越界编号 |
| 所有数值在 `[0, 1]` | YOLO 用归一化坐标 |
| `w > 0 and h > 0` | 零面积框没有意义 |
| `x - w/2 >= -1e-6`、`x + w/2 <= 1 + 1e-6` | 框不能越出图像边界（留浮点容差） |
| 多边形坐标数 ≥ 6 且为偶数 | 至少 3 个点才构成多边形 |

### 两个安全设计

1. **`DEST.exists()` 就 `raise FileExistsError` 停下**，绝不覆盖已有数据
2. **`except` 里不删目标目录**，留现场方便检查

### 审计报告

每次转换写 `docs/label_conversion_report.txt`，逐条记录异常行 + 原始内容。
这份报告是"转换过程可复盘"的证据，不是日志垃圾。

## 3. 推理性能 `benchmark_inference.py`

### `timed(function, warmup, runs, sync)`

```python
for _ in range(warmup):
    function()
sync()

samples = []
for _ in range(runs):
    sync()                      # ① 计时前同步
    start = time.perf_counter()
    function()
    sync()                      # ② 计时后同步
    samples.append((time.perf_counter() - start) * 1000)
```

**两个关键点**：

| 点 | 原因 |
|---|---|
| **前后各加一次 `sync()`** | CUDA 调用是**异步**的。不同步，测到的是"下发指令花了多久"，不是"算完花了多久" |
| **必须先 `warmup`** | 第一次推理要加载 kernel、分配显存，比稳定值大几个数量级 |

`sync` 由调用方传入：GPU 时是 `torch.cuda.synchronize`，CPU 时是空函数。

### 三种计时口径

| 口径 | 实现 | 含什么 |
|---|---|---|
| 纯 GPU 前向 | `torch.cuda.Event` 卡在 GPU 上 | 只算模型前向 |
| 推理阶段 | `result.speed["inference"]` | Ultralytics 自带口径 |
| 完整端到端 | `perf_counter` + `synchronize` | 读图、letterbox、前后处理、NMS 全含 |

本机实测同一权重同一张图：**15.8 / 11.3 / 16.4 ms**——
所以"延迟 xx ms"不加条件是没有意义的。

## 4. 数据泄漏检查 `check_data_leakage.py`

三重检查，强度递增：

### `md5_of(path)`

按**文件内容**算 md5（分块读，1 MB 一块）。
改名不影响结果，能抓到"同一张图换名字复制到别的划分"。

### `dhash_of(path, size=DHASH_SIZE)`

差值哈希（dHash）：

```python
small = im.convert("L").resize((size + 1, size), Image.LANCZOS)
pixels = np.asarray(small, dtype=np.int16)
for value in (pixels[:, 1:] > pixels[:, :-1]).flatten():
    bits = (bits << 1) | int(value)
```

缩到 `(size+1) × size` 灰度图，**逐行比较相邻像素**：左边比右边亮记 1，否则记 0，
拼成 `size × size` 位的整数指纹。

能抓到 md5 抓不到的：缩放、重压缩造成的重复。判断靠**汉明距离**（阈值 5）。

### `stamp_of(name)`

从文件名前缀取 13 位毫秒时间戳，`human()` 转成可读时间。
能抓到"同一段采集被切到不同划分"。

### 本机实测结论（要分两句说）

| 检查 | 结果 |
|---|---|
| 内容 md5 | 跨划分 **0 对**，划分内 0 组 → 干净 |
| dHash | 跨划分 0 对；最接近一对距离 **15**（阈值 5）→ 检查有区分度 |
| 时间戳 | 三划分时间跨度**完全重叠**（2022-05-23 ~ 2022-07-01），最近一对只隔 **4.9 秒**，1 分钟内 **158 对** |

**结论必须分两句**：内容层面没有重复；但划分是按图随机切的，
**可能存在轻度场景重叠，测试指标要按偏乐观理解**。
时间戳靠得近只是弱证据，不等于画面重复——这句边界必须一起说。

## 5. 参数速查

### 数据集

| 项 | 值 |
|---|---|
| 来源 | Roboflow Universe · road-traffic v1（release-640） |
| 许可 | CC BY 4.0（公开使用需署名） |
| 划分 | train 494 / valid 187 / test 133 |
| 类别 | 12 类 |
| 框总数 | 1804 / 839 / 522 |

### 训练

| 参数 | 值 | 说明 |
|---|---|---|
| `model` | `yolo26n.pt` | nano 版，迁移学习 |
| `epochs` | 50 | |
| `batch` | 8 | 受 8 GB 显存限制 |
| `imgsz` | 640 | |
| `lr0` / `lrf` | 0.01 / 0.01 | 1e-2 → 1e-4 |
| `warmup_epochs` | 3.0 | |
| `close_mosaic` | 10 | 最后 10 轮关 Mosaic |
| `seed` / `deterministic` | 0 / true | 便于复现 |
| `workers` | 0 | Windows 上便于看报错 |

### 环境

Python 3.11.16 / PyTorch 2.11.0+cu128 / CUDA 12.8 / Ultralytics 8.4.176 /
RTX 5060 Laptop 8 GB。训练耗时 660.4 s（约 11 分钟）。

## 6. 原理补充

### 为什么改标签后必须删 `.cache`

YOLO 第一次读数据后会在标签目录生成 `train.cache` / `val.cache`，
里面是**解析好的标签索引**。改了 `.txt` 却不删缓存，训练读的还是旧索引——等于没改。

```bash
rm datasets/road-traffic-detect/train/labels/train.cache
rm datasets/road-traffic-detect/valid/labels/val.cache
```

### 为什么 `path: .` 会导致 `missing path`

见第 1 节 `resolve_dataset_root` 的说明。一句话：**相对路径按进程 cwd 展开，不按 yaml 位置**。
从项目根启动训练，它会去找 `项目根/valid/images`，那里没有 valid 目录。

### 为什么不能用文件大小判断权重是否相同

`best.pt` 和 `last.pt` 都是 5,469,061 字节，但 md5 不同
（`974c53ee…` / `e56d3064…`）。第 48 轮指标最好，之后又跑了 2 轮，
权重数值变了、序列化后大小没变。比较必须用哈希。

### 为什么 dHash 比 md5 更适合查"重复图片"

md5 对**任何一个字节**的变化都敏感——重新压缩一次 JPEG，md5 完全变掉，
但画面几乎一样。dHash 只保留"亮暗相对关系"的结构信息，
所以能吃下缩放和重压缩，代价是可能误报（靠汉明距离阈值控制）。
