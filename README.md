# YOLO 道路目标检测项目

用 Ultralytics YOLO26n 在 road-traffic 数据集上做多类别目标检测的完整流程：
**数据检查 → 标注格式统一 → 数据验收 → 训练 → 评估 → 推理演示 → 视频跟踪**

这个项目的目的不只是"把模型训出来"，而是把一条**可复现、可追溯、别人能接手**的
数据处理链路走通。每一步都有对应脚本，每个脚本都会打印可核对的数字；
所有结论性的数字都会**连同它的测量条件**一起记录下来。

---

## 1. 目录结构

```
YOLO_Road_Detection/
├── scripts/                          # 全部脚本（可在任意目录下运行）
│   ├── dataset_config.py             # 共享配置：路径 + 类别，从 data.yaml 读
│   ├── 01_check_dataset.py           # 结构检查（图片/标签配对、格式、数值）
│   ├── 02_inspect_label_formats.py   # 标注格式分类统计
│   ├── 03_find_bad_coordinates.py    # 多边形标注数值专项排查
│   ├── 04_convert_labels_to_detection.py  # 多边形 -> 矩形框，生成新数据集
│   ├── 05_validate_detection_dataset.py   # 转换后数据集验收 + 类别分布
│   ├── visualize_labels.py           # 把**人工标注**画回原图，抽检数据质量
│   ├── 06_predict.py                 # 推理入口：图 / 目录 / 视频 / 摄像头
│   ├── 07_make_demo_figure.py        # 生成「人工标注 vs 模型预测」对照拼图
│   ├── 08_track.py                   # 视频多目标跟踪（ByteTrack）
│   ├── 09_evaluate.py                # 正式评估 + 自动生成评估报告
│   ├── benchmark_inference.py        # 推理性能测试（三档计时 + 6 项口径）
│   ├── check_data_leakage.py         # 数据泄漏检查（md5 + 感知哈希 + 时间戳）
│   └── _backup_2026-10-14/           # 改造前的脚本原件（留档）
├── datasets/
│   ├── road-traffic/                 # 原始数据集（Roboflow 导出）
│   │   ├── train/ valid/ test/       # 各自含 images/ 与 labels/
│   │   ├── data.yaml                 # 源头配置：names / nc / path
│   │   └── README*.txt               # 来源与许可说明
│   └── road-traffic-detect/          # 04 号脚本生成的检测框数据集
│       ├── train/ valid/ test/
│       └── data.yaml                 # 训练实际使用的配置
├── docs/                             # 各类记录
│   ├── RESUME_PROJECT.md                  # 简历条目与面试材料
│   ├── label_conversion_report.txt        # 04 号脚本生成的转换报告（审计日志）
│   ├── inference_benchmark.json           # 性能测试原始统计量 + 完整测量条件
│   ├── data_leakage_report.json           # 数据泄漏检查的完整结果
│   └── args_failed_run_2026-10-13.yaml    # 那次失败运行的参数快照（证据）
├── runs/detect/runs/road_baseline/   # 正式训练产物（曲线、混淆矩阵、权重）
├── results/
│   ├── label_check/                  # 人工标注抽检图
│   ├── predict/                      # 06 号脚本的推理结果（带框的图 / 视频）
│   ├── demo/demo_grid.png            # 07 号脚本的对照拼图（可直接当简历配图）
│   ├── track/                        # 08 号脚本的跟踪视频与轨迹统计
│   └── eval_report/                  # 09 号脚本的评估报告与图表
├── assets/                           # 演示素材（示例图；自己录的视频放 assets/videos/）
├── weights/yolo26n.pt                # 预训练权重
├── requirements.txt                  # 依赖清单（版本已固定）
└── LICENSE                           # 代码 MIT；数据集 CC BY 4.0（须署名）
```

> **权重为什么要写全路径**：Ultralytics 只按命令里写的路径找权重，找不到就**自动下载一份到当前目录**。
> 如果命令写 `model=yolo26n.pt` 而当前目录没有，它会再下一份；想避免重复下载，就写
> `model=weights/yolo26n.pt`。（原来根目录那份重复权重就是这么来的，2026-10-14 已清掉。）

---

## 2. 环境与依赖

本机实测通过的版本（Windows，内部版本 10.0.26300）：

| 组件 | 版本 |
|---|---|
| Python | 3.11.16（conda 环境名 `yolo_road`） |
| PyTorch | 2.11.0+cu128 |
| CUDA 运行时 | 12.8 |
| Ultralytics | 8.4.176 |
| NumPy | 2.4.6 |
| OpenCV | 5.0.0 |
| Pillow | 12.3.0 |
| Matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| lap | 0.5.13（多目标跟踪的线性分配求解器） |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8 GB，驱动 591.91 |

完整的版本清单在 `requirements.txt`（已固定版本，可直接复用）。

从零重建环境：

```bash
conda create -n yolo_road python=3.11 -y
conda activate yolo_road

# 第 1 步：PyTorch 版本要和本机 CUDA 对应。下面这条对应 cu128 构建。
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 第 2 步：其余依赖全部照 requirements.txt 装
pip install -r requirements.txt
```

> 国内下载慢就加镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`。
> 注意要**先装 PyTorch 再装其余**，否则 pip 可能顺手装一个 CPU 版的 torch 上去。

装完自检：

```bash
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
```

要看到形如 `2.11.0+cu128 True 8.4.176` 的输出才能用 GPU 训练。
如果 `cuda.is_available()` 打印 `False`，说明装成了 CPU 版，卸载重装 CUDA 版，
不要在 CPU 上硬跑。

> **关于 `workers`（Windows 用户注意）**
> 所有脚本和训练都用了 `workers=0`。这是**为了便于排查问题而采取的保守设置**：
> Windows 上 DataLoader 用 spawn 方式开子进程，一旦配错，程序会长时间没有输出、
> 看起来像卡死。`workers=0` 让数据加载在主进程里跑，报错能直接打出来。
> 它不是 Windows 的硬性要求，熟悉之后可以调大换速度。

---

## 3. 数据集

- **来源**：[Roboflow Universe · road-traffic](https://universe.roboflow.com/roboflow-100/road-traffic/dataset/1)
  （workspace `roboflow-100`，版本 v1，release-640）
- **许可**：**CC BY 4.0**。公开使用需署名，出处文件保留在
  `datasets/road-traffic/README.dataset.txt`、`README.roboflow.txt`、`README.md`。
- **原始作者**：Muntaser Al Abdulla Aljouma；本数据集属于 RF100 基准计划。

划分与规模：

| 划分 | 图片数 | 标签文件数 | 标注框总数 |
|---|---|---|---|
| train | 494 | 494 | 1804 |
| valid | 187 | 187 | 839 |
| test | 133 | 133 | 522 |

**12 个类别**（编号顺序即 `data.yaml` 里 `names` 的顺序）：

```
0 bicycles   1 bridges    2 buses      3 chimneys
4 crosswalks 5 fire hydrants 6 motorcycles 7 stairs
8 taxis      9 tractors   10 traffic lights 11 vehicles
```

### 这个数据集的三个已知特点

1. **标注格式不统一**：原始标注里混着"矩形框"（5 字段）和"多边形"（实例分割格式）。
   train 有 3 行多边形、valid 有 804 行、test 有 53 行，合计 860 行，需要统一转换。
2. **类别严重不均衡**：train 里 `vehicles` 有 763 个框，而 `tractors` 只有 **1** 个；
   test 里 `tractors` 是 **0** 个。所以总体 mAP 好看，不代表每个类别都可靠。
3. **有 1 个空标签文件**：`datasets/road-traffic-detect/valid/labels/1653753597698_jpg.rf.830c9b5213f4259a90da83dd73551633.txt`
   （0 字节）。空标签可能表示"这张图确实没有目标"，也可能是漏标，需要看原图判断。
   用 `visualize_labels.py` 可以直接把那批图调出来看。

> **数据集适配度提醒**：`bridges`、`chimneys`、`stairs` 这几类和典型的道路感知任务
> 并不完全对应。这个数据集适合用来**打通和验证检测流程**，但不能据此宣称
> 做出了可靠的自动驾驶感知系统。

---

## 4. 脚本运行顺序

所有脚本都可以**在任意目录下执行**（内部会自动定位项目根），
下面的命令以项目根目录为当前目录。

### 第 1 步 · 检查原始数据集

```bash
python scripts/01_check_dataset.py
```

看四件事：图片与标签是否一一对应、标签行字段数是否为 5、类别编号是否越界、
坐标是否在 `[0, 1]` 内。输出会分 train / valid / test 三段。

### 第 2 步 · 看清格式构成

```bash
python scripts/02_inspect_label_formats.py
```

把每一行按字段数归类，回答"到底混了几种格式"。
预期结果：train 1801 个标准框 + 3 个多边形；valid 35 + 804；test 469 + 53。

### 第 3 步 · 排查多边形本身有没有坏

```bash
python scripts/03_find_bad_coordinates.py
```

三个划分都应该是 **异常多边形行数：0**。
这一步的意义是：先证明"多边形只是格式不同、数值本身是好的"，
后面才敢放心把它转换成矩形框。

### 第 4 步 · 统一格式，生成检测数据集

```bash
python scripts/04_convert_labels_to_detection.py
```

把多边形取最小外接矩形压成检测框，图片一并复制，并生成新的 `data.yaml`，
同时写下转换报告 `docs/label_conversion_report.txt`。

**注意**：如果 `datasets/road-traffic-detect/` 已存在，脚本会主动停下并报错，
不覆盖已有数据。要重跑请先把它改名或移走。

### 第 5 步 · 验收新数据集

```bash
python scripts/05_validate_detection_dataset.py
```

比第 1 步更严格的六道校验，并且**打印每个类别的框数**——这组数字就是
"类别不均衡"的直接证据。预期结果：三个划分都 0 异常、0 空标签（valid 那 1 个）。

### 第 6 步 · 把标签画回原图，人工抽检

```bash
# 三个划分各抽 8 张（默认），结果存到 results/label_check/
python scripts/visualize_labels.py

# 每个划分抽 30 张，换一批随机样本
python scripts/visualize_labels.py --n 30 --seed 42

# 只看验证集
python scripts/visualize_labels.py --split valid
```

为什么必须有这一步：前面几个脚本**只能证明"格式合法"**，
证明不了"每个框都准确地包住了正确目标"。类别标错、框拖偏、整张漏标，
脚本一个都发现不了，只能靠人眼看。

产物（`results/label_check/`）：

| 文件 | 内容 |
|---|---|
| `train_sample.png` / `valid_sample.png` / `test_sample.png` | 各划分的抽检拼图 |
| `empty_label_check.png` | 空标签对应的原图，专门放大核对 |
| `summary.txt` | 本次抽检的文字记录，可直接截图进报告 |

### 第 7 步 · 测推理性能

```bash
# 默认用 best.pt，FP32 与 FP16 各测一遍
python scripts/benchmark_inference.py

# 多跑几次，把结果存成 JSON 存档
python scripts/benchmark_inference.py --runs 500 --warmup 50 --json-out docs/inference_benchmark.json

# 只测 FP32，或换输入尺寸
python scripts/benchmark_inference.py --half fp32 --imgsz 320
```

这个脚本会把 **5 项口径**全部打印出来（硬件型号 / 输入尺寸 / 预热次数 / 计时方式 / 精度模式），
并且把三种计时口径**分开报**：

| 口径 | 算什么 |
|---|---|
| 纯 GPU 前向 | 用 `torch.cuda.Event` 卡在 GPU 上，只算模型前向 |
| 推理阶段 | Ultralytics 自带的 `result.speed["inference"]` |
| 完整端到端 | `perf_counter` + `torch.cuda.synchronize()`，含读图、letterbox、前后处理、NMS |

跑之前先记住一点：**"延迟 xx ms" 这句话不加条件是没有意义的。** 同一份权重、同一张图，
换个测法就能差一倍以上（本机实测：15.8 / 11.3 / 16.4 ms）。

还有两条不在参数里、但同样影响结果的条件：**测的时候机器要空闲**（并行跑别的任务会让
端到端数字虚高 40% 以上），以及**至少重复两轮**（本机 FP16 单张两轮差 57%，只测一次很容易
把偶然值当结论）。

### 第 8 步 · 检查有没有数据泄漏

```bash
python scripts/check_data_leakage.py
python scripts/check_data_leakage.py --json-out docs/data_leakage_report.json
```

三重检查，强度递增：

| 检查 | 原理 | 能抓到什么 |
|---|---|---|
| 内容完全相同 | 文件内容 md5 | 同一张图换名字复制到别的划分 |
| 画面近似 | dHash 感知哈希（64 位指纹 + 汉明距离） | 缩放、重压缩造成的重复 |
| 时间戳邻域 | 文件名前缀是 13 位毫秒时间戳 | 同一段采集被切到不同划分 |

本机跑出来的结果：

- 内容 md5：**跨划分 0 对**，划分内也 0 组 → 干净
- dHash：跨划分 0 对；抽样里最接近的一对距离 **15**（阈值 5）→ 说明检查有区分度
- 时间戳：三个划分的时间跨度**完全重叠**（都是 2022-05-23 ~ 2022-07-01），
  跨划分最近的一对只隔 **4.9 秒**，1 分钟内有 **158 对**

> **结论要分两句说**：内容层面没有重复；但划分是按图随机切的，
> **可能存在轻度场景重叠，测试指标要按偏乐观来理解**。
> 时间戳靠得近只是弱证据，不等于画面重复——这句边界必须一起说，
> 否则就是把检查结果说过了头。

---

### 训练之后的四步（第 9~12 步）

下面这四步都需要先有一个训练好的模型，所以请在完成第 5 节的训练之后再跑。
脚本编号和运行顺序不完全一致，按这里的顺序走最顺。

### 第 9 步 · 在测试集上做一次正式评估

```bash
python scripts/09_evaluate.py --split test
```

**这一步只能跑一次，跑之前请想清楚。** 验证集（valid）用来挑模型、调阈值；
测试集如果也反复看，它就不再是"没见过的数据"了，报出来的数字会虚高。

产出（`results/eval_report/`）：

| 文件 | 内容 |
|---|---|
| `evaluation_report.md` | **先看这个**：结论 + 逐类表 + 能说明什么 / 不能说明什么 |
| `metrics_per_class.csv` | 逐类 P / R / mAP50 / mAP50-95 + 训练集与测试集框数 |
| `fig_per_class_ap.png` | 逐类 AP 柱状图 |
| `fig_confusion_matrix.png` | 归一化混淆矩阵（按列归一化，对角线就是召回率） |
| `fig_f1_curve.png` | F1 随置信度阈值的变化，标出最优点 |
| `ultralytics_raw/` | 框架原生输出的英文评估图（留档用） |

### 第 10 步 · 跑推理看效果

```bash
# 不给参数：默认跑验证集前 8 张
python scripts/06_predict.py

# 跑某个划分若干张
python scripts/06_predict.py --split valid --limit 20

# 跑自己放进 assets/ 的图
python scripts/06_predict.py --source assets/demo

# 跑视频
python scripts/06_predict.py --source 你的视频.mp4

# 摄像头实时（0 是默认摄像头编号）
python scripts/06_predict.py --source 0
```

结果存到 `results/predict/<name>/`，同时打印每张图检出多少个目标、类别分布、
以及一份 `summary.json`。

> **注意**：这里打印的是**模型输出了几个框**，不是正确率。
> 正确率看第 9 步的评估报告。这两件事必须分开说——**模型敢不敢报，和它报得对不对，是两回事。**

### 第 11 步 · 生成「人工标注 vs 模型预测」对照拼图

```bash
python scripts/07_make_demo_figure.py
```

按明确的规则自动挑图（不是手挑好看的）：检测完整 2 张 + 典型漏检 2 张 + 典型误检 2 张，
拼成一张图存到 `results/demo/demo_grid.png`。

- 绿框 = 检出正确；黄框 = 漏检（有标注、模型没报）；红框 = 误检（模型报了、没有标注）
- 挑选规则写在 `results/demo/demo_grid.json` 里，任何人都能复算

这张图**可以直接当简历配图**，也是面试时讲"模型哪里行、哪里不行"最好用的材料。

### 第 12 步 · 视频多目标跟踪

```bash
python scripts/08_track.py --source assets/videos/你录的视频.mp4 --name real_road
```

给同一个目标分配固定 ID，输出带轨迹的视频到 `results/track/<name>/track.mp4`，
并统计轨迹总数、轨迹寿命、"只活了 1 帧的轨迹"占比。

> ⚠️ **本数据集不能用来做跟踪评估。** `datasets/road-traffic` 的 814 张图不是连续帧——
> 实测相邻两张时间戳最小间隔 1.09 秒，普遍 4.8~5.5 秒。它是按固定间隔抽帧的快照集。
> 拿它跑跟踪，目标早就跑出画面了，ID 会疯狂新建，数字没有意义。
>
> 想真正演示跟踪，**用手机录 20~30 秒道路视频**放进 `assets/videos/` 再跑。
> 目录里那个 `_pipeline_check_pan.mp4` 是用单张图合成出来的，只用来验证链路能跑通，
> 见 `assets/videos/README.md`。

---

## 5. 训练

```bash
yolo detect train \
  model=yolo26n.pt \
  data=.\datasets\road-traffic-detect\data.yaml \
  epochs=50 batch=8 imgsz=640 device=0 workers=0 \
  project=.\runs name=road_baseline
```

> 这条命令是按训练产出的 `args.yaml` **还原**出来的。没在命令里写的参数都取了
> Ultralytics 的默认值（比如 `patience=100`、`close_mosaic=10`、`mosaic=1.0`、
> `amp=true`、`pretrained=true`）。**唯一权威是 `args.yaml`**，本 README 只是
> 方便复制。

### 关键参数

| 参数 | 取值 | 说明 |
|---|---|---|
| `model` | `yolo26n.pt` | nano 版预训练权重，从它做迁移学习 |
| `data` | `.\datasets\road-traffic-detect\data.yaml` | 用转换后的数据集 |
| `epochs` | 50 | |
| `batch` | 8 | 受 8 GB 显存限制 |
| `imgsz` | 640 | 输入边长 |
| `device` | 0 | 第 0 号 GPU |
| `workers` | 0 | 见上文说明 |
| `lr0` / `lrf` | 0.01 / 0.01 | 初始学习率 → 末轮降到的比例，即 1e-2 → 1e-4 |
| `warmup_epochs` | 3.0 | 前 3 轮学习率从接近 0 爬升 |
| `close_mosaic` | 10 | 最后 10 轮关闭 Mosaic 增强 |
| `seed` `deterministic` | 0 / true | 固定随机性，便于复现 |

### 这次的训练结果

- **耗时**：660.4 s（约 11 分钟，RTX 5060 Laptop）
- **mAP50**：最高 **0.644**（第 19 轮），末轮 0.637
- **mAP50-95**：最高 **0.416**（第 48 轮），末轮 0.409
- **best.pt 对应第 48 轮**（按 `fitness = 0.1 × mAP50 + 0.9 × mAP50-95` 选出）

以上都是**验证集**上的数字。在**留出的测试集**上做一次性评估的结果是
**mAP50 0.680 / mAP50-95 0.423**，两者不一样是正常的（不同划分、样本不同）。
完整结论见 `results/eval_report/evaluation_report.md`，或按第 9 步自己复现。

产物在 `runs/detect/runs/road_baseline/`：

```
args.yaml results.csv            # 参数快照 + 逐轮指标
weights/best.pt weights/last.pt  # 两份权重（字节数相同但内容不同，见下）
results.png confusion_matrix*.png
train_batch*.jpg val_batch*_pred.jpg val_batch*_labels.jpg
```

> **`runs/detect/runs/road_baseline/` 为什么有两层 `runs`？**
> Ultralytics 的输出目录 = **当前工作目录 + project + name**。
> 那次启动时工作目录已经在 `runs/detect` 里了，于是 `project=.\runs` 又被套了一层。
> 另外 `runs/detect/runs/detect/road_baseline/` 是更早一次失败的尝试留下的半个目录
> （只有 `args.yaml`，`weights/` 是空的）。**2026-10-14 已把它移入回收站**，
> 里面的参数快照先归档到了 `docs/args_failed_run_2026-10-13.yaml`——那份文件是上面
> 这段路径推断的唯一证据，删了就没依据了。
> **教训**：训练前先确认当前工作目录，或者干脆用绝对路径写 `project`。

> **`best.pt` 和 `last.pt` 字节数相同，内容并不相同。**
> 两份都是 5,469,061 字节，但 md5 不同（`974c53ee…` / `e56d3064…`）。
> 训练到第 48 轮时指标最好，之后又跑了 2 轮，权重有变化、大小没变而已。
> 不能只看文件大小就下结论。

---

## 6. 已知的坑

### 6.1 改了标签一定要删缓存

YOLO 第一次读数据后会在标签目录下生成 `train.cache` / `val.cache`。
**改了标签文件却不删这两个缓存，等于没改**——训练读的还是旧缓存。

```bash
rm datasets/road-traffic-detect/train/labels/train.cache
rm datasets/road-traffic-detect/valid/labels/val.cache
```

### 6.2 `path: .` 会按"当前工作目录"解析，不是按 yaml 所在目录

`data.yaml` 里的 `path` 如果是相对的，Ultralytics 会相对**进程当前工作目录**去解析。
从项目根目录启动训练时，它会去找 `项目根/valid/images`，找不到就报
`missing path '...\YOLO_Road_Detection\valid\images'`。

现有的处理方式：`datasets/road-traffic-detect/data.yaml` 里 `path` 写成绝对路径。

注意 `04_convert_labels_to_detection.py` 自动生成的 `data.yaml` 写的仍然是 `path: .`。
如果你重跑过第 4 步，记得把首行改回绝对路径。

### 6.3 别只看文件大小判断权重是否相同

`best.pt` 和 `last.pt` 字节数完全一样，但内容不同。要比较就用 md5 或哈希，
不要用文件大小。

---

## 7. 复现说明

`args.yaml` 记录了训练参数，是复现的重要依据。但**要复现整个实验，光有它不够**，
还需要同时固定：

- **数据版本**：数据集来自 Roboflow v1，且经过第 4 步转换。原始数据或转换脚本变了，结果就不可比。
- **模型权重**：`yolo26n.pt` 的具体版本。
- **软件环境**：Python / PyTorch / CUDA / Ultralytics 版本（见第 2 节）。
- **代码版本**：脚本改动过就要重新记录。
- **随机性设置**：`seed=0`、`deterministic=true`。

---

## 8. 待办

- [x] 人工核对 `empty_label_check.png` —— 已核实：画面里有一辆载两人的摩托车没被标，
      **判定为漏标**（见解析文档 6.8 ③）
- [ ] 抽检数量加大到 20~30 张（当前每个划分 8 张，`--n` 可调），并记录问题总数
- [x] 补齐推理性能测试口径：硬件型号、输入尺寸、预热次数、计时方式、精度模式（FP32/FP16）
      —— 已落成 `scripts/benchmark_inference.py`，实测结果见解析文档 10.3 节
- [x] 清理 `runs/detect/runs/detect/road_baseline/` 这个失败运行留下的空目录
      —— 已移入回收站；里面的 `args.yaml` 先归档到 `docs/args_failed_run_2026-10-13.yaml`
- [x] 删除根目录重复的 `yolo26n.pt` —— 已移入回收站（先确认 md5 与 `weights/` 里那份一致）
- [x] 检查数据泄漏 —— 已写 `scripts/check_data_leakage.py`（md5 + 感知哈希 + 时间戳邻域），
      内容层面 0 重复；但三个划分时间跨度完全重叠，**不能宣称测试集完全独立**
- [ ] 补一条场景级去重的办法（按采集批次分组后再划分），彻底解决上一个问题
      —— **注意：重新划分后现有指标就不能直接对比了**，得重新评估一次并注明「旧划分下的结果」
- [x] 在测试集上做一次正式评估 —— 已跑，**mAP50 0.680 / mAP50-95 0.423**，
      报告在 `results/eval_report/evaluation_report.md`
- [x] 推理与演示脚本 —— `06_predict.py`（图 / 目录 / 视频 / 摄像头）、
      `07_make_demo_figure.py`（对照拼图）、`08_track.py`（视频多目标跟踪）
      均已完成并实跑验证
- [x] 简历与面试材料 —— `docs/RESUME_PROJECT.md`
- [x] 工程基线 —— `requirements.txt`（版本已固定）、`LICENSE`（代码 MIT + 数据集 CC BY 4.0）
- [ ] **录一段 20~30 秒的真实道路视频**放进 `assets/videos/`，跑一次真正的跟踪演示
      （现在目录里那个是合成片段，只能验证链路能跑通，不能当效果演示）
- [ ] 阶段四：ONNX 导出与 TensorRT 推理优化 —— **本次暂缓**，是下一步计划

详细的代码逐行解析、面试问答、项目原理说明见解析文档（**不在本仓库内**，2026-10-14 起统一放在）：

```
D:\工程\方向与规划\项目\YOLO_Road_Detection\
├── YOLO_Road_Detection_代码解析.html    # 主文档（自带图表，单文件可离线打开）
└── YOLO_Road_Detection_代码解析.pdf     # 同内容 PDF，便于打印或发给别人
```
