# YOLO 道路目标检测项目

**在线演示页**：https://dourenchi.github.io/YOLO-road-detection/
**项目说明书**（零基础版，点开即读）：https://dourenchi.github.io/YOLO-road-detection/manual.html

不用下载仓库，点开链接就能看检测效果、逐类评估结果和完整的项目讲解。

用 Ultralytics YOLO26n 在 road-traffic 数据集上做多类别目标检测的完整流程：
**数据检查 → 标注格式统一 → 数据验收 → 训练 → 评估 → 推理演示 → 视频跟踪**

这个项目的目的不只是"把模型训出来"，而是把一条**可复现、可追溯、别人能接手**的
数据处理链路走通。每一步都有对应脚本，每个脚本都会打印可核对的数字；
所有结论性的数字都会**连同它的测量条件**一起记录下来。

---

## 1. 目录结构

```
YOLO_Road_Detection/
├── scripts/                          # 全部脚本（共 23 个，可在任意目录下运行）
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
│   │
│   │   # ↓ 以下 8 个属于部署阶段，详见第 7 节
│   ├── 10_export_onnx.py             # 导出 ONNX，并留痕（环境快照 / sha256 / 读回结构）
│   ├── 11_verify_onnx.py             # ONNX 三层验证：结构 / 数值 / 结果
│   ├── 12_benchmark_pt_vs_onnx.py    # PyTorch 与 ONNX Runtime 同条件性能对比
│   ├── 13_compare_cpp_backends.py    # C++ 两种图像后端对比（手写双线性 vs OpenCV）
│   ├── 14_export_engine.py           # 构建 TensorRT FP32 / FP16 引擎
│   ├── 15_verify_engine.py           # 引擎四层验证（结构/数值/结果/精度）+ 性能
│   ├── 16_eval_valid_split.py        # valid 上对比五种格式精度（含 INT8，刻意不碰 test）
│   ├── 17_make_test_video.py         # 合成测试视频（仅验证链路，不用于报延迟）
│   ├── 18_export_engine_int8.py      # 构建 INT8 引擎（PTQ，校准集用 train）
│   └── 19_verify_int8.py             # INT8 四层验证 + 量化覆盖率 + 延迟对照
├── datasets/
│   ├── road-traffic/                 # 原始数据集（Roboflow 导出）
│   │   ├── train/ valid/ test/       # 各自含 images/ 与 labels/
│   │   ├── data.yaml                 # 源头配置：names / nc / path
│   │   └── README*.txt               # 来源与许可说明
│   └── road-traffic-detect/          # 04 号脚本生成的检测框数据集
│       ├── train/ valid/ test/
│       └── data.yaml                 # 训练实际使用的配置
├── cpp/                              # C++ 推理工程（Windows + MSYS2，见 cpp/README.md）
│   ├── src/main.cpp                  # 三种模式：单图 / 端到端（--repeat）/ 视频流（--video）
│   ├── experiments/trt_abi_probe.cpp # ABI 探测：证明 MinGW 不能直接调用 TensorRT 虚函数
│   ├── collect_deps.py               # 递归读导入表收依赖 + sha256 体检 + --gpu 选项
│   ├── third_party/                  # ONNX Runtime C/C++ 头文件 + stb
│   ├── build/                        # 开发态构建产物（.gitignore）
│   └── dist/                         # 分发包 143 文件 / 212.9 MB（.gitignore）
├── ros2/                             # ROS2 封装（WSL2 + Humble，已实测；实录见 docs/ros2/）
│   └── src/yolo_trt_ros/             # 节点 + 自定义消息 + launch + 单元测试 + fake_camera
├── docs/                             # 各类记录
│   ├── architecture.md               # 架构：模块职责与调用关系
│   ├── workflow.md                   # 端到端流程：输入到输出
│   ├── code_notes.md                 # 关键函数、参数、数据结构
│   ├── debug_log.md                  # 问题、原因、解法、验证结果
│   ├── deployment/                   # 部署阶段：实施记录 + 12 份实测 JSON + 两份分包清单
│   ├── label_conversion_report.txt   # 04 号脚本生成的转换报告（审计日志）
│   ├── inference_benchmark.json      # 性能测试原始统计量 + 完整测量条件
│   ├── data_leakage_report.json      # 数据泄漏检查的完整结果
│   └── args_failed_run_2026-10-13.yaml    # 那次失败运行的参数快照（证据）
├── runs/detect/runs/road_baseline/   # 正式训练产物（曲线、混淆矩阵、权重）
├── results/
│   ├── label_check/                  # 人工标注抽检图
│   ├── predict/                      # 06 号脚本的推理结果（带框的图 / 视频）
│   ├── demo/demo_grid.png            # 07 号脚本的对照拼图（三档示例）
│   ├── track/                        # 08 号脚本的跟踪视频与轨迹统计
│   └── eval_report/                  # 09 号脚本的评估报告与图表
├── assets/                           # 演示素材（videos/ 下是稳定性测试用的真实视频，不入库）
├── site/                             # GitHub Pages 站点（会被发布到公网）
│   ├── index.html                    # 在线演示页
│   ├── manual.html                   # 项目说明书（零基础版，单文件自包含）
│   └── media/                        # 站点用的图与视频（演示拼图、评估图表、跟踪视频）
├── weights/
│   ├── yolo26n.pt                    # 预训练权重
│   ├── best.onnx                     # 导出产物 9.4 MB，部署链路的起点
│   ├── best_fp32.engine              # TensorRT FP32 17.9 MB（绑本机显卡与 TRT 版本）
│   ├── best_fp16.engine              # TensorRT FP16 11.8 MB（同上，不能跨机器拷）
│   ├── best_int8.engine              # TensorRT INT8 11.8 MB（PTQ；实测不推荐，见 7.2.1）
│   └── trt_cache/                    # ORT TensorRT EP 的引擎缓存（.gitignore）
├── .github/workflows/pages.yml       # 站点自动部署（push 后自动更新在线版）
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

这张图把"哪里行、哪里不行"放在同一张画布上，比只看数字直观得多。

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

> **部署阶段另有 8 个脚本（10–17 号）**，用途和运行顺序见第 7 节。
> 它们不属于「训练与评估」这条线，所以没排在上面的 1–12 步里。

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

## 7. 部署：从 best.pt 到能交付的程序

训练和评估回答"模型准不准"，部署回答"它在系统里能不能用"。这条链路有四个独立的验收点，
**不能跳着做**——因为"没有框"这一个症状，可能是通信配错、格式转换错、后端没生效、或者模型在这个场景里真检测不到。

| 段落 | 产出 | 验收标准 | 详细记录 |
|---|---|---|---|
| ONNX 导出 | `weights/best.onnx`（9.4 MB） | 结构 / 数值 / 结果三层验证 | `docs/deployment/` |
| TensorRT | `best_fp32.engine` / `best_fp16.engine` | 四层验证 + 精度对照 | 同上 |
| C++ 工程 | `cpp/`（单图 / 端到端 / 视频流） | 与 Python 侧结果一致 + 延迟 | 同上 |
| **INT8 量化** | `weights/best_int8.engine`（11.8 MB） | 四层验证 + 覆盖率 + 五格式精度 | 同上（结论见 7.2.1） |
| ROS2 封装 | `ros2/src/yolo_trt_ros` | **WSL2 实测通过**（A1/A2/A3/A5；A4 缺 nvinfer） | `docs/ros2/README.md` |

每一步的逐步操作、每个数字怎么读、踩过的 27 个坑，都在 **`docs/deployment/README.md`**。
下面是结论。

### 7.1 四档性能：同一个模型，差 11 倍

同一张图、预热 30 次、n=200、每次前后同步：

| 路径 | 纯推理中位 | 相对 ONNX |
|---|---|---|
| PyTorch（框架计时） | 12.65 ms | — |
| ONNX Runtime FP32 | 8.19 ms | 1.00× |
| TensorRT FP32 引擎 | 2.88 ms | 2.84× |
| TensorRT FP16 引擎 | **1.97 ms** | **4.16×** |
| C++ + ORT CUDA EP | 6.23 ms | 1.31× |
| C++ + ORT TensorRT EP | **3.22 ms** | 2.54× |

**跨表比较绝对值没有意义**：ONNX 这一项在不同时段测到过 7.10 和 8.19，差的 15% 是机器状态
（GPU 频率、后台负载），不是代码。**只比同一次测出来的相对值。**

### 7.2 精度：两个落差必须分开算

在 **valid** 上比（test 只评过一次，全程不碰）：

| 格式 | mAP50 | mAP50-95 |
|---|---|---|
| best.pt | 0.6407 | 0.4169 |
| best.onnx | 0.6329 | 0.4111 |
| engine FP32 | 0.6327 | 0.4111 |
| engine FP16 | **0.6323** | 0.4103 |
| engine INT8 | 0.6048 | 0.3935 |

- **pt → ONNX 掉 0.0078**：这部分**不该全算作 ONNX 的精度损失**。pt 走的是训练时那条原生前向 + NMS，
  onnx/engine 走的是另一条解码 + 后处理路径——这是「换了一条推理路径」，不是「算错了」。
- **FP32 → FP16 只掉 0.0004（−0.06%）**：这才是 FP16 的净代价，小到可以忽略。

**所以判断 FP16 能不能用，基准必须是「同路径的 FP32 引擎」，不能拿 pt 的 0.641。**
拿 0.641 当基准，会把格式转换的代价误算到 FP16 头上，得出「FP16 不能用」的错误结论。

### 7.2.1 INT8：测过了，结论是不值得

| 指标 | FP32 | FP16 | INT8 |
|---|---|---|---|
| mAP50（valid） | 0.6327 | 0.6323 | **0.6048（−4.41%）** |
| mAP50-95 | 0.4111 | 0.4103 | 0.3935 |
| 精确率 P / 召回率 R | 0.6255 / 0.6020 | 0.6249 / 0.5925 | 0.5673 / 0.5840 |
| 纯推理 p50（两次运行） | 2.61–2.70 ms | 1.86–1.94 ms | **1.83–1.91 ms（只快 1.5%–3%）** |
| 卷积层 INT8 覆盖 | — | — | 83/102（81%） |

**结论：掉 4.41% 的 mAP，换 1.5%–3% 的延迟，本机不值得。** 三个必须说清的点：

0. **延迟差小于测量波动**（两次运行各自差 1.5%–3%，而运行间本身波动 ±5%），
   所以只能说"看不出加速"；而 mAP 的 −4.41% 是可重复的。这两件事的可信度不一样。

1. **量化确实生效了**（81% 卷积是 INT8），所以"不快"不能赖"没量化下去"。
   但引擎里仍有 41 个 Float / 50 个 Half 张量，层间精度切换本身有开销。
2. **失效方式是弥散的，不是某一类崩掉**：buses −0.120、bicycles −0.069、vehicles 只 −0.008。
   如果某一类掉到 0，才说明校准范围被压坏；现在是量化噪声的正常表现。
3. **P 掉 5.8 点、R 只掉 1.8 点**：主要是置信度分布被压低，不是大面积漏检。

原因**推测**（未验证）：2.39 M 参数的小模型在桌面 GPU 上是访存/调度受限，不是算力受限，
INT8 的算力优势用不上。要验证只有两条路——换更大的模型，或换 Jetson 这类算力受限设备。

⚠️ 上表的延迟是**另一次测量**（与 7.1 的四档不同场次），跨场次不比绝对值，只比同场内的相对关系。
校准集用的是 **train**（不是 valid）：用 valid 校准会让 mAP 偏乐观。

### 7.3 端到端：推理只占 38%

`--repeat` 模式跑完整链路（每轮重新读图），拆五段：

| 配置 | 纯推理 | **端到端** | 读图 | 预处理 | 推理 | 后处理 |
|---|---|---|---|---|---|---|
| TRT + OpenCV | 2.76 ms | **7.94 ms** | 1.48 | 2.61 | 3.02 | 0.065 |
| TRT + 手写双线性 | 2.72 ms | 11.74 ms | 1.73 | 6.26 | 2.99 | 0.064 |
| CUDA + OpenCV | 6.55 ms | 13.31 ms | 1.56 | 2.40 | 8.76 | 0.071 |

三条结论：

1. **端到端是纯推理的 2.9 倍。** 只报 2.76 ms，被问一句"含不含读图"就答不上来。
2. **推理只占 38%，预处理 33%，读图 19%。** 后处理才 0.8%——直觉上以为 NMS 很贵，实测不是。
3. **阿姆达尔定律的实例**：CUDA→TRT 纯推理快 2.37 倍，端到端只快 1.68 倍，因为另外 62% 没变。

另外换图像后端比换推理引擎更划算：手写双线性 → `cv::resize`，预处理 6.26 → 2.61 ms，一次省 3.80 ms。

### 7.4 视频流与 1000 帧稳定性

用两段真实公开视频（Wikimedia Commons，CC0 / CC BY 3.0），各跑两轮 1000 帧：

| 输入 | 吞吐 | 中位 | p99 | 预处理 |
|---|---|---|---|---|
| 1280×720 | 100.5–101.6 FPS | 9.82–9.90 ms | 13.25–14.11 ms | 5.16–5.48 ms |
| 1920×1080 | 72.5–75.4 FPS | 13.15–13.73 ms | 17.63–18.08 ms | 7.87–8.36 ms |

- **p99 比中位高 22%–34%**。按 10 ms 设计控制回路的话，100 帧里就有 1 帧超时。
- **内存与显存都不涨**：RSS 1000 帧净增 4–6 MB、显存 `1691 → 1691`（净增 0），两三次采样内持平。
- **分辨率是主要成本项**：像素涨 8 倍（400×640 → 1920×1080），预处理涨约 3 倍，推理几乎不变。
- 合成视频（图片拼接）会把 p99 抬高 3–4 ms，**报延迟必须用真实视频**，合成只用来验链路。

显存是运行时动态加载 `cudaMemGetInfo` 采的。**不能用 `nvidia-smi` 前后各读一次求差值**——
单次调用 101–107 ms（比一帧推理大 30–50 倍），而且程序退出时显存全释放，
实测前后差 **−9 MB**，会得出"模型几乎不占显存"的错误结论。

### 7.5 交付：什么能交，什么不该交

| 形态 | 体积 | 状态 |
|---|---|---|
| **CPU 自包含包**（`cpp/dist/`） | 143 文件 / 212.9 MB | ✅ 干净 PATH 下验证通过（回落 CPU，19 ms） |
| GPU 自包含包 | 163 文件 / 3.0 GB | ❌ 补齐全部依赖后仍 `0xC0000005`，已放弃 |
| **轻包 + 目标机装运行时** | ≈ 40 MB | ✅ 推荐用于 GPU / TRT 场景 |

**为什么不打包 GPU 运行时**：`nvinfer` 418 + `cublasLt` 643 + cuDNN 全套 904 + `cufft` 263 +
`providers_cuda` 307 ≈ 3 GB，而且版本必须严丝合缝——缺 `nvrtc` 时 TensorRT 会**静默回落**
（你还以为在用），cudnn 子库少一个就直接崩、报错还不提"缺文件"。

打包用 `cpp/collect_deps.py`（递归读导入表 + 生成 sha256 MANIFEST + `--check` 体检），
**不要手工拷 DLL**——本项目实测手工拷漏过两次，其中一次是包里的 exe 是旧版，
跑 `--ep trt` 静默走了 CUDA EP（5.25 ms 而非 2.35 ms），完全不报错。

### 7.6 ROS2：已在 WSL2 跑通四关

`ros2/src/yolo_trt_ros` 订阅图像话题、发布结构化检测结果与分段延迟，
`depth=1` 背压、双时钟分离、启动自检抓静默回落——这些机制都已实测。

2026-11-24 在 WSL2 + Humble 上完成分阶段验收：

| 关卡 | 状态 | infer p50 |
|---|---|---|
| A1 dummy | ✅ | 0 ms |
| A2 CPU | ✅ | 39.0 ms |
| A3 CUDA | ✅ | **7.1 ms（5.5×）** |
| A4 TensorRT | 根因已定位，依赖未装 | 回落 CUDA |
| A5 worker 对比 | ✅ 负结果（无收益） | 37.7 vs 39.0 ms |

编译阶段修了 6 个坑（CMake 签名混用、`project()` 缺 C、`find_path` 不递归、
`printf` 类型、头文件 namespace 位置等），运行阶段修了 4 个（SONAME、numpy 2.x、
ORT 字符串 API 过期、CUDA 运行库缺失）。
**完整实录、原始日志与取舍说明见 `docs/ros2/README.md`**，配套教程《YOLO_ROS2部署教程》。

### 7.7 报数字的三条纪律

1. **一律带条件**：哪个划分（test / valid）、哪台机器、batch 多少、含不含后处理。
2. **跨时点不比绝对值**：只在同一次测量里比相对值。
3. **精度基准用同路径**：判断 FP16 的好坏，跟同路径 FP32 比，不跟 pt 比。

---

## 8. 复现说明

`args.yaml` 记录了训练参数，是复现的重要依据。但**要复现整个实验，光有它不够**，
还需要同时固定：

- **数据版本**：数据集来自 Roboflow v1，且经过第 4 步转换。原始数据或转换脚本变了，结果就不可比。
- **模型权重**：`yolo26n.pt` 的具体版本。
- **软件环境**：Python / PyTorch / CUDA / Ultralytics 版本（见第 2 节）。
- **代码版本**：脚本改动过就要重新记录。
- **随机性设置**：`seed=0`、`deterministic=true`。

---

## 9. 待办

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
- [x] 工程基线 —— `requirements.txt`（版本已固定）、`LICENSE`（代码 MIT + 数据集 CC BY 4.0）
- [ ] **录一段 20~30 秒的真实道路视频**放进 `assets/videos/`，跑一次真正的跟踪演示
      —— 稳定性测试已用两段 Wikimedia 的 CC0 / CC BY 视频完成（见 7.4 节），
      但那两段是为了测延迟，**帧间连续不等于适合做跟踪效果演示**；跟踪演示仍在等一段自己拍的
- [x] 阶段四：ONNX 导出 → TensorRT → C++ 部署 —— **已完成**，详见第 7 节。
      四档性能对比、精度对照、端到端五段拆解、1000 帧稳定性、CPU 分发包都有实测数据
- [x] **ROS2 包在 WSL2 里编译跑通** —— A1/A2/A3/A5 已实测通过，A4 TRT 定位到缺
      `libnvinfer.so.10` 后按取舍停止（Windows 侧 TRT 已有完整实测）。见 `docs/ros2/`
- [ ] **GPU 轻包**（`≈40 MB`：exe + onnx + ORT/OpenCV 少数 DLL）——
      3 GB 的自包含方案实测崩溃已放弃，轻包本身还没打出来
- [ ] **INT8 校准** —— 需要校准集（只能用 valid），且要接受「某一类突然全没了」的风险，
      端到端收益约 1 ms，性价比不高，暂缓

## 10. 相关文档

**在线版**（不用下载仓库，点开就能看）：

- 演示页 —— https://dourenchi.github.io/YOLO-road-detection/
- 项目说明书（零基础版）—— https://dourenchi.github.io/YOLO-road-detection/manual.html

这两份由 `site/` 目录发布；push 到 `main` 之后由 `.github/workflows/pages.yml` 自动更新，
所以在线版永远和仓库里的内容一致，不需要手动同步。

> **站点部署（只需做一次）**：仓库启用 Pages 这一步只能人工完成，工作流自己没有这个权限。
> 到 Settings → Pages，把 **Source** 选成 **GitHub Actions**；之后每次改动 `site/` 都会自动发布。
> 在启用之前，工作流会跑通但给出一条 `Pages 尚未启用` 的提示，不会报错。
>
> 站点结构：`site/index.html`（演示页）、`site/manual.html`（说明书）、`site/media/`（图表与视频）。
> 本地想预览，直接在浏览器里打开 `site/index.html` 即可——全部用相对路径，不需要起服务器。

**《完整手册》**（本地，不在本仓库内）——由原来的《项目说明书》和《代码解析》两份整合而成，
覆盖从数据、训练、评估到部署的完整链路，另含代码逐行解读与整条链路的复盘总结。

体积较大（图表以 base64 内嵌，单文件自包含），因此不随仓库分发，单独维护。
仓库内在线的说明文档是 `site/manual.html`，内容侧重项目本身与各个数字的来历，
和本地这份手册是两份不同的文档，改动时注意区分。

