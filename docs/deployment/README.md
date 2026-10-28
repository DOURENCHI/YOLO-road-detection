# 模型部署执行记录 · 阶段 4（第 1 步：ONNX）

> 这份文件是**执行留痕**，不是教程。教程（原理、概念、坑的解释）在
> `D:\工程\方向与规划\项目\YOLO_Road_Detection\YOLO_模型部署教学文档.html`。
> 这里只记：**做了什么、为什么这么做、得到了什么、怎么复现**。

- 执行日期：2026-10-27
- 执行人：AI 代操作（命令与本文件同源可查）
- 机器：RTX 5060 Laptop（8150 MiB，cc 12.0 / sm_120）+ Windows
- 权重：`runs/detect/runs/road_baseline/weights/best.pt`（epoch 48，sha256 `a5cdefaa39d43fa4…`）

---

## 0. 一句话结论

`best.pt` 已导出为 `weights/best.onnx`，**与 PyTorch 输出等价**（逐框完全一致），
**纯推理快 39%**（11.70 → 7.10 ms），端到端快 15%（14.96 → 12.71 ms）。
过程中踩了 4 个坑，其中 2 个会导致「静默变慢但结果正确」，已写进第 5 节。

---

## 1. 本次做了什么

| # | 步骤 | 产出 |
|---|---|---|
| 1 | 装依赖（onnx / onnxslim / onnxruntime-gpu） | 环境可用 |
| 2 | 导出 ONNX | `weights/best.onnx` + `docs/deployment/onnx_export_report.json` |
| 3 | 一致性与等价性验证（三层） | `docs/deployment/onnx_verify_report.json` |
| 4 | PyTorch vs ONNX 同条件性能对比 | `docs/deployment/benchmark_pt_vs_onnx.json` |
| 5 | 脚本入库（可重跑） | `scripts/10_export_onnx.py`、`11_verify_onnx.py`、`12_benchmark_pt_vs_onnx.py` |

**没做的事**：TensorRT、C++、INT8、test 集评估（test 只评一次，见第 6 节）。

---

## 2. 环境变更记录（含一次返工）

| 时间 | 动作 | 结果 |
|---|---|---|
| 09:33 | 检查镜像：阿里云 0.58s / 中科大 2.4s / 清华 403（curl 被拦） | 选阿里云 `mirrors.aliyun.com/pypi/simple` |
| 09:37 | `pip install onnx onnxslim onnxruntime-gpu` | onnx 1.23.2 / onnxslim 0.1.98 本来就有；**onnxruntime-gpu 1.31.0**（168 MB，2:51） |
| 09:39 | 首次验证 | ❌ **CUDA EP 加载失败，静默回落 CPU**（见坑 1） |
| 09:44 | 降级到 `onnxruntime-gpu==1.22.0` | ✅ CUDA EP 可用 |

最终生效版本：

```
python 3.11.16 · torch 2.11.0+cu128 · ultralytics 8.4.176
onnx 1.23.2 · onnxruntime-gpu 1.22.0 · onnxslim 0.1.98
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## 3. 步骤一：导出

**做了什么**

```bash
python scripts/10_export_onnx.py
# 等价于：yolo export model=…/best.pt format=onnx imgsz=640 simplify=True dynamic=False
```

**为什么这样做**

- `imgsz=640`：必须与评估时一致，否则 mAP 不可比。
- `dynamic=False`：先用固定形状跑通；动态 shape 会牵出 TensorRT 的 profile 配置，属于下一步的事。
- `simplify=True`：让 onnxslim 合并冗余算子，减少节点数（对 ORT/TRT 都有利）。
- 单独写脚本而不是敲一行命令：为了把**导出参数 + 环境版本 + 产物 sha256** 一起存档，否则三天后无法判断手里这个 `.onnx` 是哪来的。

**结果**

| 项 | 值 |
|---|---|
| 产物 | `weights/best.onnx` · 9.41 MB · sha256 `c676ea9adfc00293…` |
| 导出耗时 | 3.6 s |
| opset | ai.onnx **18** |
| 输入 | `images` FLOAT `[1, 3, 640, 640]` |
| 输出 | `output0` FLOAT **`[1, 16, 8400]`** |
| 图规模 | 368 节点 / 223 个 initializer |
| 参数（fused） | 2,392,408 |

**关于 `[1, 16, 8400]` 这个形状（重要）**

- `8400` = 80² + 40² + 20²，三个尺度（stride 8/16/32）的网格点总数，说明这是**标准 anchor-free 输出**。
- `16` = 4（cx, cy, w, h）+ 12（类别数），**没有 objectness 通道**，也**没有把 NMS 编进图里**。
- 结论：后处理仍需自己做「解码 + 置信度过滤 + NMS」。不要照抄网上 `[1, 84, 8400]`（那是 80 类的 COCO），
  也不要假设它已经过端到端 NMS。

---

## 4. 步骤二：一致性与等价性验证

**做了什么**（`python scripts/11_verify_onnx.py`）

三层，顺序不能换：

1. **结构层**：读 `.onnx` 的 opset / 输入输出形状 / 节点数，与导出报告对上。
2. **数值层**：固定随机输入（seed=0），torch 与 ORT 各跑一次，比原始输出张量。
3. **结果层**：valid 前 16 张真实图片，两边都用 Ultralytics 同参数 `predict`（conf=0.25, iou=0.7），比框数 / 类别 / 分数。

**为什么这么做**

- 只比「能跑通 / 能出框」不够：框偏移 3 像素、分数低 0.05、某一类被换列，肉眼和图都看不出来。
- 只比张量也不够：原始输出未过 sigmoid，个别单元的差值会被放大，判断标准不该定在一个固定的绝对阈值上。
  所以**判定依据放在结果层**，张量层只作为诊断。

**结果**

| 层 | 指标 | 值 | 判定 |
|---|---|---|---|
| 结构 | 输入/输出 | `[1,3,640,640]` → `[1,16,8400]` | 与报告一致 |
| 数值 | 最大绝对差 | 0.907（输出量级 mean\|y\| = 51.1，即 0–640 像素坐标） | 诊断项 |
| 数值 | p99.9 / 平均 / **相对误差** | 0.192 / 5.47e-3 / **1.07e-4** | 诊断项 |
| 结果 | 框数（16 张） | pt 70 · onnx 70 | ✅ 差 0.00% |
| 结果 | 逐框类别 | 全对 | ✅ |
| 结果 | 最大分数差 | 1.00e-3 | ✅ ≤1e-2 |

**结论：通过。** ONNX 与 PyTorch 在这个任务上是同一个模型。

---

## 5. 步骤三：性能对比

**做了什么**（`python scripts/12_benchmark_pt_vs_onnx.py`）

同条件：同一张图、预热 30 次、重复 200 次、每次前后 `cuda.synchronize()`、机器空闲。

**结果（单位 ms）**

| 项目 | 中位数 | p95 | 均值 | std |
|---|---|---|---|---|
| PyTorch 纯前向 | 11.70 | 17.46 | 12.25 | 3.26 |
| PyTorch 端到端 | 14.96 | 23.19 | 16.11 | 3.63 |
| **ORT 纯推理** | **7.10** | 11.08 | 7.75 | 1.82 |
| **ORT 端到端** | **12.71** | 16.43 | 12.92 | 2.23 |

| 口径 | ONNX / PyTorch | 含义 |
|---|---|---|
| 纯推理 | **0.607** | ONNX 快 **39%** |
| 端到端 | **0.850** | ONNX 快 **15%** |

**两个值得记住的观察**

1. **端到端收益远小于纯推理收益**（0.85 vs 0.61）。因为预处理 + 后处理约占 2.5 ms，
   而且 ONNX 的收益只发生在「算」的那一段。这直接说明：**下一步的瓶颈不在模型，在前后处理**。
2. **框架计时与实际调用差 5.5 ms**：走 Ultralytics 的 `YOLO(onnx).predict()` 时，
   它自己记录的 `inference` 是 12.65 ms，而直接 `session.run()` 只要 7.10 ms。
   差的这 5.5 ms 是封装层开销（输入张量构造、结果包装）。
   → 到 C++ 里直接调 session，这部分就没有了，**这是 C++ 部署的真正收益之一**。

**与旧数据的口径差异（必须说明）**

`docs/inference_benchmark.json` 里的「纯前向 15.5 ms」是 `torch.cuda.Event` 卡在 GPU 上测的；
本脚本的 11.70 ms 是 `perf_counter` 包住一次调用（中位数）。
**两者不可直接相减**，只能在同一脚本内比较。这也是为什么对比必须同条件。

---

## 6. 步骤四：四个真实踩到的坑

### 坑 1（最隐蔽）：ORT 1.31 是 CUDA 13 构建，装了但用不上 GPU

现象：`onnxruntime-gpu 1.31.0` 装成功，`get_available_providers()` 也列出了 `CUDAExecutionProvider`，
但创建 Session 时：

```
Error loading "onnxruntime_providers_cuda.dll" which depends on "cublasLt64_13.dll" which is missing.
Require cuDNN 9.* and CUDA 13.*
WARNING CUDA requested but CUDAExecutionProvider not available. Using CPU...
```

原因：ORT ≥1.23 的 Windows 轮子按 **CUDA 13** 编译；而本机 torch 是 **cu128**。
**程序不报错、结果也对，只是慢 10 倍**——最容易被误判成「ONNX 就是慢」。

处理：降到 `onnxruntime-gpu==1.22.0`（CUDA 12 线）。

### 坑 2：ORT 找不到 cuBLAS / cuDNN

即使版本对了，`cublasLt64_12.dll`、`cudnn_*.dll` 也不在系统 PATH 上——它们躺在 torch 的 `lib` 目录里。
处理：创建 Session 前注册目录：

```python
import os, torch
from pathlib import Path
os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
```

脚本 `11`/`12` 里的 `register_cuda_dlls()` 就是这件事，并把实际生效的 EP 写进报告。

### 坑 3：拿「未 fuse」的模型去比「已 fuse」的导出图

第一次数值层算出最大绝对差 **4.92**，看着像导出错了。
实际原因：导出时 Ultralytics 会 fold BN（日志里的 `summary (fused)`），而我拿 `.pt` 未 fuse 的模型去比，
差异里混进了 BN 折叠的舍入。fuse 之后再比，最大差降到 **0.907**。

处理：脚本里显式 `model.fuse()`，并在注释里写明原因。

### 坑 4：绝对误差看起来很大，是因为坐标是像素量级

输出张量前 4 列是 0–640 的框坐标，`mean|y| = 51.1`。所以 0.9 的绝对差其实是 **1.07e-4 的相对误差**。
处理：报告里同时给 `相对误差 = mean(|d|)/mean(|y|)`，并说明判定以结果层为准。

> 附带的小坑：`weights/` 下只有 `yolo26n.pt`，`best.pt` 在 `runs/...` 里；
> 脚本默认路径一开始写错，跑起来是 `FileNotFoundError`。

---

## 7. 产物与复现

```
weights/best.onnx                              9.41 MB   ONNX 模型
docs/deployment/onnx_export_report.json         导出参数+环境+sha256+结构
docs/deployment/onnx_verify_report.json         三层验证结果，逐图明细
docs/deployment/benchmark_pt_vs_onnx.json       同条件性能对比，含统计量
scripts/10_export_onnx.py                       导出
scripts/11_verify_onnx.py                       三层验证
scripts/12_benchmark_pt_vs_onnx.py              pt vs onnx 性能
```

一键复现：

```bash
conda activate yolo_road
python scripts/10_export_onnx.py
python scripts/11_verify_onnx.py
python scripts/12_benchmark_pt_vs_onnx.py
```

---

## 8. 步骤五：C++ 最小闭环（2026-10-27 新增）

**做了什么**（代码在 `cpp/`）

把 Python 里由 `ultralytics` 代劳的四件事自己实现一遍：预处理（letterbox）、`Run` 推理、解码 `[1,16,8400]`、
按类 NMS + 逆变换。三种运行模式：

| 模式 | 用途 |
|---|---|
| `--image x.jpg` | 完整链路（含预处理），出框并画图 |
| `--input-bin x.bin` | 吃 float32 NCHW 张量，**跳过预处理**，专门验证「推理+后处理」 |
| `--json out.json` | 结果落盘，便于与 Python 自动比对 |

**关键取舍：不引 OpenCV。** 读图写图用 stb_image 单头文件，双线性缩放自己写 30 行。
理由是 C++ 部署真正的成本在「预处理一致性」，用库会把这件事藏起来；等对齐之后再换库也不会错。

**ORT 的取法**：不下 200 MB 官方 C++ 包。只从 `raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/include/`
取 6 个头文件，链接直接指向 pip 装好的 `onnxruntime.dll`（MinGW 的 ld 能读 DLL 导出表）。

**结果：与 Python 逐框一致**

| 项 | 值 |
|---|---|
| 检出框数 | Python 6 · C++ 6（6/6 全部配对） |
| 最小 IoU（配对框） | **0.995** |
| 最大分数差 | 0.0051 |
| 类别 | 全对 |
| 坐标最大差 | **0.07 px** |
| C++ 纯推理（GPU） | **6.15 ms**（中位，n=200，p95 7.14） |
| C++ 纯推理（CPU 回落） | 17.29 ms |

**三个口径放一起看，这才是 C++ 的收益**

| 路径 | 纯推理中位数 |
|---|---|
| C++ 直调 `Session::Run` | **6.15 ms** |
| Python 直调 `session.run()` | 7.10 ms |
| Python 走 `YOLO(onnx).predict()` | 12.65 ms（框架自记） |

从 12.65 → 6.15，省下的 6.5 ms **不是模型变快了，是封装消失了**。

**这一步踩的四个坑**

1. **只拷 `onnxruntime.dll`**：CUDA EP 因缺 `onnxruntime_providers_shared.dll` 静默回落 CPU（17.3 ms vs 6.15 ms）。
   现在 CMake 里用 `file(GLOB ...)` 把 `onnxruntime*.dll` 全部拷到 exe 旁边。
2. **有了 shared 还缺 `cublasLt64_12.dll`**：CUDA 运行库不在 PATH。C++ 程序没有 torch 帮忙，
   运行时要么把 CUDA runtime 一起随包发，要么把 `torch/lib` 加进 PATH（本次用后者验证）。
3. **头文件路径**：从源码树取的头文件比官方包多一层 `onnxruntime/core/session`，CMake 的 include 要写全。
4. **数据集全是 640×640**：在这份数据上 letterbox 是空操作，缩放分支根本没被执行到。
   必须自己裁一张 400×640 的图（`docs/deployment/_test_odd.jpg`）才能真正验证 pad 与逆变换
   ——这也是 0.07 px 那点差异唯一可能出现的场景。

**这一层为什么重要（一句话总结）**

> 「C++ 里我没有用 OpenCV，letterbox 的 gain 和 pad 是我自己算并传给后处理的。
> 验证方法是把 Python 预处理好的张量直接喂给 C++，先分离出推理层的差异；再单独喂原图，验证预处理。
> 最后 6 个框全部配上，最小 IoU 0.995，坐标最大差 0.07 像素。」

这句话里同时有：动手能力、分层验证思路、可量化结果。

---

## 8.5 步骤六：接入 OpenCV（2026-10-27 同日）

**做了什么**：把 C++ 的「读图 / 缩放 / 写图」做成可切换后端，编译期自动探测、运行时可覆盖。

| 后端 | 实现 | 何时用 |
|---|---|---|
| `stb` | `stbi_load` + 手写双线性 + `stbi_write_png` | 零依赖，永远可用 |
| `opencv` | `cv::imread` + `cv::resize(INTER_LINEAR)` + `cv::imwrite` | 装上 OpenCV 后默认启用 |

画框和解码**仍然自己写**：画框要让两个后端的输出图逐像素可比，解码是模型语义，不该被库改变。

### 装 OpenCV 的真实代价（实测，不是估计）

| 项 | 数 |
|---|---|
| `pacman` 连带安装包数 | **133 个**（ffmpeg / tesseract / ogre3d / hdf5 / ceres …） |
| 安装后占盘 | **1966 MiB** |
| exe 直接依赖的 OpenCV DLL | 3 个（core 6.7 MB / imgproc 12.0 MB / imgcodecs 0.8 MB） |
| 递归后需要随包发的 DLL | **44 个，78.7 MB**（多出来的是 libaom 9.2 MB、libSvtAv1Enc 7.2 MB、libjxl 5.9 MB、libOpenEXR、libjpeg、libtiff、libwebp、libavif…） |
| 装之前 exe 的依赖 | 8 个 DLL |

`imgcodecs` 支持多少图像格式，就要带多少编解码器。这就是「引一个库」的账单。

### 三条硬约束（都踩过）

1. **必须与编译器 ABI 一致**。OpenCV 的 C++ API 大量传 `std::vector` / `std::string`，布局由编译器决定。
   把 MSVC 编的官方 OpenCV 包用 `dlltool` 转 `.a` 给 MinGW 用，**链接能过、运行会崩**。
   要么整个项目用 MSVC（配 vcpkg），要么用 MSYS2 这份同一套 GCC 构建的 OpenCV。
2. **MSYS2 不提供 `.a` 静态库**，只有 `.dll.a` 导入库，所以没法全静态链接，只能随包发 DLL。
3. **别用 `-static-libstdc++` 偷懒**。静态链接后 exe 会额外依赖 UCRT 的
   `api-ms-win-crt-utility-l1-1-0.dll`，在干净 PATH 下反而起不来（exit 127）。
   正确做法是把 `libstdc++-6 / libgcc_s_seh-1 / libwinpthread-1` 拷到 exe 旁边。

### 三层等价性验证（`scripts/13_compare_cpp_backends.py`）

| 层 | 结果 | 结论 |
|---|---|---|
| 预处理张量 | 最大绝对差 **1.18e-02**，2.21% 的元素不同，9687 个元素差 > 1/255 | **两个 resize 实现确实不一样** |
| 检测框 | 6 vs 6，全配对；最小 IoU **0.9946**，坐标最大差 **0.07 px**，分数最大差 0.005，类别全对 | 结果一致 |
| 输出图 | 最大像素差 3，3.49% 像素不同 | 底图有 ±3 灰度差 |

**这个结果比「完全一致」有价值**：OpenCV 的 `INTER_LINEAR` 内部是定点整数运算，
和浮点手写版本数学同源但不逐像素相等。差 ±3 灰度传进网络后，最高分只变了 0.005，
没有一个框越过 `conf=0.25` 的边界——所以结果一致。**这件事不能靠假设，只能分层测出来。**

### 分发：`cpp/collect_dlls.py`

从 exe 出发用 `objdump -p` 递归读导入表，把非系统 DLL 收进 `cpp/dist/`。
比手写清单可靠，也和 Windows 加载器的查找顺序一致。

```bash
python cpp/collect_dlls.py     # → cpp/dist/（exe + 44 DLL，78.7 MB）
```

验证：把 PATH 清成只剩系统目录，`cpp/dist/yolo_cpp.exe` 能正常跑出 6 个框，
但 **CUDA EP 会加载失败并回落 CPU**（缺 `cublasLt64_12.dll`）。
要 GPU 就得再带 CUDA 12 运行库（cublas / cublasLt / cuDNN，约 1–2 GB），
或者要求目标机器已装 CUDA——这是部署时必须先问清楚的一件事。

### 这一步新踩的坑

1. **C++ 写 JSON 时没转义反斜杠**：`"image": "..\docs\x.jpg"` 写成单反斜杠后不是合法 JSON，
   Python 侧 `json.loads` 直接报 `Invalid \escape`。路径字段必须转义。
2. **只拷 OpenCV 的 DLL 不够**：它的依赖树会继续向下拉 30 多个库，必须递归收集。
3. **MSYS2 需要先全量升级才能装 OpenCV**：`pacman -S opencv` 会报
   `libgcc=16.2.0-4 与 16.2.0-6 冲突`，因为库里已是新版而本地还是旧版。
   正确顺序是 `pacman -Su`（本次 42 包 / 130 MiB / 约 10 分钟）之后再装。
4. **管道会杀死 pacman**：`pacman -Su | head -30` 触发 SIGPIPE，进程被杀并留下
   `db.lck`，下一次就报 `unable to lock database`。要重定向到文件，不要接 `head`。

## 8.6 步骤七：TensorRT FP32 / FP16（2026-10-27 同日）

**结论先放前面**：FP16 比 FP32 快 **1.46 倍**、比 ONNX FP32 快 **4.16 倍**，
而 mAP50 只掉 **0.0004**（−0.06%）。这是整个部署链路上收益最大的一步。

### 装 TensorRT：Windows 上 pip 走不通

| 尝试 | 结果 |
|---|---|
| `pip install tensorrt` | PyPI 上 Windows wheel **只有 `10.0.0b6`，而且是个 0 字节空壳**（不支持 sm_120） |
| `pip install tensorrt-cu12` / `cu13` | Windows 没有 wheel，只有 Linux |
| **NVIDIA 官方 zip（本次采用）** | `TensorRT-10.8.0.43.Windows.win10.cuda-12.8.zip`，**1.72 GB，免登录可下** |

三步装好：

```bash
# 1) 解压：1.72 GB 的 zip → 解压后 2.6 GB
# 2) 装 python 包（zip 里自带 wheel，按 Python 版本选 cp311）
pip install --no-deps tensorrt-10.8.0.43-cp311-none-win_amd64.whl \
                   tensorrt_lean-10.8.0.43-cp311-none-win_amd64.whl \
                   tensorrt_dispatch-10.8.0.43-cp311-none-win_amd64.whl
# 3) DLL 搜索路径加 lib/（不是 bin/，bin 里只有 trtexec.exe）
```

环境正好对上：TensorRT 10.8 要 CUDA 12.x + cuDNN 9.x，本机 torch 2.11 是 cu128 + cuDNN 9.19。

### 构建

| | FP32 | FP16 |
|---|---|---|
| 产物大小 | **17.91 MB** | **11.79 MB** |
| 构建耗时 | **111.7 s** | **616.7 s（10.3 分钟）** |
| 输入 / 输出 dtype | FLOAT | **FLOAT（没变）** |
| shape | `[1,3,640,640]` → `[1,16,8400]` | 同 |

**FP16 构建慢 5.5 倍不是异常**：FP16 模式下可选的 tactic 多得多，builder 要逐个实测再挑最快的。

**一个容易误判的点：FP16 引擎的输入输出仍然是 FLOAT。**
量化发生在网络内部，IO 保持 FP32——调用方不用改 API，也不用自己 cast。

### 四层验证

**第 1 层 结构**：`images` FLOAT `[1,3,640,640]` → `output0` FLOAT `[1,16,8400]`，与 ONNX 完全一致。

**第 2 层 数值**（原始输出张量 vs ONNX FP32，单图）

| | 最大绝对差 | 平均 | 相对（mean / 量级） |
|---|---|---|---|
| engine FP32 | 24.66 | 0.0337 | **7.09e-04** |
| engine FP16 | 12.96 | 0.0525 | **1.10e-03** |

FP32 的**最大**差反而比 FP16 大——这不是精度问题，是不同 kernel 的累加顺序不同，
个别 anchor 的框坐标偏得多。**看平均和相对量才准**（输出量级 mean|y| = 47.5，单位是像素）。

**第 3 层 结果**（16 张 valid 图，conf 0.25 / iou 0.7）

| | 框总数 | 框数一致 | 类别 | 最小 IoU | 坐标最大差 | 分数最大差 |
|---|---|---|---|---|---|---|
| ONNX FP32 | 70 | — | — | — | — | — |
| engine FP32 | 70 | 16/16 页 | 全对 | 0.9957 | 0.88 px | 0.0034 |
| engine FP16 | 70 | 16/16 页 | 全对 | **0.9733** | **3.36 px** | 0.0112 |

**第 4 层 精度**（valid 187 图 / 839 框，`scripts/16_eval_valid_split.py`）

| 格式 | mAP50 | mAP50-95 | P | R | ultralytics 记的 inference |
|---|---|---|---|---|---|
| best.pt | 0.6407 | 0.4169 | 0.6724 | 0.6048 | 10.0 ms |
| best.onnx | 0.6329 | 0.4111 | 0.6235 | 0.6028 | 6.0 ms |
| engine FP32 | 0.6327 | 0.4111 | 0.6255 | 0.6020 | **1.5 ms** |
| engine FP16 | 0.6323 | 0.4103 | 0.6249 | 0.5925 | **0.9 ms** |

**两个必须分开看的落差**：

- **pt → ONNX：mAP50 −0.0078**。这部分**不该全算作 ONNX 的精度损失**——
  pt 走的是训练时那条原生前向 + NMS，onnx / engine 走的是另一条解码 + 后处理路径。
  P 从 0.6724 掉到 0.6235 也是同一来源。这是「换了一条推理路径」，不是「算错了」。
- **FP32 → FP16：mAP50 −0.0004（−0.06%）**。这才是 FP16 的净代价，小到可以忽略。

要更干净地归因，得让 pt 也走同一个后处理路径——这一步没做，所以上面的区分是**推断**，不是实测。

### 性能（`scripts/15_verify_engine.py`，同图、预热 30、n=200、每次前后同步）

| 路径 | 中位数 | p95 | 相对 ONNX |
|---|---|---|---|
| ONNX Runtime FP32 | 8.19 ms | 11.70 | 1.00× |
| TensorRT FP32 | **2.88 ms** | 4.03 | **2.84×** |
| TensorRT FP16 | **1.97 ms** | 3.80 | **4.16×** |

注意这里 ONNX 是 8.19 ms，而 §5 里是 7.10 ms——同一台机器不同时段差 15%。
**跨表比较绝对值没意义，只比同一次测出来的相对值。**

### 这一步踩的两个坑

1. **Ultralytics 的 `.engine` 不是纯 plan**。它在前面写了 `[4 字节长度][JSON 元数据]`
   （本项目是 0x30F = 783 字节）。直接 `deserialize_cuda_engine(整个文件)` 会报
   `plan->header.magicTag == rt::kPLAN_MAGIC_TAG failed ... incompatible serialization version`——
   这个报错完全不提「文件前面有头」，极容易被误判成 TensorRT 版本不对，白折腾半小时。
   正确做法是先探测 `4 + N` 再跳过；能直接读就是纯 plan。
2. **DLL 在 `lib/` 不在 `bin/`**。TRT 10 的 Windows 包 `bin/` 里只有 `trtexec.exe`，
   `nvinfer_10.dll` 全在 `lib/`。加错目录时 `import tensorrt` 直接崩，报错只说 DLL 加载失败。

### 仍然没做

- **C++ 接 TensorRT**：`nvinfer_10.lib` 与 `NvInfer.h` 都在 zip 里，
  把 ORT 的 Session 换成 TRT 的 Engine + Context。这是把 1.97 ms 变成「端到端可交付」的最后一步。
- **INT8**：需要校准集（只能用 valid），且要接受更大的精度风险。

## 8.7 步骤八：C++ 接 TensorRT（2026-10-27 同日）

目标很明确：把 §8.6 那 1.97 ms 的纯 TRT 引擎，接到 C++ 里跑成端到端。

**结论先放**：纯 TRT 的 C++ 走不通（不是不会写，是 ABI 不允许）；
改用 ORT 自带的 TensorRT EP 后，C++ 侧从 **6.23 ms 降到 3.22 ms**（约 1.9×），
结果与 CUDA EP 逐框一致。

### 为什么纯 TRT 的 C++ 在 MinGW 下不能做

先看符号，这是判断依据而不是猜：
`nvinfer_10.dll` 的导出表里 `createInferRuntime` 等工厂函数是**未修饰的 C 名字**，
其余 API 全是虚函数（走 vtable，不需要链接期符号）。

所以「链接」这一关居然过了。写了个最小探针（`cpp/experiments/trt_abi_probe.cpp`）：

| 步骤 | 结果 |
|---|---|
| `createInferRuntime(logger)` | ✅ 成功 |
| 探测 Ultralytics 头 + `deserializeCudaEngine` | ✅ 成功（781 字节 JSON，偏移 785） |
| `engine->getNbIOTensors()` | ❌ **返回 1215455368**（垃圾值） |

原因：**GCC 和 MSVC 的 vtable 布局不同**。Itanium ABI 的 vtable 前面有 `offset-to-top` 与 `typeinfo`
两个槽，MSVC 没有。工厂函数是 C ABI 所以能调通，一进对象方法就整体错位。

**结论：能链接 ≠ 能调用。** 用 MSVC 编译的 C++ 库，必须用 MSVC 编译调用方；
MinGW 只能安全地调它的 **C 接口**（这也解释了为什么 ORT 的 DLL 我们在 MinGW 下用得好好的——
`onnxruntime.dll` 导出的就是 C API）。

要走纯 TRT 只有一条路：整个项目改用 MSVC（或者装 Visual Studio Build Tools）。
本机没有 MSVC，为这一步装几个 GB 的编译器不划算，所以换方案。

### 换的方案：ORT 的 TensorRT EP

`onnxruntime_providers_tensorrt.dll` 本来就和 ORT 一起装好了（1 MB）。
它在 ORT 内部调用 TensorRT，对外仍然是 ORT 的 C API——**跨编译器安全**。

Python 侧先验证（首次要构建引擎）：

| 路径 | 中位 | 说明 |
|---|---|---|
| ORT + CUDA EP（FP32） | 8.45 ms | 基线 |
| ORT + TRT EP（FP16） | **3.32 ms** | **2.54×** |

首次建会话 **605 s**（10 分钟），engine 缓存 11.1 MB + timing cache 0.9 MB；
之后加载只要 14 s。

### C++ 侧结果（同图、预热 30、n=200，三轮）

| 轮次 | CUDA EP 中位 | TRT EP 中位 |
|---|---|---|
| 1 | 7.13 ms | 3.28 ms |
| 2 | 6.15 ms | 3.10 ms |
| 3 | 5.40 ms | 3.29 ms |
| **均值** | **6.23 ms** | **3.22 ms** |

**加速约 1.9×**，但要注意：TRT EP 三轮稳定在 3.10–3.29（±3%），
CUDA EP 在 5.40–7.13 之间波动（32%）。所以「加速多少倍」取决于拿哪个 CUDA 数当分母，
说 **1.7–2.2×、中位约 1.9×** 比说一个精确值诚实。

结果一致性：**6 个框全部配对，最小 IoU 0.9861，坐标最大差 0.23 px，类别全对**。

横向看整条链路（纯推理，各自最好条件）：

| 路径 | 中位 |
|---|---|
| Python 走 `YOLO(onnx).predict()` | 12.65 ms |
| Python 直调 `session.run()`（CUDA EP） | 8.19 ms |
| **C++ + CUDA EP** | 6.23 ms |
| **C++ + TRT EP（本次）** | **3.22 ms** |
| 纯 TRT（Python，§8.6） | 1.97 ms |
| 纯 TRT（C++，需 MSVC，未做） | — |

**3.22 与 1.97 之间差的 1.25 ms 就是 ORT 的封装成本**（EP 抽象、每 run 的 shape 检查、张量拷贝）。
这笔钱换来的是「不用 MSVC、不用重写解码和后处理」。

### 三个坑

1. **通用的 `SessionOptionsAppendExecutionProvider(按名字)` 不认识 TensorRT**。
   它会报 `Unknown provider name 'TensorrtExecutionProvider'`，
   可支持列表里明明有 DML / QNN / OpenVINO / CoreML —— 唯独 TRT 走独立的 V2 接口。
2. **`OrtTensorRTProviderOptionsV2` 只是前向声明**，直接 `OptionsV2 opts{}` 会报 `incomplete type`。
   必须 `CreateTensorRTProviderOptions(&p)` → `UpdateTensorRTProviderOptions(p, keys, vals, n)` →
   `ReleaseTensorRTProviderOptions(p)`。头文件在
   `include/onnxruntime/core/providers/tensorrt/tensorrt_provider_options.h`，需要单独补。
3. **TRT EP 下相对路径模型找不到**。同一句 `--model ..\weights\best.onnx`，
   CUDA EP 正常，TRT EP 报 `Load model ... failed: File doesn't exist`。
   现在传进 Session 前统一 `std::filesystem::absolute()`。

### 怎么用

```bash
# CUDA EP（默认）
yolo_cpp --model weights/best.onnx --image x.jpg

# TensorRT EP（FP16，首次会构建引擎约 10 分钟，之后走缓存）
yolo_cpp --model weights/best.onnx --image x.jpg --ep trt --trt-cache weights/trt_cache
```

运行时 `nvinfer_10.dll` 必须在 PATH 里（它在 `D:\TensorRT\TensorRT-10.8.0.43\lib`）。

## 8.8 步骤九：C++ 端到端延迟（2026-10-27 同日）

前面所有「纯推理」的数字都只包住 `session.Run`。真实部署里还要读图、letterbox、解码、NMS，
所以这一节把完整链路测出来——**结论是纯推理数字会严重低估实际延迟**。

### 方法

`--repeat N` 跑完整链路，**每轮都重新读图**（不是拿同一张张量反复跑）：

| 段 | 内容 | 计入端到端 |
|---|---|---|
| 读图+解码 | `stbi_load` / `cv::imread` | ✅ |
| 预处理 | letterbox（缩放+补边）+ HWC→NCHW + /255 | ✅ |
| 推理+张量封装 | `CreateTensor` + `session.Run` | ✅ |
| 后处理 | 解码 `[1,16,8400]` + 按类 NMS + 逆变换 | ✅ |
| 画框 | `draw_detections`（只画到内存） | ✅ |
| 写 PNG | `stbi_write_png` / `cv::imwrite` | ❌ 一次性动作，混进来会让数字失真 |

### 结果（n=200，预热 5 次；三种配置对比）

| 配置 | 纯推理 | **端到端** | 读图 | 预处理 | 推理+封装 | 后处理 | 画框 |
|---|---|---|---|---|---|---|---|
| **TRT EP + OpenCV** | 2.76 | **7.94** | 1.48 | 2.61 | 3.02 | 0.065 | 0.005 |
| TRT EP + stb | 2.72 | 11.74 | 1.73 | **6.26** | 2.99 | 0.064 | 0.005 |
| CUDA EP + OpenCV | 6.55 | 13.31 | 1.56 | 2.40 | **8.76** | 0.071 | 0.007 |

单位 ms，取多轮中位数的均值。

### 四个结论

**① 端到端是纯推理的 2.9 倍。**
7.94 / 2.76。只报 2.76 ms 会让评价失真——被问「你这 2.76 ms 含不含读图」时，
答不上来就是硬伤。

**② 推理只占 38%，预处理占 33%，读图占 19%。**

```
读图+解码   1.48 ms  19%
预处理      2.61 ms  33%   ← 第二大开销
推理+封装   3.02 ms  38%
后处理     0.065 ms  0.8%
画框       0.005 ms  0.1%
```

**③ 阿姆达尔定律的实例。**
纯推理 CUDA→TRT 快 **2.37 倍**，端到端只快 **1.68 倍**——因为另外 62% 的时间一点没变。
这就是为什么「只优化模型」在后期几乎没收益。

**④ 换图像后端的收益，比换推理引擎还大。**

| 改动 | 段 | 前后 |
|---|---|---|
| stb → OpenCV | 预处理 | **6.26 → 2.61 ms** |
| CUDA → TRT | 推理 | 8.76 → 3.02 ms |
| stb → OpenCV | **端到端** | **11.74 → 7.94 ms（省 3.80 ms）** |
| CUDA → TRT | 端到端 | 13.31 → 7.94 ms（省 5.37 ms） |

手写双线性在 Release 下仍然比 `cv::resize` 慢 2.4 倍：OpenCV 走了 SIMD，
而 `resize_bilinear` 是逐像素双线性。**这时候「用库」不是为了省事，是因为它真的快。**

### 一个容易忽略的开销

端到端（7.94）减去五段之和（7.18）差 **0.73 ms / 9%**。
这部分是每轮 `Image`（1.2 MB）+ `boxed`（1.2 MB）+ `inp`（4.9 MB）的分配与释放，约 7 MB。
它不落在任何一段计时里，但确实在端到端里。要消掉就得做 buffer 复用（下一节）。

### 所以下一步该优化什么（按数据排）

| 优先级 | 项 | 现状 | 手段 | 预期 |
|---|---|---|---|---|
| 1 | 预处理 | 2.61 ms | 预分配缓冲（去掉每轮 4.9 MB 分配）+ 融合 /255 与 HWC→NCHW 到一次遍历 | 可能砍到 1 ms 内 |
| 2 | 读图 | 1.48 ms | 换成更快的 JPEG 解码，或直接吃摄像头帧（省掉解码） | 视频流场景下接近 0 |
| 3 | 推理 | 3.02 ms | 纯 TRT（需 MSVC）或 INT8 | 1.97 / 更低，但收益已被上面两段稀释 |

**继续压推理的边际收益最低**——这正是测端到端的价值。

## 8.9 步骤十：视频流与 1000 帧稳定性（2026-10-27 同日）

前面所有数字都是「单张图片」。真实部署面对的是**连续的帧**，要回答三个新问题：
帧率够不够、**尾部延迟**有多差、跑久了内存会不会涨。

### 视频从哪来：最后用的是两段真实公开视频

试过但不可用的：

| 尝试 | 结果 |
|---|---|
| Ultralytics 示例 `decelera_landscape_min.mov` | 内容是「Decelera Mayakoba」活动演讲现场，不是道路；只有 11 帧 |
| 合成视频（valid 187 张图循环拼接） | 能测通链路，但**帧间跨场景跳变**让滤波/缓存行为失真，延迟数字不可信 |
| Pexels / Pixabay | 取直链需 API key；BDD100K 需签授权 |

最终用 Wikimedia Commons 上两段**真实视频**（都能免密钥直下）：

| 文件 | 规格 | 授权 | 用途 |
|---|---|---|---|
| `dashcam_cc0_720p.ogv` | 1280×720 @30fps，3601 帧/120s，69.8 MB | **CC0**（无需署名） | 主测试 |
| `road_traffic_cobblestone.webm` | 1920×1080 @25fps，2652 帧/106s，51.6 MB | **CC BY 3.0**（需署名） | 高分辨率对照 |

后者第一次跑出来「每帧检出 20–30 个目标」，看着像严重误检——**实际是我忘了做 NMS**，
加 NMS 后正好稳定检出 2–3 个 `vehicles`，与画面里的两三辆车对得上。
（教训：看到异常数字先怀疑自己的统计代码，别急着怀疑模型。）

### 实现

| 改动 | 说明 |
|---|---|
| CMake 加 `videoio` 组件 | `cv::VideoCapture` 走 FFmpeg 后端；OpenCV DLL 从 3 个变 4 个 |
| 新增 `--video` / `--frames` / `--warmup-frames` / `--no-draw` | 独立分支，不复用 `--image` 路径 |
| BGR→RGB 后复用同一份预处理 | 保证视频与图片测的是同一条链路 |
| `GetProcessMemoryInfo` 每 100 帧采一次 RSS | 链接 `psapi`，看有无泄漏 |

### 结果（各两轮 × 1000 帧，预热 30 帧）

**720p（1280×720，CC0 视频）**

| 项 | 第 A 轮 | 第 B 轮 |
|---|---|---|
| 吞吐 | **101.6 FPS** | **100.5 FPS** |
| 中位延迟 | 9.90 ms | 9.82 ms |
| p95 | 13.02 ms | 13.62 ms |
| **p99** | **13.25 ms** | **14.11 ms** |
| 解码 | 0.66 ms | 0.65 ms |
| **预处理** | **5.48 ms** | **5.16 ms** |
| 推理 | 3.63 ms | 3.82 ms |
| 后处理 | 0.08 ms | 0.08 ms |
| 内存净增 | **+5 MB** | **+6 MB** |

**1080p（1920×1080，CC BY 3.0 视频）**

| 项 | 第 A 轮 | 第 B 轮 |
|---|---|---|
| 吞吐 | **72.5 FPS** | **75.4 FPS** |
| 中位延迟 | 13.73 ms | 13.15 ms |
| p95 | 16.30 ms | 15.72 ms |
| **p99** | **18.08 ms** | **17.63 ms** |
| 最大 | 21.94 ms | 23.10 ms |
| 解码 | 1.23 ms | 1.24 ms |
| **预处理** | **8.36 ms** | **7.87 ms** |
| 推理 | 3.97 ms | 3.89 ms |
| 内存净增 | **+17 MB** | **+18 MB** |

内存采样（每 100 帧一次，MB）：
720p `3997 → 4012 → 4014 → 4014 → 4014 → …`；
1080p `3997 → 4012 → 4014 → 4014 → 4014 → …`。
**前 2–3 次采样涨完就持平**，后面 700 帧一动不动 → 没有泄漏（涨的部分是分配器预热）。

### 四条结论

**① 720p 能跑到 100 FPS，1080p 掉到 73–75 FPS。**
差的 27 FPS 几乎全在预处理：**5.2–5.5 → 7.9–8.4 ms**，而推理两边都是 3.6–4.0 ms。

**② 预处理随分辨率近似线性增长。**
把三档数据放在一起看：

| 输入 | 像素数 | 预处理 | 推理 |
|---|---|---|---|
| 400×640（图片） | 25.6 万 | 2.61 ms | 3.6 ms |
| 1280×720（视频） | 92 万 | **5.2–5.5 ms** | 3.6–3.8 ms |
| 1920×1080（视频） | 207 万 | **7.9–8.4 ms** | 3.9–4.0 ms |

像素涨 8 倍（25.6 万 → 207 万），预处理涨约 3 倍 —— 次线性，但仍然是**唯一随输入规模变大的那一段**。
**这条曲线就是「实机换 1080p 摄像头要付多少钱」的直接答案。**

**③ 尾部延迟比中位高 32–34%（1080p）。**
p99 是 17.6–18.1 ms，中位 13.2–13.7 ms。如果控制回路按中位设计，**每 100 帧就有 1 帧超时**。
720p 的尾部差距小一些（约 34%），绝对值更低。
**结论不变：实时系统必须报 p99，只报均值会低估最坏情况。**

**④ 内存没有泄漏。**
1000 帧只涨 5–18 MB，且 2–3 次采样内就持平。这条对「能不能长跑」是决定性的，
而且**只有跑够帧数才测得出来**。

### 这一步的坑

**① 只有 OpenCV 的 `imgcodecs` 打不开视频。**
视频解码属于 `videoio`，CMake 的 `find_package(OpenCV COMPONENTS ...)` 里漏了它，编译能过但 `VideoCapture` 用不了。
videoio 依赖 FFmpeg 后端（`avcodec` / `avformat` / `swscale` 等一串 DLL），分发包因此变大。

**② 报延迟只能用真实视频。**
合成视频两轮 96.0 / 107.5 FPS，真实 720p 两轮 101.6 / 100.5 FPS——数值接近但不可互换：
合成视频的帧间跳变会让解码和缓存行为失真。**它的用途只有一个：验证链路通不通。**

### 补充：显存该怎么测（nvidia-smi 前后差值是错的）

视频模式现在除了 RSS，还会采显存（`vram_mb` 字段）。为什么不用 `nvidia-smi` 前后各读一次：

| 项 | 实测 |
|---|---|
| `nvidia-smi` 单次调用 | **101–107 ms** |
| 连续采样上限 | 71 ms/次（约 13 Hz） |
| 单帧推理 | **2–3 ms** |

测量开销是推理时长的 30–50 倍。更关键的是**差值本身没有意义**：
程序退出时 CUDA 全部释放，所以「推理前 1621 MB → 推理后 1612 MB」，差 **−9 MB**，
会得出「模型几乎不占显存」的错误结论。

改为运行时动态加载 `cudaMemGetInfo`（Windows `LoadLibrary` / Linux `dlopen`，找不到就降级为不可用），
每 100 帧采一次。1000 帧实测：

| 指标 | 值 | 含义 |
|---|---|---|
| `at_start` | 1665 MB | 视频循环开始前（模型已加载、Session 已建） |
| `first_sample` | 1691 MB | 预热 30 帧后 → **加载+预热只多占 26 MB** |
| `last_sample` | 1691 MB | 第 1000 帧 |
| `peak` / `delta` | 1691 MB / **0 MB** | **稳态 1000 帧零增长** |

10 次采样值完全一致（1691 × 10），说明显存在预热阶段就分配完毕，进入稳态后不动。
**这也是「显存泄漏」的正确判据：看 first → last 是否增长，而不是看程序前后的总数。**

顺带一个副作用：`cudaMemGetInfo` 是跨平台的，将来在 WSL/Linux 上跑同一份代码也能采显存，
不需要像 RSS 那样为 Linux 单独写 `/proc/self/statm`。

## 8.10 步骤十一：优化预处理（2026-10-27 同日）

§8.9 的结论是「预处理占 49–55%，是最大头」。这一节把它压掉。

### 原来的三个浪费

| 问题 | 具体 |
|---|---|
| 每帧重新分配 | `resized`(1.2 MB) + `boxed`(1.2 MB) + `nchw`(4.9 MB) ≈ **7.3 MB/帧**的分配释放 |
| 三次遍历 | resize → 填充 + 逐行 memcpy → to_nchw 逐像素除 255 |
| 缓存不友好 | to_nchw 手写循环，写时跨平面跳（`out[i]` / `out[plane+i]` / `out[2*plane+i]`） |

### 改动

| 改动 | 做法 |
|---|---|
| ① 缓冲复用 | `im.rgb`、输入张量等提到循环外；同一路视频尺寸固定，只在首帧分配 |
| ② 交给 SIMD | OpenCV 后端改为 `setTo(114)` → `resize` 直接写进 canvas 的 ROI → `split` 拆平面 → `convertTo(1/255)`，全程 OpenCV 实现 |
| ③ 留后路 | 加 `--legacy-pre` 开关走旧实现，专门用于一致性对比 |

关键细节：`split` 出来的平面顺序就是 canvas 的通道顺序（RGB），与原来 `to_nchw` 的 R/G/B 平面顺序一致，所以后处理不用改。

### 一致性验证（这一步不能省）

同一张图，`--legacy-pre` 与新路径各导出一次预处理张量（4.9 MB / 122.9 万个 float32）：

| 项 | 结果 |
|---|---|
| 元素数 | 1228800 = 1228800 |
| **最大绝对差** | **5.96e-08**（1 ULP） |
| 平均绝对差 | 2.74e-08 |
| **差 > 1/255 的元素** | **0** |
| 检测框 | 6 vs 6，**最小 IoU 1.000000**，分数差 0，类别全对 |

差异来源很明确：旧实现用 `/255.f`（除法），新实现用 `convertTo(1/255)`（乘法），两者差 1 个浮点末位。
**没有元素超过 1/255，检测结果逐框相同** —— 可以放心替换。

### 性能（同一时刻，720p 视频各 1000 帧 × 2 轮）

| 版本 | 吞吐 | 单帧中位 | p99 | 解码 | **预处理** | 推理 |
|---|---|---|---|---|---|---|
| legacy（旧算法） | 170.3 FPS | 5.83 ms | 7.2 ms | 0.54 | **2.47 ms** | 2.70 ms |
| **fast（新算法）** | **211.9 FPS** | **4.71 ms** | 5.6 ms | 0.45 | **1.14 ms** | 2.98 ms |

**预处理 2.47 → 1.14 ms（−54%），单帧 5.83 → 4.71 ms（−19%），吞吐 170 → 212 FPS（1.2×）。**

### 两个必须说清的口径问题

**① 这个对比里 legacy 也不是「原始版」。**
我改循环结构时顺手把缓冲提到了循环外，所以 legacy 也享受了这部分收益。
全部改动应拆成两段：**缓冲复用**（历史 9.90 → 现在 legacy 5.83 ms）+ **SIMD**（5.83 → 4.71 ms）。

**② 跨时间的对比不可靠。**
这次测出来的推理时间是 2.70–2.98 ms，而 §8.9 那次是 3.63–3.97 ms —— **推理代码一行没改却快了 25%**，
说明机器状态（GPU 频率、后台负载）变了。所以「9.90 → 4.71」不能全部算作代码优化的功劳。
**只有同一时刻的 legacy vs fast 是严格可比的**，这也是为什么我留了 `--legacy-pre` 开关。

### 结论

优化后**预处理只占单帧的 24%（原 55%），瓶颈转移到了推理（63%）**。
下一步就是 §8.6/§8.7 那两条：纯 TRT（需 MSVC）或 INT8 —— 但那两条的性价比现在才开始变高。

## 8.11 交付形态：什么能交、什么不该交（2026-10-27 同日）

**结论先放前面**：

| 形态 | 体积 | 状态 | 适用 |
|---|---|---|---|
| **CPU 自包含包** | **212.9 MB / 143 个文件** | ✅ 干净 PATH 下验证通过 | 目标机没 GPU，或只做功能验证 |
| GPU 自包含包 | 3.0 GB / 163 个文件 | ❌ 补齐全部依赖后仍崩 | **不推荐**，改用下面的轻包 |
| **轻包 + 目标机装运行时** | 约 40 MB | 推荐做法 | GPU / TensorRT 场景 |

### 先暴露出来的一件事：分发包里的 exe 是旧的

我一开始拿 `cpp/dist/yolo_cpp.exe` 去测 TRT，跑出 **5.25 ms**，看着"能用但慢"。

查下来：`dist` 里那个 exe 是 **387 KB**，而 `build` 里是 **443 KB**。
`strings` 一比，dist 版**根本没有 `--ep` 这个字符串**——它是加 TensorRT 支持之前打的包。

于是 `--ep trt` 被忽略，程序走了默认的 CUDA EP。那个 5.25 ms 从头到尾都不是 TRT 的成绩。

> **教训**：手工维护分发包，第一件会漏的事就是"忘了更新 exe"。
> 而且它**不会报错**——程序照常运行，只是跑的不是你以为的那条路径。

### 为什么要写脚本收依赖

修好 exe 之后，新问题来了：**`0xC0000135`（STATUS_DLL_NOT_FOUND）**。

原因是加了视频模式后，exe 链接了 `libopencv_videoio`，而 dist 里连这个文件都没有（旧 exe 用不到它）。
手工补上 videoio，它又拉出 `avcodec` / `avformat` / `swscale` / GStreamer 一整串。

所以写了 `cpp/collect_deps.py`：从 exe 出发，用 `objdump -p` 读导入表，**递归**收集。
它同时支持体检：

```bash
python cpp/collect_deps.py --check        # 只看缺什么，不复制
python cpp/collect_deps.py                # 打 CPU 版 → cpp/dist
python cpp/collect_deps.py --gpu          # 打 GPU 版 → cpp/dist_gpu
```

脚本还会生成 `MANIFEST.txt`（每个文件的 sha256 + 来源路径），以及"换机器怎么验"的说明。

### CPU 版：从 79 MB 涨到 212.9 MB，但终于跑得全

| | 改之前 | 改之后 |
|---|---|---|
| 文件数 | 45 | **143** |
| 体积 | 79 MB | **212.9 MB** |
| 视频模式 | ❌ 缺 videoio | ✅ |
| exe 版本 | ❌ 旧版（无 `--ep`） | ✅ 与 build 一致 |

涨的那 134 MB 主要是 FFmpeg 与 GStreamer：`libopencv_videoio` 硬依赖它们，
而 MSYS2 的 OpenCV 默认编了 GStreamer 后端，导入表里写着就必须能找到，哪怕运行时根本不用。

**验证方式**：PATH 只留 `System32 + Windows + dist`，不放 MSYS2、TensorRT、torch：

```
退出码 0
CUDA EP 不可用（缺 providers_cuda，按预期回落 CPU）
检出 6 个目标   ← 与源机一致
纯推理中位 19.03 ms
```

### GPU 版：3.0 GB，七步失败链

这是本次最有价值的一段，因为它说明**这条路本身不对**。逐步记录：

| # | 现象 | 真因 | 处理 |
|---|---|---|---|
| ① | 延迟 5.25 ms，看着"能用" | dist 里的 exe 是旧版，`--ep` 被忽略，跑的是 CUDA EP | 换成 build 版 exe |
| ② | `0xC0000135` DLL 找不到 | 新 exe 依赖 `libopencv_videoio`，dist 里没有 | 加 videoio + FFmpeg + GStreamer |
| ③ | CUDA EP 加载失败（Error 126） | 缺 `onnxruntime_providers_cuda.dll` | 补（307 MB） |
| ④ | 仍失败 | 缺 `cufft64_11.dll`——**导入表里没写**，是间接依赖 | 补（263 MB） |
| ⑤ | `0xC0000409` 直接崩，报 `Cannot load symbol cudnnCreate` | `cudnn64_9.dll` 只是 **0.3 MB 的加载器**，真正干活的是 7 个 `cudnn_*64_9.dll` 子库 | 全补（904 MB） |
| ⑥ | TRT 不生效，回落 CUDA EP | TensorRT 自报 `Could not locate nvrtc64_120_0.dll`（tactic 需要运行时 JIT） | 补 nvrtc + nvJitLink（157 MB） |
| ⑦ | **`0xC0000005` 访问冲突** | 移除 nvrtc 后仍崩，未能定位到单一 DLL | **停止** |

到这一步依赖已经是 **3.0 GB**（`nvinfer` 418 MB + `cublasLt` 643 MB + cudnn 900 MB + `cufft` 263 MB + `providers_cuda` 307 MB…），
而**同一时刻**用 build 版 + 完整开发环境跑，基线是 **2.35 ms、TRT 正常**。

也就是说：机器、模型、代码都没问题，**崩的是这个 3 GB 的拼装包**。

### 所以正确的交付方式是哪种

**不是把运行时打包，而是让目标机装运行时。**

原因有三条，都是这次实测出来的：

1. **体积不成比例**。CPU 版 213 MB 覆盖了全部功能；GPU 版 3 GB 只换来 2.35 ms 与 19 ms 的差别。
2. **版本必须严丝合缝**。缺 nvrtc 时 TRT 静默失效（回落 CUDA 你还以为是 TRT）；
   cudnn 子库少一个直接崩，而报错只说"符号加载失败"，完全看不出是缺文件。
3. **engine 本来就不能跨机器拷**（绑显卡架构 / TRT 版本 / 驱动）。
   GPU 场景的交付方式一直是"到目标机重建一次"（`scripts/14_export_engine.py`，约 10 分钟），
   那顺便装个官方运行时，成本远低于维护一套 3 GB 的 DLL 组合。

**轻包清单**（约 40 MB）：

```
yolo_cpp.exe
best.onnx
onnxruntime.dll
onnxruntime_providers_shared.dll
onnxruntime_providers_cuda.dll        ← 只装 CUDA EP 时
onnxruntime_providers_tensorrt.dll    ← 要用 TRT 时
libopencv_core-500.dll
libopencv_imgcodecs-500.dll
libopencv_imgproc-500.dll
libopencv_videoio-500.dll             ← 要跑视频时
libgcc_s_seh-1.dll / libstdc++-6.dll / libwinpthread-1.dll
```

其余（CUDA / cuDNN / TensorRT / FFmpeg）由目标机的安装包提供。

### 换机器后的五步验收

```bash
# 1. 先验 CPU 路径通不通
yolo_cpp.exe --model best.onnx --image test.jpg --runs 20 --warmup 5

# 2. 检出数量要和源机一致（本项目基准图 _test_odd.jpg 是 6 个框）

# 3. 再验 GPU：看输出有没有这一行
yolo_cpp.exe --model best.onnx --image test.jpg --ep trt --runs 50 --warmup 10
#   EP：TensorRT（FP16）+ CUDA 兜底，缓存 xxx
#   ↑ 没有这行就是没成功，别信延迟

# 4. 延迟落在 2–4 ms 才算 TRT 真的生效；> 15 ms 说明回落到 CPU

# 5. 最后验视频（这条以前在旧包里是坏的）
yolo_cpp.exe --video clip.mp4 --frames 1000 --ep trt
```

**第 3 步是重点**：不能只看"没报错"就认为 TRT 生效了。
本次就是靠"输出里没有 `EP：TensorRT` 那一行"才发现跑了半天 CUDA EP。

### 这一步的三个坑

1. **旧 exe 不会报错，只会静默走错路径**。所以 `--check` 里加了"exe 与 build 的 sha256 是否一致"。
2. **`cudnn64_9.dll` 是加载器不是库**。0.3 MB 那个文件必须配 7 个子库，缺一个就 `0xC0000409`。
   这个报错（`Cannot load symbol cudnnCreate`）完全不提"缺文件"，极容易误判成版本不兼容。
3. **间接依赖不会出现在导入表里**。`cufft64_11` 是 `providers_cuda` 运行时才拉的，
   静态分析发现不了。所以脚本里专门有一份 `GPU_EXTRA` 手工清单，注释写清每一项为什么在。

## 9. 下一步（还没做）

**已完成**：ONNX 导出 → 三层验证 → 性能对比 → C++ 最小闭环 → OpenCV → TensorRT FP32/FP16 → C++ 接 TRT EP → 端到端延迟 → 视频流与 1000 帧稳定性 → 预处理优化 → 交付打包与跨机验证（§3–§8.11）。

| 事项 | 前置条件 | 备注 |
|---|---|---|
| 纯 TRT 的 C++ 版本 | 需要 MSVC（MinGW 与 MSVC 的 vtable 布局不兼容，见 §8.7） | 只比现在的 3.22 ms 快 1.25 ms，性价比低 |
| INT8 | FP16 通过之后 | 校准集只能用 valid，不能用 test |

**一条纪律**：无论哪一步，`test` 集都不要再跑。验证一致性只用 `valid`（mAP50 = 0.641）。
对外报的 0.680 是 test 上唯一的一次。
