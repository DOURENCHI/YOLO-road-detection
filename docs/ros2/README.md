# ROS2 部署实施记录（WSL2 · 2026-11-24 起）

这份记录与 `docs/deployment/README.md`（Windows 侧部署）平级，记的是**同一个项目在 ROS2 侧的落地过程**。
所有数字都能回溯到 `ros2/runlogs/` 里的原始日志——按标签查对应 `.log` 文件。

**当前状态**：编译通过（带 ORT）· A1 dummy ✅ · A2 CPU ✅ · A3 CUDA ✅ · A5 线程对比 ✅（负结果）· A4 TRT 已诊断到根因、依赖未装（取舍：见下）

---

## 0. 编译：六个坑，全部是实录

从第一次 `colcon build` 到带 ORT 的干净编译，一共失败 5 次、成功 2 次。
这六个坑不在原教程的"预判 8 坑"里，是这次实跑跑出来的。

| # | 现象（日志原文关键句） | 真因 | 修法 | 日志 |
|---|---|---|---|---|
| 1 | `The keyword signature for target_link_libraries has already been used` / `The plain signature ... already been used` | `ament_target_dependencies` 内部用 plain 签名，代码 82/85 行用 `PRIVATE`（keyword）。CMake 规定同一 target 不能混用 | 去掉 82/85 行的 `PRIVATE`，统一 plain | `B-colcon_150519.log` |
| 2 | `CMAKE_C_COMPILE_OBJECT` / `CMAKE_C_CREATE_SHARED_LIBRARY` 缺失，报 "cmake may not be built correctly" | `project(yolo_trt_ros CXX)` 只启用 C++，但 `rosidl_generate_interfaces` 生成的类型支持内部需要 C。**错误文案误导性极强**，看着像 CMake 安装坏了 | 改成 `project(yolo_trt_ros C CXX)`，并 `rm -rf build install log` 清缓存 | `B-colcon2_150733.log` |
| 3 | `format '%u' expects unsigned int, but argument has type uint64_t` | `empty_streak_max_` 是 `uint64_t`，格式串写 `%u` | 改成 `%lu`（32 位上会打印错数字，属 UB） | `B-colcon3_150910.log` |
| 4 | 设了 `ORT_ROOT` 仍报"未找到 ONNX Runtime，只编译假检测器" | `find_path(NAMES onnxruntime_cxx_api.h)` **只查 HINTS 第一层**，不递归；真实文件在 `include/onnxruntime/core/session/` 第三层 | `NAMES` 改成 `onnxruntime/core/session/onnxruntime_cxx_api.h` | `B-colcon-ort_155755.log` |
| 5 | `fatal error: onnxruntime_cxx_api.h: No such file` | 源码写扁平 `#include "onnxruntime_cxx_api.h"`，但 include 根指向 `include/`，扁平头不存在 | `target_include_directories` 额外加 `${ORT_INCLUDE_DIR}/onnxruntime/core/session` | `B-colcon-ort2_155916.log` |
| 6 | `/usr/include/c++/11/variant` 报几十条 `expected template-name`、`std::forward is not a member of yolo_trt_ros::std` | `#include "onnxruntime_cxx_api.h"` 写在 `namespace yolo_trt_ros {` **内部**。语法合法，但第三方头里的 `std::xxx` 被解析成 `yolo_trt_ros::std::xxx` | 把 ORT include 移到文件顶部、namespace 之外 | `B-colcon-ort3_160036.log` |

**最终成功**：`B-colcon-ort4_160501.log`，退出码 0，21.4 s，**零 warning**。

### 六个坑的共同规律

1. **看第一条错误，不要看最后一条**。CMake 会把同一个问题从两个方向报两遍（坑 1），
   最后一行往往是 `gmake: *** Error 2` 这种废话。
2. **错误文案会指向错误的地方**。坑 2 说 "cmake may not be built correctly"（其实项目没启用 C）；
   坑 6 指向 `/usr/include/c++/11/variant`（其实是你自己的 include 位置）。
3. **"找不到文件"先查路径层级**。坑 4 和坑 5 是同一个问题的两面：一个查得太浅，一个引得太浅。

---

## 1. ONNX Runtime：不用 curl，用 pip + 软链

### 现状

`cpp/third_party/onnxruntime/` 里**只有 `include/`，没有 `lib/`**（2026-10-27 做 Windows C++ 时下的）。
所以只够 dummy 阶段；A2 起必须补 `.so`。

### 网络诊断（实测，2026-11-24）

| 源 | 结果 | 说明 |
|---|---|---|
| `curl` GitHub release | **404** | 连 v1.19.2 都 404；HEAD 和 GET Range 都返回 "Not Found" 9 字节。不是版本号错 |
| 清华 PyPI 源 | **HTTP 403** | pip 报 `from versions: none`——**不是"没有这个版本"，是被挡了拿不到列表** |
| 官方 `pypi.org/simple` | HTTP 200（8 s） | 索引能看 |
| `files.pythonhosted.org` | HTTP 000 | 文件服务器不通，pip 就算查到版本也下不了 |

### 采用的方案

1. 从可联网侧下载 cp310 manylinux wheel（15.6 MB）到 `ros2/_wheelhouse/`，
   校验 ZIP 完整（内含 `libonnxruntime.so.1.22.0`，20.1 MB）
2. WSL 里**本地安装**（不加 `-i`，否则又走清华源）：
   ```bash
   python3 -m pip install --user \
     "/mnt/d/VC Code/YOLO_Road_Detection/ros2/_wheelhouse/onnxruntime-1.22.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
   ```
3. 软链成 `find_library(NAMES onnxruntime)` 认识的名字：
   ```bash
   mkdir -p ~/onnxruntime-linux-x64-1.22.0/lib
   ln -sf ~/.local/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.22.0 \
          ~/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so
   export ORT_ROOT=$HOME/onnxruntime-linux-x64-1.22.0
   ```
4. **必须清缓存重编**：`rm -rf build install log`，否则 CMake 沿用 `HAVE_ORT=FALSE`

拼装结果：头文件用项目里现成的，`.so` 用 pip 的，合成完整 `ORT_ROOT`。**全程不碰 GitHub release**。


### A2 启动阶段的两个坑（2026-11-24 实测）

| 现象 | 真因 | 修法 |
|---|---|---|
| `error while loading shared libraries: libonnxruntime.so.1` | `.so` 内部 SONAME 是 `libonnxruntime.so.1`，运行时按 SONAME 找；只建 `libonnxruntime.so` 软链（给链接器用）不够 | 再建一条 `libonnxruntime.so.1` 软链指向同一个 `.so.1.22.0` |
| `AttributeError: _ARRAY_API not found` / `numpy.core.multiarray failed to import` | pip 装 ORT 时顺带把 numpy 升到 **2.2.6**；ROS2 Humble 的 cv_bridge/rclpy 是按 numpy 1.x 编译的，ABI 不兼容 | 本地装 `numpy-1.24.4`（wheel 已放 `_wheelhouse/`），满足 ORT 的 `numpy>=1.21.6` 且兼容 ROS2 |

**坑点**：numpy 2.x 不报"版本不对"，而是报 `_ARRAY_API not found`——完全看不出是版本问题。
**规则**：ROS2 Humble 环境下，`pip install` 任何包后都要确认 numpy 没被升到 2.x。

---

## 2. A1 · dummy 检测器（通过）

见 `A1_dummy.json`。核心数字：

- 话题名正确：`/yolo_trt_node/detections`（带节点名，不是 `/yolo/detections`）
- `topic hz` 稳态 **9.6–10.4 Hz**（相机设 10）
- 节点自报 fps 1–9 是**另一口径**（瞬时计数 vs 滑动窗口平均）——报数字必须说口径
- `transport p50 ≈ 58 ms`，dummy 推理为 0，这 58 ms 全来自时间戳链路（待查 `fake_camera` 的 stamp）
- `dropped_stale` 累计 2（仅启动期，稳态不涨）；`vram` 2.7→3.2 GB 是整卡总量，不是进程占用
- 无 error / exception

证据：`B-colcon3_150910.log`、`A1-camera_151401.log`、`A1-node_151413.log`、
`A1-list_151424.log`、`A1-hz_151425.log`

---

## 3. A2 · ORT CPU（进行中）

编译已就绪（`B-colcon-ort4` 零 warning）。下一步：启动 `detector:=ort ep:=cpu`，
验证真模型推理链路和逐段延迟。


## 3. A3 CUDA：卡在 CUDA 运行库（2026-11-24，进行中）

### 已完成的前置

- `E-nvidia`：RTX 5060 Laptop / 8 GB / 驱动 591.91 / CUDA 13.1，WSL 通路正常
- `E-providers-cpu`：`['AzureExecutionProvider','CPUExecutionProvider']`（基线）
- 下载 ORT-GPU 1.22.0 wheel（270 MB）：官方源和阿里镜像单连接都被限速到 ~90 KB/s，
  写 `ros2/tools/fetch_wheel.py` 用 **16 线程 Range 分块**并发，34 块 296 s 下完，ZIP 校验 OK
  内含 `libonnxruntime_providers_cuda.so`（解压 389 MB）
- 本地 pip 装 GPU 版 + 降 numpy 回 1.24.4（`--force-reinstall` 会把它升回 2.2.6）
- `E-providers-gpu`：`['TensorrtExecutionProvider','CUDAExecutionProvider','CPUExecutionProvider']`
- `E-ortgpu`：建 `~/ort_gpu/` 目录，软链 3 个 provider + `libonnxruntime.so{,.1,.1.22.0}`

### 两个新坑（都是"A3 跑不通"的真因）

| # | 现象 | 真因 | 修法 |
|---|---|---|---|
| 7 | `Unknown provider name 'CUDA'. Currently supported values are DML/QNN/...` | ROS2 包用字符串版 `AppendExecutionProvider("CUDA")`，ORT 1.22 的 C++ API 只认识 DML/QNN/Azure/CoreML 等；CUDA/TRT 必须用类型化 V2 API。**Windows 侧 `cpp/src/main.cpp` 注释里早写过这个坑，但 ROS2 包是另一套代码，没同步** | 改成 `CreateCUDAProviderOptions` + `UpdateCUDAProviderOptions` + `SessionOptionsAppendExecutionProvider_CUDA_V2`，TRT 同理；失败回落 CPU 并写 err |
| 8 | 节点打印 `ep=CPU`，启动自检警告"请求 CUDA 但推理 41 ms，疑似静默回落"，**但看不到原因** | `load()` 降级时写了 `err` 却返回 true，`init_detector()` 只在返回 false 时打印 err —— 原因被吞 | 在 `init_detector()` 里 `load()` 成功后也检查 `err` 非空并 `RCLCPP_WARN` |

### 决定性诊断：不是 C++ 代码的问题

`E-cuda-python` 绕过 C++/ROS2，直接问 Python ORT：

```
Failed to load library libonnxruntime_providers_cuda.so with error:
libcublasLt.so.12: cannot open shared object file: No such file or directory
实际 providers: ['CPUExecutionProvider']
```

**根因**：ORT 1.22 用 CUDA 12.8 + cuDNN 9，wheel 只带 provider 本身，**不带 CUDA 运行库**。
驱动（`/usr/lib/wsl/lib/libcuda.so`）由 WSL 自动提供，但 BLAS/cuDNN 等用户态库要单独装。
教程里"用预编译包不用装 CUDA Toolkit"只对**驱动**成立，对**运行库**不成立。

`E-cuda-libs` 确认系统里完全没有；`E-ldd-cuda` 用 `ldd` 精确列出缺失的 7 个：

```
libcublasLt.so.12 / libcublas.so.12 / libcurand.so.10 / libcufft.so.11
libcudart.so.12 / libcudnn.so.9 / libnvrtc.so.12
```

对应 6 个 nvidia pip 包（约 1.6 GB），用 `ros2/tools/fetch_cuda_wheels.py` 并发下载中。

### ⚠️ 对教程的修正

ROS2 部署教程 A3 那句"用 ORT 预编译包不需要 CUDA Toolkit"**不够准确**，需要补一句：
> 预编译包不需要的是**驱动**和完整 **CUDA Toolkit**（nvcc 等编译工具），
> 但 CUDA **运行库**（cublas/cudnn/curand/fft/rt/nvrtc）仍需提供。
> 最省事的是 `pip install nvidia-*-cu12`，并把它们的 `lib/` 加进 `LD_LIBRARY_PATH`。


## 4. 收尾：五关结果一览与取舍（2026-11-24）

| 关卡 | 状态 | 推理 p50 | 核心证据 |
|---|---|---|---|
| 编译（带 ORT） | ✅ | — | `B-colcon-ort4_160501.log`（零 warning） |
| A1 dummy | ✅ | 0 ms | `A1_dummy.json` |
| A2 ORT CPU | ✅ | 39.0 ms | `A2_ort_cpu.json` |
| A3 CUDA | ✅ | **7.1 ms（5.5×）** | `A3_cuda.json` |
| A4 TensorRT | ⚠️ 根因已定位 | — | `A4_tensorrt.json`（缺 libnvinfer.so.10） |
| A5 worker 对比 | ✅ 负结果 | 37.7 vs 39.0 ms | `A5_threads.json` |

### A4 为什么停在这里

请求 `ep:=trt` 后节点回落 CUDA，日志明确给出原因：`libnvinfer.so.10: cannot open shared object file`。
这和 A3 是同一类问题——provider 在、它依赖的运行库不在：

- A3 缺 cublas/cudnn → 装 6 个 `nvidia-*-cu12` 包（1.65 GB）解决
- A4 缺 nvinfer → 要装 TRT 10.x（SONAME 必须是 `.so.10`）

不再继续的两个理由：

1. **渠道不确定**：PyPI 官方 `tensorrt-cu12` 的 10.x 只有源码包；11.x 的 wheel SONAME 是 `.so.11`，对不上 ORT provider 要求的 `.so.10`。TRT 10 要从 NVIDIA 官方索引另下，几百 MB。
2. **数字重复**：Windows 侧已经完整验证过 TensorRT（FP32/FP16 引擎四层验证、C++ 接 TRT EP 3.22 ms），
   同一张 RTX 5060、同一个 best.onnx、同一个 TRT 10.8。在 WSL 重做一遍，边际收益低。
   把时间投在边缘设备 profiling（缺口③）更有价值。

**这不是"没做"，是"知道为什么不做"**——和 INT8 的结论一个性质。日志里那句降级原因就是证据。

### 本次一共踩了 12 条坑

构建 6 条（见 §0）+ 运行 3 条（SONAME / numpy / 降级原因被吞）+ 依赖 2 条（CUDA 运行库 / nvinfer，后者未修）+ 观测性 1 条（与"降级原因被吞"同源，但性质是"代码没错却把原因藏了"）。

> 早前这里写的是"10 个坑"，是漏数：A3 的 CUDA 运行库和 A4 的 nvinfer 两条写在正文里、没进编号表。
> 逐条清点后统一为 **12**，《全项目答辩级学习手册》14.5 用的是同一口径。

其中最值得记的三条：

1. **同一个坑在两个子系统各踩一次**：ORT 字符串版 `AppendExecutionProvider("CUDA")` 不能用，
   Windows 侧 `cpp/src/main.cpp` 注释里早就写了，但 ROS2 包是另一套代码，没同步。
2. **错误信息会指向错误的地方**：`std::forward is not a member of yolo_trt_ros::std` 指向标准库，
   真因是 `#include` 写在 namespace 里；`cmake may not be built correctly` 指向 CMake 安装，
   真因是 `project()` 没启用 C。
3. **观测性本身要花钱建**：`load()` 吞掉降级原因、`runlog.sh` 把输出全重定向——
   两处都是"信息在那儿但你看不到"。修完之后排查速度才真正上来。

---

## 附：记录方式

`ros2/tools/runlog.sh` 提供 `log <标签> <命令...>`，自动把完整命令、时间、cwd、环境变量、
stdout/stderr、退出码写进 `ros2/runlogs/<日期>/`，并维护 `INDEX.md`。
WSL 的 `/mnt/d` 就是 D 盘，所以日志写下来后我能直接读，不需要手工复制终端输出。

**规则：每开一个新终端，用 `log` 之前先 `source` 一次**——shell 函数只对当前终端有效。
