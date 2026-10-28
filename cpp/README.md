# C++ 推理最小闭环（ONNX Runtime）

**目的**：把 Python 里由 `ultralytics` 代劳的四件事（预处理 / 推理 / 解码 / NMS）在 C++ 里自己实现一遍，
并证明输出与 Python 一致。这是「部署」从「能跑」走到「能交付」的那一步。

## 关于 OpenCV：先手写，再换库（现在两者都支持）

C++ 部署真正的成本在**预处理一致性**，不在「调用哪个库」。用 OpenCV 会把 letterbox、
归一化、通道重排全藏进一行 `cv::resize` 里，出错时无从定位。

所以这个程序的第一版是**零依赖**的：读图写图用 `stb_image`（单头文件），双线性缩放自己写
（约 30 行）。等这套跑通、结果和 Python 对齐之后，再引入 OpenCV——此时每一步在做什么已经清楚，
换库只是换实现，不是换黑盒。

现在两种后端都在，**编译时自动探测**、**运行时可以切换**：

| 后端 | 读图 / 缩放 / 写图 | 说明 |
|---|---|---|
| `stb` | `stbi_load` + 手写双线性 + `stbi_write_png` | 零依赖，永远可用 |
| `opencv` | `cv::imread` + `cv::resize` + `cv::imwrite` | 需装 OpenCV；默认启用 |

只有这三件事交给库。**画框和解码仍然自己写**——画框要让两个后端的输出图逐像素可比，
解码是模型语义，不该被库改变。

```bash
yolo_cpp --model ../weights/best.onnx --image x.jpg                 # 有 OpenCV 就用 OpenCV
yolo_cpp --model ../weights/best.onnx --image x.jpg --backend stb   # 强制手写实现
yolo_cpp --model ../weights/best.onnx --image x.jpg --dump-input t.bin   # 导出预处理张量用于逐值对比
```

两个后端等价性由 `scripts/13_compare_cpp_backends.py` 验证（张量 / 框 / 输出图三层）。

### 装 OpenCV 的两个硬约束

1. **必须和编译器 ABI 一致**。OpenCV 的 C++ API 大量传 `std::vector` / `std::string`，
   这些类型的二进制布局由编译器决定。所以不能把 MSVC 编译的官方 OpenCV 包用 `dlltool` 转成 `.a`
   给 MinGW 用——**能在链接期混过去，会在运行期崩**。要么整个项目用 MSVC（配 vcpkg），
   要么用 MSYS2 这份「同一套 GCC 构建」的 OpenCV。本项目选后者。
2. **会连带装 100+ 个包**。`mingw-w64-ucrt-x86_64-opencv` 依赖 ffmpeg、tesseract、ogre3d、
   hdf5、ceres 等，即使你只用 `core` + `imgproc` + `imgcodecs` 也躲不掉。这是 OpenCV 的现实代价。

## 目录

```
cpp/
├── CMakeLists.txt
├── src/main.cpp
└── third_party/
    ├── stb_image.h / stb_image_write.h       读图写图（stb 后端）
    └── onnxruntime/include/...               ORT C++ 头文件（v1.22.0）
```

> 头文件是从 `raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/include/` 取的，
> **不需要**下载 200 MB 的官方 C++ 包——链接直接指向 pip 装好的 `onnxruntime.dll`。

## 依赖

| 项 | 说明 |
|---|---|
| 编译器 | MSYS2 UCRT64 `g++` 16.2（`D:\msys64\ucrt64\bin`） |
| CMake | 3.16+ |
| ONNX Runtime DLL | 用 conda 环境里 `onnxruntime-gpu` 自带的那份 |
| OpenCV（可选） | `pacman -S mingw-w64-ucrt-x86_64-opencv`；**装之前要先 `pacman -Su` 全量升级**，否则 libgcc 版本冲突 |

OpenCV 是可选项：CMake 自动探测，找不到就用 stb。注意它连带装 133 个包、占盘 1966 MiB。

DLL 的实际位置：

```
D:\Miniconda\envs\yolo_road\Lib\site-packages\onnxruntime\capi\onnxruntime.dll
D:\Miniconda\envs\yolo_road\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll
D:\Miniconda\envs\yolo_road\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_shared.dll
```

## 构建

```bash
cd cpp
export PATH="/d/msys64/ucrt64/bin:$PATH"

cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

产物：`cpp/build/yolo_cpp.exe`

**运行前**把三个 DLL 复制到 exe 同目录（Windows 找不到就启动即崩，编译器不报错）：

```bash
cp /d/Miniconda/envs/yolo_road/Lib/site-packages/onnxruntime/capi/onnxruntime*.dll build/
```

## 用法

```bash
# 用图片跑（含预处理），还会存一张画好框的 png
./build/yolo_cpp.exe --model ../weights/best.onnx --image ../assets/xxx.jpg --out ../results/cpp/demo

# 用预处理好的张量跑（跳过预处理，专门验证「推理+后处理」这一层）
./build/yolo_cpp.exe --model ../weights/best.onnx --input-bin ../docs/deployment/_tensor.bin

# 计时参数
./build/yolo_cpp.exe --model ../weights/best.onnx --image xxx.jpg --runs 200 --warmup 30

# 切换图像后端（装了 OpenCV 后默认就是 opencv）
./build/yolo_cpp.exe --model ../weights/best.onnx --image xxx.jpg --backend stb
./build/yolo_cpp.exe --model ../weights/best.onnx --image xxx.jpg --backend opencv

# 导出预处理后的张量，用来逐值对比两个后端
./build/yolo_cpp.exe --model ../weights/best.onnx --image xxx.jpg --dump-input /tmp/in.bin
```

两个后端是否等价，用 `scripts/13_compare_cpp_backends.py` 一次跑完（张量 / 框 / 输出图三层）。

## TensorRT：为什么用 ORT 的 EP 而不是直接调 TRT

想把纯 TensorRT 的 1.97 ms 接到 C++ 里，第一反应是直接 `#include <NvInfer.h>`。
**在 MinGW 下这条路走不通**，而且失败方式很有迷惑性：

| 步骤 | 结果 |
|---|---|
| `createInferRuntime()` | ✅ 成功（工厂函数是未修饰的 C 名字，链接能过） |
| `deserializeCudaEngine()` | ✅ 成功 |
| `engine->getNbIOTensors()` | ❌ 返回 1215455368（垃圾值） |

原因是 **GCC 与 MSVC 的 vtable 布局不同**（Itanium ABI 前面多 `offset-to-top` 与 `typeinfo` 两个槽）。
C 接口能调，一进对象方法就整体错位。**能链接 ≠ 能调用。**

所以改用 `onnxruntime_providers_tensorrt.dll`（ORT 自带，1 MB）——
它内部调 TensorRT，对外还是 ORT 的 C API，跨编译器安全：

```bash
export PATH="/d/TensorRT/TensorRT-10.8.0.43/lib:$PATH"   # 需要 nvinfer_10.dll
./build/yolo_cpp.exe --model ../weights/best.onnx --image x.jpg --ep trt
```

实测（同图、预热 30、n=200、三轮）：CUDA EP 5.40–7.13 ms → TRT EP **3.10–3.29 ms**，
**约 1.9×**，6/6 框一致（最小 IoU 0.9861，坐标差 0.23 px）。

想更快的唯一办法是纯 TRT（1.97 ms），代价是整套改用 MSVC。为了 1.25 ms 装几个 GB 的编译器，
当前不值得。探针代码留在 `experiments/trt_abi_probe.cpp`，哪天换 MSVC 可以直接接着用。

## 两种运行环境（别混淆）

| 场景 | 目录 | DLL 来源 |
|---|---|---|
| **开发态** | `cpp/build/` | 只拷了 3 个 OpenCV 核心 DLL 与 ORT；**其余第三方库（libpng / libjpeg / libtiff …）从 `D:\msys64\ucrt64\bin` 走 PATH** |
| **交付态** | `cpp/dist/` | `collect_deps.py` 递归收集的**全部**非系统 DLL（143 个 / 212.9 MB），在没装 MSYS2 的机器上也能跑 |

开发时这样跑：

```bash
export PATH="/d/msys64/ucrt64/bin:/d/Miniconda/envs/yolo_road/Lib/site-packages/torch/lib:$PATH"
./build/yolo_cpp.exe --model ../weights/best.onnx --image ../docs/deployment/_test_odd.jpg
```

第二个路径是给 CUDA EP 找 `cublasLt64_12.dll` 的，少了它只会静默回落 CPU。

## 打包给别人

```bash
python collect_deps.py            # → cpp/dist/       （CPU 版，143 文件 / 212.9 MB，已验证）
python collect_deps.py --gpu      # → cpp/dist_gpu/   （含 TRT/cuDNN，3.0 GB，见下方警告）
python collect_deps.py --check    # 只体检：缺什么、exe 是否与 build 一致
```

从 exe 出发用 `objdump -p` 递归读导入表，把非系统 DLL 全部收进来，并生成 `MANIFEST.txt`
（每个文件的 sha256 + 来源路径 + 换机器怎么验）。

**为什么必须用脚本**：手工拷一定会漏，而且漏了不一定报错。本次实测踩到两处——
① `dist` 里的 exe 是加 `--ep` 之前的旧版，跑 `--ep trt` 时**静默走了 CUDA EP**（5.25 ms，
而不是 TRT 的 2.35 ms），完全不报错；② 加了视频模式后 exe 依赖 `libopencv_videoio`，
而 dist 里没有这个文件，直接 `0xC0000135`。

### 交付形态怎么选

| 形态 | 体积 | 状态 |
|---|---|---|
| **CPU 自包含包**（`dist/`） | 212.9 MB | ✅ 干净 PATH 下验证通过（回落 CPU，19.03 ms，检出 6 框） |
| GPU 自包含包（`dist_gpu/`） | 3.0 GB | ❌ 补齐全部依赖后仍 `0xC0000005`，**不推荐** |
| **轻包 + 目标机装运行时** | 约 40 MB | ✅ 推荐用于 GPU / TRT 场景 |

**为什么不打包 GPU 运行时**：`nvinfer` 418 MB + `cublasLt` 643 MB + cuDNN 全套 904 MB +
`cufft` 263 MB + `providers_cuda` 307 MB ≈ 3 GB；而且版本必须严丝合缝——缺 `nvrtc` 时
TensorRT 会**静默失效**（回落 CUDA，你还以为是 TRT），`cudnn_*64_9` 子库少一个就
`0xC0000409` 直接崩，报错还说 `Cannot load symbol cudnnCreate`（完全不提缺文件）。

轻包清单（约 40 MB）：`yolo_cpp.exe`、`best.onnx`、`onnxruntime.dll`、
`onnxruntime_providers_*.dll`、4 个 `libopencv_*`、3 个 MinGW 运行时 DLL。
其余由目标机的 CUDA / cuDNN / TensorRT 安装包提供；engine 到目标机用
`scripts/14_export_engine.py` 重建（约 10 分钟）。

### 换机器验收五步

```bash
# 1. CPU 路径先通
yolo_cpp.exe --model best.onnx --image test.jpg --runs 20 --warmup 5
# 2. 检出数量应与源机一致（基准图 _test_odd.jpg 是 6 个框）

# 3. GPU：必须看到这一行才算成功
yolo_cpp.exe --model best.onnx --image test.jpg --ep trt --runs 50 --warmup 10
#   EP：TensorRT（FP16）+ CUDA 兜底，缓存 xxx

# 4. 延迟 2–4 ms 才算 TRT 生效；> 15 ms 说明回落到 CPU
# 5. 视频链路
yolo_cpp.exe --video clip.mp4 --frames 1000 --ep trt
```

详细数据与失败链记录在 `docs/deployment/README.md` §8.11 与 `delivery_report.json`。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | 必填 | `.onnx` 路径 |
| `--image` | — | 输入图片（走完整预处理） |
| `--input-bin` | — | float32 NCHW 张量文件（跳过预处理） |
| `--backend` | 自动 | `stb` 或 `opencv`；编进 OpenCV 时默认 opencv |
| `--ep` | `cuda` | `cuda` 或 `trt`。`trt` 走 ORT 的 TensorRT EP（FP16），首次要构建引擎约 10 分钟 |
| `--trt-cache` | `weights/trt_cache` | TRT EP 的 engine 缓存目录，第二次起加载约 14 s |
| `--dump-input` | — | 把预处理后的张量写成 float32 文件 |
| `--runs` | 100 | 计时重复次数 |
| `--warmup` | 20 | 预热次数 |
| `--repeat` | 0 | >0 时跑完整链路（每轮重新读图），输出端到端与分段耗时 |
| `--video` | — | 视频流模式：读视频逐帧处理，输出 FPS / p99 / 内存曲线 |
| `--frames` | 0 | 视频模式处理帧数上限（0 = 全部） |
| `--warmup-frames` | 30 | 视频模式前 N 帧不计入统计 |
| `--no-draw` | 关 | 视频模式跳过画框（测纯感知链路） |
| `--legacy-pre` | 关 | 走旧的预处理实现（逐像素 + 每帧分配），用于和新路径做一致性对比 |
| `--conf` | 0.25 | 置信度阈值 |
| `--classes` | 12 | 类别数（决定输出通道 4+nc） |
| `--out` | — | 输出前缀，会生成 `<前缀>_boxes.png` |
| `--json` | — | 结果与耗时写成 JSON（路径会做转义） |

### 视频模式的输出字段

除延迟与分段耗时外，`--video` 还会输出两组内存数据到 JSON：

| 字段 | 采集方式 | 说明 |
|---|---|---|
| `memory_mb` | `GetProcessMemoryInfo`（Windows）/ 待补（Linux） | 进程 RSS，看有没有泄漏 |
| `vram_mb` | 运行时 `cudaMemGetInfo` | GPU 显存，包含 `available` / `at_start` / `first_sample` / `last_sample` / `peak` / `delta` / `samples` |

`vram_mb` 是通过 `LoadLibrary`（Windows）或 `dlopen`（Linux）动态取 `cudaMemGetInfo`，
**不增加编译期依赖**：找不到 `cudart` 时 `available=false`，终端显示「显存不可用」，程序照常跑完。

之所以不用 `nvidia-smi` 前后各读一次：单次调用实测 **101–107 ms**，而一帧推理只有 2–3 ms；
且程序退出时显存全释放，前后差值约等于 0，会得出错误结论。详见 `docs/deployment/README.md` §8.9。

## 输出形状说明

`--classes 12` 对应输出 `[1, 16, 8400]`：

- `16 = 4`（cx, cy, w, h）+ `12`（类别分数）
- `8400 = 80² + 40² + 20²`，stride 8/16/32 三个尺度的网格点
- **没有 objectness**，类别分数已过 sigmoid，**NMS 不在图里**（要自己算）

## 已知限制

- 路径只支持 ASCII（`widen()` 是逐字节扩展，中文路径会乱码）
- 画框不写文字标签（没引字体），只按类别配色
- NMS 是 O(n²) 朴素实现，只在候选框几十个时够用
- 计时只包住 `Session::Run`，与 Python 侧「纯推理」口径对齐；端到端请自己加读图与前后处理
