# yolo_trt_ros — 把 YOLO 的 C++ 推理接进 ROS2

> ✅ **状态：已在 WSL2 + ROS2 Humble 上实测（2026-11-24）。**
> 编译通过（带 ORT，零 warning），A1 dummy / A2 CPU / A3 CUDA / A5 线程对比全部跑通，
> A4 TensorRT 已定位到根因（`libnvinfer.so.10` 未安装）后按取舍停止。
> 完整实录、原始日志与 10 个真实踩坑见 **`docs/ros2/README.md`** 和
> `docs/ros2/A{1,2,3,4,5}_*.json`；日志在 `ros2/runlogs/`。
>
> **实测数字**：CPU infer p50 39.0 ms → CUDA 7.1 ms（5.5×）；
> worker 线程对比无实质收益（负结果，已如实记录）。
>
> 📌 **2026-10-30 补充**：新增 `scripts/fake_camera.py`，用项目里的真实道路图按帧率发图像话题，
> **不装 Gazebo 也能验证整条链路**（见 §8.1）。同时修正了 `use_sim_time` 的默认值（§8.3）。

## 1. 这个节点解决什么问题

已有的 `cpp/yolo_cpp.exe` 证明了"单张图/单段视频能跑出框"。它缺的是**在系统里跑**：

- 输入从"文件"变成"30 fps 的实时话题流"
- 输出要变成下游能订阅的结构化消息
- 要能回答"延迟多少、抖不抖、显存多少、丢帧怎么办"

所以这个包**不重写推理**，只做三件事：**接话题、打时间戳、量化异常**。

### 验收标准（量化，不是"能启动"）

| # | 指标 | 通过线 | 怎么测 |
|---|---|---|---|
| 1 | 检测话题帧率 | ≥ 20 Hz（相机 30 fps 时不丢超过 1/3） | `ros2 topic hz /yolo_trt_node/detections` |
| 2 | `infer_ms` P99 | < 15 ms（TRT EP） | `/yolo_trt_node/latency` 的 `infer_ms_p99` |
| 3 | `dropped_stale` | 5 分钟内为 0 | `/yolo_trt_node/latency` |
| 4 | 显存增量 | < 1.5 GB | `vram_mib` 与空载对比 |
| 5 | 稳定性 | 连续 5 分钟不崩、不 OOM、`ros2 node list` 仍在 | 人工看 |

**注意第 3 条**：如果 `dropped_stale` 持续增长，说明节点处理不过来，即使 `infer_ms` 很小，
系统也是坏的。这是本节点最重要的一个设计点，见 §3。

## 2. 接口

### 2.1 话题

| 方向 | 话题 | 类型 | QoS |
|---|---|---|---|
| 订阅 | `/camera/image_raw`（可配） | `sensor_msgs/Image` | **SensorData + keep_last(1)** |
| 发布 | `~/detections` | `yolo_trt_ros/DetectionArray` | reliable, depth 10 |
| 发布 | `~/latency` | `yolo_trt_ros/LatencyReport` | SensorData + keep_last(1) |
| 发布 | `~/debug_image` | `sensor_msgs/Image` | SensorData + keep_last(1) |

用 `~/` 相对名，节点放在哪个命名空间就自动跟随，**多实例可并存**。

### 2.2 为什么输入 QoS 必须是 best_effort + depth 1

**best_effort**：Gazebo 的相机插件发布的就是 best_effort。订阅端若用默认的 reliable，
QoS 不兼容，**一条消息都收不到，而且不报错** —— 这是 ROS2 排查成本最高的一类问题。

**depth 1（关键）**：这是一条**背压**设计。

```
depth=10 时：节点慢 → 队列积压 → 排队时间线性增长
             infer_ms 依然 3 ms（看不出来），但端到端已经 300 ms
             现象：数据"都在"，系统实际不可用

depth=1  时：节点慢 → 旧帧被中间件直接丢弃 → 只处理最新帧
             现象：帧率下降，但端到端延迟不涨
```

对实时系统，**丢帧优于积压**。这也是为什么 `LatencyReport` 里要单独报 `transport_ms`
（排队+传输）而不只报 `infer_ms`。

### 2.3 消息

```yaml
# DetectionArray.msg
header          # 直接沿用输入图像的 header，保留原始时间戳
model_id        # "yolo26n"
ep              # 实际生效的后端：CPU / CUDA / TensorRT
image_width/image_height
infer_ms        # 本帧纯推理耗时
detections[]    # Detection: class_name, class_id, score, x1,y1,x2,y2
```

用自定义消息而不是 `vision_msgs/Detection2DArray`，原因：
1. `vision_msgs` 的 `ObjectHypothesis` 字段在不同发行版之间改过，直接依赖会把代码锁在某个版本上；
2. 裸的 `x1/y1/x2/y2` 比 `BoundingBox2D.center + size_x/size_y` 少一次换算，下游画框和算 IoU 都更直接；
3. **接口是自己设计的** —— 感知与规划之间怎么定接口边界，是这里最需要说清取舍的地方。

需要和 RViz2 或第三方工具互通时，再加一个 adapter 节点转 `vision_msgs` 即可，不必改核心。

### 2.4 参数

见 `config/params.yaml`。关键几项：

| 参数 | 默认 | 说明 |
|---|---|---|
| `detector` | `dummy` | `dummy` 固定框（验证通信层）／`ort` 真推理 |
| `ep` | `cpu` | `cpu` / `cuda` / `trt` |
| `image_topic` | `/camera/image_raw` | TurtleBot3 waffle 相机 |
| `max_transport_ms` | `200` | 超过则丢弃该帧（过期帧保护） |
| `use_worker_thread` | `false` | v2 优化开关，见 §6 |
| `class_names` | 12 类道路要素 | **顺序必须与训练时 `data.yaml` 一致** |

`class_names` 的顺序不能错位 —— 索引就是 `class_id`，错了不会报错，只会全部分错类。

## 3. 延迟的量法（本节点最核心的设计）

### 3.1 打点

```
t0  header.stamp        相机曝光时刻（仿真时钟）
t1  回调进入            steady_clock
t2  预处理开始
t3  推理开始（Session::Run 前）
t4  推理结束
t5  后处理结束（decode + NMS）
t6  publish 返回
```

派生指标：

| 指标 | 公式 | 时段 | 时钟 |
|---|---|---|---|
| `transport_ms` | (t1 的仿真时刻 − t0) | 相机 → 节点 | **仿真时钟** |
| `pre_ms` | t3 − t2 | letterbox + 归一化 + CHW | steady_clock |
| `infer_ms` | t4 − t3 | `Session::Run` | steady_clock |
| `post_ms` | t5 − t4 | decode + NMS | steady_clock |
| `serve_ms` | t6 − t2 | 节点内总处理 | steady_clock |

### 3.2 为什么把时钟拆成两套

在 Gazebo 里 `use_sim_time=true` 时，`node->now()` 走仿真时钟，会随仿真倍速、暂停而变。
如果拿"仿真时间戳"去减"真实 `steady_clock`"，仿真暂停时 `transport_ms` 会算出几万毫秒，
结论全废。

**launch 里 `use_sim_time` 默认 `false`**（2026-10-30 从 true 改过来）：用 fake_camera
或真实相机时系统里没有 `/clock`，开着它会让 `now()` 恒为 0。只有接 Gazebo 时才显式
`use_sim_time:=true`。见 §8.3。

所以：
- **跨进程的那一段（transport）只用仿真时钟**，两边都在同一时间轴上，有意义；
- **节点内部的各段（pre/infer/post/serve）只用 `steady_clock`**，它不受 NTP 跳变影响，
  也不会被仿真暂停拉长。

**不在指标里硬凑一个 `e2e_ms`**，因为仿真环境下它没有物理意义。真机上把两套时钟统一后
再报 `e2e`。这一条写在代码注释里：我知道自己的数字在什么前提下成立。

### 3.3 为什么报 P99 而不是均值

平均 6 ms、P99 30 ms 的系统在 30 fps 下会周期性丢帧，但**均值看起来很美**。
部署验收看尾部，`LatencyReport` 里 infer / transport 都同时给 P50/P95/P99。

## 4. 分层

```
yolo_node.cpp       ROS2 层：订阅/发布/参数/统计/异常
      ↓  只通过 IDetector 接口依赖下层
yolo_detector.cpp   推理层：ORT 会话 + 预处理 + decode + NMS
letterbox.cpp       纯几何：缩放、填充、坐标反变换、CHW 布局
latency_stats.hpp   纯统计：环形缓冲 + 最近秩分位数
```

**`yolo_detector.hpp` 与 `letterbox.hpp` 不 include 任何 `rclcpp` / `sensor_msgs`。**
这条约束带来四件事：

1. 能脱离 ROS 做 benchmark（不必启动 Gazebo）
2. 能写纯 gtest（本包含 2 个测试文件、11 个用例）
3. 换框架（Apollo / 自研 / 裸进程）时这一层整块搬走
4. 改一行只重编核心库，不重编 ROS 全栈

模块化怎么做的，答案就是这条约束，而不是"我分了文件夹"。

## 5. 异常处理矩阵

| 异常 | 检测 | 处理 | 计数 |
|---|---|---|---|
| 编码不支持 | `encoding ∉ {bgr8,rgb8,mono8}` | 丢弃 + 限频 warn | `dropped_bad_encoding` |
| 尺寸过小 | `w<32 or h<32` | 丢弃 | `dropped_bad_size` |
| 帧过期 | `transport_ms > max_transport_ms` | 丢弃 | `dropped_stale` |
| 推理抛异常 | `try/catch` 包 `infer` | 发布空结果 + 计数，**不退出** | `infer_errors` |
| 模型加载失败 | 启动时 | **抛异常退出**（无模型无法工作） | — |
| 长时空检测 | 连续 >30 帧 | 限频 warn：域漂移，不是故障 | `empty_streak_max` |
| 后端静默回落 | 启动自检 | warn | — |

**两条设计取舍要说清：**

- **模型加载失败就退出，推理失败不退出**。前者说明配置错了，早崩早发现；后者是运行时
  偶发问题，机器人上不该因为一帧推理出错让整个系统塌。
- **"长时空检测"只 warn 不报错**。因为你已知这个模型是道路场景训练的、跑在 Gazebo 里，
  检测不到是**预期结果**，不是 bug。把它当错误处理会让人误以为节点坏了。
  这个区分本身就是"懂自己系统边界"的表现。

### 后端静默回落自检

`OrtDetector::sanity_check()` 在启动时跑 25 帧空白图，若请求 TRT 但实测 > 15 ms
（或请求 CUDA 但 > 25 ms），打印警告。

存在的理由来自实际踩过的坑：缺 `nvrtc` / `cudnn_*` 时 ORT 会**静默回落到 CUDA 或 CPU**，
不报错、不抛异常，只是变慢。只看"加载成功"会误以为 TRT 生效了。

## 6. v1 → v2：加线程

`use_worker_thread=false`（v1）：回调里直接处理，积压交给 DDS 的 depth=1 丢弃。

`use_worker_thread=true`（v2）：回调只把 `shared_ptr<Image>` 存进**单槽邮箱**
（新帧覆盖未处理的旧帧），worker 线程取最新帧处理。用 `mutex + condition_variable`。

**为什么用"覆盖式单槽"而不是"队列"**：队列会积压，单槽永远只保留最新，语义与 depth=1 一致。

**v2 的预期结论可能反直觉**：如果瓶颈是推理本身，加线程**不会降低端到端延迟**，
只是把积压从 DDS 挪到自己的邮箱里，`queue_ms` 换个地方出现。真正的收益是
**解耦了接收与处理**，让节点在推理卡顿时仍能更新"最新帧"。这个对比本身就是一次
有价值的实验 —— 也是 2027.12 要手写的那个线程池的 1/N 版本。

## 7. 分阶段实施（每阶段单独验收，不要跳）

| 阶段 | 内容 | 验收 | 依赖 |
|---|---|---|---|
| **A1** | `detector:=dummy` 跑通通信层 | `ros2 topic echo /yolo_trt_node/detections` 有 1 个固定框（640×640 输入时 x1=240/y1=240/x2=400/y2=400）；`debug_image` 里框在画面中心 | 只需 rclcpp |
| **A2** | `detector:=ort ep:=cpu` | 出真实框；`infer_ms` ≈ 15–25 ms | 需 ORT |
| **A3** | `ep:=cuda` | `ep=CUDA`，`infer_ms` ≈ 5–8 ms | 需 CUDA EP |
| **A4** | `ep:=trt` | `ep=TensorRT`，`infer_ms` ≈ 3 ms；首次构建 engine 约 10 分钟 | 需 TRT + 依赖齐全 |
| **A5** | `use_worker_thread:=true` | 给出 v1/v2 的 `serve_ms` / 丢帧数对比 | A2 以上 |

**A1 为什么必须先做**：它把"ROS2 接口对不对"和"推理对不对"两件事分开。
一上来就接真推理，出问题时你不知道是 QoS 配错还是模型路径错还是张量布局错，
排查时间会翻好几倍。

**A1 的预期**：Gazebo 里 dummy 框固定画在画面中心 —— 这一步能同时验证
坐标变换、画框、消息发布三件事，而且**不依赖任何 GPU 组件**。

## 8. 编译与运行

```bash
# 1) 建工作区，把本包软链进去（源码留在 Windows 那侧，单一来源）
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
ln -sfn "/mnt/d/VC Code/YOLO_Road_Detection/ros2/src/yolo_trt_ros" yolo_trt_ros

# 2) （可选）准备 ONNX Runtime 的 C++ 头；pip 包只带 .so 不带头
cd ~ && curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar xf onnxruntime-linux-x64-1.22.0.tgz
export ORT_ROOT=$HOME/onnxruntime-linux-x64-1.22.0

# 3) 编译
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select yolo_trt_ros --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

# 4) 单元测试（不需要 ROS 运行时）
colcon test --packages-select yolo_trt_ros && colcon test-result --verbose

# 5) 起一路图像源（二选一，见 8.1）——另开终端
#    5a. 推荐：用项目里的真实道路图，不需要 Gazebo
python3 ~/ros2_ws/src/yolo_trt_ros/scripts/fake_camera.py --fps 10
#    5b. 或者：Gazebo + TurtleBot3（要几 GB 依赖 + 图形加速）
# export TURTLEBOT3_MODEL=waffle
# ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# 6) 起节点 —— 先 A1
ros2 launch yolo_trt_ros yolo_trt.launch.py detector:=dummy

# 7) 验证（注意话题名带节点名，见 8.2）
ros2 topic hz /yolo_trt_node/detections
ros2 topic echo /yolo_trt_node/latency --once
```

### 8.1 两种图像源怎么选

| 方案 | 依赖 | 适合什么 |
|---|---|---|
| **`scripts/fake_camera.py`** | 只要 rclpy + OpenCV | **默认选这个。** 不装 Gazebo、帧率可控、画面是模型真正认识的场景（187 张验证图） |
| Gazebo + TurtleBot3 | 几 GB + WSLg 图形加速 | 需要机器人本体和运动时。WSL2 里 3D 渲染可能很卡，甚至起不来 |

`fake_camera.py` 读 `datasets/road-traffic-detect/valid/images`，按 `--fps` 发到 `/camera/image_raw`，
循环播放。常用参数：

```bash
python3 scripts/fake_camera.py --fps 30            # 目标帧率
python3 scripts/fake_camera.py --size 960x540      # 缩放到非方形，用来验证 letterbox 的 pad
python3 scripts/fake_camera.py --max-frames 300    # 发够就停
```

**为什么它用 best_effort + depth 1 发布**：本节点订阅的是 `SensorDataQoS().keep_last(1)`。
DDS 的规则是「发布 best_effort、订阅 reliable」会**一条都收不到且不报错**，
所以这里故意取更保守的一侧。

⚠️ **640×640 的图验证不出 letterbox 的 bug**：`scale = min(640/640, 640/640) = 1.0`、`pad = 0`，
整个坐标变换退化成恒等映射。要验 pad/scale 必须给非方形输入（`--size 960x540`，
此时 `pad_y = 140`；若框整体上下偏 140 像素就是 `pad_y` 错了）。

### 8.2 话题名带节点名（最容易猜错的一步）

发布用的是相对名 `~/detections`，`~/` 在 ROS2 里展开成 `<命名空间>/<节点名>/`。
本节点名是 `yolo_trt_node` 且没设命名空间，所以真实话题是：

```
/yolo_trt_node/detections
/yolo_trt_node/latency
/yolo_trt_node/debug_image
```

**不是 `/yolo/detections`。** 猜错的时候 `ros2 topic hz` 会一直打印
`does not appear to be published yet` 然后永远等下去，**不报错**。

拿不准就不要猜：

```bash
ros2 topic list | grep detections
ros2 node info /yolo_trt_node        # 列出这个节点全部发布/订阅及完整话题名
```

用相对名（而不是写死绝对名）是有意的：起两个不同命名空间的实例时，
话题会自动分开、互不干扰。

### 8.3 `use_sim_time` 什么时候该开

**默认 `false`**（用 fake_camera 或真实相机时）。只有系统里**有 `/clock` 发布者**
（也就是接 Gazebo）时才设 `true`。

开了 `true` 但收不到 `/clock` 的后果很好认：
`node->now()` 恒为 0 → `transport_ms = (0 − 图像时间戳)` 为负 → 全部被当成过期帧，
现象是 **`frames_total` 在涨、`dropped_stale` 也在涨、但 `fps` 一直是 0**。

```bash
ros2 param get /yolo_trt_node use_sim_time     # 确认为 false
```

`ROS_DOMAIN_ID` 若同时跑多个仿真，记得分开，否则话题会串。

## 9. 已知限制

1. **A4 TensorRT 未在 WSL 跑通**：ORT 的 TRT EP 需要 `libnvinfer.so.10`，
   该运行库未装（见开头状态与 `docs/ros2/A4_tensorrt.json`）。Windows 侧 TRT 已完整验证。
2. `OrtDetector::load` 用的是 `model_path.c_str()`（窄字符）。**Linux 下正确，Windows 下需换
   `std::wstring`**。本包只面向 WSL2，不打算跨到 Windows。
3. NMS 是 O(n²) 朴素实现，与 `cpp/` 里的版本一致；候选框几十个时够用。
4. 预处理用 OpenCV 的 `cv::resize`，没有搬 `cpp/` 里手写的那套双线性。
   理由：手写版的价值是"证明我懂"（已在 CLI 里证明过），节点这里要的是**少出错**。
   两者等价性由 `scripts/13_compare_cpp_backends.py` 在 CLI 侧保证。
5. `read_vram` 每秒 popen 一次 `nvidia-smi`。追求极简可以关掉，但那样就没有显存数字了。
6. 只支持方形输入（`input_size`），不支持动态 batch。

## 10. 与你另外两个项目的关系

```
YOLO 训练 (mAP50 0.680)
  └─ cpp/  C++ ORT 单图/视频推理  (已有，fp32/fp16 engine)
       └─ ros2/  本包：接进 ROS2 实时系统  (A 阶段)
            └─ 后续：加深度反投影 + 3D 定位 → 与 Nav2 costmap 对接  (A3，暂不做)
```

简历上这**是一条线**，不是三个项目。讲述顺序：训练 → 部署 → 系统集成。
