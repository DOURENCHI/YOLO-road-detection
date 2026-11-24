# ros2/ — 一个独立的 colcon 工作区根

这个目录**与同级 `cpp/` 互不影响**：

```
cpp/            已有的 C++ 单图/视频推理（Windows + MSYS2，已验证可用）
ros2/           ROS2 节点（WSL2 + Ubuntu 22.04 + ROS2 Humble）
  └── src/yolo_trt_ros/     唯一的包
```

`cpp/` 里的 `yolo_cpp.exe` 不参与 ROS2 编译，也不被本工作区改动。
两条线共享同一份 `../weights/best.onnx`。

## 为什么单独一个工作区

- `cpp/` 用 MSYS2 UCRT64 + MinGW Makefiles，在 Windows 侧；
- `ros2/` 用 WSL2 里的 GCC + colcon，在 Linux 侧。

两边的工具链、依赖、构建方式完全不同，混在一个 CMake 工程里只会互相拖累。
共享的是**模型权重**和**项目叙事**，不是构建系统。

## 快速开始

```bash
# WSL2 里
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
ln -sfn "/mnt/d/VC Code/YOLO_Road_Detection/ros2/src/yolo_trt_ros" yolo_trt_ros
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select yolo_trt_ros --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

> `-DCMAKE_BUILD_TYPE=Release` **不能省**。不加是默认无优化，推理会慢几倍，
> 而且你会误以为"ROS2 里比单机慢"。

## 先跑通通信层，再接真模型

不需要装 Gazebo：`scripts/fake_camera.py` 会把项目里的 187 张道路图按帧率发成图像话题。

```bash
# 终端 1
python3 ~/ros2_ws/src/yolo_trt_ros/scripts/fake_camera.py --fps 10
# 终端 2
ros2 launch yolo_trt_ros yolo_trt.launch.py detector:=dummy
# 终端 3
ros2 topic hz /yolo_trt_node/detections
```

细节、分阶段验收（A1–A5）、话题名为什么带节点名、`use_sim_time` 怎么设，
见 `src/yolo_trt_ros/README.md` §7–§8。
