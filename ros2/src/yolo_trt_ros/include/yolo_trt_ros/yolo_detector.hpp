#pragma once

// 推理层接口 —— 这一族头文件不 include 任何 rclcpp / sensor_msgs。
// 好处：可脱离 ROS 单独 benchmark、可写纯 gtest、换框架时整层搬走。

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "yolo_trt_ros/detection.hpp"

namespace yolo_trt_ros {

class IDetector {
 public:
  virtual ~IDetector() = default;

  // 加载模型。失败返回 false 并把原因写进 err。
  virtual bool load(const std::string& model_path,
                    EpKind ep,
                    const std::string& trt_cache_dir,
                    int device_id,
                    int intra_threads,
                    std::string* err) = 0;

  // 输入 BGR 三通道原图，返回原图坐标系下的框 + 分段耗时。
  virtual DetectorOutput infer(const cv::Mat& bgr,
                               float conf_thres,
                               float iou_thres) = 0;

  // "实际生效"的后端。用于把静默回落暴露出来。
  virtual EpKind actual_ep() const = 0;

  // 模型输入边长（正方形）
  virtual int input_size() const = 0;

  virtual std::string model_id() const = 0;

  // 启动后跑几十帧，若 infer_ms 明显偏离目标区间则返回一句警告（空串表示正常）。
  // 存在的理由：ORT 的 TensorRT EP 缺 nvrtc 之类依赖时会静默回落，
  // 只用返回值判断会误以为 TRT 生效了。
  virtual std::string sanity_check() = 0;
};

// A0/A1 阶段用的假检测器：不依赖 ONNX Runtime，固定输出一个框。
// 用途是把"ROS2 通信层是否正确"与"推理是否正确"两件事分开验证。
std::unique_ptr<IDetector> make_dummy_detector(int input_size, int num_classes);

// 真检测器（需编译时找到 ONNX Runtime）
std::unique_ptr<IDetector> make_ort_detector(
    const std::vector<std::string>& class_names);

}  // namespace yolo_trt_ros
