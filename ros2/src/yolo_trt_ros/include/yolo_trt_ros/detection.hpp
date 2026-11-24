#pragma once

#include <string>
#include <vector>

namespace yolo_trt_ros {

// 一个检测框，坐标已反变换回原图（像素，左上为原点）
struct Detection {
  int class_id = -1;
  std::string class_name;
  float score = 0.f;
  float x1 = 0.f;
  float y1 = 0.f;
  float x2 = 0.f;
  float y2 = 0.f;

  float width() const { return x2 - x1; }
  float height() const { return y2 - y1; }
};

// 一帧的完整输出，含分段耗时。
// 耗时由检测器自己记录，节点层不再插桩，避免"测的是节点开销还是推理开销"混淆。
struct DetectorOutput {
  std::vector<Detection> detections;
  int image_width = 0;
  int image_height = 0;
  float pre_ms = 0.f;
  float infer_ms = 0.f;
  float post_ms = 0.f;
};

enum class EpKind { kCpu, kCuda, kTensorRT };

inline const char* ep_to_string(EpKind ep) {
  switch (ep) {
    case EpKind::kCpu:
      return "CPU";
    case EpKind::kCuda:
      return "CUDA";
    case EpKind::kTensorRT:
      return "TensorRT";
  }
  return "Unknown";
}

inline bool parse_ep(const std::string& s, EpKind* out) {
  if (!out) return false;
  if (s == "cpu" || s == "CPU") {
    *out = EpKind::kCpu;
    return true;
  }
  if (s == "cuda" || s == "CUDA") {
    *out = EpKind::kCuda;
    return true;
  }
  if (s == "trt" || s == "TRT" || s == "tensorrt" || s == "TensorRT") {
    *out = EpKind::kTensorRT;
    return true;
  }
  return false;
}

}  // namespace yolo_trt_ros
