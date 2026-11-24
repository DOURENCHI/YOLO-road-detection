// 推理层实现。
// 本文件不 include 任何 ROS 头 —— 可以脱离 ROS 单独编译与 benchmark。

#include "yolo_trt_ros/yolo_detector.hpp"

#include "yolo_trt_ros/letterbox.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

// ⚠️ ORT 头必须在 namespace yolo_trt_ros 之外 include。
// 它内部大量使用 std::xxx，若在命名空间内 include，会被解析成 yolo_trt_ros::std::xxx，
// 导致 /usr/include/c++/11/variant 报几十条 "expected template-name" 错误。
#ifdef YOLO_TRT_ROS_WITH_ORT
#include <array>
#include <memory>
#include <unordered_map>
#include "onnxruntime_cxx_api.h"
#endif

namespace yolo_trt_ros {
namespace {

using Clock = std::chrono::steady_clock;

inline float ms_since(Clock::time_point t0) {
  return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
}

float iou(const Detection& a, const Detection& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float w = std::max(0.f, xx2 - xx1);
  const float h = std::max(0.f, yy2 - yy1);
  const float inter = w * h;
  const float area_a = std::max(0.f, a.width()) * std::max(0.f, a.height());
  const float area_b = std::max(0.f, b.width()) * std::max(0.f, b.height());
  const float uni = area_a + area_b - inter;
  return uni > 0.f ? inter / uni : 0.f;
}

// 按类别做 NMS（与 ultralytics 默认 agnostic=False 一致）
std::vector<Detection> nms_per_class(std::vector<Detection> in, float iou_thres) {
  std::vector<Detection> out;
  std::vector<int> classes;
  for (const auto& d : in) {
    if (std::find(classes.begin(), classes.end(), d.class_id) == classes.end())
      classes.push_back(d.class_id);
  }
  for (int c : classes) {
    std::vector<Detection> group;
    for (const auto& d : in)
      if (d.class_id == c) group.push_back(d);
    std::sort(group.begin(), group.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<char> dead(group.size(), 0);
    for (std::size_t i = 0; i < group.size(); ++i) {
      if (dead[i]) continue;
      out.push_back(group[i]);
      for (std::size_t j = i + 1; j < group.size(); ++j) {
        if (!dead[j] && iou(group[i], group[j]) > iou_thres) dead[j] = 1;
      }
    }
  }
  std::sort(out.begin(), out.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  return out;
}

// ---------------------------------------------------------------- 假检测器

class DummyDetector : public IDetector {
 public:
  DummyDetector(int input_size, int num_classes)
      : input_size_(input_size), num_classes_(num_classes) {}

  bool load(const std::string&, EpKind, const std::string&, int, int,
            std::string* err) override {
    if (err) err->clear();
    return true;
  }

  DetectorOutput infer(const cv::Mat& bgr, float, float) override {
    DetectorOutput o;
    o.image_width = bgr.cols;
    o.image_height = bgr.rows;
    // 固定框：图像中心 1/4 大小，方便肉眼确认坐标变换与画框是对的
    Detection d;
    d.class_id = 0;
    d.class_name = "dummy";
    d.score = 0.99f;
    d.x1 = bgr.cols * 0.375f;
    d.y1 = bgr.rows * 0.375f;
    d.x2 = bgr.cols * 0.625f;
    d.y2 = bgr.rows * 0.625f;
    o.detections.push_back(d);
    o.pre_ms = 0.f;
    o.infer_ms = 0.f;
    o.post_ms = 0.f;
    return o;
  }

  EpKind actual_ep() const override { return EpKind::kCpu; }
  int input_size() const override { return input_size_; }
  std::string model_id() const override { return "dummy"; }
  std::string sanity_check() override { return {}; }

 private:
  int input_size_ = 640;
  int num_classes_ = 12;
};

}  // namespace

std::unique_ptr<IDetector> make_dummy_detector(int input_size, int num_classes) {
  return std::make_unique<DummyDetector>(input_size, num_classes);
}

// ---------------------------------------------------------------- ORT 检测器

#ifdef YOLO_TRT_ROS_WITH_ORT

namespace {

class OrtDetector : public IDetector {
 public:
  explicit OrtDetector(std::vector<std::string> class_names)
      : class_names_(std::move(class_names)) {}
  ~OrtDetector() override = default;

  bool load(const std::string& model_path, EpKind ep, const std::string& trt_cache_dir,
            int device_id, int intra_threads, std::string* err) override {
    try {
      requested_ep_ = ep;
      env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "yolo_trt_ros");

      Ort::SessionOptions so;
      so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      if (intra_threads > 0) so.SetIntraOpNumThreads(intra_threads);

      // ⚠️ ORT 1.22 的 C++ API：不能用字符串版 AppendExecutionProvider("CUDA"/"Tensorrt")。
      // 那个接口只认识 DML/QNN/OpenVINO/Azure/CoreML 等，传 "CUDA" 会直接抛
      // "Unknown provider name 'CUDA'"，启动时 FATAL。
      // 正确写法与 Windows 侧 cpp/src/main.cpp 一致：用类型化 V2 options + V2 挂载函数。
      OrtApi const& ort_api = Ort::GetApi();

      if (ep == EpKind::kCuda) {
        OrtCUDAProviderOptionsV2* cuda_opts = nullptr;
        try {
          Ort::ThrowOnError(ort_api.CreateCUDAProviderOptions(&cuda_opts));
          const std::string dev = std::to_string(device_id);
          const char* keys[] = {"device_id"};
          const char* vals[] = {dev.c_str()};
          Ort::ThrowOnError(ort_api.UpdateCUDAProviderOptions(cuda_opts, keys, vals, 1));
          Ort::ThrowOnError(ort_api.SessionOptionsAppendExecutionProvider_CUDA_V2(so, cuda_opts));
          actual_ep_ = EpKind::kCuda;
        } catch (const std::exception& e) {
          // CUDA 挂不上就回落 CPU，不让节点直接死掉
          if (err) {
            *err = std::string("CUDA EP 挂载失败，回落 CPU: ") + e.what();
          }
          actual_ep_ = EpKind::kCpu;
        }
        if (cuda_opts) ort_api.ReleaseCUDAProviderOptions(cuda_opts);
      } else if (ep == EpKind::kTensorRT) {
        OrtTensorRTProviderOptionsV2* trt_opts = nullptr;
        try {
          Ort::ThrowOnError(ort_api.CreateTensorRTProviderOptions(&trt_opts));
          const std::string dev = std::to_string(device_id);
          const char* keys[] = {"device_id", "trt_fp16_enable", "trt_engine_cache_enable",
                                "trt_engine_cache_path", "trt_timing_cache_enable"};
          const char* vals[] = {dev.c_str(), "1", "1", trt_cache_dir.c_str(), "1"};
          Ort::ThrowOnError(ort_api.UpdateTensorRTProviderOptions(trt_opts, keys, vals, 5));
          Ort::ThrowOnError(ort_api.SessionOptionsAppendExecutionProvider_TensorRT_V2(so, trt_opts));
          actual_ep_ = EpKind::kTensorRT;
        } catch (const std::exception& e) {
          // TRT 挂不上先回落 CUDA；CUDA 也挂不上则在 session 创建时走 CPU
          if (err) {
            *err = std::string("TensorRT EP 挂载失败，尝试 CUDA: ") + e.what();
          }
          OrtCUDAProviderOptionsV2* cuda_opts = nullptr;
          try {
            Ort::ThrowOnError(ort_api.CreateCUDAProviderOptions(&cuda_opts));
            const std::string dev = std::to_string(device_id);
            const char* keys[] = {"device_id"};
            const char* vals[] = {dev.c_str()};
            Ort::ThrowOnError(ort_api.UpdateCUDAProviderOptions(cuda_opts, keys, vals, 1));
            Ort::ThrowOnError(ort_api.SessionOptionsAppendExecutionProvider_CUDA_V2(so, cuda_opts));
            actual_ep_ = EpKind::kCuda;
          } catch (const std::exception&) {
            actual_ep_ = EpKind::kCpu;
          }
          if (cuda_opts) ort_api.ReleaseCUDAProviderOptions(cuda_opts);
        }
        if (trt_opts) ort_api.ReleaseTensorRTProviderOptions(trt_opts);
      } else {
        actual_ep_ = EpKind::kCpu;
      }

      session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), so);

      Ort::AllocatorWithDefaultOptions alloc;
      auto in_name = session_->GetInputNameAllocated(0, alloc);
      auto out_name = session_->GetOutputNameAllocated(0, alloc);
      input_name_ = in_name.get();
      output_name_ = out_name.get();

      auto in_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
      if (in_shape.size() == 4 && in_shape[2] > 0)
        input_size_ = static_cast<int>(in_shape[2]);
      return true;
    } catch (const std::exception& e) {
      if (err) *err = e.what();
      if (requested_ep_ != EpKind::kCpu) {
        if (err) *err += "  | 建议先降到 ep:=cpu 验证链路，再逐级上 CUDA / TensorRT";
      }
      return false;
    }
  }

  DetectorOutput infer(const cv::Mat& bgr, float conf_thres, float iou_thres) override {
    DetectorOutput o;
    o.image_width = bgr.cols;
    o.image_height = bgr.rows;

    auto t_pre = Clock::now();
    const Letterbox lb = make_letterbox(bgr, input_size_, /*rgb=*/true);
    blob_.resize(static_cast<std::size_t>(3) * input_size_ * input_size_);
    to_blob(lb.canvas, blob_.data());
    std::array<int64_t, 4> shape{1, 3, input_size_, input_size_};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        mem, blob_.data(), blob_.size(), shape.data(), shape.size());
    o.pre_ms = ms_since(t_pre);

    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};

    auto t_infer = Clock::now();
    auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
    o.infer_ms = ms_since(t_infer);

    auto t_post = Clock::now();
    auto info = outs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> os = info.GetShape();
    if (os.size() != 3) {
      o.post_ms = ms_since(t_post);
      return o;  // 形状不符：返回空结果而不是抛异常，让上层按"空检测"处理
    }
    const int ch = static_cast<int>(os[1]);
    const int na = static_cast<int>(os[2]);
    const int nc = ch - 4;
    if (nc <= 0) {
      o.post_ms = ms_since(t_post);
      return o;
    }
    const float* d = outs[0].GetTensorMutableData<float>();

    std::vector<Detection> cand;
    for (int a = 0; a < na; ++a) {
      float best = -1.f;
      int best_c = -1;
      for (int c = 0; c < nc; ++c) {
        const float s = d[static_cast<std::size_t>(4 + c) * na + a];
        if (s > best) {
          best = s;
          best_c = c;
        }
      }
      if (best < conf_thres || best_c < 0) continue;
      const float cx = d[static_cast<std::size_t>(0) * na + a];
      const float cy = d[static_cast<std::size_t>(1) * na + a];
      const float w = d[static_cast<std::size_t>(2) * na + a];
      const float h = d[static_cast<std::size_t>(3) * na + a];

      Detection det;
      det.class_id = best_c;
      det.class_name = (best_c < static_cast<int>(class_names_.size())) ? class_names_[best_c]
                                                                       : std::to_string(best_c);
      det.score = best;
      // 先回到原图坐标，再做 NMS —— 这样 iou_thres 的语义与原图尺度一致
      det.x1 = cx - w * 0.5f;
      det.y1 = cy - h * 0.5f;
      det.x2 = cx + w * 0.5f;
      det.y2 = cy + h * 0.5f;
      unletterbox(&det.x1, &det.y1, &det.x2, &det.y2, lb);
      // unletterbox 故意不裁剪：落在填充区的框会算出负坐标，这里统一 clamp
      det.x1 = std::max(0.f, det.x1);
      det.y1 = std::max(0.f, det.y1);
      det.x2 = std::min(static_cast<float>(bgr.cols), det.x2);
      det.y2 = std::min(static_cast<float>(bgr.rows), det.y2);
      if (det.x2 <= det.x1 || det.y2 <= det.y1) continue;
      cand.push_back(det);
    }
    o.detections = nms_per_class(std::move(cand), iou_thres);
    o.post_ms = ms_since(t_post);
    return o;
  }

  EpKind actual_ep() const override { return actual_ep_; }
  int input_size() const override { return input_size_; }
  std::string model_id() const override { return "yolo26n"; }

  std::string sanity_check() override {
    if (requested_ep_ == EpKind::kCpu) return {};
    // 用一帧空白图跑 20 次，看延迟量级是否与所请求后端相符。
    // 这是为了抓"缺 nvrtc / cudnn 时静默回落"——它不报错，只是变慢。
    cv::Mat blank(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    for (int i = 0; i < 5; ++i) infer(blank, 1.0f, 0.45f);
    float sum = 0.f;
    for (int i = 0; i < 20; ++i) sum += infer(blank, 1.0f, 0.45f).infer_ms;
    const float avg = sum / 20.f;
    const float limit = (requested_ep_ == EpKind::kTensorRT) ? 15.f : 25.f;
    if (avg > limit) {
      char buf[256];
      std::snprintf(buf, sizeof(buf),
                    "请求后端 %s 但实测推理 %.2f ms 超过 %.0f ms，很可能已静默回落，"
                    "请确认 nvrtc / cudnn / nvinfer 是否齐全",
                    ep_to_string(requested_ep_), avg, limit);
      return std::string(buf);
    }
    return {};
  }

 private:
  std::vector<std::string> class_names_;
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
  std::vector<float> blob_;
  int input_size_ = 640;
  EpKind requested_ep_ = EpKind::kCpu;
  EpKind actual_ep_ = EpKind::kCpu;
};

}  // namespace

std::unique_ptr<IDetector> make_ort_detector(const std::vector<std::string>& class_names) {
  return std::make_unique<OrtDetector>(class_names);
}

#else  // !YOLO_TRT_ROS_WITH_ORT

std::unique_ptr<IDetector> make_ort_detector(const std::vector<std::string>&) {
  throw std::runtime_error(
      "本包编译时未找到 ONNX Runtime（未定义 YOLO_TRT_ROS_WITH_ORT）。"
      "请设置 ORT_ROOT 后重新 colcon build，或使用 detector:=dummy 验证通信层。");
}

#endif

}  // namespace yolo_trt_ros
