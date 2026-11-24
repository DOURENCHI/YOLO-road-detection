// ROS2 层。唯一知道 ROS 存在的文件。
//
// 三个设计决定（也是评审时最容易被问到的）：
// 1) 输入 QoS 用 SensorData + depth=1。best_effort 是为了和 Gazebo 相机对得上；
//    depth=1 是背压 —— 节点慢时自动丢旧帧，避免 queue_ms 线性增长而 infer_ms 看不出来。
// 2) 延迟拆成 transport / pre / infer / post / serve 五段分别统计，不只报一个 e2e。
//    任何一段异常都能被单独定位。
// 3) 仿真时钟与真实时钟分开用：transport 用仿真时间（两边都可对齐），
//    其余用 steady_clock（不受 NTP 跳变影响，也不会被仿真暂停拉长）。

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#include "yolo_trt_ros/msg/detection.hpp"
#include "yolo_trt_ros/msg/detection_array.hpp"
#include "yolo_trt_ros/msg/latency_report.hpp"

#include "yolo_trt_ros/latency_stats.hpp"
#include "yolo_trt_ros/yolo_detector.hpp"

namespace yolo_trt_ros {
namespace {

using Clock = std::chrono::steady_clock;

inline float ms_since(Clock::time_point t0) {
  return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
}

// 直接从 sensor_msgs/Image 包一层 cv::Mat，不引入 cv_bridge。
// 理由：Gazebo 相机只给 bgr8 / rgb8 / mono8 三种编码，手写 15 行比多一个编译依赖划算，
// 而且少一层黑盒，出问题时能一眼看到 stride 怎么算的。
bool wrap_image(const sensor_msgs::msg::Image& msg, cv::Mat* bgr, std::string* err) {
  const std::string& enc = msg.encoding;
  if (msg.width < 32 || msg.height < 32) {
    if (err) *err = "size_too_small";
    return false;
  }
  if (enc == "bgr8" || enc == "rgb8") {
    cv::Mat view(static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC3,
                 const_cast<uint8_t*>(msg.data.data()), static_cast<std::size_t>(msg.step));
    if (enc == "rgb8") {
      cv::cvtColor(view, *bgr, cv::COLOR_RGB2BGR);
    } else {
      *bgr = view;  // 浅拷贝，生命周期由 msg 的 shared_ptr 保证
    }
    return true;
  }
  if (enc == "mono8") {
    cv::Mat view(static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC1,
                 const_cast<uint8_t*>(msg.data.data()), static_cast<std::size_t>(msg.step));
    cv::cvtColor(view, *bgr, cv::COLOR_GRAY2BGR);
    return true;
  }
  if (err) *err = "unsupported_encoding:" + enc;
  return false;
}

cv::Scalar class_color(int id) {
  // 稳定配色：同一 class_id 每次颜色一致，方便肉眼比对多帧
  return cv::Scalar(60 + (id * 97) % 180, 60 + (id * 53) % 180, 60 + (id * 151) % 180);
}

std::uint32_t read_vram_mib() {
  std::FILE* f = popen(
      "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
  if (!f) return 0;
  int v = 0;
  if (std::fscanf(f, "%d", &v) != 1) v = 0;
  pclose(f);
  return v > 0 ? static_cast<std::uint32_t>(v) : 0;
}

}  // namespace

class YoloNode : public rclcpp::Node {
 public:
  YoloNode() : rclcpp::Node("yolo_trt_node") {
    declare_params();
    init_detector();

    const auto qos_in = rclcpp::SensorDataQoS().keep_last(1);
    sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_, qos_in,
        [this](sensor_msgs::msg::Image::ConstSharedPtr m) { on_image(std::move(m)); });

    pub_det_ = create_publisher<msg::DetectionArray>("~/detections", rclcpp::QoS(10));
    pub_lat_ = create_publisher<msg::LatencyReport>(
        "~/latency", rclcpp::SensorDataQoS().keep_last(1));
    if (publish_debug_image_) {
      pub_dbg_ = create_publisher<sensor_msgs::msg::Image>(
          "~/debug_image", rclcpp::SensorDataQoS().keep_last(1));
    }

    if (use_worker_thread_) {
      worker_ = std::thread([this] { worker_loop(); });
    }

    const double period = 1.0 / std::max(0.1, display_rate_hz_);
    timer_ = create_wall_timer(std::chrono::duration<double>(period),
                               [this] { report(); });

    RCLCPP_INFO(get_logger(),
                "yolo_trt_node ready | detector=%s ep=%s input=%d classes=%zu "
                "worker_thread=%s image=%s",
                detector_kind_.c_str(), ep_to_string(detector_->actual_ep()),
                detector_->input_size(), class_names_.size(),
                use_worker_thread_ ? "on" : "off", image_topic_.c_str());

    const std::string warn = detector_->sanity_check();
    if (!warn.empty()) {
      RCLCPP_WARN(get_logger(), "启动自检：%s", warn.c_str());
    }
  }

  ~YoloNode() override {
    {
      std::lock_guard<std::mutex> lk(mbox_mtx_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

 private:
  // ------------------------------------------------------------ 参数与初始化

  void declare_params() {
    model_path_ = declare_parameter<std::string>("model_path", "");
    detector_kind_ = declare_parameter<std::string>("detector", "ort");
    const std::string ep_str = declare_parameter<std::string>("ep", "cpu");
    trt_cache_dir_ = declare_parameter<std::string>("trt_cache_dir", "/tmp/trt_cache");
    device_id_ = declare_parameter<int>("device_id", 0);
    intra_threads_ = declare_parameter<int>("intra_threads", 2);
    input_size_ = declare_parameter<int>("input_size", 640);
    num_classes_ = declare_parameter<int>("num_classes", 12);
    class_names_ = declare_parameter<std::vector<std::string>>(
        "class_names", std::vector<std::string>{"class_0"});
    conf_thres_ = static_cast<float>(declare_parameter<double>("conf_thres", 0.25));
    iou_thres_ = static_cast<float>(declare_parameter<double>("iou_thres", 0.45));
    publish_debug_image_ = declare_parameter<bool>("publish_debug_image", true);
    const int window = declare_parameter<int>("stats_window", 200);
    max_transport_ms_ = static_cast<float>(declare_parameter<double>("max_transport_ms", 200.0));
    use_worker_thread_ = declare_parameter<bool>("use_worker_thread", false);
    display_rate_hz_ = declare_parameter<double>("display_rate_hz", 1.0);
    read_vram_ = declare_parameter<bool>("read_vram", true);
    image_topic_ = declare_parameter<std::string>("image_topic", "/camera/image_raw");

    if (!parse_ep(ep_str, &ep_)) {
      RCLCPP_ERROR(get_logger(), "无法识别的 ep 参数: %s（可选 cpu / cuda / trt）", ep_str.c_str());
      throw std::invalid_argument("bad ep parameter");
    }

    st_transport_ = std::make_unique<LatencyStats>(window);
    st_pre_ = std::make_unique<LatencyStats>(window);
    st_infer_ = std::make_unique<LatencyStats>(window);
    st_post_ = std::make_unique<LatencyStats>(window);
    st_serve_ = std::make_unique<LatencyStats>(window);
  }

  void init_detector() {
    if (detector_kind_ == "dummy") {
      detector_ = make_dummy_detector(input_size_, num_classes_);
      std::string err;
      if (!detector_->load("", EpKind::kCpu, "", device_id_, intra_threads_, &err)) {
        RCLCPP_ERROR(get_logger(), "dummy 检测器初始化失败: %s", err.c_str());
      }
      return;
    }
    detector_ = make_ort_detector(class_names_);
    std::string err;
    if (!detector_->load(model_path_, ep_, trt_cache_dir_, device_id_, intra_threads_, &err)) {
      RCLCPP_ERROR(get_logger(),
                   "检测器加载失败: %s。可用 detector:=dummy + ep:=cpu 先验证通信层。",
                   err.c_str());
      throw std::runtime_error("detector load failed");
    }
    // load 成功但发生降级时也会写 err —— 必须打出来，否则"为什么 ep=CPU"无从查起。
    // 这是实测踩出来的观测性坑：CUDA 挂载失败被静默吞掉，只剩一句"疑似回落"的猜测。
    if (!err.empty()) {
      RCLCPP_WARN(get_logger(), "检测器降级: %s", err.c_str());
    }
    input_size_ = detector_->input_size();
  }

  // ------------------------------------------------------------ 数据通路

  void on_image(sensor_msgs::msg::Image::ConstSharedPtr msg) {
    const auto t_recv = Clock::now();
    ++frames_total_;

    if (use_worker_thread_) {
      {
        std::lock_guard<std::mutex> lk(mbox_mtx_);
        if (pending_) ++dropped_overrun_;  // 上一帧还没处理完就来了新的：覆盖式丢帧
        pending_ = msg;
      }
      cv_.notify_one();
      return;
    }
    process(msg, t_recv, /*from_worker=*/false);
  }

  void worker_loop() {
    while (true) {
      sensor_msgs::msg::Image::ConstSharedPtr msg;
      {
        std::unique_lock<std::mutex> lk(mbox_mtx_);
        cv_.wait(lk, [this] { return stop_ || static_cast<bool>(pending_); });
        if (stop_) return;
        msg = pending_;
        pending_.reset();
      }
      const auto t_recv = Clock::now();
      process(msg, t_recv, /*from_worker=*/true);
    }
  }

  void process(const sensor_msgs::msg::Image::ConstSharedPtr& msg,
               Clock::time_point t_recv,
               bool from_worker) {
    const auto t_serve_begin = Clock::now();

    // ---- 传输延迟（仿真时钟）
    float transport_ms = -1.f;
    const auto& stamp = msg->header.stamp;
    if (stamp.sec != 0 || stamp.nanosec != 0) {
      const double dt = (now() - rclcpp::Time(stamp)).seconds() * 1000.0;
      transport_ms = dt >= 0.0 ? static_cast<float>(dt) : 0.f;
      if (max_transport_ms_ > 0.f && transport_ms > max_transport_ms_) {
        ++dropped_stale_;
        return;  // 过期帧不处理：处理它只会让延迟雪上加霜
      }
    }

    // ---- 解包
    cv::Mat bgr;
    std::string err;
    if (!wrap_image(*msg, &bgr, &err)) {
      if (err.rfind("size_too_small", 0) == 0) {
        ++dropped_bad_size_;
      } else {
        ++dropped_bad_encoding_;
        // 限频打印：坏编码可能每帧都来，不限频会把日志刷爆
        if (dropped_bad_encoding_ % 100 == 1) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "丢弃图像: %s", err.c_str());
        }
      }
      return;
    }

    // ---- 推理
    DetectorOutput out;
    try {
      out = detector_->infer(bgr, conf_thres_, iou_thres_);
    } catch (const std::exception& e) {
      ++infer_errors_;
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 3000, "推理异常: %s", e.what());
      out.image_width = bgr.cols;
      out.image_height = bgr.rows;
    }

    const float serve_ms = ms_since(t_serve_begin);

    // ---- 统计
    if (transport_ms >= 0.f) st_transport_->add(transport_ms);
    st_pre_->add(out.pre_ms);
    st_infer_->add(out.infer_ms);
    st_post_->add(out.post_ms);
    st_serve_->add(serve_ms);

    if (out.detections.empty()) {
      ++empty_streak_;
      empty_streak_max_ = std::max(empty_streak_max_, empty_streak_);
    } else {
      empty_streak_ = 0;
    }

    publish_detections(*msg, out);
    if (pub_dbg_ && publish_debug_image_) publish_debug(*msg, bgr, out);

    if (from_worker) ++processed_by_worker_;
    ++processed_;
  }

  void publish_detections(const sensor_msgs::msg::Image& src, const DetectorOutput& out) {
    msg::DetectionArray m;
    m.header = src.header;  // 保留原始时间戳，下游才能算真正的端到端
    m.model_id = detector_->model_id();
    m.ep = ep_to_string(detector_->actual_ep());
    m.image_width = static_cast<std::uint32_t>(out.image_width);
    m.image_height = static_cast<std::uint32_t>(out.image_height);
    m.infer_ms = out.infer_ms;
    m.detections.reserve(out.detections.size());
    for (const auto& d : out.detections) {
      msg::Detection dm;
      dm.class_name = d.class_name;
      dm.class_id = d.class_id;
      dm.score = d.score;
      dm.x1 = d.x1;
      dm.y1 = d.y1;
      dm.x2 = d.x2;
      dm.y2 = d.y2;
      m.detections.push_back(std::move(dm));
    }
    pub_det_->publish(m);
  }

  void publish_debug(const sensor_msgs::msg::Image& src, const cv::Mat& bgr,
                     const DetectorOutput& out) {
    cv::Mat canvas = bgr.clone();
    for (const auto& d : out.detections) {
      const cv::Rect r(cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                       cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)));
      cv::rectangle(canvas, r, class_color(d.class_id), 2);
      char label[128];
      std::snprintf(label, sizeof(label), "%s %.2f", d.class_name.c_str(), d.score);
      cv::putText(canvas, label, cv::Point(r.x, std::max(12, r.y - 5)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, class_color(d.class_id), 1, cv::LINE_AA);
    }
    cv::putText(canvas,
                "ep=" + std::string(ep_to_string(detector_->actual_ep())) +
                    " infer=" + std::to_string(static_cast<int>(out.infer_ms)) + "ms",
                cv::Point(8, canvas.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

    sensor_msgs::msg::Image im;
    im.header = src.header;
    im.height = static_cast<std::uint32_t>(canvas.rows);
    im.width = static_cast<std::uint32_t>(canvas.cols);
    im.encoding = "bgr8";
    im.is_bigendian = 0;
    im.step = static_cast<std::uint32_t>(canvas.cols * 3);
    im.data.assign(canvas.data, canvas.data + static_cast<std::size_t>(im.step) * im.height);
    pub_dbg_->publish(im);
  }

  // ------------------------------------------------------------ 统计报告

  void report() {
    const auto t = Clock::now();
    const double dt = std::chrono::duration<double>(t - last_report_).count();
    last_report_ = t;

    const auto processed_now = processed_.load();
    const float fps = dt > 0 ? static_cast<float>(processed_now - last_processed_) / static_cast<float>(dt)
                             : 0.f;
    last_processed_ = processed_now;

    msg::LatencyReport r;
    r.header.stamp = now();
    r.ep = ep_to_string(detector_->actual_ep());
    r.window = static_cast<std::uint32_t>(st_infer_->window_count());
    r.frames_total = static_cast<std::uint32_t>(frames_total_);
    r.dropped_stale = static_cast<std::uint32_t>(dropped_stale_);
    r.dropped_bad_encoding = static_cast<std::uint32_t>(dropped_bad_encoding_);
    r.dropped_bad_size = static_cast<std::uint32_t>(dropped_bad_size_);
    r.infer_errors = static_cast<std::uint32_t>(infer_errors_);
    r.fps = fps;
    r.transport_ms_p50 = st_transport_->percentile(50);
    r.transport_ms_p95 = st_transport_->percentile(95);
    r.transport_ms_p99 = st_transport_->percentile(99);
    r.pre_ms_p50 = st_pre_->percentile(50);
    r.pre_ms_p99 = st_pre_->percentile(99);
    r.infer_ms_p50 = st_infer_->percentile(50);
    r.infer_ms_p95 = st_infer_->percentile(95);
    r.infer_ms_p99 = st_infer_->percentile(99);
    r.post_ms_p50 = st_post_->percentile(50);
    r.post_ms_p99 = st_post_->percentile(99);
    r.serve_ms_p50 = st_serve_->percentile(50);
    r.serve_ms_p99 = st_serve_->percentile(99);
    r.vram_mib = read_vram_ ? read_vram_mib() : 0;
    r.empty_streak_max = empty_streak_max_;
    pub_lat_->publish(r);

    RCLCPP_INFO(get_logger(),
                "%.1f fps ep=%s | transport p50 %.1f p99 %.1f | pre %.1f | infer p50 %.1f p95 %.1f "
                "p99 %.1f | post %.1f | serve p50 %.1f p99 %.1f | drop stale %u enc %u | vram %u MiB",
                fps, ep_to_string(detector_->actual_ep()), r.transport_ms_p50, r.transport_ms_p99,
                r.pre_ms_p50, r.infer_ms_p50, r.infer_ms_p95, r.infer_ms_p99, r.post_ms_p50,
                r.serve_ms_p50, r.serve_ms_p99, r.dropped_stale, r.dropped_bad_encoding,
                r.vram_mib);

    if (empty_streak_max_ > 30 && fps > 0.1f) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                           "连续 %lu 帧无任何检测 —— 模型可能与该场景不匹配（域漂移），"
                           "这不等于节点故障",
                           empty_streak_max_);
    }
  }

  // ------------------------------------------------------------ 成员

  std::unique_ptr<IDetector> detector_;

  std::string model_path_, detector_kind_, trt_cache_dir_, image_topic_;
  int device_id_ = 0, intra_threads_ = 2, input_size_ = 640, num_classes_ = 12;
  EpKind ep_ = EpKind::kCpu;
  std::vector<std::string> class_names_;
  float conf_thres_ = 0.25f, iou_thres_ = 0.45f, max_transport_ms_ = 200.f;
  bool publish_debug_image_ = true, use_worker_thread_ = false, read_vram_ = true;
  double display_rate_hz_ = 1.0;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<msg::DetectionArray>::SharedPtr pub_det_;
  rclcpp::Publisher<msg::LatencyReport>::SharedPtr pub_lat_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_dbg_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<LatencyStats> st_transport_, st_pre_, st_infer_, st_post_, st_serve_;

  std::atomic<std::uint64_t> frames_total_{0}, processed_{0}, processed_by_worker_{0};
  std::uint64_t dropped_stale_ = 0, dropped_bad_encoding_ = 0, dropped_bad_size_ = 0,
                infer_errors_ = 0, empty_streak_ = 0, empty_streak_max_ = 0, dropped_overrun_ = 0;

  std::thread worker_;
  std::mutex mbox_mtx_;
  std::condition_variable cv_;
  sensor_msgs::msg::Image::ConstSharedPtr pending_;
  bool stop_ = false;

  Clock::time_point last_report_ = Clock::now();
  std::uint64_t last_processed_ = 0;
};

}  // namespace yolo_trt_ros

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<yolo_trt_ros::YoloNode>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("yolo_trt_node"), "启动失败: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
