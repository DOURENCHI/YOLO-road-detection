#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace yolo_trt_ros {

// 定长环形缓冲 + 最近秩分位数。
//
// 为什么不用 std::vector 一直 push_back：
//   节点要跑几小时，累计几十万帧，无界容器会让内存线性增长。
// 为什么报 P99 而不只报均值：
//   实时系统怕的是抖动。均值 6ms、P99 30ms 的系统在 30fps 下会周期性丢帧，
//   但均值看起来很漂亮 —— 部署验收必须看尾部。
class LatencyStats {
 public:
  explicit LatencyStats(std::size_t capacity = 200)
      : buf_(capacity > 0 ? capacity : 1, 0.f), capacity_(buf_.size()) {}

  void add(float v) {
    std::lock_guard<std::mutex> lk(mtx_);
    buf_[head_] = v;
    head_ = (head_ + 1) % capacity_;
    if (count_ < capacity_) ++count_;
    ++total_;
  }

  std::size_t window_count() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return count_;
  }

  std::uint64_t total() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return total_;
  }

  // p 取 0~100。样本不足时返回已有样本上的分位数。
  float percentile(double p) const {
    std::lock_guard<std::mutex> lk(mtx_);
    if (count_ == 0) return 0.f;
    // 顺序无关，只取前 count_ 个有效槽位即可
    std::vector<float> tmp(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(count_));
    std::sort(tmp.begin(), tmp.end());
    double rank = p / 100.0 * static_cast<double>(tmp.size() - 1);
    std::size_t idx = static_cast<std::size_t>(rank + 0.5);
    if (idx >= tmp.size()) idx = tmp.size() - 1;
    return tmp[idx];
  }

  void reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    std::fill(buf_.begin(), buf_.end(), 0.f);
    head_ = 0;
    count_ = 0;
    total_ = 0;
  }

 private:
  mutable std::mutex mtx_;
  std::vector<float> buf_;
  std::size_t capacity_ = 0;
  std::size_t head_ = 0;
  std::size_t count_ = 0;
  std::uint64_t total_ = 0;
};

}  // namespace yolo_trt_ros
