#include "yolo_trt_ros/letterbox.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace yolo_trt_ros {

Letterbox make_letterbox(const cv::Mat& bgr, int size, bool rgb) {
  Letterbox lb;
  const int w = bgr.cols;
  const int h = bgr.rows;
  lb.scale = std::min(static_cast<float>(size) / static_cast<float>(w),
                      static_cast<float>(size) / static_cast<float>(h));
  const int new_w = static_cast<int>(std::lround(w * lb.scale));
  const int new_h = static_cast<int>(std::lround(h * lb.scale));
  lb.pad_x = 0.5f * static_cast<float>(size - new_w);
  lb.pad_y = 0.5f * static_cast<float>(size - new_h);

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  lb.canvas = cv::Mat(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(lb.canvas(cv::Rect(static_cast<int>(lb.pad_x),
                                    static_cast<int>(lb.pad_y), new_w, new_h)));
  if (rgb) cv::cvtColor(lb.canvas, lb.canvas, cv::COLOR_BGR2RGB);
  return lb;
}

void unletterbox(float* x1, float* y1, float* x2, float* y2, const Letterbox& lb) {
  const float inv = lb.scale > 0.f ? 1.f / lb.scale : 0.f;
  *x1 = (*x1 - lb.pad_x) * inv;
  *y1 = (*y1 - lb.pad_y) * inv;
  *x2 = (*x2 - lb.pad_x) * inv;
  *y2 = (*y2 - lb.pad_y) * inv;
}

void to_blob(const cv::Mat& canvas, float* out) {
  const int size = canvas.rows;
  std::vector<cv::Mat> ch(3);
  for (int c = 0; c < 3; ++c) {
    ch[c] = cv::Mat(size, size, CV_32FC1, out + static_cast<std::size_t>(c) * size * size);
  }
  cv::Mat f;
  canvas.convertTo(f, CV_32FC3, 1.0 / 255.0);
  // 用 Mat* 重载而非 OutputArrayOfArrays：后者的 create() 语义依赖"尺寸相同即不重分配"，
  // 显式传首地址可以避免把预分配好的缓冲区换成新内存。
  cv::split(f, ch.data());
}

}  // namespace yolo_trt_ros
