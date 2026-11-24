#pragma once

// letterbox 与张量布局是 YOLO 部署里最容易出错的一环：
// 缩放系数、填充偏移、通道序、归一化，任何一处错都不会报错，只会让框整体偏移。
// 所以把它从检测器里抽出来，单独用单元测试盯住。

#include <opencv2/core.hpp>

namespace yolo_trt_ros {

struct Letterbox {
  cv::Mat canvas;       // size × size，三通道 uint8
  float scale = 1.f;    // 原图 → canvas 的等比缩放系数
  float pad_x = 0.f;    // 左侧填充宽度（canvas 像素）
  float pad_y = 0.f;    // 顶部填充高度（canvas 像素）
};

// rgb=true 时输出通道序为 RGB（ultralytics 训练用的就是 RGB）
Letterbox make_letterbox(const cv::Mat& bgr, int size, bool rgb);

// canvas 坐标 → 原图坐标。故意不做裁剪：
// 落在填充区的框会算出负数，调用方必须自己 clamp —— 这个测试里专门验了。
void unletterbox(float* x1, float* y1, float* x2, float* y2, const Letterbox& lb);

// HWC uint8 → CHW float32，并除以 255。
// out 由调用方预分配 3 * size * size 个 float。
void to_blob(const cv::Mat& canvas, float* out);

}  // namespace yolo_trt_ros
