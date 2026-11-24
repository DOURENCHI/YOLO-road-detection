#include <gtest/gtest.h>

#include <vector>

#include <opencv2/core.hpp>

#include "yolo_trt_ros/letterbox.hpp"

using yolo_trt_ros::Letterbox;
using yolo_trt_ros::make_letterbox;
using yolo_trt_ros::to_blob;
using yolo_trt_ros::unletterbox;

namespace {
constexpr int kSize = 640;
constexpr int kPad = 114;
}

// 横图（1280x720）→ 640：宽度先顶满，上下留灰边
TEST(Letterbox, WideImagePadsVertically) {
  cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
  const Letterbox lb = make_letterbox(img, kSize, /*rgb=*/false);
  EXPECT_EQ(lb.canvas.rows, kSize);
  EXPECT_EQ(lb.canvas.cols, kSize);
  EXPECT_FLOAT_EQ(lb.scale, 0.5f);
  EXPECT_FLOAT_EQ(lb.pad_x, 0.f);
  EXPECT_FLOAT_EQ(lb.pad_y, 140.f);
}

// 竖图（720x1280）→ 640：高度先顶满，左右留灰边
TEST(Letterbox, TallImagePadsHorizontally) {
  cv::Mat img(1280, 720, CV_8UC3, cv::Scalar(0, 0, 0));
  const Letterbox lb = make_letterbox(img, kSize, /*rgb=*/false);
  EXPECT_FLOAT_EQ(lb.scale, 0.5f);
  EXPECT_FLOAT_EQ(lb.pad_x, 140.f);
  EXPECT_FLOAT_EQ(lb.pad_y, 0.f);
}

// 正方形输入：无填充
TEST(Letterbox, SquareImageNoPad) {
  cv::Mat img(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  const Letterbox lb = make_letterbox(img, kSize, /*rgb=*/false);
  EXPECT_FLOAT_EQ(lb.scale, 1.f);
  EXPECT_FLOAT_EQ(lb.pad_x, 0.f);
  EXPECT_FLOAT_EQ(lb.pad_y, 0.f);
}

// 填充区必须真的是 114 —— 这个值若与训练时不一致，模型精度会莫名下降
TEST(Letterbox, PadValueIs114) {
  cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(255, 255, 255));
  const Letterbox lb = make_letterbox(img, kSize, /*rgb=*/false);
  const cv::Vec3b top = lb.canvas.at<cv::Vec3b>(0, 320);      // 上灰边
  const cv::Vec3b mid = lb.canvas.at<cv::Vec3b>(320, 320);    // 图像区（应为白）
  EXPECT_EQ(static_cast<int>(top[0]), kPad);
  EXPECT_EQ(static_cast<int>(top[1]), kPad);
  EXPECT_EQ(static_cast<int>(top[2]), kPad);
  EXPECT_EQ(static_cast<int>(mid[0]), 255);
}

// 坐标往返：canvas 中心 → 原图中心
TEST(Letterbox, RoundTripCenter) {
  cv::Mat wide(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
  const Letterbox lb = make_letterbox(wide, kSize, false);
  float x1 = 320.f, y1 = 320.f, x2 = 320.f, y2 = 320.f;
  unletterbox(&x1, &y1, &x2, &y2, lb);
  EXPECT_NEAR(x1, 640.f, 0.01f);   // 1280 / 2
  EXPECT_NEAR(y1, 360.f, 0.01f);   // 720 / 2

  cv::Mat tall(1280, 720, CV_8UC3, cv::Scalar(0, 0, 0));
  const Letterbox lb2 = make_letterbox(tall, kSize, false);
  float a1 = 320.f, b1 = 320.f, a2 = 320.f, b2 = 320.f;
  unletterbox(&a1, &b1, &a2, &b2, lb2);
  EXPECT_NEAR(a1, 360.f, 0.01f);   // 720 / 2
  EXPECT_NEAR(b1, 640.f, 0.01f);   // 1280 / 2
}

// 整块落在填充区的框，反变换后必须得到越界坐标（负数），
// 以此证明"调用方必须自己 clamp"这个契约是真实的。
TEST(Letterbox, PadRegionMapsOutsideImage) {
  cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
  const Letterbox lb = make_letterbox(img, kSize, false);
  // canvas 上边缘一条框，完全在灰边里
  float x1 = 100.f, y1 = 10.f, x2 = 200.f, y2 = 60.f;
  unletterbox(&x1, &y1, &x2, &y2, lb);
  EXPECT_LT(y1, 0.f);   // (10 - 140) / 0.5 = -260
  EXPECT_NEAR(y2, -160.f, 0.01f);
}

// 张量布局：CHW + /255
TEST(ToBlob, ProducesChwNormalized) {
  const int s = 4;
  cv::Mat canvas(s, s, CV_8UC3);
  // 全部填 BGR = (10, 20, 30)
  for (int y = 0; y < s; ++y)
    for (int x = 0; x < s; ++x) canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(10, 20, 30);

  std::vector<float> blob(static_cast<std::size_t>(3) * s * s);
  to_blob(canvas, blob.data());

  const std::size_t plane = static_cast<std::size_t>(s) * s;
  for (std::size_t i = 0; i < plane; ++i) {
    EXPECT_NEAR(blob[i], 10.f / 255.f, 1e-6f);            // 通道 0
    EXPECT_NEAR(blob[plane + i], 20.f / 255.f, 1e-6f);    // 通道 1
    EXPECT_NEAR(blob[2 * plane + i], 30.f / 255.f, 1e-6f); // 通道 2
  }
}
