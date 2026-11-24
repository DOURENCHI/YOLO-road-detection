#include <gtest/gtest.h>

#include <vector>

#include "yolo_trt_ros/latency_stats.hpp"

using yolo_trt_ros::LatencyStats;

TEST(LatencyStats, EmptyReturnsZero) {
  LatencyStats s(10);
  EXPECT_EQ(s.window_count(), 0u);
  EXPECT_FLOAT_EQ(s.percentile(50), 0.f);
  EXPECT_FLOAT_EQ(s.percentile(99), 0.f);
}

TEST(LatencyStats, PartiallyFilled) {
  LatencyStats s(10);
  for (int i = 1; i <= 5; ++i) s.add(static_cast<float>(i));
  EXPECT_EQ(s.window_count(), 5u);
  EXPECT_EQ(s.total(), 5u);
  // 1..5 的中位数
  EXPECT_FLOAT_EQ(s.percentile(50), 3.f);
  EXPECT_FLOAT_EQ(s.percentile(0), 1.f);
  EXPECT_FLOAT_EQ(s.percentile(100), 5.f);
}

TEST(LatencyStats, WindowDropsOldest) {
  LatencyStats s(4);
  for (int i = 1; i <= 4; ++i) s.add(static_cast<float>(i));
  s.add(100.f);  // 挤掉 1
  EXPECT_EQ(s.window_count(), 4u);
  EXPECT_EQ(s.total(), 5u);
  EXPECT_FLOAT_EQ(s.percentile(0), 2.f);   // 1 已被挤掉
  EXPECT_FLOAT_EQ(s.percentile(100), 100.f);
}

TEST(LatencyStats, PercentileOfTail) {
  // 100 个样本里放 1 个 1000ms，验证 P99 能抓到尾部
  LatencyStats s(100);
  for (int i = 0; i < 99; ++i) s.add(5.f);
  s.add(1000.f);
  EXPECT_LT(s.percentile(50), 6.f);
  EXPECT_GT(s.percentile(99), 900.f);
}

TEST(LatencyStats, ResetClearsEverything) {
  LatencyStats s(8);
  for (int i = 0; i < 8; ++i) s.add(10.f);
  s.reset();
  EXPECT_EQ(s.window_count(), 0u);
  EXPECT_EQ(s.total(), 0u);
  EXPECT_FLOAT_EQ(s.percentile(50), 0.f);
}
