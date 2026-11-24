#!/usr/bin/env python3
"""fake_camera —— 把项目里的真实道路图片按固定帧率发成 ROS2 图像话题。

它代替 Gazebo，作用是给 yolo_trt_node 提供一路"按时到达的图像流"。

    python3 fake_camera.py                       # 30 fps，无限循环
    python3 fake_camera.py --fps 15 --loop 3     # 15 fps，跑 3 轮
    python3 fake_camera.py --size 1280x720       # 统一缩放到 720p

为什么不用 Gazebo
----------------
Gazebo 在 WSL2 里要几 GB 依赖，还要图形加速，起不来的时候"是驱动问题还是
QoS 问题还是模型问题"三件事混在一起排查。而验证节点真正需要的只是
"一路按帧率到达的图像"，不关心图像从哪来。等通信链路确认没问题，再上仿真器。

为什么用 best_effort 发布（最容易踩的一步）
------------------------------------------
yolo_trt_node 的输入订阅是 rclcpp::SensorDataQoS().keep_last(1)，
即 **best_effort + depth 1**。DDS 的 QoS 兼容规则是：

    发布 best_effort  → 订阅 best_effort   ✅ 通
    发布 reliable     → 订阅 best_effort   ✅ 通（可靠端可以降级服务尽力端）
    发布 best_effort  → 订阅 reliable      ❌ 不通，而且两边都不报错

所以这里用 best_effort 一定是安全的；用 reliable 也通，但语义上没必要
（图像丢一帧无所谓，重传一帧反而增加延迟）。

时间戳
------
header.stamp 必须填。yolo_trt_node 在 stamp 为 0 时会跳过 transport_ms 统计
（它没法算"从相机到节点花了多久"）。这里用 node 的系统时钟，
所以节点侧必须 use_sim_time:=false，两边才在同一时间轴上。
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

DEFAULT_IMAGES = "/mnt/d/VC Code/YOLO_Road_Detection/datasets/road-traffic-detect/valid/images"


def parse_size(text):
    """'1280x720' -> (1280, 720)；'none' -> None"""
    if not text or text.lower() == "none":
        return None
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("尺寸格式应为 1280x720")


class FakeCamera(Node):
    def __init__(self, topic, fps):
        super().__init__("fake_camera")
        # depth=1 + best_effort：与节点的 SensorDataQoS().keep_last(1) 对齐。
        # depth=1 在这里也有意义——发图比消费快时，旧帧直接在中间件里被丢掉，
        # 而不是排队，否则节点收到的永远是"几秒前的画面"。
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(Image, topic, qos)
        self.fps = fps
        self.sent = 0
        self.get_logger().info(
            "fake_camera -> %s  (%d fps, best_effort/depth=1)" % (topic, fps)
        )

    def publish_one(self, bgr):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height, msg.width = bgr.shape[:2]
        msg.encoding = "bgr8"          # OpenCV 读出来就是 BGR，不用转
        msg.is_bigendian = 0
        msg.step = int(bgr.strides[0])  # = width * 3，显式给出来避免行对齐歧义
        msg.data = bgr.tobytes()
        self.pub.publish(msg)
        self.sent += 1


def main():
    ap = argparse.ArgumentParser(description="把道路图片按帧率发成 ROS2 图像话题")
    ap.add_argument("--images", default=DEFAULT_IMAGES, help="图片目录")
    ap.add_argument("--topic", default="/camera/image_raw", help="话题名")
    ap.add_argument("--fps", type=float, default=30.0, help="发布帧率")
    ap.add_argument("--loop", type=int, default=0, help="循环几轮，0 表示无限")
    ap.add_argument("--max-frames", type=int, default=0, help="最多发多少帧，0 表示不限")
    ap.add_argument("--size", type=parse_size, default=None,
                    help="统一缩放到 WxH，例如 1280x720；默认保持原图")
    ap.add_argument("--limit-images", type=int, default=0,
                    help="只用前 N 张图（想固定画面内容时用）")
    args = ap.parse_args()

    root = Path(args.images)
    if not root.is_dir():
        raise SystemExit(
            "图片目录不存在：%s\n"
            "Windows 里的路径在 WSL 下是 /mnt/d/...，注意盘符要小写、空格要转义。" % root
        )
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.limit_images > 0:
        files = files[: args.limit_images]
    if not files:
        raise SystemExit("目录里没有 jpg/png：%s" % root)

    rclpy.init()
    node = FakeCamera(args.topic, args.fps)
    period = 1.0 / max(0.1, args.fps)

    print("图片 %d 张，来源 %s" % (len(files), root))
    if args.size:
        print("统一缩放到 %dx%d" % args.size)

    rounds = 0
    t_start = time.monotonic()
    next_t = t_start
    try:
        while rclpy.ok():
            for f in files:
                if args.max_frames and node.sent >= args.max_frames:
                    break

                bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if bgr is None:
                    node.get_logger().warn("读不出这张图，跳过：%s" % f.name)
                    continue
                if args.size:
                    bgr = cv2.resize(bgr, args.size, interpolation=cv2.INTER_LINEAR)
                bgr = np.ascontiguousarray(bgr)  # tobytes() 要求连续内存

                node.publish_one(bgr)

                # 按绝对时间推进，而不是"发一帧睡 1/fps"——
                # 后者会把读图+编码的耗时代代累积，实际帧率越跑越低。
                next_t += period
                sleep_s = next_t - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.monotonic()  # 已经落后，不追帧，重新对齐

            rounds += 1
            if args.max_frames and node.sent >= args.max_frames:
                break
            if args.loop and rounds >= args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = max(1e-6, time.monotonic() - t_start)
        print(
            "\n共发 %d 帧 / %.1f 秒，实测 %.1f fps（目标 %.1f）"
            % (node.sent, elapsed, node.sent / elapsed, args.fps)
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
