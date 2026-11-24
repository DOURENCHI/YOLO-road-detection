#!/usr/bin/env python3
"""采集 /yolo/latency 报告并追加到对比表。

用法（Gazebo 与节点已经在跑的前提下）：

    python3 bench_latency.py --label trt --duration 60
    python3 bench_latency.py --label cuda --duration 60
    python3 bench_latency.py --label cpu  --duration 60

三次跑完，本脚本把结果追加到 --out 指定的 CSV，并打印 Markdown 表格，
可直接贴进 docs 或报告。

为什么不在脚本里替你启动/杀节点：
  启动顺序、Gazebo 是否就绪、engine 首次构建要 10 分钟 —— 这些用脚本自动串起来
  很脆，一旦某步失败你会分不清是脚本问题还是系统问题。手动分三次跑，每次都有
  明确的成功判据，出错定位成本低得多。
"""

import argparse
import csv
import os
import statistics
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from yolo_trt_ros.msg import LatencyReport

FIELDS = [
    ('fps', 'fps'),
    ('transport_ms_p50', 'transport p50'),
    ('transport_ms_p95', 'transport p95'),
    ('transport_ms_p99', 'transport p99'),
    ('pre_ms_p50', 'pre p50'),
    ('pre_ms_p99', 'pre p99'),
    ('infer_ms_p50', 'infer p50'),
    ('infer_ms_p95', 'infer p95'),
    ('infer_ms_p99', 'infer p99'),
    ('post_ms_p50', 'post p50'),
    ('post_ms_p99', 'post p99'),
    ('serve_ms_p50', 'serve p50'),
    ('serve_ms_p99', 'serve p99'),
    ('vram_mib', 'vram MiB'),
]


class Collector(Node):
    def __init__(self, duration):
        super().__init__('bench_latency_collector')
        # 必须与发布端 QoS 一致，否则一条都收不到且不报错
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.reports = []
        self.sub = self.create_subscription(
            LatencyReport, '/yolo/latency', self.on_report, qos)
        self.deadline = self.get_clock().now().nanoseconds + int(duration * 1e9)

    def on_report(self, msg):
        self.reports.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True, help='这一轮的名字，如 cpu / cuda / trt')
    ap.add_argument('--duration', type=float, default=60.0)
    ap.add_argument('--out', default='latency_bench.csv')
    ap.add_argument('--topic', default='yolo_trt_node')
    args = ap.parse_args()

    rclpy.init()
    node = Collector(args.duration)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            if node.get_clock().now().nanoseconds >= node.deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        reports = node.reports
        ep = reports[-1].ep if reports else 'n/a'
        node.destroy_node()
        rclpy.shutdown()

    if not reports:
        print(f'[FAIL] {args.duration:.0f}s 内没有收到任何 /yolo/latency。\n'
              '       检查：① 节点是否在跑 ② ros2 topic list 有没有 /yolo/latency\n'
              '       ③ ROS_DOMAIN_ID 是否与仿真一致 ④ QoS 是否 best_effort', file=sys.stderr)
        return 1

    row = {'label': args.label, 'ep_reported': ep, 'samples': len(reports)}
    for attr, _ in FIELDS:
        vals = [float(getattr(r, attr)) for r in reports]
        row[attr] = round(statistics.median(vals), 3)
    row['dropped_stale'] = reports[-1].dropped_stale
    row['dropped_bad_encoding'] = reports[-1].dropped_bad_encoding
    row['infer_errors'] = reports[-1].infer_errors
    row['empty_streak_max'] = max(r.empty_streak_max for r in reports)

    exists = os.path.exists(args.out)
    with open(args.out, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

    print(f'\n=== {args.label} (ep={ep}, {len(reports)} 份报告, 中位数) ===')
    for attr, title in FIELDS:
        print(f'  {title:<16} {row[attr]:>10.3f}')
    print(f'  {"dropped_stale":<16} {row["dropped_stale"]:>10d}')
    print(f'  {"infer_errors":<16} {row["infer_errors"]:>10d}')
    print(f'\n已追加到 {os.path.abspath(args.out)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
