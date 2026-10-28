# assets/ 素材说明

这个目录放**演示素材**，不参与训练和评估。

```
assets/
├── demo/
│   └── bus.jpg                         Ultralytics 官方示例图（阶段一做链路验证时下载的）
└── videos/
    ├── README.md                       本文件
    └── _pipeline_check_pan.mp4         合成视频，用来验证跟踪链路（见下）
```

## 关于 `_pipeline_check_pan.mp4`

这不是真实拍摄的视频，**不能当作跟踪效果演示**。

它的来历：本项目的数据集 `datasets/road-traffic` 是按约 5 秒一张抽帧的快照集
（实测相邻两张最小间隔 1.09 秒），**没有连续帧**，所以没法用数据集验证跟踪。
于是取了一张真实道路图，把它放大 1.25 倍后做一个缓慢的平移窗口，
合成出 120 帧 @25fps 的片段 —— 画面里的车会平滑移动，跟踪器能正常关联。

**用途**：证明 `scripts/08_track.py` 这条链路（读视频 → 逐帧检测 → ByteTrack 关联 → 写回视频）
是通的。实测结果：120 帧共 373 个检测框，收敛成 **4 条轨迹**，
其中 3 条跟满全程，1 条在第 13 帧因目标移出画面而结束。

**局限**：画面是同一张图平移出来的，没有真实的光照变化、遮挡、目标进出。
它的数字不能用来评价跟踪质量。

## 想真正演示跟踪，怎么做

用手机拍一段 20~30 秒的道路视频（路口、马路、停车场都行，手持拍摄也可以），
放进这个目录，然后：

```bash
python scripts/08_track.py --source assets/videos/你拍的视频.mp4 --name real_road
```

结果会出现在 `results/track/real_road/track.mp4`，带框和稳定的轨迹 ID。

拍的时候注意两点：
- **别把镜头怼在静止画面上**——跟踪要看的是运动，画面不动就没有意义
- 尽量顺光拍，逆光和夜间会让检出率明显下降

---

## `dashcam_cc0_720p.ogv` —— 真实行车记录仪视频（CC0）

用来做**视频流 + 长时间稳定性测试**（FPS / p99 延迟 / 内存增长）。

| 项 | 值 |
|---|---|
| 标题 | Dashcam recording example - Panorama II BlackBox Dashcam daylight |
| 作者 | Fernost |
| 授权 | **CC0**（公共领域贡献，无需署名、可商用） |
| 来源 | Wikimedia Commons |
| 规格 | 1280×720 @ 30 fps，3601 帧（120 s），69.8 MB |
| 内容 | 地下车库 → 驶出 → 城市街道（一镜到底，覆盖暗到亮的光照变化） |

原页面：https://commons.wikimedia.org/wiki/File:Dashcam_recording_example_-_Panorama_II_BlackBox_Dashcam_daylight.mp4.ogv

**为什么选它**：Commons 上的行车视频大多来自事故/执法记录，正在被批量提删（版权争议）；
这一条是明确 CC0 的行车记录仪示例，授权最干净。项目里测的是第 0–999 帧。

## `_stability_test.mp4` —— 合成视频（备选，已被真实视频取代）

由 `scripts/17_make_test_video.py` 用 valid 的 187 张图循环拼接而成，1280×720、1000 帧。
**不纳入版本库**（69 MB，脚本可重新生成）。

实测发现一个差异，值得记：**合成视频的帧间跨场景跳变会让 p99 偏高**——
同样的程序跑 1000 帧，合成视频两轮 96.0 / 107.5 FPS（p99 16.18 / 13.24），
真实视频两轮 **100.8 / 100.8 FPS**（p99 12.51 / 12.19）。
所以：**合成视频可以用来验证链路通不通，但报延迟数据要用真实视频。**

## `road_traffic_cobblestone.webm` —— 真实道路视频（CC BY 3.0，1080p）

用来做**高分辨率对照**：同样 1000 帧，1080p 比 720p 慢多少。

| 项 | 值 |
|---|---|
| 标题 | Road traffic – cobblestone road |
| 作者 | barbara.grilc（Sound: Boštjan Troha, Video: Barbara Rezar Grilc, Photo: Dušan Oblak） |
| 授权 | **CC BY 3.0**（需署名，不能去掉本条）+ 来源 soundsofchanges.eu |
| 来源 | Wikimedia Commons |
| 规格 | 1920×1080 @ 25 fps，2652 帧（106 s），51.6 MB |
| 内容 | 固定机位拍摄的道路车流，NMS 后稳定检出 2–3 个 `vehicles` |

原页面：https://commons.wikimedia.org/wiki/File:Road_traffic_–_cobblestone_road.webm

**两个真实视频的分工**：

| 视频 | 分辨率 | 用途 | 实测 |
|---|---|---|---|
| `dashcam_cc0_720p.ogv` | 1280×720 | 主测试（CC0 授权最干净） | 100.5–101.6 FPS，p99 13.3–14.1 ms |
| `road_traffic_cobblestone.webm` | 1920×1080 | 高分辨率对照 | 72.5–75.4 FPS，p99 17.6–18.1 ms |

差别的来源几乎全在**预处理**：720p 是 5.2–5.5 ms，1080p 是 7.9–8.4 ms，而推理两边都是 3.6–4.0 ms。


