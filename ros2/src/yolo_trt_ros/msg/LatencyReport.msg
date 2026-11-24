# 滑窗统计报告，默认每秒一次
std_msgs/Header header

string ep
uint32 window
uint32 frames_total

# 异常计数（每次报告累计值）
uint32 dropped_stale
uint32 dropped_bad_encoding
uint32 dropped_bad_size
uint32 infer_errors

float32 fps

# 分段延迟分位数（毫秒）。transport 用仿真时钟，其余用 steady_clock
float32 transport_ms_p50
float32 transport_ms_p95
float32 transport_ms_p99
float32 pre_ms_p50
float32 pre_ms_p99
float32 infer_ms_p50
float32 infer_ms_p95
float32 infer_ms_p99
float32 post_ms_p50
float32 post_ms_p99
float32 serve_ms_p50
float32 serve_ms_p99

# 显存（MiB），拿不到时为 0
uint32 vram_mib

# 连续空检测的最大帧数（判断"模型在这个场景里彻底失效"用）
uint32 empty_streak_max
