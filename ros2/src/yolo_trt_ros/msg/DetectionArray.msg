# 一帧的检测结果
std_msgs/Header header

# 模型标识与"实际生效"的执行后端（CPU / CUDA / TensorRT）
string model_id
string ep

# 原图尺寸，供下游反算与画框
uint32 image_width
uint32 image_height

# 本帧纯推理耗时（仅 Session::Run，毫秒）
float32 infer_ms

Detection[] detections
