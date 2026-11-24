// ORT 链接占位。
// 存在的唯一理由：让 CMake 在 HAVE_ORT 时把 libonnxruntime 链进 core 库，
// 同时把"未启用 ORT"的编译分支保持在另一条路径上。
// 不放任何业务代码。

#include <cstddef>

namespace yolo_trt_ros {
const char* ort_link_stub() { return "yolo_trt_ros: ONNX Runtime linked"; }
}  // namespace yolo_trt_ros
