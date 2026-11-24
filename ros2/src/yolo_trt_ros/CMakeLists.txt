cmake_minimum_required(VERSION 3.16)
# 必须同时声明 C：rosidl 生成的消息类型支持（rosidl_typesupport_cpp）依赖 C，
# 只写 CXX 会在 generate 阶段报 CMAKE_C_COMPILE_OBJECT / CMAKE_C_CREATE_SHARED_LIBRARY 缺失。
project(yolo_trt_ros C CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  "msg/LatencyReport.msg"
  DEPENDENCIES std_msgs
)

# ---------------------------------------------------------------------------
# ONNX Runtime 是可选依赖。
#   找不到时只编假检测器，节点仍可启动（detector:=dummy）。
#   这样"通信层对不对"和"推理对不对"可以分两次验证，出问题不必两边一起猜。
#
# pip 装的 onnxruntime 只带 .so、不带 C++ 头。头文件要单独取：
#   curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
#   tar xf onnxruntime-linux-x64-1.22.0.tgz
#   export ORT_ROOT=$PWD/onnxruntime-linux-x64-1.22.0
# ---------------------------------------------------------------------------
# NAMES 必须用相对 include 根的完整路径：find_path 不会递归进子目录，
# 只写 onnxruntime_cxx_api.h 会找不到（真实位置在 include/onnxruntime/core/session/）。
# 找到后 ORT_INCLUDE_DIR = .../include，正好是 target_include_directories 需要的根。
find_path(ORT_INCLUDE_DIR
  NAMES onnxruntime/core/session/onnxruntime_cxx_api.h
  HINTS "${ORT_ROOT}/include" "$ENV{ORT_ROOT}/include"
)
find_library(ORT_LIBRARY
  NAMES onnxruntime
  HINTS "${ORT_ROOT}/lib" "$ENV{ORT_ROOT}/lib"
)

set(HAVE_ORT FALSE)
if(ORT_INCLUDE_DIR AND ORT_LIBRARY)
  set(HAVE_ORT TRUE)
  message(STATUS "[yolo_trt_ros] ONNX Runtime: ${ORT_LIBRARY}")
else()
  message(WARNING
    "[yolo_trt_ros] 未找到 ONNX Runtime，只编译假检测器。\n"
    "  设置 ORT_ROOT 后重新 colcon build 可启用真推理。\n"
    "  当前状态下 detector:=dummy 可用，detector:=ort 会抛异常。")
endif()

set(YOLO_TRT_ROS_SOURCES
  src/letterbox.cpp
  src/yolo_detector.cpp
)

if(HAVE_ORT)
  list(APPEND YOLO_TRT_ROS_SOURCES src/ort_link_stub.cpp)
endif()

add_library(yolo_trt_ros_core ${YOLO_TRT_ROS_SOURCES})
target_include_directories(yolo_trt_ros_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(yolo_trt_ros_core PUBLIC ${OpenCV_LIBS})
target_compile_options(yolo_trt_ros_core PRIVATE -Wall -Wextra)
if(HAVE_ORT)
  # 源码里写的是扁平 #include "onnxruntime_cxx_api.h"，所以 include 路径必须
  # 直接指到头文件所在目录（core/session），而不是 include 根。
  # 有些 release 包会在 include/ 下同时放一份扁平头，那种情况下 ${ORT_INCLUDE_DIR} 就够；
  # 这里两者都加，兼容两种布局。
  target_include_directories(yolo_trt_ros_core PUBLIC
    ${ORT_INCLUDE_DIR}
    ${ORT_INCLUDE_DIR}/onnxruntime/core/session
  )
  target_link_libraries(yolo_trt_ros_core PUBLIC ${ORT_LIBRARY})
  target_compile_definitions(yolo_trt_ros_core PUBLIC YOLO_TRT_ROS_WITH_ORT=1)
endif()

# ---------------------------------------------------------------------------
# 节点可执行文件
# ---------------------------------------------------------------------------
add_executable(yolo_trt_node src/yolo_node.cpp)
# 注意：ament_target_dependencies 内部用的是 plain 签名（不带 PRIVATE/PUBLIC），
# CMake 不允许同一 target 混用 keyword 和 plain 两种签名，所以这里也必须用 plain。
# 混用会报 "The keyword signature ... has already been used" / "plain signature ... already been used"。
target_link_libraries(yolo_trt_node yolo_trt_ros_core)
ament_target_dependencies(yolo_trt_node rclcpp std_msgs sensor_msgs)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(yolo_trt_node "${cpp_typesupport_target}")

install(TARGETS yolo_trt_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# ---------------------------------------------------------------------------
# 测试：纯 C++ 层，不启动 ROS
#   跑法：colcon test --packages-select yolo_trt_ros
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_latency_stats test/test_latency_stats.cpp)
  target_link_libraries(test_latency_stats yolo_trt_ros_core)

  ament_add_gtest(test_letterbox test/test_letterbox.cpp)
  target_link_libraries(test_letterbox yolo_trt_ros_core)
endif()

ament_export_include_directories(include)
ament_export_libraries(yolo_trt_ros_core)
ament_package()
