cmake_minimum_required(VERSION 3.16)
project(yolo_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ------------------------------------------------------------------ ONNX Runtime
# 头文件：third_party/onnxruntime/include（从 onnxruntime v1.22.0 源码取的）
# 导入库：直接链 pip 装好的 onnxruntime.dll。MinGW 的 ld 能读 DLL 的导出表，
#        所以不需要官方包里的 .lib，也不用 gendef/dlltool 生成。
set(ORT_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include")
set(ORT_DLL "D:/Miniconda/envs/yolo_road/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
    CACHE FILEPATH "onnxruntime.dll 的完整路径")

if(NOT EXISTS "${ORT_DLL}")
  message(FATAL_ERROR "找不到 onnxruntime.dll：${ORT_DLL}\n用 -DORT_DLL=<路径> 指定")
endif()

add_executable(yolo_cpp src/main.cpp)
# 官方包的布局是 include/ 下平铺头文件；这里是从源码树取的，多加一层 session 目录
target_include_directories(yolo_cpp PRIVATE
  "${ORT_INCLUDE}/onnxruntime/core/session"
  "${ORT_INCLUDE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(yolo_cpp PRIVATE "${ORT_DLL}")

# 读进程内存（稳定性测试要报 RSS 增长）。Windows 上是 psapi。
if(WIN32)
  target_link_libraries(yolo_cpp PRIVATE psapi)
endif()

# 把 MinGW 运行时 DLL 拷到 exe 旁边。
# 为什么不用 -static-libstdc++：静态链接后 exe 会额外依赖 UCRT 的 apiset
# （api-ms-win-crt-utility-l1-1-0.dll），在干净的 PATH 下反而更难搞。
# 拷 DLL 是 MSYS2 官方推荐的分发方式，行为最可预测。
if(MINGW)
  get_filename_component(_mingw_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
  foreach(_rt libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll)
    if(EXISTS "${_mingw_bin}/${_rt}")
      add_custom_command(TARGET yolo_cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_mingw_bin}/${_rt}" "$<TARGET_FILE_DIR:yolo_cpp>/${_rt}"
        COMMENT "复制 ${_rt}")
    endif()
  endforeach()
endif()

# 构建后把 DLL 拷到 exe 旁边。
# 注意必须拷「全部」三个：只拷 onnxruntime.dll 时，CUDA EP 会因为找不到
# onnxruntime_providers_shared.dll 而**静默回落到 CPU**（程序照跑，慢 2 倍多）。
get_filename_component(ORT_DIR "${ORT_DLL}" DIRECTORY)
file(GLOB ORT_DLLS "${ORT_DIR}/onnxruntime*.dll")
foreach(dll ${ORT_DLLS})
  get_filename_component(dll_name "${dll}" NAME)
  add_custom_command(TARGET yolo_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${dll}" "$<TARGET_FILE_DIR:yolo_cpp>/${dll_name}"
    COMMENT "复制 ${dll_name}")
endforeach()

message(STATUS "ORT headers : ${ORT_INCLUDE}")
message(STATUS "ORT runtime : ${ORT_DLL}")

# ------------------------------------------------------------------ OpenCV（可选）
# 只用来做三件事：读图、缩放、写图。找不到就退回 stb_image，功能不受影响。
#
# 为什么不给 MSVC 编译的官方包用 dlltool 转 .a：
#   OpenCV 的 C++ API 大量传 std::vector / std::string，这些类型的二进制布局由编译器决定。
#   MinGW 链接 MSVC 构建的 OpenCV 会在运行期崩溃或行为诡异。
#   所以要么整个项目用 MSVC（配 vcpkg），要么就用 MSYS2 这份「同一套 GCC 构建」的 OpenCV。
option(YOLO_WITH_OPENCV "用 OpenCV 做读图/缩放/写图（找不到会自动退回 stb）" ON)
if(YOLO_WITH_OPENCV)
  # MSYS2 的 UCRT64 前缀；用别的工具链时可用 -DOpenCV_DIR=... 覆盖
  list(APPEND CMAKE_PREFIX_PATH "D:/msys64/ucrt64")
  # videoio：读视频（cv::VideoCapture）。依赖 FFmpeg 后端，会多带一批 av*.dll
  find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs videoio)
  if(OpenCV_FOUND)
    target_compile_definitions(yolo_cpp PRIVATE YOLO_HAVE_OPENCV)
    target_include_directories(yolo_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(yolo_cpp PRIVATE ${OpenCV_LIBS})
    message(STATUS "OpenCV      : ${OpenCV_VERSION}  已启用（--backend stb 可切回手写实现）")

    # 把 OpenCV 运行时 DLL 也拷到 exe 旁边。
    # MSYS2 只提供导入库（.dll.a），没有 .a 静态库，所以没法全静态链接；
    # 不拷的话 exe 一离开 MSYS2 环境就报 "libopencv_imgproc-500.dll not found"（退出码 127）。
    # 只拷真正用到的三个模块：core / imgproc / imgcodecs（共约 20 MB）。
    # 注意 OpenCV 5 的头文件目录是 include/opencv5，config 在 lib/cmake/opencv5。
    if(DEFINED OpenCV_INSTALL_PATH)
      file(GLOB OCV_RUNTIME_DLLS
        "${OpenCV_INSTALL_PATH}/bin/libopencv_core-*.dll"
        "${OpenCV_INSTALL_PATH}/bin/libopencv_imgproc-*.dll"
        "${OpenCV_INSTALL_PATH}/bin/libopencv_imgcodecs-*.dll"
        "${OpenCV_INSTALL_PATH}/bin/libopencv_videoio-*.dll")
      foreach(dll ${OCV_RUNTIME_DLLS})
        get_filename_component(dll_name "${dll}" NAME)
        add_custom_command(TARGET yolo_cpp POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                  "${dll}" "$<TARGET_FILE_DIR:yolo_cpp>/${dll_name}"
          COMMENT "复制 ${dll_name}")
      endforeach()
      list(LENGTH OCV_RUNTIME_DLLS _ocv_n)
      message(STATUS "OpenCV DLL  : ${OpenCV_INSTALL_PATH}/bin（${_ocv_n} 个模块拷到输出目录）")
    endif()
  else()
    message(STATUS "OpenCV      : 未找到，退回 stb_image（功能相同，只是自己实现）")
  endif()
endif()
