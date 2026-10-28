// YOLO26n 道路检测 · C++ 推理最小闭环（ONNX Runtime）
//
// 这个程序的唯一目的：把 Python 里由库代劳的四件事，用 C++ 自己实现一遍，并证明结果一致。
//     ① 预处理：读图 → letterbox 到 640 → /255 → HWC 转 CHW
//     ② 推理  ：Ort::Session::Run
//     ③ 解码  ：[1,16,8400] → 框 + 分数 + 类别
//     ④ 后处理：置信度过滤 → 按类 NMS → 逆变换回原图坐标
//
// 设计上的三个刻意选择：
//     · 不依赖 OpenCV。读图和写图用 stb_image 单头文件，双线性缩放自己写 30 行。
//       理由是：C++ 部署真正的成本在「预处理一致性」，而不是在调一个库；
//       用 OpenCV 会把这件事藏起来，反而学不到东西。
//     · 支持 --input-bin：直接吃一个 float32 NCHW 张量文件。
//       这样可以把「预处理差异」和「推理+后处理差异」分开验证，出问题能定位到层。
//     · 计时只统计 Session::Run，与 Python 侧的「纯推理」口径对齐。
//
// 用法
//     yolo_cpp --model weights/best.onnx --image xxx.jpg
//     yolo_cpp --model weights/best.onnx --input-bin docs/deployment/_tensor.bin
//     yolo_cpp --model weights/best.onnx --image xxx.jpg --runs 200 --warmup 30

#include <onnxruntime_cxx_api.h>
// OrtTensorRTProviderOptionsV2 的完整定义只在这里（c_api.h 里是前向声明）
#include <onnxruntime/core/providers/tensorrt/tensorrt_provider_options.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// OpenCV 是可选依赖：编译时找到就用（-DYOLO_WITH_OPENCV=ON 或自动探测），
// 找不到就退回 stb。运行时还可以用 --backend 覆盖。
#ifdef YOLO_HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>   // 视频流（cv::VideoCapture）
#endif

// 读进程内存：稳定性测试要报 RSS 增长，判断有没有泄漏
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace {

constexpr int kInputSize = 640;   // 与训练/评估一致，不能改
constexpr int kPadValue = 114;    // letterbox 填充值：Ultralytics 用的就是 114
constexpr int kNumAnchors = 8400; // 80*80 + 40*40 + 20*20

// 图像 IO 与缩放由哪个库来做。
// 只有这三件事交给库：读图、缩放、写图。画框和解码仍然自己写——
// 画框要保证两个后端输出逐像素可比，解码是模型语义，不该被库改变。
enum class Backend { Stb, OpenCv };
Backend g_backend = Backend::Stb;

const char* backend_name(Backend b) { return b == Backend::OpenCv ? "OpenCV" : "stb"; }

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int cls;
};

struct LetterboxInfo {
    float gain = 1.f;  // 缩放比
    int pad_x = 0;     // 左侧填充（像素，640 空间）
    int pad_y = 0;
};

// ---------------------------------------------------------------- 图像

struct Image {
    int w = 0, h = 0;
    std::vector<unsigned char> rgb;  // RGB8
};

bool load_image(const std::string& path, Image& out) {
#ifdef YOLO_HAVE_OPENCV
    if (g_backend == Backend::OpenCv) {
        // imread 默认给 BGR，必须先转 RGB：模型训练时喂的是 RGB，
        // 顺序错了不会报错，只会让精度莫名其妙地掉几个点。
        cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
        if (bgr.empty()) {
            std::cerr << "读图失败：" << path << "（cv::imread 返回空，路径或格式有问题）\n";
            return false;
        }
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        out.w = rgb.cols;
        out.h = rgb.rows;
        // 保证连续再拷：ROI 之类的 Mat 可能带行间距（step > width*3）
        if (!rgb.isContinuous()) rgb = rgb.clone();
        out.rgb.assign(rgb.data, rgb.data + static_cast<size_t>(rgb.total()) * 3);
        return true;
    }
#endif
    int w = 0, h = 0, c = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3);  // 强制 3 通道
    if (!data) {
        std::cerr << "读图失败：" << path << "（" << stbi_failure_reason() << "）\n";
        return false;
    }
    out.w = w;
    out.h = h;
    out.rgb.assign(data, data + static_cast<size_t>(w) * h * 3);
    stbi_image_free(data);
    return true;
}

// 双线性缩放（单通道或多通道通用）。自己写而不是引库，见文件头说明。
void resize_bilinear(const unsigned char* src, int sw, int sh,
                     unsigned char* dst, int dw, int dh, int channels) {
    const float scale_x = static_cast<float>(sw) / dw;
    const float scale_y = static_cast<float>(sh) / dh;

    for (int y = 0; y < dh; ++y) {
        float fy = (y + 0.5f) * scale_y - 0.5f;
        fy = std::max(0.f, fy);
        int y0 = static_cast<int>(fy);
        int y1 = std::min(y0 + 1, sh - 1);
        float wy = fy - y0;

        for (int x = 0; x < dw; ++x) {
            float fx = (x + 0.5f) * scale_x - 0.5f;
            fx = std::max(0.f, fx);
            int x0 = static_cast<int>(fx);
            int x1 = std::min(x0 + 1, sw - 1);
            float wx = fx - x0;

            for (int ch = 0; ch < channels; ++ch) {
                float p00 = src[(static_cast<size_t>(y0) * sw + x0) * channels + ch];
                float p01 = src[(static_cast<size_t>(y0) * sw + x1) * channels + ch];
                float p10 = src[(static_cast<size_t>(y1) * sw + x0) * channels + ch];
                float p11 = src[(static_cast<size_t>(y1) * sw + x1) * channels + ch];
                float top = p00 + (p01 - p00) * wx;
                float bot = p10 + (p11 - p10) * wx;
                float val = top + (bot - top) * wy;
                dst[(static_cast<size_t>(y) * dw + x) * channels + ch] =
                    static_cast<unsigned char>(std::lround(std::min(255.f, std::max(0.f, val))));
            }
        }
    }
}

// 缩放分发：OpenCV 后端用 cv::resize(INTER_LINEAR)，否则用上面的手写版本。
//
// 注意：两者都遵循「半像素中心对齐」（dst = (src + 0.5) * scale - 0.5），
// 但 OpenCV 内部是定点整数运算，**不保证逐像素相等**，一般会有 ±1 的灰度差。
// 这个差别要不要紧，必须自己实测（脚本 13 就是干这个的）。
void resize_rgb(const unsigned char* src, int sw, int sh,
                unsigned char* dst, int dw, int dh) {
#ifdef YOLO_HAVE_OPENCV
    if (g_backend == Backend::OpenCv) {
        cv::Mat s(sh, sw, CV_8UC3, const_cast<unsigned char*>(src));
        cv::Mat d;
        cv::resize(s, d, cv::Size(dw, dh), 0.0, 0.0, cv::INTER_LINEAR);
        std::memcpy(dst, d.data, static_cast<size_t>(dw) * dh * 3);
        return;
    }
#endif
    resize_bilinear(src, sw, sh, dst, dw, dh, 3);
}

// letterbox：等比缩放到长边 640，居中补灰边。必须把 gain/pad 带出去，后处理要用。
void letterbox(const Image& img, std::vector<unsigned char>& out, LetterboxInfo& info) {
    info.gain = std::min(static_cast<float>(kInputSize) / img.w,
                         static_cast<float>(kInputSize) / img.h);
    const int new_w = static_cast<int>(std::lround(img.w * info.gain));
    const int new_h = static_cast<int>(std::lround(img.h * info.gain));

    std::vector<unsigned char> resized(static_cast<size_t>(new_w) * new_h * 3);
    resize_rgb(img.rgb.data(), img.w, img.h, resized.data(), new_w, new_h);

    out.assign(static_cast<size_t>(kInputSize) * kInputSize * 3, kPadValue);
    info.pad_x = (kInputSize - new_w) / 2;
    info.pad_y = (kInputSize - new_h) / 2;
    for (int y = 0; y < new_h; ++y) {
        std::memcpy(out.data() + (static_cast<size_t>(y + info.pad_y) * kInputSize + info.pad_x) * 3,
                    resized.data() + static_cast<size_t>(y) * new_w * 3,
                    static_cast<size_t>(new_w) * 3);
    }
}

// ---------------------------------------------------------------- 快速预处理（OpenCV 后端）
//
// 旧路径（letterbox + to_nchw）有三个浪费：
//   ① 每帧新建 3 个 vector：resized(1.2 MB) + boxed(1.2 MB) + nchw(4.9 MB) ≈ 7.3 MB 反复分配释放
//   ② 三次遍历：resize → 填充+逐行 memcpy → to_nchw 逐像素除 255
//   ③ to_nchw 是手写循环，写的时候跨平面跳（out[i] / out[plane+i] / out[2*plane+i]），缓存不友好
//
// 快速路径全部交给 OpenCV 的 SIMD 实现，并且所有缓冲只分配一次：
//   setTo(114) 填边 → resize 直接写进 canvas 的 ROI → split 拆平面 → convertTo 做 /255
// 注意 split 出来的平面顺序就是 canvas 的通道顺序（RGB），与 to_nchw 的 R/G/B 平面一致。
#ifdef YOLO_HAVE_OPENCV
struct PreWs {
    cv::Mat canvas;      // 640x640 CV_8UC3，letterbox 后的画布
    cv::Mat planes[3];   // split 的结果（只做头，不复制数据）
    cv::Mat f32;         // 640x640 CV_32F，convertTo 的暂存
    PreWs()
        : canvas(kInputSize, kInputSize, CV_8UC3),
          f32(kInputSize, kInputSize, CV_32F) {}
};

void preprocess_fast(const Image& img, PreWs& ws, LetterboxInfo& info, std::vector<float>& out) {
    info.gain = std::min(static_cast<float>(kInputSize) / img.w,
                         static_cast<float>(kInputSize) / img.h);
    const int nw = static_cast<int>(std::lround(img.w * info.gain));
    const int nh = static_cast<int>(std::lround(img.h * info.gain));
    info.pad_x = (kInputSize - nw) / 2;
    info.pad_y = (kInputSize - nh) / 2;

    ws.canvas.setTo(cv::Scalar(kPadValue, kPadValue, kPadValue));
    // Image 是 RGB，这里零拷贝包装成 Mat（不复制像素）
    cv::Mat src(img.h, img.w, CV_8UC3, const_cast<unsigned char*>(img.rgb.data()));
    cv::Mat roi = ws.canvas(cv::Rect(info.pad_x, info.pad_y, nw, nh));
    cv::resize(src, roi, roi.size(), 0.0, 0.0, cv::INTER_LINEAR);

    cv::split(ws.canvas, ws.planes);
    const size_t plane = static_cast<size_t>(kInputSize) * kInputSize;
    if (out.size() != plane * 3) out.resize(plane * 3);   // 只在第一帧分配
    for (int c = 0; c < 3; ++c) {
        ws.planes[c].convertTo(ws.f32, CV_32F, 1.0 / 255.0);
        std::memcpy(out.data() + c * plane, ws.f32.data, plane * sizeof(float));
    }
}
#endif

// HWC(0-255, RGB) → NCHW(0-1, RGB)，按通道排布
void to_nchw(const std::vector<unsigned char>& hwc, std::vector<float>& out) {
    const size_t plane = static_cast<size_t>(kInputSize) * kInputSize;
    out.assign(plane * 3, 0.f);
    for (size_t i = 0; i < plane; ++i) {
        out[i] = hwc[i * 3 + 0] / 255.f;             // R
        out[plane + i] = hwc[i * 3 + 1] / 255.f;     // G
        out[plane * 2 + i] = hwc[i * 3 + 2] / 255.f; // B
    }
}

// 统一入口：有 OpenCV 走 SIMD 路径，否则退回手写。
// legacy=true 时强制走旧路径，用来验证两条实现结果一致。
// （必须定义在 to_nchw 之后）
void preprocess_into(const Image& img, LetterboxInfo& info, std::vector<float>& out, bool legacy) {
#ifdef YOLO_HAVE_OPENCV
    if (!legacy && g_backend == Backend::OpenCv) {
        static PreWs ws;                  // 静态：缓冲只分配一次，后续帧复用
        preprocess_fast(img, ws, info, out);
        return;
    }
#endif
    (void)legacy;
    std::vector<unsigned char> boxed;
    letterbox(img, boxed, info);
    to_nchw(boxed, out);
}

// ---------------------------------------------------------------- 后处理

float iou(const Detection& a, const Detection& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float w = std::max(0.f, x2 - x1);
    const float h = std::max(0.f, y2 - y1);
    const float inter = w * h;
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

// 解码输出张量 [1, 4+nc, 8400]。
// 注意：输出不含 objectness，且类别分数已经是 sigmoid 后的概率（导出时已做），
// 所以直接取最大类别分数与阈值比较，不需要再 sigmoid 一次。
std::vector<Detection> decode(const float* data, int channels, int anchors,
                              float conf_thres, int num_classes,
                              const LetterboxInfo& info, int img_w, int img_h) {
    std::vector<Detection> dets;
    dets.reserve(64);

    for (int i = 0; i < anchors; ++i) {
        float best = 0.f;
        int best_cls = -1;
        for (int c = 0; c < num_classes; ++c) {
            const float s = data[static_cast<size_t>(4 + c) * anchors + i];
            if (s > best) {
                best = s;
                best_cls = c;
            }
        }
        if (best < conf_thres || best_cls < 0) continue;

        const float cx = data[0 * anchors + i];
        const float cy = data[1 * anchors + i];
        const float w = data[2 * anchors + i];
        const float h = data[3 * anchors + i];

        // 逆变换：先减 pad，再除 gain
        float x1 = (cx - w * 0.5f - info.pad_x) / info.gain;
        float y1 = (cy - h * 0.5f - info.pad_y) / info.gain;
        float x2 = (cx + w * 0.5f - info.pad_x) / info.gain;
        float y2 = (cy + h * 0.5f - info.pad_y) / info.gain;

        x1 = std::min(std::max(x1, 0.f), static_cast<float>(img_w - 1));
        y1 = std::min(std::max(y1, 0.f), static_cast<float>(img_h - 1));
        x2 = std::min(std::max(x2, 0.f), static_cast<float>(img_w - 1));
        y2 = std::min(std::max(y2, 0.f), static_cast<float>(img_h - 1));
        if (x2 <= x1 || y2 <= y1) continue;

        dets.push_back({x1, y1, x2, y2, best, best_cls});
    }

    // 按分数降序，再逐类 NMS（agnostic=false，和 Ultralytics 默认一致）
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<Detection> kept;
    std::vector<char> removed(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j] || dets[j].cls != dets[i].cls) continue;
            if (iou(dets[i], dets[j]) > 0.7f) removed[j] = 1;
        }
    }
    return kept;
}

// ---------------------------------------------------------------- 画框

void draw_rect(Image& img, int x1, int y1, int x2, int y2,
               unsigned char r, unsigned char g, unsigned char b) {
    auto put = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= img.w || y >= img.h) return;
        unsigned char* p = &img.rgb[(static_cast<size_t>(y) * img.w + x) * 3];
        p[0] = r; p[1] = g; p[2] = b;
    };
    for (int t = 0; t < 2; ++t) {  // 线宽 2 像素，缩略图也能看清
        for (int x = x1; x <= x2; ++x) { put(x, y1 + t); put(x, y2 - t); }
        for (int y = y1; y <= y2; ++y) { put(x1 + t, y); put(x2 - t, y); }
    }
}

void draw_detections(Image& img, const std::vector<Detection>& dets) {
    // 固定调色板，只为了区分不同类别，不追求好看
    static const unsigned char palette[12][3] = {
        {230, 25, 75},  {60, 180, 75},  {255, 225, 25}, {0, 130, 200},
        {245, 130, 48}, {145, 30, 180}, {70, 240, 240}, {240, 50, 230},
        {210, 245, 60}, {250, 190, 190}, {0, 128, 128}, {170, 110, 40}};
    for (const auto& d : dets) {
        const auto& c = palette[d.cls % 12];
        draw_rect(img, static_cast<int>(d.x1), static_cast<int>(d.y1),
                  static_cast<int>(d.x2), static_cast<int>(d.y2), c[0], c[1], c[2]);
    }
}

// ---------------------------------------------------------------- 写图

bool save_png(const std::string& path, const Image& img) {
#ifdef YOLO_HAVE_OPENCV
    if (g_backend == Backend::OpenCv) {
        // imwrite 认的是 BGR，和 imread 对称，别拿 RGB 直接写，否则红蓝互换
        cv::Mat rgb(img.h, img.w, CV_8UC3, const_cast<unsigned char*>(img.rgb.data()));
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return cv::imwrite(path, bgr);
    }
#endif
    return stbi_write_png(path.c_str(), img.w, img.h, 3, img.rgb.data(), img.w * 3) != 0;
}

// ---------------------------------------------------------------- 工具

// JSON 字符串转义。Windows 路径里的反斜杠如果不转义，写出来的就不是合法 JSON
// （Python 侧 json.loads 会报 Invalid \escape），排查起来很费时间。
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out;
}

std::vector<float> read_floats(const std::string& path, size_t expect) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "打不开 " << path << "\n";
        std::exit(2);
    }
    std::vector<float> buf(expect);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(expect * sizeof(float)));
    if (static_cast<size_t>(f.gcount()) != expect * sizeof(float))
        std::cerr << "警告：张量文件长度不足，可能被截断\n";
    return buf;
}

std::wstring widen(const std::string& s) {
    return std::wstring(s.begin(), s.end());  // 路径只含 ASCII 时够用
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t idx = static_cast<size_t>(std::min<double>(v.size() - 1, p * (v.size() - 1)));
    return v[idx];
}

// 当前进程的常驻内存（MB）。稳定性测试每 100 帧采一次，看有没有泄漏。
size_t rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<size_t>(pmc.WorkingSetSize / (1024 * 1024));
#endif
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, image_path, bin_path, out_prefix, json_path, dump_input;
    std::string ep = "cuda", trt_cache = "weights/trt_cache", video_path;
    int runs = 100, warmup = 20, num_classes = 12, repeat = 0;
    int frames = 0, warmup_frames = 30;
    bool no_draw = false, legacy_pre = false;
    float conf = 0.25f;

#ifdef YOLO_HAVE_OPENCV
    // 编译期就编进了 OpenCV → 默认用它；要对比就显式 --backend stb
    g_backend = Backend::OpenCv;
#endif

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--model") model_path = next();
        else if (a == "--image") image_path = next();
        else if (a == "--input-bin") bin_path = next();
        else if (a == "--runs") runs = std::stoi(next());
        else if (a == "--repeat") repeat = std::stoi(next());   // >0 时跑完整链路（每轮重新读图）
        else if (a == "--video") video_path = next();            // 视频流模式
        else if (a == "--frames") frames = std::stoi(next());    // 视频模式处理帧数上限（0=全部）
        else if (a == "--warmup-frames") warmup_frames = std::stoi(next());
        else if (a == "--no-draw") no_draw = true;
        else if (a == "--legacy-pre") legacy_pre = true;   // 走旧的预处理实现（用于一致性对比）
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--conf") conf = std::stof(next());
        else if (a == "--classes") num_classes = std::stoi(next());
        else if (a == "--out") out_prefix = next();
        else if (a == "--json") json_path = next();
        else if (a == "--dump-input") dump_input = next();
        else if (a == "--ep") {
            ep = next();
            if (ep != "cuda" && ep != "trt") {
                std::cerr << "--ep 只接受 cuda 或 trt\n";
                return 2;
            }
        }
        else if (a == "--trt-cache") trt_cache = next();
        else if (a == "--backend") {
            const std::string v = next();
            if (v == "stb") {
                g_backend = Backend::Stb;
            } else if (v == "opencv") {
#ifndef YOLO_HAVE_OPENCV
                std::cerr << "本次编译没有启用 OpenCV，只能用 stb。\n"
                             "（CMake 没找到 OpenCV 时会自动退回 stb；装了 OpenCV 后重跑 cmake 即可）\n";
                return 2;
#endif
                g_backend = Backend::OpenCv;
            } else {
                std::cerr << "--backend 只接受 stb 或 opencv\n";
                return 2;
            }
        }
    }
    if (model_path.empty()) {
        std::cerr << "必须指定 --model\n";
        return 2;
    }

    // ---- Session ----
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_cpp");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    OrtCUDAProviderOptions cuda_opts{};   // device_id 默认 0
    if (ep == "trt") {
        // TensorRT EP：ORT 会把能交给 TRT 的子图切出去，剩下的仍由 CUDA/CPU 跑。
        //
        // 这里有两个坑，都踩过：
        //  ① 不能用通用的 SessionOptionsAppendExecutionProvider(按名字)：那个接口认识
        //     DML/QNN/OpenVINO，但**不认识 TensorRT**，会报 Unknown provider name。
        //  ② OrtTensorRTProviderOptionsV2 在 c_api.h 里只有前向声明，直接 {} 初始化会
        //     报 incomplete type。必须用 CreateTensorRTProviderOptions 创建，再用
        //     UpdateTensorRTProviderOptions 逐项设置（结构体内部有非平凡成员）。
        const std::string cache = trt_cache;
        OrtApi const& api = Ort::GetApi();
        OrtTensorRTProviderOptionsV2* trt_opts = nullptr;
        try {
            Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt_opts));
            const char* keys[] = {"device_id", "trt_fp16_enable", "trt_engine_cache_enable",
                                  "trt_engine_cache_path", "trt_timing_cache_enable"};
            const char* vals[] = {"0", "1", "1", cache.c_str(), "1"};
            Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt_opts, keys, vals, 5));
            Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(opts, trt_opts));
            api.ReleaseTensorRTProviderOptions(trt_opts);
            std::cout << "EP：TensorRT（FP16）+ CUDA 兜底，缓存 " << cache << "\n";
        } catch (const std::exception& e) {
            if (trt_opts) api.ReleaseTensorRTProviderOptions(trt_opts);
            std::cerr << "TensorRT EP 不可用，回落到 CUDA：" << e.what() << "\n";
        }
        try {
            opts.AppendExecutionProvider_CUDA(cuda_opts);
        } catch (const std::exception& e) {
            std::cerr << "CUDA EP 也不可用：" << e.what() << "\n";
        }
    } else {
        try {
            opts.AppendExecutionProvider_CUDA(cuda_opts);
        } catch (const std::exception& e) {
            std::cerr << "CUDA EP 不可用，将回落到 CPU：" << e.what() << "\n";
        }
    }

    // Session 构造会真正加载 EP 插件（如 onnxruntime_providers_tensorrt.dll），
    // 失败原因常常是缺 nvinfer_10.dll 而不是模型有问题，所以要把 what() 打出来。
    //
    // 模型路径先转绝对路径：TRT EP 内部解析相对路径会失败
    // （报 "Load model ..\weights\best.onnx failed: File doesn't exist"），
    // 而 CUDA EP 下同样的相对路径是好的——这种「换 EP 就找不到文件」很难查。
    std::unique_ptr<Ort::Session> session_holder;
    try {
        const std::string abs_model = std::filesystem::absolute(model_path).string();
        session_holder = std::make_unique<Ort::Session>(env, widen(abs_model).c_str(), opts);
    } catch (const std::exception& e) {
        std::cerr << "创建 Session 失败：" << e.what() << "\n"
                  << "（EP 插件加载失败时通常是缺 DLL：nvinfer_10.dll / cublasLt64_12.dll / cudnn64_9.dll）\n";
        return 3;
    }
    Ort::Session& session = *session_holder;
    Ort::AllocatorWithDefaultOptions allocator;
    const std::string in_name = session.GetInputNameAllocated(0, allocator).get();
    const std::string out_name = session.GetOutputNameAllocated(0, allocator).get();
    std::cout << "输入 " << in_name << "　输出 " << out_name << "\n";
    std::cout << "图像后端：" << backend_name(g_backend) << "（只负责读图 / 缩放 / 写图）\n";

    // ---- 视频流模式：稳定性测试（FPS / p99 延迟 / 内存增长）----
    //
    // 和 --repeat 的区别：--repeat 读同一张图片重复处理，测的是「单帧成本」；
    // 视频模式带真正的解码（VideoCapture），连续跑 N 帧，才能看出
    // 帧率稳不稳、尾部延迟有多差、内存会不会缓慢上涨。
    if (!video_path.empty()) {
#ifdef YOLO_HAVE_OPENCV
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "打不开视频：" << video_path
                      << "\n（VideoCapture 走 FFmpeg 后端，缺 av*.dll 时会打不开）\n";
            return 2;
        }
        const int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        const int vw = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int vh = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double vfps = cap.get(cv::CAP_PROP_FPS);
        const int limit = (frames > 0 && frames < total) ? frames : total;
        std::cout << "\n视频：" << video_path << "（" << total << " 帧，" << vw << "x" << vh
                  << "，" << vfps << " fps）本次处理 " << limit << " 帧，预热 " << warmup_frames
                  << " 帧\n";

        std::array<int64_t, 4> vshape{1, 3, kInputSize, kInputSize};
        Ort::MemoryInfo vmem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const char* vin[] = {in_name.c_str()};
        const char* vout[] = {out_name.c_str()};
        const unsigned long long mem0 = static_cast<unsigned long long>(rss_mb());

        std::vector<double> v_tot, v_dec, v_pre, v_inf, v_post, v_draw;
        std::vector<int> mem_at;
        std::vector<size_t> mem_val;

        auto now = [] { return std::chrono::steady_clock::now(); };
        auto span = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };

        // 这几个缓冲提到循环外：同一路视频尺寸固定，复用可避免每帧 7 MB 的分配释放
        cv::Mat bgr, rgb;
        Image im;
        std::vector<float> inp;
        LetterboxInfo lb;
        int done = 0, valid = 0, last_count = 0;
        while (done < limit) {
            const auto c0 = now();
            if (!cap.read(bgr) || bgr.empty()) break;
            const auto c1 = now();

            im.w = bgr.cols; im.h = bgr.rows;
            const size_t need = static_cast<size_t>(im.w) * im.h * 3;
            if (im.rgb.size() != need) im.rgb.resize(need);   // 只在首帧分配
            cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
            std::memcpy(im.rgb.data(), rgb.data, need);

            preprocess_into(im, lb, inp, legacy_pre);
            const auto c2 = now();

            Ort::Value tv = Ort::Value::CreateTensor<float>(vmem, inp.data(), inp.size(),
                                                           vshape.data(), vshape.size());
            auto outs = session.Run(Ort::RunOptions{nullptr}, vin, &tv, 1, vout, 1);
            const auto c3 = now();

            const auto os = outs[0].GetTensorTypeAndShapeInfo().GetShape();
            auto dets = decode(outs[0].GetTensorData<float>(), static_cast<int>(os[1]),
                               static_cast<int>(os[2]), conf, num_classes, lb, im.w, im.h);
            const auto c4 = now();

            if (!no_draw) draw_detections(im, dets);
            const auto c5 = now();

            last_count = static_cast<int>(dets.size());
            if (done >= warmup_frames) {
                v_tot.push_back(span(c5 - c0));
                v_dec.push_back(span(c1 - c0));
                v_pre.push_back(span(c2 - c1));
                v_inf.push_back(span(c3 - c2));
                v_post.push_back(span(c4 - c3));
                v_draw.push_back(span(c5 - c4));
                ++valid;
            }
            if (done % 100 == 0) { mem_at.push_back(done); mem_val.push_back(rss_mb()); }
            ++done;
        }
        cap.release();

        if (valid == 0) {
            std::cerr << "有效帧为 0（视频太短或预热帧数设太大）\n";
            return 2;
        }

        const double total_ms = std::accumulate(v_tot.begin(), v_tot.end(), 0.0);
        const double med_v = percentile(v_tot, 0.5);
        const unsigned long long m_first = mem_val.front();
        const unsigned long long m_last = mem_val.back();
        unsigned long long m_max = 0;
        for (auto v : mem_val) m_max = std::max(m_max, static_cast<unsigned long long>(v));
        const long long m_delta = static_cast<long long>(m_last) - static_cast<long long>(m_first);

        std::printf("\n稳定性测试：共处理 %d 帧（其中 %d 帧计入统计）\n", done, valid);
        std::printf("  吞吐      %.1f FPS\n", 1000.0 * valid / total_ms);
        std::printf("  每帧延迟  中位 %.2f ms　p95 %.2f　p99 %.2f　最大 %.2f\n",
                    med_v, percentile(v_tot, 0.95), percentile(v_tot, 0.99),
                    *std::max_element(v_tot.begin(), v_tot.end()));
        std::printf("  分段中位  解码 %.2f　预处理 %.2f　推理 %.2f　后处理 %.2f　画框 %.2f\n",
                    percentile(v_dec, 0.5), percentile(v_pre, 0.5), percentile(v_inf, 0.5),
                    percentile(v_post, 0.5), percentile(v_draw, 0.5));
        std::printf("  内存     起始 %llu MB　结束 %llu MB　峰值 %llu MB　净增 %+lld MB\n",
                    m_first, m_last, m_max, m_delta);
        std::cout << "  采样     ";
        for (size_t i = 0; i < mem_val.size(); ++i)
            std::cout << mem_val[i] << (i + 1 < mem_val.size() ? " → " : " MB\n");
        std::cout << "  最后一帧 检出 " << last_count << " 个目标\n";
        std::cout << "  判读　　";
        if (m_delta > 50 && mem_val.size() >= 3 && mem_val.back() > mem_val.front())
            std::cout << "内存净增 " << m_delta << " MB 且趋势向上，需要复查是否泄漏\n";
        else
            std::cout << "内存净增 " << m_delta << " MB 且在正常波动范围（含预热与分配器缓存）\n";

        if (!json_path.empty()) {
            std::ofstream jf(json_path);
            jf << "{\n  \"mode\": \"video\",\n  \"video\": \"" << json_escape(video_path)
               << "\",\n  \"source_frames\": " << total
               << ",\n  \"processed\": " << done << ",\n  \"valid\": " << valid
               << ",\n  \"warmup_frames\": " << warmup_frames
               << ",\n  \"fps\": " << 1000.0 * valid / total_ms
               << ",\n  \"latency_ms\": {\"median\": " << med_v
               << ", \"p95\": " << percentile(v_tot, 0.95)
               << ", \"p99\": " << percentile(v_tot, 0.99)
               << ", \"max\": " << *std::max_element(v_tot.begin(), v_tot.end())
               << ", \"mean\": " << total_ms / valid << "}"
               << ",\n  \"stages_ms\": {\"decode\": " << percentile(v_dec, 0.5)
               << ", \"preprocess\": " << percentile(v_pre, 0.5)
               << ", \"infer\": " << percentile(v_inf, 0.5)
               << ", \"postprocess\": " << percentile(v_post, 0.5)
               << ", \"draw\": " << percentile(v_draw, 0.5) << "}"
               << ",\n  \"memory_mb\": {\"at_start\": " << mem0
               << ", \"first_sample\": " << m_first << ", \"last_sample\": " << m_last
               << ", \"peak\": " << m_max << ", \"delta\": " << m_delta
               << ",\n    \"samples\": [";
            for (size_t i = 0; i < mem_val.size(); ++i)
                jf << (i ? ", " : "") << mem_val[i];
            jf << "]}\n}\n";
            std::cout << "结果：" << json_path << "\n";
        }
        return 0;
#else
        std::cerr << "--video 需要 OpenCV（本次编译未启用），无法读视频\n";
        return 2;
#endif
    }

    // ---- 准备输入张量 ----
    std::vector<float> input;               // NCHW
    Image img;                              // 用于画图；bin 模式下留空
    LetterboxInfo info;

    if (!bin_path.empty()) {
        input = read_floats(bin_path, static_cast<size_t>(3) * kInputSize * kInputSize);
        std::cout << "输入来源：" << bin_path << "（已预处理好，跳过预处理）\n";
    } else if (!image_path.empty()) {
        if (!load_image(image_path, img)) return 2;
        preprocess_into(img, info, input, legacy_pre);
        std::cout << "输入来源：" << image_path << "（" << img.w << "x" << img.h
                  << "，gain " << info.gain << "，pad " << info.pad_x << "," << info.pad_y
                  << "，预处理 " << (legacy_pre ? "旧实现" : "快速路径") << "）\n";
    } else {
        std::cerr << "需要 --image 或 --input-bin\n";
        return 2;
    }

    std::array<int64_t, 4> shape{1, 3, kInputSize, kInputSize};

    // 把预处理后的 float32 张量原样落盘：两个后端各 dump 一次再逐值相减，
    // 就能回答「OpenCV 的 resize 和手写双线性到底一不一样」这个具体问题。
    if (!dump_input.empty()) {
        std::ofstream df(dump_input, std::ios::binary);
        df.write(reinterpret_cast<const char*>(input.data()),
                 static_cast<std::streamsize>(input.size() * sizeof(float)));
        std::cout << "预处理张量：" << dump_input << "（" << input.size() << " 个 float32）\n";
    }

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                       shape.data(), shape.size());
    const char* in_names[] = {in_name.c_str()};
    const char* out_names[] = {out_name.c_str()};

    // ---- 预热 ----
    for (int i = 0; i < warmup; ++i) {
        session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
    }

    // ---- 计时：只包住 Run，与 Python 的「纯推理」对齐 ----
    std::vector<double> samples;
    samples.reserve(runs);
    std::vector<float> last_output;
    std::vector<int64_t> out_shape;

    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        if (i == runs - 1) {
            out_shape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
            const float* p = outs[0].GetTensorData<float>();
            last_output.assign(p, p + std::accumulate(out_shape.begin(), out_shape.end(), int64_t{1},
                                                      std::multiplies<int64_t>()));
        }
    }

    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    std::cout << "\n输出形状 [";
    for (size_t i = 0; i < out_shape.size(); ++i)
        std::cout << out_shape[i] << (i + 1 < out_shape.size() ? ", " : "");
    std::cout << "]\n";

    // ---- 解码（需要原图尺寸才能逆变换；bin 模式下用 640 假定无 padding）----
    if (img.w == 0) { img.w = kInputSize; img.h = kInputSize; info = LetterboxInfo{}; }
    const int channels = static_cast<int>(out_shape[1]);
    const int anchors = static_cast<int>(out_shape[2]);
    auto dets = decode(last_output.data(), channels, anchors, conf, num_classes, info, img.w, img.h);

    std::cout << "检出 " << dets.size() << " 个目标（conf>=" << conf << "）\n";
    for (size_t i = 0; i < dets.size() && i < 10; ++i) {
        std::printf("  cls %2d  score %.3f  [%.1f, %.1f, %.1f, %.1f]\n",
                    dets[i].cls, dets[i].score, dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);
    }

    std::printf("\n纯推理 %d 次：中位数 %.2f ms　p95 %.2f ms　均值 %.2f ms　最小 %.2f ms\n",
                runs, percentile(samples, 0.5), percentile(samples, 0.95), mean,
                *std::min_element(samples.begin(), samples.end()));

    // ---- 端到端：每轮都重新读图 ----
    //
    // 为什么必须重新读图：上面那个「纯推理」只包住了 Session::Run。
    // 真实部署里读图、letterbox、解码、NMS 一样要花时间，
    // 只测 Run 会让人误以为换个引擎就能解决延迟问题。
    //
    // 画框只画到内存，不写盘：写 PNG 是一次性动作，混进延迟会让数字失去意义。
    struct E2E {
        std::vector<double> total, load, pre, infer, post, draw;
    } e2e;

    if (repeat > 0 && !image_path.empty()) {
        auto now_ms = [] { return std::chrono::steady_clock::now(); };
        auto span = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };

        auto pipeline = [&](double& t_load, double& t_pre, double& t_infer,
                            double& t_post, double& t_draw) {
            const auto c0 = now_ms();
            Image im;
            if (!load_image(image_path, im)) {
                std::cerr << "端到端：读图失败\n";
                std::exit(2);
            }
            const auto c1 = now_ms();

            std::vector<unsigned char> boxed;
            LetterboxInfo lb;
            std::vector<float> inp;
            preprocess_into(im, lb, inp, legacy_pre);
            const auto c2 = now_ms();

            // CreateTensor 计入「推理」段：它是每次调用 ORT 都要付的封装成本
            Ort::Value tv = Ort::Value::CreateTensor<float>(
                mem, inp.data(), inp.size(), shape.data(), shape.size());
            auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &tv, 1, out_names, 1);
            const auto c3 = now_ms();

            const auto os = outs[0].GetTensorTypeAndShapeInfo().GetShape();
            auto d = decode(outs[0].GetTensorData<float>(), static_cast<int>(os[1]),
                            static_cast<int>(os[2]), conf, num_classes, lb, im.w, im.h);
            const auto c4 = now_ms();

            draw_detections(im, d);
            const auto c5 = now_ms();

            t_load = span(c1 - c0); t_pre = span(c2 - c1); t_infer = span(c3 - c2);
            t_post = span(c4 - c3); t_draw = span(c5 - c4);
        };

        const int warm_e2e = std::max(3, warmup / 4);
        for (int i = 0; i < warm_e2e; ++i) {
            double a, b, c, d, e; pipeline(a, b, c, d, e);
        }
        for (int i = 0; i < repeat; ++i) {
            double a, b, c, d, e;
            const auto t0 = now_ms();
            pipeline(a, b, c, d, e);
            const auto t1 = now_ms();
            e2e.total.push_back(span(t1 - t0));
            e2e.load.push_back(a); e2e.pre.push_back(b); e2e.infer.push_back(c);
            e2e.post.push_back(d); e2e.draw.push_back(e);
        }

        const double med_e2e = percentile(e2e.total, 0.5);
        std::printf("\n端到端 %d 次（每轮重新读图，预热 %d）：中位 %.2f ms　p95 %.2f　最小 %.2f　均值 %.2f\n",
                    repeat, warm_e2e, med_e2e, percentile(e2e.total, 0.95),
                    *std::min_element(e2e.total.begin(), e2e.total.end()),
                    std::accumulate(e2e.total.begin(), e2e.total.end(), 0.0) / e2e.total.size());
        std::cout << "  分段中位：\n";
        struct Seg { const char* name; const std::vector<double>* v; };
        const Seg segs[] = {{"读图+解码", &e2e.load}, {"预处理", &e2e.pre},
                            {"推理(+张量封装)", &e2e.infer}, {"后处理(解码+NMS)", &e2e.post},
                            {"画框", &e2e.draw}};
        for (const auto& s : segs) {
            const double v = percentile(*s.v, 0.5);
            std::printf("    %-20s %6.3f ms  %5.1f%%\n", s.name, v, 100.0 * v / med_e2e);
        }
    }

    // ---- 存结果 JSON：便于和 Python 侧自动比对 ----
    if (!json_path.empty()) {
        std::ofstream jf(json_path);
        jf << "{\n  \"image\": \"" << json_escape(image_path.empty() ? std::string("(tensor)") : image_path)
           << "\",\n  \"w\": " << img.w << ", \"h\": " << img.h
           << ",\n  \"conf_thres\": " << conf
           << ",\n  \"runs\": " << runs
           << ",\n  \"median_ms\": " << percentile(samples, 0.5)
           << ",\n  \"p95_ms\": " << percentile(samples, 0.95)
           << ",\n  \"mean_ms\": " << mean;
        if (!e2e.total.empty()) {
            jf << ",\n  \"end_to_end\": {\n    \"repeat\": " << repeat
               << ",\n    \"median_ms\": " << percentile(e2e.total, 0.5)
               << ",\n    \"p95_ms\": " << percentile(e2e.total, 0.95)
               << ",\n    \"min_ms\": " << *std::min_element(e2e.total.begin(), e2e.total.end())
               << ",\n    \"stages_median_ms\": {\"load_and_decode\": " << percentile(e2e.load, 0.5)
               << ", \"preprocess\": " << percentile(e2e.pre, 0.5)
               << ", \"infer_and_wrap\": " << percentile(e2e.infer, 0.5)
               << ", \"postprocess\": " << percentile(e2e.post, 0.5)
               << ", \"draw\": " << percentile(e2e.draw, 0.5) << "}\n  }";
        }
        jf << ",\n  \"count\": " << dets.size() << ",\n  \"boxes\": [";
        for (size_t i = 0; i < dets.size(); ++i) {
            const auto& d = dets[i];
            jf << (i ? ",\n    " : "\n    ")
               << "[" << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2
               << ", " << d.score << ", " << d.cls << "]";
        }
        jf << "\n  ]\n}\n";
        std::cout << "结果：" << json_path << "\n";
    }

    // ---- 存图与结果 ----
    if (!out_prefix.empty() && !image_path.empty()) {
        Image vis = img;
        draw_detections(vis, dets);
        const std::string png = out_prefix + "_boxes.png";
        if (save_png(png, vis)) {
            std::cout << "图：" << png << "\n";
        } else {
            std::cerr << "写图失败：" << png << "\n";
        }
    }
    return 0;
}
