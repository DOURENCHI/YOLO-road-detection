// 最小验证：MinGW 能不能链接 MSVC 构建的 TensorRT？
//
// 判断依据是符号风格，不是猜：
//   nvinfer_10.dll 的导出表里 createInferRuntime 是**未修饰的 C 名字**
//   （createInferBuilder_INTERNAL / createInferRuntime_INTERNAL ...）。
//   TRT 其余 API 全是虚函数，通过 vtable 调用，不需要链接期符号。
//   所以理论可行 —— 但"理论可行"要跑一次才知道。
//
// 另外要验证：Ultralytics 的 .engine 前面有 [4 字节长度][JSON] 头，
// 直接丢给 deserializeCudaEngine 会失败，这里顺便确认。

#include <NvInfer.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity sev, char const* msg) noexcept override {
        if (sev <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
    }
};

static std::vector<char> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    const auto n = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(n);
    f.read(buf.data(), static_cast<std::streamsize>(n));
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: trt_probe <engine>\n";
        return 2;
    }
    Logger logger;

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "createInferRuntime 返回空\n";
        return 1;
    }
    std::cout << "IRuntime 创建成功（说明 MinGW 链接 MSVC 的 TRT 可行）\n";

    auto raw = read_file(argv[1]);
    if (raw.empty()) {
        std::cerr << "读不到文件\n";
        return 1;
    }
    std::cout << "文件大小 " << raw.size() << " 字节\n";

    // 探测 Ultralytics 的头：[4 字节小端长度][JSON]
    size_t offset = 0;
    if (raw.size() > 8) {
        uint32_t n = 0;
        std::memcpy(&n, raw.data(), 4);
        if (n > 0 && n < raw.size() - 4 && raw[4] == '{') {
            offset = 4 + n;
            std::cout << "检测到 Ultralytics 头：" << n << " 字节 JSON，plan 从偏移 " << offset << " 开始\n";
        }
    }

    nvinfer1::ICudaEngine* engine =
        runtime->deserializeCudaEngine(raw.data() + offset, raw.size() - offset);
    if (!engine) {
        std::cerr << "反序列化失败\n";
        return 1;
    }
    std::cout << "反序列化成功\n";

    std::cout << "IO 张量数 " << engine->getNbIOTensors() << "\n";
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        const auto mode = engine->getTensorIOMode(name);
        const auto dims = engine->getTensorShape(name);
        std::cout << "  " << (mode == nvinfer1::TensorIOMode::kINPUT ? "输入 " : "输出 ")
                  << name << "  dims=[";
        for (int d = 0; d < dims.nbDims; ++d)
            std::cout << dims.d[d] << (d + 1 < dims.nbDims ? ", " : "");
        std::cout << "]\n";
    }
    return 0;
}
