#pragma once

#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <string>

/**
 * @brief CUDA 调用检查宏。
 *
 * 使用：`CUDA_CHECK(cudaMalloc(...))`。
 * 如果返回值不是 `cudaSuccess`，会打印错误信息并 `std::exit(1)`。
 */
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        const cudaError_t error_code = (call);                                                                         \
        if (error_code != cudaSuccess) {                                                                               \
            std::cerr << "CUDA Error:\n"                                                                               \
                                << "    File:       " << __FILE__ << "\n"                                              \
                                << "    Line:       " << __LINE__ << "\n"                                              \
                                << "    Error code: " << static_cast<int>(error_code) << "\n"                          \
                                << "    Error text: " << cudaGetErrorString(error_code) << std::endl;                  \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

namespace trt_yolo {

/**
 * @brief TensorRT 日志输出（实现 nvinfer1::ILogger）。
 *
 * `reportableSeverity` 用于控制最小输出等级：
 * - severity > reportableSeverity 时不输出
 */
class Logger : public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO)
    : reportableSeverity(severity)
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity > reportableSeverity) {
            return;
        }

        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "VERBOSE: ";
                break;
        }

        std::cerr << msg << std::endl;
    }
};

inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType)
{
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 4;
    }
}

inline float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

inline bool IsPathExist(const std::string& path)
{
    return access(path.c_str(), 0) == F_OK;
}

inline bool IsFile(const std::string& path)
{
    if (!IsPathExist(path)) {
        std::cerr << "Path not exist: " << path << std::endl;
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

inline bool IsFolder(const std::string& path)
{
    if (!IsPathExist(path)) {
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

namespace det {
/**
 * @brief TensorRT binding 描述（输入/输出 tensor 元信息）。
 */
struct Binding {
    size_t         size  = 1;
    size_t         dsize = 1;
    nvinfer1::Dims dims;
    std::string    name;
};

/**
 * @brief 单个检测目标。
 */
struct Object {
    cv::Rect_<float> rect;
    int              label    = 0;
    float            prob     = 0.0F;
    float            x_center = 0.0F;
    bool             pass;
};

/**
 * @brief 预处理（letterbox）产生的参数，用于将框坐标映射回原图。
 */
struct PreParam {
    float ratio  = 1.0F;
    float dw     = 0.0F;
    float dh     = 0.0F;
    float height = 0.0F;
    float width  = 0.0F;
};
}  // namespace det

}  // namespace trt_yolo
