#pragma once

#include <NvInferPlugin.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include <trt_yolo/common.hpp>

namespace trt_yolo {

/**
 * @brief 用于后处理排序：按检测框中心 x 从大到小排序。
 *
 * 典型用途：取“最靠右/最靠左”的目标。
 */
inline bool Compition(const det::Object& a, const det::Object& b)
{
    return a.x_center > b.x_center;
}

/**
 * @brief YOLOv8 推理相关的可配置参数（初始化时传入）。
 *
 * 说明：
 * - `class_names/colors` 用于可视化（draw），不影响推理结果。
 * - `num_labels` 用于后处理时的类别数；若 `class_names` 非空，类别数将以其长度为准。
 */
struct YOLOv8Config {
    /// 类别数量（当 class_names 为空时使用）。
    /*
    * @param num_labels 类别数量
    * @param class_names 类别名称列表
    * @param colors 每个类别的框颜色（BGR）
    * @param input_size 网络输入尺寸（通常与导出的 TensorRT engine 一致，例如 640x640）
    * @param topk 后处理时最多保留的框数量
    * @param score_thres 置信度阈值（越大越“严格”）
    * @param iou_thres NMS IOU 阈值（越大越“宽松”）
    */
    YOLOv8Config(int num_labels_, const std::vector<std::string>& class_names_, const std::vector<std::vector<unsigned int>>& colors_,
                cv::Size input_size_, int topk_, float score_thres_, float iou_thres_) 
                : num_labels(num_labels_), class_names(class_names_), colors(colors_),
                input_size(input_size_), topk(topk_), score_thres(score_thres_), iou_thres(iou_thres_) {}

    YOLOv8Config() = default;
    int num_labels = 1;

    /// 类别名称列表（用于绘制文本）。若非空，类别数以该 vector 长度为准。
    std::vector<std::string> class_names{"obj"};

    /// 每个类别的框颜色（BGR）。若数量不足会循环使用。
    std::vector<std::vector<unsigned int>> colors{{0U, 114U, 189U}};

    /// 网络输入尺寸（通常与导出的 TensorRT engine 一致，例如 640x640）。
    cv::Size input_size{640, 640};

    /// NMS 后最多保留的框数量。
    int   topk        = 100;

    /// 置信度阈值（越大越“严格”）。
    float score_thres = 0.6F;

    /// NMS IOU 阈值（越大越“宽松”）。
    float iou_thres   = 0.65F;
};

/**
 * @brief TensorRT 加速的 YOLOv8 推理封装（C++）。
 *
 * 使用方式（典型）：
 * 1) 构造：加载 engine 并初始化 TensorRT runtime/context
 * 2) 调用 `make_pipe()`：分配 GPU/CPU 缓冲区并可选 warmup
 * 3) 对每帧调用 `detect()`：预处理 -> 推理 -> 后处理 -> 可视化
 *
 * 线程安全：
 * - 同一个 `YOLOv8` 实例不建议多线程并发调用 `detect()`（内部复用 buffer）。
 * - 多线程使用建议：每线程独立实例，或外部加锁。
 */
class YOLOv8 {
public:
    /**
     * @param engine_file_path TensorRT engine 路径（.engine）
     * @param config           可配置参数（类别、颜色、阈值、输入尺寸等）
     */
    explicit YOLOv8(const std::string& engine_file_path, YOLOv8Config config = {});
    ~YOLOv8();

    /**
     * @brief 对单张图像执行检测。
     *
     * 调用前置条件：必须先调用 `make_pipe()` 完成显存/页锁内存分配。
     * 结果输出：
     * - `objs`：后处理后的检测结果
     * - `res`：绘制后的可视化结果
     * - `rects`：绘制/下游逻辑使用的矩形列表（与 objs 对应的部分信息）
     */
    void detect(const cv::Mat& image);

    /**
     * @brief 初始化推理管道（分配 buffer）。
     * @param warmup 是否执行 warmup（默认 true，通常能减少首帧延迟）。
     */
    void make_pipe(bool warmup = true);

    /// 更新类别数（通常会同步生成默认 class_names/colors）。
    void setNumLabels(int num_labels);
    /// 更新类别名称列表（长度将覆盖 num_labels）。
    void setClassNames(std::vector<std::string> class_names);
    /// 更新颜色表（BGR）。
    void setColors(std::vector<std::vector<unsigned int>> colors);
    const std::vector<std::string>& classNames() const { return class_names_; }
    const std::vector<std::vector<unsigned int>>& colors() const { return colors_; }

    cv::Mat              res;
    std::vector<det::Object> objs;
    std::vector<cv::Rect>    rects;
    bool                 resultFlag  = false;

private:
    cv::Size             size        = cv::Size{640, 640};
    int                  num_labels  = 1;
    int                  topk        = 100;
    float                score_thres = 0.6F;
    float                iou_thres   = 0.65F;
    det::PreParam        pparam;
    
    int                  num_bindings = 0;
    int                  num_inputs   = 0;
    int                  num_outputs  = 0;
    std::vector<det::Binding> input_bindings;
    std::vector<det::Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;
    /// 将 config 写入成员变量，并补全默认 class_names/colors。
    void applyConfig(YOLOv8Config config);

    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};

    std::vector<std::string> class_names_;
    std::vector<std::vector<unsigned int>> colors_;
private:
    /// 将 Mat 预处理后拷贝到 GPU 输入（使用内部 input_size）。
    void copy_from_Mat(const cv::Mat& image);

    /// 将 Mat 预处理后拷贝到 GPU 输入（使用指定 size）。
    void copy_from_Mat(const cv::Mat& image, cv::Size& size);

    /// letterbox 预处理：保持比例缩放 + 填充，并输出 NCHW float blob。
    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);

    /// 执行一次推理（enqueueV3），并将输出异步拷回 host。
    void infer();

    /**
     * @brief 后处理：解析输出、阈值过滤、NMS。
     * @param objs       输出检测结果
     * @param score_thres 置信度阈值
     * @param iou_thres   NMS IOU 阈值
     * @param topk        最多保留框数量
     * @param num_labels  类别数；<0 时使用实例内 `num_labels`
     */
    void postprocess(
        std::vector<det::Object>& objs,
        float score_thres = 0.6F,
        float iou_thres   = 0.65F,
        int   topk        = 100,
        int   num_labels  = -1);

    void draw_objects(
        const cv::Mat&                  image,
        cv::Mat&                        res,
        const std::vector<det::Object>& objs,
        bool&                           resultFlag);

    void draw_objects(
        const cv::Mat&                                image,
        cv::Mat&                                      res,
        const std::vector<det::Object>&               objs,
        bool&                                         resultFlag,
        const std::vector<std::string>&               class_names,
        const std::vector<std::vector<unsigned int>>& colors);

};

}  // namespace trt_yolo
