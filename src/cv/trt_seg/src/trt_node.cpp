#include "trt_seg/trt_node.hpp"
#include "trt_unit.cpp"

static Logger gLogger;

// =========================================================
// 构造函数
// =========================================================
TRTNode::TRTNode(const std::string& engine_path) 
    : Node("trt_seg_node"), engine_path_(engine_path)
{
    // 空间线段初始化（保持不变）
    plane_grid_w_.line_1_ = {{150, 9950, 2420}, {150, 9950, 800}};
    plane_grid_w_.line_2_ = {{150, 10480, 2420}, {150, 10480, 800}};
    plane_grid_w_.line_3_ = {{150, 11020, 2420}, {150, 11020, 800}};
    plane_grid_w_.line_4_ = {{150, 11550, 2420}, {150, 11550, 800}};

    plane_grid_w_.line_5_ = {{150, 9940, 2410}, {150, 11560, 2410}};
    plane_grid_w_.line_6_ = {{150, 9940, 1880}, {150, 11560, 1880}};
    plane_grid_w_.line_7_ = {{150, 9940, 1340}, {150, 11560, 1340}};
    plane_grid_w_.line_8_ = {{150, 9940, 810}, {150, 11560, 810}};
    
    // 【核心修改】直接使用头文件中定义的 IMG_W (960) 和 IMG_H (720)
    W_ = IMG_W;
    H_ = IMG_H;

    // 预先创建与原图等比尺寸的 Host 端接收 Mask
    mask_.create(H_, W_, CV_8U);

    // 【修改】移除了分块专用的 build_windows() 函数调用
    init_trt();
    
    std::cout << "Unet++ 全图单张白边推理模式初始化完成！" << std::endl;
    running_ = true;
    // run();
}

// =========================================================
// 初始化 TensorRT 引擎及显存分配
// =========================================================
void TRTNode::init_trt() {
    std::ifstream file(engine_path_, std::ios::binary);
    if (!file.good()) throw std::runtime_error("Engine file not found: " + engine_path_);

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_  = runtime_->deserializeCudaEngine(engine_data.data(), size);
    context_ = engine_->createExecutionContext();

    input_name_  = engine_->getIOTensorName(0);
    output_name_ = engine_->getIOTensorName(1);

    cudaStreamCreate(&stream_);

    // 【核心修改】精简显存分配，全面服务于 1x3x736x960 紧凑全图
    cudaMalloc(&d_input_,    1 * 3 * NET_W * NET_H * sizeof(float)); // 网络输入张量
    cudaMalloc(&d_output_,   1 * 1 * NET_W * NET_H * sizeof(float)); // 网络输出概率图
    cudaMalloc(&d_img_full_, W_ * H_ * sizeof(uchar3));              // 存放 960x720 原始BGR图像
    cudaMalloc(&d_mask_out_, W_ * H_ * sizeof(uint8_t));             // 存放 960x720 裁剪后的二值化掩码

    context_->setInputTensorAddress(input_name_, d_input_);
    context_->setTensorAddress(output_name_, d_output_);

    // 【修改】彻底删除了坐标拷贝 cudaMemcpy(d_xs_...) 与 d_score_ 等历史代码
}

// =========================================================
// 析构与显存释放
// =========================================================
TRTNode::~TRTNode() {
    running_ = false;
    cv_frame_.notify_all(); 

    if (th_camera_.joinable())  th_camera_.join();
    if (th_process_.joinable()) th_process_.join();
    if (is_registered_) {
        cudaHostUnregister(resized_host_.data);
    }

    release();
}

void TRTNode::release() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
    
    // 【核心修改】清理多余的释放，仅保留现有的四个指针
    if (d_input_)     cudaFree(d_input_);
    if (d_output_)    cudaFree(d_output_);
    if (d_img_full_)  cudaFree(d_img_full_);
    if (d_mask_out_)  cudaFree(d_mask_out_);
    // if (d_score_)  cudaFree(d_score_); // 已移除
    // if (d_xs_)     cudaFree(d_xs_);    // 已移除
    // if (d_ys_)     cudaFree(d_ys_);    // 已移除

    if (context_) delete context_;
    if (engine_)  delete engine_;
    if (runtime_) delete runtime_;
}

// int main(int argc, char** argv) {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<TRTNode>();
//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }