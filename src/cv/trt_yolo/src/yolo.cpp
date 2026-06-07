#include <trt_yolo/yolo.hpp>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>

namespace trt_yolo {

// -----------------------------
// 默认类别/颜色：当用户未提供 class_names/colors 时使用。
// 说明：这些默认值仅影响可视化，不影响推理输出。
// -----------------------------
static std::vector<std::string> makeDefaultClassNames(int num_labels)
{
    if (num_labels <= 0) {
        num_labels = 1;
    }
    std::vector<std::string> names;
    names.reserve(static_cast<size_t>(num_labels));
    for (int i = 0; i < num_labels; ++i) {
        names.push_back("cls" + std::to_string(i));
    }
    return names;
}

static std::vector<std::vector<unsigned int>> makeDefaultColors(int num_labels)
{
    if (num_labels <= 0) {
        num_labels = 1;
    }
    static const std::vector<std::vector<unsigned int>> kPalette = {
        {0U, 114U, 189U},
        {217U, 83U, 25U},
        {237U, 177U, 32U},
        {126U, 47U, 142U},
        {119U, 172U, 48U},
        {77U, 190U, 238U},
        {162U, 20U, 47U},
    };
    std::vector<std::vector<unsigned int>> colors;
    colors.reserve(static_cast<size_t>(num_labels));
    for (int i = 0; i < num_labels; ++i) {
        colors.push_back(kPalette[static_cast<size_t>(i) % kPalette.size()]);
    }
    return colors;
}

YOLOv8::YOLOv8(const std::string& engine_file_path, YOLOv8Config config)
{
    // 构造函数主要做两件事：
    // 1) 应用配置（阈值/类别/颜色/输入尺寸）
    // 2) 加载 TensorRT engine 并初始化 runtime/engine/context/bindings
    applyConfig(std::move(config));

    std::ifstream file(engine_file_path, std::ios::binary);//以二进制模式打开文件
    assert(file.good());

    file.seekg(0, std::ios::end);
    auto size = file.tellg();//总字节大小
    file.seekg(0, std::ios::beg);

    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    initLibNvInferPlugins(&this->gLogger, "");//初始化并注册 TensorRT 官方提供的所有内置插件（Plugins）
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);//创建推理运行时环境 (Runtime)
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);//反序列化模型，生成推理引擎
    assert(this->engine != nullptr);
    delete[] trtModelStream;

    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    CUDA_CHECK(cudaStreamCreate(&this->stream));//创建 CUDA 异步执行流

    this->num_bindings = this->engine->getNbIOTensors();//获取模型的输入输出张量总数

    for (int i = 0; i < this->num_bindings; ++i) {
        det::Binding binding;
        const char*  name = engine->getIOTensorName(i);

        nvinfer1::DataType dtype = this->engine->getTensorDataType(name);

        binding.name  = name;
        binding.dsize = type_to_size(dtype);

        nvinfer1::TensorIOMode ioMode  = engine->getTensorIOMode(name);
        bool                   isInput = (ioMode == nvinfer1::TensorIOMode::kINPUT);

        if (isInput) {
            this->num_inputs += 1;
            nvinfer1::Dims dims = this->engine->getTensorShape(name);
            binding.size        = get_size_by_dims(dims);
            binding.dims        = dims;
            this->input_bindings.push_back(binding);
            this->context->setInputShape(name, dims);
        } else {
            nvinfer1::Dims dims = this->context->getTensorShape(name);
            binding.size        = get_size_by_dims(dims);
            binding.dims        = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

YOLOv8::~YOLOv8()
{
    if (this->context) {
        delete this->context;
        this->context = nullptr;
    }
    if (this->engine) {
        delete this->engine;
        this->engine = nullptr;
    }
    if (this->runtime) {
        delete this->runtime;
        this->runtime = nullptr;
    }

    if (this->stream) {
        CUDA_CHECK(cudaStreamDestroy(this->stream));
        this->stream = nullptr;
    }

    for (auto& ptr : this->device_ptrs) {
        if (ptr) {
            CUDA_CHECK(cudaFree(ptr));
        }
    }

    for (auto& ptr : this->host_ptrs) {
        if (ptr) {
            CUDA_CHECK(cudaFreeHost(ptr));
        }
    }
}

void YOLOv8::applyConfig(YOLOv8Config config)
{
    // 将对外 config 写入成员变量；并在必要时补齐默认 class_names/colors。
    // Core runtime parameters
    this->size        = config.input_size;
    this->topk        = config.topk;
    this->score_thres = config.score_thres;
    this->iou_thres   = config.iou_thres;

    // Classes
    if (!config.class_names.empty()) {
        this->class_names_ = std::move(config.class_names);
        this->num_labels   = static_cast<int>(this->class_names_.size());
    } else {
        this->num_labels   = (config.num_labels > 0) ? config.num_labels : 1;
        this->class_names_ = makeDefaultClassNames(this->num_labels);
    }

    if (!config.colors.empty()) {
        this->colors_ = std::move(config.colors);
    } else {
        this->colors_ = makeDefaultColors(this->num_labels);
    }
}

void YOLOv8::setNumLabels(int num_labels)
{
    if (num_labels <= 0) {
        return;
    }
    this->num_labels = num_labels;
    if (this->class_names_.empty() || static_cast<int>(this->class_names_.size()) != this->num_labels) {
        this->class_names_ = makeDefaultClassNames(this->num_labels);
    }
    if (this->colors_.empty()) {
        this->colors_ = makeDefaultColors(this->num_labels);
    }
}

void YOLOv8::setClassNames(std::vector<std::string> class_names)
{
    if (class_names.empty()) {
        return;
    }
    this->class_names_ = std::move(class_names);
    this->num_labels   = static_cast<int>(this->class_names_.size());
    if (this->colors_.empty()) {
        this->colors_ = makeDefaultColors(this->num_labels);
    }
}

void YOLOv8::setColors(std::vector<std::vector<unsigned int>> colors)
{
    if (colors.empty()) {
        return;
    }
    this->colors_ = std::move(colors);
}

void YOLOv8::detect(const cv::Mat& image)
{
    // 单帧检测主流程：preprocess -> infer -> postprocess -> draw
    this->objs.clear();
    this->copy_from_Mat(image, this->size);
    this->infer();
    this->postprocess(this->objs, this->score_thres, this->iou_thres, this->topk, this->num_labels);
    this->draw_objects(image, this->res, this->objs, this->resultFlag);
}

void YOLOv8::make_pipe(bool warmup)
{
    for (auto& bindings : this->input_bindings) {
        void* d_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ptr, bindings.size * bindings.dsize));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->output_bindings) {
        void*  d_ptr = nullptr;
        void*  h_ptr = nullptr;
        size_t size  = bindings.size * bindings.dsize;

        CUDA_CHECK(cudaMalloc(&d_ptr, size));
        CUDA_CHECK(cudaHostAlloc(&h_ptr, size, cudaHostAllocDefault));

        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (int in_idx = 0; in_idx < this->num_inputs; in_idx++) {
                auto&  in_binding = this->input_bindings[in_idx];
                size_t bytes      = in_binding.size * in_binding.dsize;
                CUDA_CHECK(cudaMemsetAsync(this->device_ptrs[in_idx], 0, bytes, this->stream));
            }
            this->infer();
        }
        std::cout << "model warmup 10 times" << std::endl;
    }
}

void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    } else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0F;
    dh /= 2.0F;
    int top    = int(std::round(dh - 0.1F));
    int bottom = int(std::round(dh + 0.1F));
    int left   = int(std::round(dw - 0.1F));
    int right  = int(std::round(dw + 0.1F));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});//中性灰色

    //是否交换 Red（红）和 Blue（蓝）通道
    //是否进行中心裁剪
    cv::dnn::blobFromImage(tmp, out, 1 / 255.F, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);

    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
}

void YOLOv8::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat nchw;

    auto&    in_binding = this->input_bindings[0];
    int      width      = static_cast<int>(in_binding.dims.d[3]);
    int      height     = static_cast<int>(in_binding.dims.d[2]);
    cv::Size size{width, height};

    this->letterbox(image, nchw, size);

    const char* input_name = this->engine->getIOTensorName(0);
    this->context->setInputShape(input_name, nvinfer1::Dims4(1, 3, size.height, size.width));

    size_t bytes = nchw.total() * nchw.elemSize();
    CUDA_CHECK(cudaMemcpyAsync(
        this->device_ptrs[0],
        nchw.ptr<float>(),
        bytes,
        cudaMemcpyHostToDevice,
        this->stream));
}

void YOLOv8::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);

    const char* input_name = this->engine->getIOTensorName(0);
    this->context->setInputShape(input_name, nvinfer1::Dims4(1, 3, size.height, size.width));

    size_t bytes = nchw.total() * nchw.elemSize();
    CUDA_CHECK(cudaMemcpyAsync(
        this->device_ptrs[0],
        nchw.ptr<float>(),
        bytes,
        cudaMemcpyHostToDevice,
        this->stream));
}

void YOLOv8::infer()
{
    for (int i = 0; i < this->num_inputs; ++i) {
        const std::string& name = this->input_bindings[i].name;
        void*              ptr  = this->device_ptrs[i];
        this->context->setInputTensorAddress(name.c_str(), ptr);
    }

    for (int i = 0; i < this->num_outputs; ++i) {
        const std::string& name = this->output_bindings[i].name;
        void* ptr = this->device_ptrs[this->num_inputs + i];
        this->context->setTensorAddress(name.c_str(), ptr);
    }

    if (!this->context->enqueueV3(this->stream)) {
        std::cerr << "Failed to enqueue inference with enqueueV3" << std::endl;
        return;
    }

    for (int i = 0; i < this->num_outputs; ++i) {
        size_t osize     = this->output_bindings[i].size * this->output_bindings[i].dsize;
        void*  dst_host  = this->host_ptrs[i];
        void*  src_dev   = this->device_ptrs[this->num_inputs + i];
        CUDA_CHECK(cudaMemcpyAsync(dst_host, src_dev, osize, cudaMemcpyDeviceToHost, this->stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(this->stream));
}

void YOLOv8::postprocess(std::vector<det::Object>& objs, float score_thres, float iou_thres, int topk, int num_labels)
{
    int labels_count = (num_labels > 0) ? num_labels : this->num_labels;
    if (labels_count <= 0) {
        labels_count = 1;
    }

    std::vector<cv::Rect> bboxes;
    std::vector<float>    scores;
    std::vector<int>      labels;
    std::vector<int>      indices;

    auto dim1 = this->output_bindings[0].dims.d[1];
    auto dim2 = this->output_bindings[0].dims.d[2];
    // std::cout << "dim1: " << dim1 << ", dim2: " << dim2 << std::endl;
    int num_channels, num_anchors;
    cv::Mat output;

    float dw = this->pparam.dw;
    float dh = this->pparam.dh;
    float w  = this->pparam.width;
    float h  = this->pparam.height;
    float r  = this->pparam.ratio;

    float* ptr = static_cast<float*>(this->host_ptrs[0]);//拿到从 GPU 拷回 CPU 的一维浮点数数组的首地址。
    if (dim1 > dim2)
    {
        num_anchors  = dim1; // 8400
        num_channels = dim2; // 6
        output = cv::Mat(num_anchors, num_channels, CV_32F, ptr);
    }
    else
    {
        num_channels = dim1; // 6
        num_anchors  = dim2; // 8400
        output = cv::Mat(num_channels, num_anchors, CV_32F, ptr);
        output = output.t();
    }

    bboxes.reserve(num_anchors);
    scores.reserve(num_anchors);
    labels.reserve(num_anchors);

    for (int i = 0; i < num_anchors; i++) 
    {
        auto  row_ptr   = output.ptr<float>(i);

        float* scores_ptr = row_ptr + 4;                       // + labels_count
        float* max_s_ptr  = std::max_element(scores_ptr, scores_ptr + labels_count);
        float  score      = *max_s_ptr;
        
        if (score < score_thres) {
            continue;
        }
        // std::cout << "score: " << std::fixed << std::setprecision(3) << score << std::endl;

        float x_center  = row_ptr[0] - dw;
        float y_center  = row_ptr[1] - dh;
        float box_w     = row_ptr[2];
        float box_h     = row_ptr[3];

        // float* scores_ptr = row_ptr + 4;
        // float* max_s_ptr  = std::max_element(scores_ptr, scores_ptr + labels_count);
        // float  score      = *max_s_ptr;
        // if (score < score_thres) {
        //     continue;
        // }

        int label = static_cast<int>(max_s_ptr - scores_ptr);

        float x0 = clamp((x_center - 0.5F * box_w) * r, 0.F, w);//边界0.F, w
        float y0 = clamp((y_center - 0.5F * box_h) * r, 0.F, h);
        float x1 = clamp((x_center + 0.5F * box_w) * r, 0.F, w);
        float y1 = clamp((y_center + 0.5F * box_h) * r, 0.F, h);

        // cv::Rect bbox;
        // bbox.x = static_cast<int>(x0);
        // bbox.y = static_cast<int>(y0);
        // bbox.width  = static_cast<int>(x1 - x0);
        // bbox.height = static_cast<int>(y1 - y0);
        // bboxes.push_back(bbox);
        bboxes.emplace_back(static_cast<int>(x0), static_cast<int>(y0), 
                                    static_cast<int>(x1 - x0), static_cast<int>(y1 - y0));
        labels.push_back(label);
        scores.push_back(score);
    }

#ifdef BATCHED_NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif
    objs.reserve(objs.size() + std::min(static_cast<int>(indices.size()), topk));
    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        det::Object obj;
        obj.rect     = bboxes[i];
        obj.prob     = scores[i];
        obj.label    = labels[i];
        obj.pass     = false;
        objs.emplace_back(std::move(obj));
        cnt += 1;
    }

    std::sort(objs.begin(), objs.end(), Compition);
}

void YOLOv8::draw_objects(
    const cv::Mat&                  image,
    cv::Mat&                        res,
    const std::vector<det::Object>& objs,
    bool&                           resultFlag)
{
    this->draw_objects(image, res, objs, resultFlag, this->class_names_, this->colors_);
}

void YOLOv8::draw_objects(
    const cv::Mat&                                image,
    cv::Mat&                                      res,
    const std::vector<det::Object>&               objs,
    bool&                                         resultFlag,
    const std::vector<std::string>&               class_names,
    const std::vector<std::vector<unsigned int>>& colors)
{
    res        = image.clone();
    resultFlag = !objs.empty();

    rects.clear();
    // std::cout << "objs.size(): " << objs.size() << std:: endl;

    for (const auto& obj : objs) 
    {
        const int label = obj.label;
        cv::Scalar color(0, 114, 189);
        if (!colors.empty()) {
            const auto& color_vec = colors[static_cast<size_t>(label < 0 ? 0 : label) % colors.size()];
            const unsigned int b  = (color_vec.size() > 0) ? color_vec[0] : 0U;
            const unsigned int g  = (color_vec.size() > 1) ? color_vec[1] : 114U;
            const unsigned int r  = (color_vec.size() > 2) ? color_vec[2] : 189U;
            color                = cv::Scalar(b, g, r);
        }
        cv::rectangle(res, obj.rect, color, 1);
        // std::cout << "Object Label: " << label 
        //                 << " -> Color (B,G,R): (" 
        //                 << static_cast<int>(color.val[0]) << ", "
        //                 << static_cast<int>(color.val[1]) << ", "
        //                 << static_cast<int>(color.val[2]) << ")" << std::endl;

        char text[256];
        const char* cls_name = "unknown";
        if (!class_names.empty() && label >= 0 && static_cast<size_t>(label) < class_names.size()) {
            cls_name = class_names[static_cast<size_t>(label)].c_str();
        }
        std::snprintf(text, sizeof(text), "%s %.1f%%", cls_name, obj.prob * 100);

        int      baseLine  = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = static_cast<int>(obj.rect.x);
        int y = static_cast<int>(obj.rect.y) - 10;
        if (y < 0) {
            y = static_cast<int>(obj.rect.y) + 20;
        }

        int x_ = static_cast<int>(obj.rect.x);
        int y_ = static_cast<int>(obj.rect.y);
        int w_ = static_cast<int>(obj.rect.width);
        int h_ = static_cast<int>(obj.rect.height);

        rects.push_back(cv::Rect(x_, y_, w_, h_));

        cv::rectangle(
            res,
            cv::Rect(x, y, label_size.width, label_size.height + baseLine),
            color,
            -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}

}  // namespace trt_yolo
