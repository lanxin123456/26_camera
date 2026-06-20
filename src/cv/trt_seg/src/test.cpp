#include "trt_seg/trt_node.hpp"

#include "trt_unit.cpp"



TRTNode::TRTNode() : Node("trt_seg_node") {

    // 预读一帧以捕获分辨率

    cv::VideoCapture temp_cap(VIDEO_PATH);

    if (!temp_cap.isOpened()) {

        throw std::runtime_error("Failed to open video file for initial setup: " + VIDEO_PATH);

    }

    cv::Mat temp_frame;

    temp_cap >> temp_frame;

    if (temp_frame.empty()) {

        throw std::runtime_error("Failed to read first frame from video: " + VIDEO_PATH);

    }

    

    W_ = temp_frame.cols;

    H_ = temp_frame.rows;

    temp_cap.release();



    build_windows();

    init_trt();



    running_ = true;

    run();

}



TRTNode::~TRTNode() {

    running_ = false;

    cv_frame_.notify_all(); // 唤醒可能阻塞在等待队列的处理线程



    if (th_camera_.joinable())  th_camera_.join();

    if (th_process_.joinable()) th_process_.join();



    release();

}



void TRTNode::run() {

    th_camera_  = std::thread(&TRTNode::camera, this);

    th_process_ = std::thread(&TRTNode::process, this);

}



void TRTNode::camera() {

    cv::VideoCapture cap(VIDEO_PATH);

    if (!cap.isOpened()) {

        RCLCPP_ERROR(this->get_logger(), "Failed to open video source sequence!");

        running_ = false;

        return;

    }



    while (rclcpp::ok() && running_) {

        cv::Mat frame;

        cap >> frame;

        if (frame.empty()) {

            // 环形循环读取视频

            cap.set(cv::CAP_PROP_POS_FRAMES, 0);

            continue;

        }



        // frame = cv::imread("/home/lx/frame/yolo_522_camera_2/1779501495_452537.jpg");

        {

            std::unique_lock<std::mutex> lock(global_mtx_);

            global_frame_ = std::move(frame); // 零拷贝转移所有权

            has_frame_ = true;

        }

        cv_frame_.notify_one();



        // 避免读取过快爆内存，匹配正常视频帧率阻尼（如30fps ≈ 33ms）

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    }

}



void TRTNode::process() {

    // 声明聚类与追踪所需的阈值

    static const float DIST_THRESH = 50.0f;     

    static const float ANGLE_THRESH = 0.174f;   



    while (rclcpp::ok() && running_) 

    {

        cv::Mat frame;

        {

            std::unique_lock<std::mutex> lock(global_mtx_);

            cv_frame_.wait(lock, [this] { return has_frame_ || !running_ || !rclcpp::ok(); });

            

            if (!rclcpp::ok() || !running_) break;

            

            frame = global_frame_.clone();

            has_frame_ = false;

        }



        // ==================== GPU 核心流水线操作 ====================

        auto t0 = std::chrono::high_resolution_clock::now();



        cudaMemsetAsync(d_score_, 0, W_ * H_ * sizeof(float), stream_);

        cudaMemsetAsync(d_count_, 0, W_ * H_ * sizeof(float), stream_);



        // 动态注册锁页内存提升传输带宽

        cudaHostRegister(frame.data, W_ * H_ * sizeof(uchar3), cudaHostRegisterMapped);

        cudaMemcpyAsync(d_img_full_, frame.data, W_ * H_ * sizeof(uchar3), cudaMemcpyHostToDevice, stream_);



        launch_preprocess_batch(d_img_full_, W_, H_, d_xs_, d_ys_, (float*)d_input_, BATCH, stream_);

        context_->setInputShape(input_name_, nvinfer1::Dims4(BATCH, 3, PATCH, PATCH));

        context_->enqueueV3(stream_);

        launch_merge_batch((float*)d_output_, d_score_, d_count_, W_, H_, BATCH, d_xs_, d_ys_, stream_);



        launch_finalize_mask(d_score_, d_count_, d_mask_out_, W_ * H_, THRESH, stream_);

        cv::Mat mask(H_, W_, CV_8U);

        cudaMemcpyAsync(mask.data, d_mask_out_, W_ * H_ * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream_);



        cudaStreamSynchronize(stream_);

        cudaHostUnregister(frame.data); // 必须在同步后释放注册



        auto t1 = std::chrono::high_resolution_clock::now();



        // ==================== 图像形态学后处理 ====================

        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));



        cv::Mat v_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 100));

        cv::Mat h_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(100, 1));

        cv::Mat vertical_lines, horizontal_lines;



        cv::morphologyEx(mask, vertical_lines, cv::MORPH_OPEN, v_kernel);

        cv::morphologyEx(mask, horizontal_lines, cv::MORPH_OPEN, h_kernel);



        cv::Mat merged_mask = cv::Mat::zeros(mask.size(), CV_8UC1);



        std::vector<LineCandidate> v_candidates = extractAndClusterLines(vertical_lines, merged_mask, true, DIST_THRESH, ANGLE_THRESH);

        std::vector<LineCandidate> h_candidates = extractAndClusterLines(horizontal_lines, merged_mask, false, DIST_THRESH, ANGLE_THRESH);



        auto t2 = std::chrono::high_resolution_clock::now();



        // ==================== 拓扑时序槽位追踪与节点输出 ====================

        trackGridAndGetNodes(h_candidates, v_candidates, frame, merged_mask);



        auto t3 = std::chrono::high_resolution_clock::now();



        std::cout << "CUDA TIME = " << std::chrono::duration<double, std::milli>(t1 - t0).count()

                << " PRE TIME = " << std::chrono::duration<double, std::milli>(t2 - t1).count()

                << "ALL TIME = " << std::chrono::duration<double, std::milli>(t3 - t0).count() << "\n";



        cv::namedWindow("Binary Mask", cv::WINDOW_NORMAL);

        cv::namedWindow("Merged Geometry", cv::WINDOW_NORMAL);

        cv::imshow("Binary Mask", mask);

        cv::imshow("Merged Geometry", merged_mask);



        std::string save_dir = "/home/lx/兰欣20241872/python/UNet++/canvas_525_2/";

        std::stringstream ss;

        ss << save_dir << "frame_" << std::setw(4) << std::setfill('0') << ++frame_idx_ << ".png";

        std::string save_path = ss.str();

        bool success = cv::imwrite(save_path, merged_mask);

        if (!success) {

            std::cout << "\033[1;31m[ERROR] 无法保存图片，请检查路径是否存在: " << save_path << "\033[0m" << std::endl;

        }

        else{

            std::cout << "\n第 " << frame_idx_ << " 号图片记录完毕！\n" << std::endl;

        }

        cv::waitKey(1);

    }

}



void TRTNode::release() {

    if (stream_) {

        cudaStreamDestroy(stream_);

    }

    if (d_input_)     cudaFree(d_input_);

    if (d_output_)    cudaFree(d_output_);

    if (d_img_full_)  cudaFree(d_img_full_);

    if (d_score_)     cudaFree(d_score_);

    if (d_count_)     cudaFree(d_count_);

    if (d_xs_)        cudaFree(d_xs_);

    if (d_ys_)        cudaFree(d_ys_);

    if (d_mask_out_)  cudaFree(d_mask_out_);



    if (context_) delete context_;

    if (engine_)  delete engine_;

    if (runtime_) delete runtime_;

}



int main(int argc, char** argv) {

    rclcpp::init(argc, argv);

    auto node = std::make_shared<TRTNode>();

    rclcpp::spin(node);

    rclcpp::shutdown();

    return 0;

}

#include "trt_seg/trt_node.hpp"



static Logger gLogger;



void TRTNode::build_windows() {

    std::cout << "[INIT] Video Frame Size: W=" << W_ << " H=" << H_ << std::endl;

    for (int y = 0; y <= H_ - PATCH; y += STRIDE) ys_.push_back(y);

    if (ys_.back() != H_ - PATCH) ys_.push_back(H_ - PATCH);



    for (int x = 0; x <= W_ - PATCH; x += STRIDE) xs_.push_back(x);

    if (xs_.back() != W_ - PATCH) xs_.push_back(W_ - PATCH);

}





void TRTNode::init_trt() {

    std::ifstream file(ENGINE_PATH, std::ios::binary);

    if (!file.good()) throw std::runtime_error("Engine file not found: " + ENGINE_PATH);



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

    size_t patch_pixels = PATCH * PATCH;



    cudaMalloc(&d_input_,  BATCH * 3 * patch_pixels * sizeof(float));

    cudaMalloc(&d_output_, BATCH * patch_pixels * sizeof(float));

    cudaMalloc(&d_img_full_, W_ * H_ * sizeof(uchar3));

    cudaMalloc(&d_xs_, BATCH * sizeof(int));

    cudaMalloc(&d_ys_, BATCH * sizeof(int));

    cudaMalloc(&d_score_, W_ * H_ * sizeof(float));

    cudaMalloc(&d_count_, W_ * H_ * sizeof(float));

    cudaMalloc(&d_mask_out_, W_ * H_);



    context_->setInputTensorAddress(input_name_, d_input_);

    context_->setTensorAddress(output_name_, d_output_);



    std::vector<cv::Rect> windows;

    for (int y : ys_) {

        for (int x : xs_) {

            windows.emplace_back(x, y, PATCH, PATCH);

        }

    }

    std::vector<int> xs_batch, ys_batch;

    for (const auto& r : windows) {

        xs_batch.push_back(r.x);

        ys_batch.push_back(r.y);

    }

    cudaMemcpy(d_xs_, xs_batch.data(), BATCH * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_ys_, ys_batch.data(), BATCH * sizeof(int), cudaMemcpyHostToDevice);

}





cv::Point2f TRTNode::computeIntersection(const cv::Vec4f& line1, const cv::Vec4f& line2) {

    float vx1 = line1[0], vy1 = line1[1], x1 = line1[2], y1 = line1[3];

    float vx2 = line2[0], vy2 = line2[1], x2 = line2[2], y2 = line2[3];

    float det = vx1 * vy2 - vy1 * vx2;

    if (std::abs(det) < 1e-5) return cv::Point2f(-1, -1); 

    float t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det;

    return cv::Point2f(x1 + vx1 * t, y1 + vy1 * t);

}



cv::Vec4f TRTNode::slotToVec4f(const TrackedLine& tl, bool is_vertical) {

    cv::Vec4f lp;

    lp[0] = std::cos(tl.angle);

    lp[1] = std::sin(tl.angle);

    if (is_vertical) {

        lp[2] = tl.intercept;

        lp[3] = H_ / 2.0f; 

    } else {

        lp[2] = W_ / 2.0f; 

        lp[3] = tl.intercept;

    }

    return lp;

}



/**

 * @brief 连通域提取、清洗与贪心聚类融合统一流水线

 * @param morph_mask  形态学处理后的二值图 (vertical_lines 或 horizontal_lines)

 * @param merged_mask 累加绘制所用掩码画布

 * @param is_vertical 标识当前是否在处理竖线

 * @param dist_thresh 聚类截距距离阈值

 * @param angle_thresh 聚类角度阈值

 */

std::vector<TRTNode::LineCandidate> TRTNode::extractAndClusterLines(

    const cv::Mat& morph_mask, 

    cv::Mat& merged_mask, 

    bool is_vertical, 

    float dist_thresh, 

    float angle_thresh) 

{

    // ==================== 1. 提取并清洗候选体 ====================

    std::vector<LineCandidate> raw_candidates;

    cv::Mat labels, stats, centroids;

    int num_components = cv::connectedComponentsWithStats(morph_mask, labels, stats, centroids);

    

    for (int i = 1; i < num_components; i++) {

        if (stats.at<int>(i, cv::CC_STAT_AREA) < 500) continue;



        cv::Mat component_mask = (labels == i);

        merged_mask.setTo(255, component_mask);



        std::vector<cv::Point> points;

        cv::findNonZero(component_mask, points);

        if (points.size() < 2) continue;



        cv::Vec4f lp;

        cv::fitLine(points, lp, cv::DIST_L2, 0, 0.01, 0.01);

        

        float intercept;

        // 添加 1e-5f 防止直线极度平滑导致的除以零崩溃

        if (is_vertical) {

            float dy = (lp[1] == 0.0f) ? 1e-5f : lp[1];

            intercept = lp[2] + (lp[0] / dy) * (H_ / 2.0f - lp[3]); 

        } else {

            float dx = (lp[0] == 0.0f) ? 1e-5f : lp[0];

            intercept = lp[3] + (lp[1] / dx) * (W_ / 2.0f - lp[2]);

        }

        float angle = std::atan2(lp[1], lp[0]);

        raw_candidates.push_back({points, intercept, angle});

    }



    // ==================== 2. 贪心聚类融合 ====================

    std::vector<LineCandidate> final_candidates; 

    std::vector<bool> visited(raw_candidates.size(), false);

    

    for (size_t i = 0; i < raw_candidates.size(); ++i) {

        if (visited[i]) continue;

        std::vector<cv::Point> merged_points = raw_candidates[i].points;

        visited[i] = true;



        for (size_t j = i + 1; j < raw_candidates.size(); ++j) {

            if (visited[j]) continue;

            float d_intercept = std::abs(raw_candidates[i].intercept - raw_candidates[j].intercept);

            float d_angle = std::abs(raw_candidates[i].angle - raw_candidates[j].angle);

            if (d_angle > CV_PI / 2) d_angle = CV_PI - d_angle;

            

            if (d_intercept < dist_thresh && d_angle < angle_thresh) {

                merged_points.insert(merged_points.end(), raw_candidates[j].points.begin(), raw_candidates[j].points.end());

                visited[j] = true;

            }

        }

        

        cv::Vec4f final_lp;

        cv::fitLine(merged_points, final_lp, cv::DIST_L2, 0, 0.01, 0.01);

        

        float final_intercept;

        if (is_vertical) {

            float dy = (final_lp[1] == 0.0f) ? 1e-5f : final_lp[1];

            final_intercept = final_lp[2] + (final_lp[0] / dy) * (H_ / 2.0f - final_lp[3]);

        } else {

            float dx = (final_lp[0] == 0.0f) ? 1e-5f : final_lp[0];

            final_intercept = final_lp[3] + (final_lp[1] / dx) * (W_ / 2.0f - final_lp[2]);

        }

        float final_angle = std::atan2(final_lp[1], final_lp[0]);

        

        // ==================== 3：基于矢量投影的长度筛选 ====================

        float min_t = std::numeric_limits<float>::max();

        float max_t = -std::numeric_limits<float>::max();

        

        float vx = final_lp[0]; // 直线的单位方向向量 X

        float vy = final_lp[1]; // 直线的单位方向向量 Y

        float x0 = final_lp[2]; // 直线上的一点 X

        float y0 = final_lp[3]; // 直线上的一点 Y



        // 将所有合并后的点投影到直线的方向矢量上，揪出两端的极值点坐标

        for (const auto& pt : merged_points) {

            float t = (pt.x - x0) * vx + (pt.y - y0) * vy;

            if (t < min_t) min_t = t;

            if (t > max_t) max_t = t;

        }

        float line_length = max_t - min_t; // 得到该线段在这帧画面里的真实物理长度



        if (line_length < 200) {

            std::cout << "\033[1;31m[LINE FILTERED] 剔除不合理短线 -> 方向: " 

                      << (is_vertical ? "竖线" : "横线") 

                      << " | 截距: " << final_intercept 

                      << " | 物理长度: " << line_length 

                      << " (低于阈值 " << 200 << ")\033[0m" << std::endl;

            for (const auto& pt : merged_points){

                merged_mask.at<uchar>(pt) = 0;

            }

            continue; // 优雅地略过，不加入最终的候选集

        }

        // =====================================================================



        final_candidates.push_back({merged_points, final_intercept, final_angle});

    }

    

    return final_candidates;

}



/**

 * @brief 单向线段时序追踪核心引擎

 * @param candidates 当前帧检测到的候选线线索

 * @param tracked_lines 历史追踪槽位（大小固定为 4）

 */

void TRTNode::trackSingleDirection(std::vector<LineCandidate>& candidates, std::vector<TrackedLine>& tracked_lines, bool is_vertical) 
{
    // ==================== 满足条件时分类输出初始截距 ====================
    if (candidates.size() >= 2) {
        std::string dir_str = is_vertical ? "Vertical" : "Horizontal";
        std::cout << "=== [Before Matching] " << dir_str << " Intercepts Base ===" << std::endl;
        
        std::cout << "   Candidates (" << candidates.size() << " lines): ";
        for (size_t i = 0; i < candidates.size(); ++i) {
            std::cout << "[" << i << "]" << candidates[i].intercept << "  ";
        }
        std::cout << std::endl;

        std::cout << "   Tracked Slots: ";
        for (int k = 0; k < 4; ++k) {
            std::cout << "[" << k << "]" << tracked_lines[k].intercept << "  ";
        }
        std::cout << "\n========================================================" << std::endl;
    }
    // ====================================================================
    
    std::vector<bool> det_matched(candidates.size(), false);
    std::vector<int> track_to_det(4, -1);
    int match_count = 0;   

    // 1. 基础数据关联（贪心匹配）
    for (int k = 0; k < 4; ++k) {
        int best_idx = -1;
        float min_dist = 150.0f;  
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (det_matched[i]) continue;
            float dist = std::abs(candidates[i].intercept - tracked_lines[k].intercept);
            if (dist < min_dist) { min_dist = dist; best_idx = i; }
        }
        if (best_idx != -1) {
            track_to_det[k] = best_idx;
            det_matched[best_idx] = true;
            match_count++;
        }
    }

    // 2. 核心改进：通过已匹配的线估计缩放比例(a)与平移(b)
    float scale_a = 1.0f;
    float shift_b = 0.0f;

    if (match_count >= 2) {
        // 利用最小二乘法拟合 y = a*x + b
        float sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
        for (int k = 0; k < 4; ++k) {
            if (track_to_det[k] != -1) {
                float x = tracked_lines[k].intercept;                  // 历史截距
                float y = candidates[track_to_det[k]].intercept;       // 当前检测截距
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;
            }
        }
        
        float denominator = match_count * sum_xx - sum_x * sum_x;
        if (std::abs(denominator) > 1e-3f) {
            scale_a = (match_count * sum_xy - sum_x * sum_y) / denominator;
            shift_b = (sum_y - scale_a * sum_x) / match_count;

            // 阻尼与安全保护：单帧间的缩放和突变不可能太离谱（设定安全阈值：0.8 ~ 1.25）
            if (scale_a < 0.8f || scale_a > 1.25f) {
                std::cout << "\033[1;33m[WARN] 估计的缩放率异常 (" << scale_a << ")，退化为纯平移模型\033[0m" << std::endl;
                scale_a = 1.0f;
                float total_shift = 0;
                for (int k = 0; k < 4; ++k) {
                    if (track_to_det[k] != -1) {
                        total_shift += (candidates[track_to_det[k]].intercept - tracked_lines[k].intercept);
                    }
                }
                shift_b = total_shift / match_count;
            }
        } else {
            // 分母过小说明匹配到的几条线截距几乎重合（异常情况），退化为均值平移
            scale_a = 1.0f;
            float total_shift = 0;
            for (int k = 0; k < 4; ++k) {
                if (track_to_det[k] != -1) total_shift += (candidates[track_to_det[k]].intercept - tracked_lines[k].intercept);
            }
            shift_b = total_shift / match_count;
        }
    } 
    else if (match_count == 1) {
        // 如果极其不幸只剩 1 条线匹配上了，无法估算缩放，只能退化为纯平移
        scale_a = 1.0f;
        for (int k = 0; k < 4; ++k) {
            if (track_to_det[k] != -1) {
                shift_b = candidates[track_to_det[k]].intercept - tracked_lines[k].intercept;
                break;
            }
        }
    } 
    else {
        // 0 条线匹配：完全盲跑，保持原样不动
        scale_a = 1.0f;
        shift_b = 0.0f;
    }

    // 打印当前帧估算出来的运动趋势
    if (match_count >= 2) {
        std::cout << "\033[1;36m[MOTION ESTIMATE] 匹配线数: " << match_count 
                  << " | 缩放系数(a): " << scale_a 
                  << " (" << (scale_a > 1.0f ? "网格扩大/靠近" : "网格缩小/远离") << ")"
                  << " | 偏移基准(b): " << shift_b << "\033[0m" << std::endl;
    }

    // 3. 更新与时序外推预测
    for (int k = 0; k < 4; ++k) {
        if (track_to_det[k] == -1) {
            // 对于当前看不见的线，利用科学的线性缩放模型预测它的新位置！
            tracked_lines[k].intercept = scale_a * tracked_lines[k].intercept + shift_b;
            tracked_lines[k].is_visible = false;
        } else {
            // 对于看得见的线，直接用实测值强制更新，防止漂移
            int idx = track_to_det[k];
            tracked_lines[k].intercept = candidates[idx].intercept;
            tracked_lines[k].angle = candidates[idx].angle;
            tracked_lines[k].is_visible = true;
        }
    }

    // ==================== 终端可视化高亮输出 ====================
    std::string direction_label = is_vertical ? "\033[1;34m[竖线追踪 (VERTICAL)]\033[0m" 
                                              : "\033[1;33m[横线追踪 (HORIZONTAL)]\033[0m";
    
    std::cout << direction_label << " -----------------------------------" << std::endl;
    std::cout << "条数：" << candidates.size() << std::endl;
    for (int k = 0; k < 4; ++k) {
        std::string status = tracked_lines[k].is_visible ? "\033[1;32mVISIBLE\033[0m" : "\033[1;31mLOCKED_LOST (预测)\033[0m";
        std::cout << " new tracked_lines [" << k << "] -> "
                  << "Intercept: " << std::fixed << std::setprecision(2) << std::setw(7) << tracked_lines[k].intercept 
                  << " | Angle: " << std::setprecision(4) << std::setw(7) << tracked_lines[k].angle
                  << " | Status: " << status << std::endl;
    }
    std::cout << "---------------------------------------------------------" << std::endl;
}


void TRTNode::trackGridAndGetNodes(std::vector<LineCandidate>& h_candidates, 

                                    std::vector<LineCandidate>& v_candidates, 

                                    const cv::Mat& current_frame,

                                    const cv::Mat& mask) 

{

    std::vector<float> final_h_intercepts, final_h_angles;

    std::vector<float> final_v_intercepts, final_v_angles;


    bool h_ok = reconstructGridLines(h_candidates, H_, false, mask, final_h_intercepts, final_h_angles);

    bool v_ok = reconstructGridLines(v_candidates, W_, true, mask, final_v_intercepts, final_v_angles);



    bool global_correction_applied = false;


    if (h_ok && v_ok) {
        bool grid_reasonable = true;
        std::string error_msg = "";
        for (int i = 0; i < 4; ++i) {
            if (std::abs(final_h_angles[i]) > 0.4f) { // 约允许 ±23 度的倾斜

                grid_reasonable = false;

                error_msg += "横线[" + std::to_string(i) + "]角度异常(" + std::to_string(final_h_angles[i]) + "); ";

            }
            float abs_v_angle = std::abs(final_v_angles[i]);
            if (std::abs(abs_v_angle - 1.5708f) > 0.4f) {
                grid_reasonable = false;

                error_msg += "竖线[" + std::to_string(i) + "]角度异常(" + std::to_string(final_v_angles[i]) + "); ";

            }
        }
        // 根据你之前提的 80 像素标准步长，我们设置一个安全的边界（如 40 ~ 150 像素）
        for (int i = 0; i < 3; ++i) {
            float h_gap = final_h_intercepts[i+1] - final_h_intercepts[i];
            float v_gap = final_v_intercepts[i+1] - final_v_intercepts[i];

            if (h_gap < 40.0f || h_gap > 300.0f) {
                grid_reasonable = false;
                error_msg += "横线间距[" + std::to_string(i) + "-" + std::to_string(i+1) + "]异常(" + std::to_string(h_gap) + "); ";
            }
            if (v_gap < 40.0f || v_gap > 300.0f) {
                grid_reasonable = false;
                error_msg += "竖线间距[" + std::to_string(i) + "-" + std::to_string(i+1) + "]异常(" + std::to_string(v_gap) + "); ";
            }
        }
        if (grid_reasonable) {
            // 【完全合理】：直接强行覆盖并重置追踪槽位（完成时序上的绝对矫正）
            for (int i = 0; i < 4; ++i) {
                tracked_h_[i] = {i, final_h_intercepts[i], final_h_angles[i], true};
                tracked_v_[i] = {i, final_v_intercepts[i], final_v_angles[i], true};
            }
            global_correction_applied = true;
            if (!grid_initialized_) {
                grid_initialized_ = true;
                std::cout << "\033[1;32m>>> [INIT SUCCESS] Grid Slots Initialized Successfully via Reasonable Geometry! <<<\033[0m" << std::endl;
            } else {
                std::cout << "\033[1;36m>>> [DYNAMIC CALIBRATION] Tracked slots corrected successfully via global reconstruction. <<<\033[0m" << std::endl;
            }
        } else {
            // 【不合理】：在终端爆红输出原因，拒绝本次全局矫正
            std::cout << "\033[1;31m>>> [GRID REJECTED] Reconstruction UNREASONABLE! Reason: " << error_msg << "\033[0m" << std::endl;
        }
    }
    // 如果系统至今连一次完美的初始化都没成功过，且本帧重建也失败/不合理，直接退场，不盲画交点
    if (!grid_initialized_) {
        return;
    }
    // 如果本帧由于各种原因（检测线少于3条、或重建网格畸变）没有应用全局矫正，

    if (!global_correction_applied) {
        trackSingleDirection(h_candidates, tracked_h_, false);
        trackSingleDirection(v_candidates, tracked_v_, true);
    }

    if (grid_initialized_) {
        std::vector<float> visible_gaps;
        // 遍历所有配对槽位，只要相互之间都是“真实可见”的，就提取其标准化单格像素距离
        for (int i = 0; i < 4; ++i) {
            for (int j = i + 1; j < 4; ++j) {
                if (tracked_h_[i].is_visible && tracked_h_[j].is_visible) {
                    visible_gaps.push_back(std::abs(tracked_h_[i].intercept - tracked_h_[j].intercept) / (j - i));
                }
                if (tracked_v_[i].is_visible && tracked_v_[j].is_visible) {
                    visible_gaps.push_back(std::abs(tracked_v_[i].intercept - tracked_v_[j].intercept) / (j - i));
                }
            }
        }

        // 准则：优先用真实可见的线计算间距
        if (!visible_gaps.empty()) {
            float sum_gap = 0;
            for (float g : visible_gaps) sum_gap += g;
            float current_measured_grid = sum_gap / visible_gaps.size();

            // 如果已有历史步长，我们可以反推出本帧实际发生的精确运动缩放因子
            if (tracked_grid_size_ > 0) {
                last_scale_a_ = current_measured_grid / tracked_grid_size_;
                // 阻尼安全保护，单帧形变不应超出合理常理
                if (last_scale_a_ < 0.8f || last_scale_a_ > 1.25f) last_scale_a_ = 1.0f;
            }
            // 写入记忆库
            tracked_grid_size_ = current_measured_grid;
        } 
        else {
            // 如果所有方向上真实可见的线都小于2（全是盲跑推理出来的线），
            // 此时直接保持 tracked_grid_size_ 现状不变（它的间距已经自动跟随 trackSingleDirection 里的各个 scale_a 进行了外推）
            // 缩放率退化为 1.0f 稳定锚定
            last_scale_a_ = 1.0f;
        }
    }
    // ===================================================================================

    // 6. 交点图绘制展现（保持原样）

    cv::Mat canvas = current_frame.clone();

    cv::Point2f nodes[4][4];

    for (int i = 0; i < 4; ++i) {

        for (int j = 0; j < 4; ++j) {

            cv::Vec4f line_h = slotToVec4f(tracked_h_[i], false);

            cv::Vec4f line_v = slotToVec4f(tracked_v_[j], true);

            nodes[i][j] = computeIntersection(line_h, line_v);



            if (std::isnan(nodes[i][j].x) || std::isnan(nodes[i][j].y) || 

                std::isinf(nodes[i][j].x) || std::isinf(nodes[i][j].y)) {

                continue; // 遇到无效点，直接跳过不画

            }



            if (nodes[i][j].x >= 0 && nodes[i][j].x < canvas.cols &&

                nodes[i][j].y >= 0 && nodes[i][j].y < canvas.rows) {

                if (tracked_h_[i].is_visible && tracked_v_[j].is_visible) {

                    cv::circle(canvas, nodes[i][j], 6, cv::Scalar(0, 255, 0), -1); // 绿点

                } else {

                    cv::circle(canvas, nodes[i][j], 6, cv::Scalar(0, 0, 255), 2);  // 红圈

                }

            }

        }

    }

    cv::namedWindow("Grid Nodes Tracker", cv::WINDOW_NORMAL);

    cv::imshow("Grid Nodes Tracker", canvas);



    std::string save_dir = "/home/lx/兰欣20241872/python/UNet++/canvas_525_2/";

    std::stringstream ss;

    ss << save_dir << "frame_" << std::setw(4) << std::setfill('0') << ++frame_idx_ << ".png";

    std::string save_path = ss.str();

    bool success = cv::imwrite(save_path, canvas);

    if (!success) {

        std::cout << "\033[1;31m[ERROR] 无法保存图片，请检查路径是否存在: " << save_path << "\033[0m" << std::endl;

    }

}



/**

 * @brief 几何与掩码联合推理：从不完整的候选线中重建出标准的4条并行线

 * @param candidates 检测到的候选线

 * @param max_dim 图像边界限制（H 或 W）

 * @param is_vertical 是否是竖线

 * @param mask 原始掩码图，用于残存特征投票

 */

bool TRTNode::reconstructGridLines(std::vector<LineCandidate>& candidates, 

                          int max_dim, 

                          bool is_vertical, 

                          const cv::Mat& mask,

                          std::vector<float>& final_intercepts, 

                          std::vector<float>& final_angles) 

{

    int raw_size = candidates.size();

    if (raw_size < 3) return false; // 巧妇难为无米之炊，单方向至少需要3条线来确定间距大小



    // 1. 按截距升序排列

    std::sort(candidates.begin(), candidates.end(), [](const LineCandidate& a, const LineCandidate& b){

        return a.intercept < b.intercept;

    });



    final_intercepts.resize(4);

    final_angles.resize(4);



// 用来标记哪个槽位是推理出来的（-1表示全是检测到的，0~3代表对应的推理索引）

    int inferred_idx = -1;

    // 情况 A：如果检测出 4 条以上，筛选出“点数最多”（最强壮）的前 4 条

    if (candidates.size() >= 4) {

        std::sort(candidates.begin(), candidates.end(), [](const LineCandidate& a, const LineCandidate& b) {

            return a.points.size() > b.points.size();

        });

        std::sort(candidates.begin(), candidates.begin() + 4, [](const LineCandidate& a, const LineCandidate& b) {

            return a.intercept < b.intercept;

        });

        for (int i = 0; i < 4; ++i) {

            final_intercepts[i] = candidates[i].intercept;

            final_angles[i] = candidates[i].angle;

        }

        return true;

    }



    // 情况 B：极其经典的“缺一门”场景（刚好检测到3条线）

    float c0 = candidates[0].intercept;

    float c1 = candidates[1].intercept;

    float c2 = candidates[2].intercept;



    float g0 = c1 - c0; // 第1个间距

    float g1 = c2 - c1; // 第2个间距



    float avg_angle = (candidates[0].angle + candidates[1].angle + candidates[2].angle) / 3.0f;



    // 定义一个 Lambda 表达式：统计指定截距位置附近是否有残存的 Mask 像素

    auto countMaskPixels = [&](float intercept) {

        int white_pixels = 0;

        int check_pos = std::round(intercept);

        if (is_vertical) {

            for (int y = 0; y < mask.rows; y += 4) {

                for (int dx = -5; dx <= 5; ++dx) {

                    int x = check_pos + dx;

                    if (x >= 0 && x < mask.cols) { // 仅在图像内统计像素

                        if (mask.at<uint8_t>(y, x) > 0) { white_pixels++; break; }

                    }

                }

            }

        } else {

            for (int x = 0; x < mask.cols; x += 4) {

                for (int dy = -5; dy <= 5; ++dy) {

                    int y = check_pos + dy;

                    if (y >= 0 && y < mask.rows) {

                        if (mask.at<uint8_t>(y, x) > 0) { white_pixels++; break; }

                    }

                }

            }

        }

        return white_pixels;

    };



    // 2. 几何推理：揪出隐藏在中间还是两端

    if (g0 > 1.6f * g1) {

        // 说明 g0 里面包庇了一条未识别的线（缺失的是第 2 条线，即 index 1）

        float G = g1; // 真正的标准网格步长

        final_intercepts[0] = c0;

        final_intercepts[1] = c0 + G;

        final_intercepts[2] = c1;

        final_intercepts[3] = c2;

        inferred_idx = 1;

    } 

    else if (g1 > 1.6f * g0) {

        // 说明 g1 里面包庇了一条未识别的线（缺失的是第 3 条线，即 index 2）

        float G = g0; 

        final_intercepts[0] = c0;

        final_intercepts[1] = c1;

        final_intercepts[2] = c1 + G;

        final_intercepts[3] = c2;

        inferred_idx = 2;

    } 

    else {

        // 两组间距接近，说明缺失的线必然在“最前”或“最后”

        float G = (g0 + g1) / 2.0f; 

        float opt_before = c0 - G;  // 预测在最前方（允许算出来是负数，即出左/上界）

        float opt_after  = c2 + G;  // 预测在最后方（允许算出来大于图像宽/高，即出右/下界）



        int score_before = countMaskPixels(c0 - 80.0f);

        int score_after  = countMaskPixels(c2 + 80.0f);



        if (score_before > score_after) {

            if(is_vertical){

                std::cout << "左得分：" << score_before << " > 右得分：" << score_after << std::endl;

            }

            else {

                std::cout << "上得分：" << score_before << " > 下得分：" << score_after << std::endl;

            }

            inferred_idx = 0;

            final_intercepts[0] = opt_before;

            final_intercepts[1] = c0; final_intercepts[2] = c1; final_intercepts[3] = c2;

        } else if (score_after > score_before) {

            if(is_vertical){

                std::cout << "右得分：" << score_after << " > 左得分：" << score_before << std::endl;

            }

            else {

                std::cout << "下得分：" << score_after << " > 上得分：" << score_before << std::endl;

            }

            inferred_idx = 3;

            final_intercepts[0] = c0; final_intercepts[1] = c1; final_intercepts[2] = c2;

            final_intercepts[3] = opt_after;

        } else {

            // =================================================================

            // 修改后的兜底：如果两端都没像素（比如完全被遮挡，或者缺失线本来就在图像外面）

            // 利用当前看到的 3 条线整体所处的相对空间位置，进行纯几何趋势外推

            // =================================================================

            float space_left_or_top = c0;                 // 0 到第一条线的留白距离

            float space_right_or_bottom = max_dim - c2;   // 最后一条线到图像边缘的留白距离



            if (space_left_or_top > space_right_or_bottom) {

                if(is_vertical){

                    std::cout << "左空间：" << space_left_or_top << " > 右空间：" << space_right_or_bottom << std::endl;

                }

                else {

                    std::cout << "上空间：" << space_left_or_top << " > 下空间：" << space_right_or_bottom << std::endl;

                }

                // 说明图像左边/上方留下的空间更大，缺的线更大概率在后面（即使 opt_before < 0 也会被正确采用！）

                final_intercepts[0] = c0; 

                final_intercepts[1] = c1; 

                final_intercepts[2] = c2;

                final_intercepts[3] = opt_after;

                inferred_idx = 3;

            } else {

                if(is_vertical){

                    std::cout << "左空间：" << space_left_or_top << " < 右空间：" << space_right_or_bottom << std::endl;

                }
                else {

                    std::cout << "上空间：" << space_left_or_top << " < 下空间：" << space_right_or_bottom << std::endl;

                }
                // 说明图像右边/下方留下的空间更大，缺的线更大概率在前面

                final_intercepts[0] = opt_before;

                final_intercepts[1] = c0; 

                final_intercepts[2] = c1; 

                final_intercepts[3] = c2;

                inferred_idx = 0;
            }
        }
    }

    for (int i = 0; i < 4; ++i) final_angles[i] = avg_angle;

    std::string dir_title = is_vertical ? "\033[1;34m[竖线几何重建 (VERTICAL RECONSTRUCT)]\033[0m" 

                                        : "\033[1;33m[横线几何重建 (HORIZONTAL RECONSTRUCT)]\033[0m";

    std::cout << "\n" << dir_title << " -----------------------------------" << std::endl;
    std::cout << "  * 原始检测到线条数 (Raw Detected Lines Count): " << raw_size << std::endl;

    
    for (int i = 0; i < 4; ++i) {

        // 根据是否为推理出的线，赋予不同的高亮颜色和标签

        std::string source_tag = (i == inferred_idx) 

            ? "\033[1;35m[INFERRED (几何推理)]\033[0m" 

            : "\033[1;32m[DETECTED (实际检测)]\033[0m";


        std::cout << "  Slot [" << i << "] -> Intercept: " 

                  << std::fixed << std::setprecision(2) << std::setw(7) << final_intercepts[i] 

                  << " | Angle: " << std::setprecision(4) << std::setw(7) << final_angles[i] 

                  << " | Source: " << source_tag << std::endl;

    }

    std::cout << "----------------------------------------------------------------------------\n" << std::endl;
    return true;
}