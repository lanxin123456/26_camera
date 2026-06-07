#include "trt_seg/trt_node.hpp"

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
        if (stats.at<int>(i, cv::CC_STAT_AREA) < 400) continue;

        cv::Mat component_mask = (labels == i);
        merged_mask_.setTo(255, component_mask);

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

        if (line_length < 100) {
            if(test_){
                std::cout << "\033[1;31m[LINE FILTERED] 剔除不合理短线 -> 方向: " 
                      << (is_vertical ? "竖线" : "横线") 
                      << " | 截距: " << final_intercept 
                      << " | 物理长度: " << line_length 
                      << " (低于阈值 " << 100 << ")\033[0m" << std::endl;
            }
            for (const auto& pt : merged_points){
                merged_mask_.at<uchar>(pt) = 0;
            }
            continue; // 优雅地略过，不加入最终的候选集
        }
        // =====================================================================

        final_candidates.push_back({merged_points, final_intercept, final_angle});
    }
    
    return final_candidates;
}

// ==================== 九宫格逆投影与坐标系复原 ====================
void TRTNode::computeGridQuads(int src_w, int src_h, int net_w, int net_h) {
    quads_.clear();
    float scale = std::min(static_cast<float>(net_w) / src_w, static_cast<float>(net_h) / src_h);
    float x_offset = (net_w - src_w * scale) * 0.5f;
    float y_offset = (net_h - src_h * scale) * 0.5f;

    // 定义一个结构体用于记录有效的可见交点
    struct VisibleNode {
        cv::Point2f pt;
        int r; // 行索引 (0~3)
        int c; // 列索引 (0~3)
    };
    std::vector<VisibleNode> visible_nodes;

    cv::Point2f nodes[4][4];
    for (int i = 0; i < 4; ++i) {
        cv::Vec4f h_line = slotToVec4f(tracked_h_[i], false);
        bool h_visible = tracked_h_[i].is_visible;

        for (int j = 0; j < 4; ++j) {
            cv::Vec4f v_line = slotToVec4f(tracked_v_[j], true);
            bool v_visible = tracked_v_[j].is_visible;

            nodes[i][j] = computeIntersection(h_line, v_line);
            nodes[i][j].x = (nodes[i][j].x - x_offset) / scale;
            nodes[i][j].y = (nodes[i][j].y - y_offset) / scale;

            // 条件：横竖 is_visible 都为 true，且点合法（非 NaN/Inf）
            if (h_visible && v_visible) {
                if (!std::isnan(nodes[i][j].x) && !std::isnan(nodes[i][j].y) &&
                    !std::isinf(nodes[i][j].x) && !std::isinf(nodes[i][j].y)) {
                    visible_nodes.push_back({nodes[i][j], i, j});
                }
            }
        }
    }

    // [原有逻辑] 生成 3x3 的 Quads
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            Unet::Quad2D quad;
            quad.pts.reserve(4);
            quad.pts.emplace_back(nodes[r][c].x, nodes[r][c].y);          // 左上
            quad.pts.emplace_back(nodes[r + 1][c].x, nodes[r + 1][c].y);  // 左下
            quad.pts.emplace_back(nodes[r + 1][c + 1].x, nodes[r + 1][c + 1].y); // 右下
            quad.pts.emplace_back(nodes[r][c + 1].x, nodes[r][c + 1].y);  // 右上

            quad.valid = true;
            for (const auto& p : quad.pts) {
                if (std::isnan(p.x) || std::isnan(p.y) || std::isinf(p.x) || std::isinf(p.y)) {
                    quad.valid = false; break;
                }
            }
            quads_.push_back(quad);
        }
    }

    // ================= 开始计算九宫格深度 =================
    grid_depth_ = -1.0f; // 默认无效值

    if (visible_nodes.size() >= 2) {
        // 物理累计距离坐标（单位：mm）
        // 间距 530, 540, 530 -> 坐标 0, 530, 1070, 1600
        const float cum_dist[4] = {0.0f, 530.0f, 1070.0f, 1600.0f};
        
        double sum_depth = 0.0;
        int count = 0;

        // 两两组合可见点，计算深度
        for (size_t i = 0; i < visible_nodes.size(); ++i) {
            for (size_t j = i + 1; j < visible_nodes.size(); ++j) {
                const auto& n1 = visible_nodes[i];
                const auto& n2 = visible_nodes[j];

                // 1. 如果两点不在同一列，可以通过横向像素差计算深度 (利用 fx_)
                if (n1.c != n2.c) {
                    float pixel_dx = std::abs(n1.pt.x - n2.pt.x);
                    float phys_dx = std::abs(cum_dist[n1.c] - cum_dist[n2.c]);
                    if (pixel_dx > 0.5f) { // 避免噪点导致的除以近 0 值
                        sum_depth += (fx_/1.5 * phys_dx) / pixel_dx;
                        count++;
                    }
                }

                // 2. 如果两点不在同一行，可以通过纵向像素差计算深度 (利用 fy_)
                if (n1.r != n2.r) {
                    float pixel_dy = std::abs(n1.pt.y - n2.pt.y);
                    float phys_dy = std::abs(cum_dist[n1.r] - cum_dist[n2.r]);
                    if (pixel_dy > 0.5f) {
                        sum_depth += (fy_/1.5 * phys_dy) / pixel_dy;
                        count++;
                    }
                }
            }
        }

        if (count > 0) {
            grid_depth_ = static_cast<float>(sum_depth / count);
        }
    }
    // std::cout << "grid_depth_: " << grid_depth_ << std::endl;

    // 此时 grid_depth_ 即为计算出的最终深度（单位：mm）
    // 你可以将其保存到类成员变量中，例如：this->depth_ = grid_depth_;

    // std::cout << "\n========== 网格交点列表 (4x4, 可见性标志) ==========" << std::endl;
    // for (int r = 0; r < 4; ++r) {
    //     for (int c = 0; c < 4; ++c) {
    //         // 重新获取可见性状态（已在前面计算 nodes 时保存到 visible_nodes 中）
    //         // 简单方法：直接根据 tracked_h_[r].is_visible 和 tracked_v_[c].is_visible 判断
    //         bool visible = tracked_h_[r].is_visible && tracked_v_[c].is_visible;
            
    //         // 获取节点坐标（之前已存储在 nodes[r][c] 中，需要将其保存为类成员或在此重新计算）
    //         // 为了简洁，这里再次计算交点（或者可以在前面将 nodes 保存为临时矩阵）
    //         cv::Vec4f h_line = slotToVec4f(tracked_h_[r], false);
    //         cv::Vec4f v_line = slotToVec4f(tracked_v_[c], true);
    //         cv::Point2f pt = computeIntersection(h_line, v_line);
            
    //         // 应用缩放和偏移还原回原始图像坐标
    //         float scale = std::min(static_cast<float>(net_w) / src_w, static_cast<float>(net_h) / src_h);
    //         float x_offset = (net_w - src_w * scale) * 0.5f;
    //         float y_offset = (net_h - src_h * scale) * 0.5f;
    //         pt.x = (pt.x - x_offset) / scale;
    //         pt.y = (pt.y - y_offset) / scale;

    //         // 设置输出颜色
    //         if (visible) {
    //             std::cout << "\033[1;32m";  // 绿色
    //         } else {
    //             std::cout << "\033[1;31m";  // 红色
    //         }
            
    //         std::cout << "节点[" << r << "][" << c << "] : (" << pt.x << ", " << pt.y << ")\033[0m" << std::endl;
    //     }
    // }
}

// ==================== 投影产生三维拓扑卡槽先验 ====================
void TRTNode::gridMasks(Eigen::Matrix3d R_bw, Eigen::Vector3d cam_pos) 
{
    // computeCameraPosWorld();
    // Eigen::Matrix3d R_bw = R_wb_.transpose();
    std::array<const std::vector<Eigen::Vector3d>*, 8> lines_w = {
        &plane_grid_w_.line_1_, &plane_grid_w_.line_2_, &plane_grid_w_.line_3_, &plane_grid_w_.line_4_,
        &plane_grid_w_.line_5_, &plane_grid_w_.line_6_, &plane_grid_w_.line_7_, &plane_grid_w_.line_8_
    };
    
    const double z_near = 0.1; 
    for(int i = 0; i < 4; ++i) { sim_v_slots_[i].valid = false; sim_h_slots_[i].valid = false; }

    for (int q = 0; q < 8; ++q) {
        if (lines_w[q]->size() < 2) {
            std::cout << "[WARN] Line_" << q + 1 << " has insufficient 3D points." << std::endl;
            continue;
        }

        Eigen::Vector3d Pc_my_0 = R_bw * ((*lines_w[q])[0] - cam_pos);
        Eigen::Vector3d P_c0(Pc_my_0.x(), -Pc_my_0.z(), Pc_my_0.y());
        Eigen::Vector3d Pc_my_1 = R_bw * ((*lines_w[q])[1] - cam_pos);
        Eigen::Vector3d P_c1(Pc_my_1.x(), -Pc_my_1.z(), Pc_my_1.y());

        if (P_c0.z() < z_near && P_c1.z() < z_near) {
            // std::cout << "[INFO] Line_" << q + 1  << " is completely behind the camera (Z < z_near)." << std::endl;
            continue; 
        }
        
        if (P_c0.z() < z_near) {
            double u = (z_near - P_c0.z()) / (P_c1.z() - P_c0.z());
            P_c0 = P_c0 + u * (P_c1 - P_c0);
        } else if (P_c1.z() < z_near) {
            double u = (z_near - P_c1.z()) / (P_c0.z() - P_c1.z());
            P_c1 = P_c1 + u * (P_c0 - P_c1);
        }

        double u0 = P_c0.x() * fx_ / P_c0.z() + ppx_;
        double v0 = P_c0.y() * fy_ / P_c0.z() + ppy_;
        double u1 = P_c1.x() * fx_ / P_c1.z() + ppx_;
        double v1 = P_c1.y() * fy_ / P_c1.z() + ppy_;

        LineCandidate lc;
        lc.pt1 = cv::Point2f(static_cast<float>(u0), static_cast<float>(v0));
        lc.pt2 = cv::Point2f(static_cast<float>(u1), static_cast<float>(v1));
        lc.angle = std::atan2(static_cast<float>(v1 - v0), static_cast<float>(u1 - u0));

        float angle_deg = lc.angle * 180.0f / M_PI;

        if (q < 4) { // 物理竖线 (1-4号)
            float dy = (static_cast<float>(v1 - v0) == 0.0f) ? 1e-5f : static_cast<float>(v1 - v0);
            lc.intercept = static_cast<float>(u0) + (static_cast<float>(u1 - u0) / dy) * (H_ / 2.0f - static_cast<float>(v0));
            sim_v_slots_[q].line = lc;
            sim_v_slots_[q].valid = true;

            if(test_){
                std::cout << "\033[1;36m[预测竖线 Line_" << q + 1 << "]\033[0m "
                        << "pt1: (" << lc.pt1.x << ", " << lc.pt1.y << ") -> "
                        << "pt2: (" << lc.pt2.x << ", " << lc.pt2.y << ") | "
                        << "X-intercept: " << lc.intercept << std::endl;
            }
        } else { // 物理横线 (5-8号)
            float dx = (static_cast<float>(u1 - u0) == 0.0f) ? 1e-5f : static_cast<float>(u1 - u0);
            lc.intercept = static_cast<float>(v0) + (static_cast<float>(v1 - v0) / dx) * (W_ / 2.0f - static_cast<float>(u0));
            sim_h_slots_[q - 4].line = lc;
            sim_h_slots_[q - 4].valid = true;

            if(test_){
                std::cout << "\033[1;35m[预测横线 Line_" << q + 1 << "]\033[0m "
                        << "pt1: (" << lc.pt1.x << ", " << lc.pt1.y << ") -> "
                        << "pt2: (" << lc.pt2.x << ", " << lc.pt2.y << ") | "
                        << "Y-intercept: " << lc.intercept << std::endl;
            }
        }
    }

    // 1. 竖线预测间距计算 (基于 X 截距)
    float sum_sim_step_v = 0.0f;
    int sim_v_pairs = 0;
    for (int i = 0; i < 4; ++i) {
        for (int k = i + 1; k < 4; ++k) {
            if (sim_v_slots_[i].valid && sim_v_slots_[k].valid) {
                // 两条有效预测线之间的像素距离
                float pixel_diff = std::abs(sim_v_slots_[k].line.intercept - sim_v_slots_[i].line.intercept);
                int index_diff = k - i; // 真实的网格索引跨度
                if (index_diff > 0) {
                    sum_sim_step_v += pixel_diff / static_cast<float>(index_diff);
                    sim_v_pairs++;
                }
            }
        }
    }
    if (sim_v_pairs > 0) sim_avg_step_v_ = sum_sim_step_v / sim_v_pairs;

    // 2. 横线预测间距计算 (基于 Y 截距)
    float sum_sim_step_h = 0.0f;
    int sim_h_pairs = 0;
    for (int i = 0; i < 4; ++i) {
        for (int k = i + 1; k < 4; ++k) {
            if (sim_h_slots_[i].valid && sim_h_slots_[k].valid) {
                float pixel_diff = std::abs(sim_h_slots_[k].line.intercept - sim_h_slots_[i].line.intercept);
                int index_diff = k - i;
                if (index_diff > 0) {
                    sum_sim_step_h += pixel_diff / static_cast<float>(index_diff);
                    sim_h_pairs++;
                }
            }
        }
    }
    if (sim_h_pairs > 0) sim_avg_step_h_ = sum_sim_step_h / sim_h_pairs;

    // ==================== 终端高亮打印间距结果 ====================
    if(test_){
        std::cout << "\n------------------ 3D PRED GRID SPACING ------------------" << std::endl;
        if (sim_v_pairs > 0) {
            std::cout << "\033[1;32m[STEP RESULT] 竖线(V)预测方向平均单格步长: " << sim_avg_step_v_ << " 像素 (样本对数: " << sim_v_pairs << ")\033[0m" << std::endl;
        } else {
            std::cout << "\033[1;31m[STEP WARN] 竖线(V)有效预测线不足2条，无法解算间距！\033[0m" << std::endl;
        }

        if (sim_h_pairs > 0) {
            std::cout << "\033[1;32m[STEP RESULT] 横线(H)预测方向平均单格步长: " << sim_avg_step_h_ << " 像素 (样本对数: " << sim_h_pairs << ")\033[0m" << std::endl;
        } else {
            std::cout << "\033[1;31m[STEP WARN] 横线(H)有效预测线不足2条，无法解算间距！\033[0m" << std::endl;
        }
        std::cout << "==================================================\n" << std::endl;
    }
}

void TRTNode::detect(const cv::Mat& frame)
{
    cv::Mat resized; 
    cv::resize(frame, resized, cv::Size(W_, H_), 0, 0, cv::INTER_LINEAR);

    // std::cout << "resized.size: " << resized.size() << std::endl;
    // cudaMemsetAsync(d_score_, 0, W_ * H_ * sizeof(float), stream_);

    // 动态注册锁页内存提升传输带宽
    cudaHostRegister(resized.data, W_ * H_ * sizeof(uchar3), cudaHostRegisterMapped);
    cudaMemcpyAsync(d_img_full_, resized.data, W_ * H_ * sizeof(uchar3), cudaMemcpyHostToDevice, stream_);

    launch_preprocess_and_pad((const uchar3*)d_img_full_, (float*)d_input_, stream_);

    // context_->setInputShape(input_name_, nvinfer1::Dims4(BATCH, 3, PATCH, PATCH));
    context_->enqueueV3(stream_);

    launch_finalize_and_crop((const float*)d_output_, d_mask_out_, THRESH, stream_);

    cudaMemcpyAsync(mask_.data, d_mask_out_, W_ * H_ * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);
    cudaHostUnregister(resized.data); // 必须在同步后释放注册
}


std::vector<Unet::Quad2D> TRTNode::getgridquads(const cv::Mat& frame, Eigen::Matrix3d R_bw, Eigen::Vector3d cam_pos)
{
    // 声明聚类与追踪所需的阈值
    static const float DIST_THRESH = 50.0f;     
    static const float ANGLE_THRESH = 0.174f;   

    // 直接在图像上生成模拟先验线，并刷新本地的 sim_v_slots_ 和 sim_h_slots_
    gridMasks(R_bw, cam_pos);

    // ==================== 图像形态学后处理 ====================
    cv::morphologyEx(mask_, mask_, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    cv::Mat v_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 90));
    cv::Mat h_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(90, 1));
    cv::Mat vertical_lines, horizontal_lines;

    cv::morphologyEx(mask_, vertical_lines, cv::MORPH_OPEN, v_kernel);
    cv::morphologyEx(mask_, horizontal_lines, cv::MORPH_OPEN, h_kernel);

    // cv::namedWindow("horizontal_lines", cv::WINDOW_NORMAL);
    // cv::imshow("horizontal_lines", horizontal_lines);

    merged_mask_ = cv::Mat::zeros(mask_.size(), CV_8UC1);

    std::vector<LineCandidate> v_candidates = extractAndClusterLines(vertical_lines, merged_mask_, true, DIST_THRESH, ANGLE_THRESH);
    std::vector<LineCandidate> h_candidates = extractAndClusterLines(horizontal_lines, merged_mask_, false, DIST_THRESH, ANGLE_THRESH);

    auto t2 = std::chrono::high_resolution_clock::now();
    
    // ==================== 索引映射：利用预测线识别检测线归属 ====================
    std::vector<LineCandidate> matched_v_lines(4);
    std::vector<bool> v_slot_has_detection(4, false);
    int v_detected_count = 0;

    for (int i = 0; i < 4; ++i) {
        if (!sim_v_slots_[i].valid) continue;

        int best_cand_idx = -1;
        float min_dist = sim_avg_step_v_ / 2.0f; // 容忍距离阈值

        // 内层遍历所有检测到的候选线，挑出最近的
        for (size_t j = 0; j < v_candidates.size(); ++j) {
            float dist = std::abs(v_candidates[j].intercept - sim_v_slots_[i].line.intercept);
            if (dist < min_dist) {
                min_dist = dist;
                best_cand_idx = j;
            }
        }

        // 如果找到了符合阈值的最近候选线
        if (best_cand_idx != -1) {
            matched_v_lines[i] = v_candidates[best_cand_idx];
            v_slot_has_detection[i] = true;
        }
    }
    for(int i = 0; i < 4; ++i) if(v_slot_has_detection[i]) v_detected_count++;

    std::vector<LineCandidate> matched_h_lines(4);
    std::vector<bool> h_slot_has_detection(4, false);
    int h_detected_count = 0;

    for (int i = 0; i < 4; ++i) {
        if (!sim_h_slots_[i].valid) continue;

        int best_cand_idx = -1;
        float min_dist = sim_avg_step_h_ / 2.0f;

        // 内层遍历所有检测到的候选线
        for (size_t j = 0; j < h_candidates.size(); ++j) {
            float dist = std::abs(h_candidates[j].intercept - sim_h_slots_[i].line.intercept);
            if (dist < min_dist) {
                min_dist = dist;
                best_cand_idx = j;
            }
        }

        if (best_cand_idx != -1) {
            matched_h_lines[i] = h_candidates[best_cand_idx];
            h_slot_has_detection[i] = true;
        }
    }
    for(int i = 0; i < 4; ++i) if(h_slot_has_detection[i]) h_detected_count++;

    if(test_)
    {
        std::cout << "\n========== 垂直方向槽位匹配结果 ==========" << std::endl;
        for (int i = 0; i < 4; ++i) {
            if (v_slot_has_detection[i]) {
                std::cout << "  Slot[" << i << "] ✓ 匹配成功"
                        << "  intercept = " << matched_v_lines[i].intercept;
                // 如果 LineCandidate 还有其他成员如 angle, rho, theta 等，也可输出
                // std::cout << ", angle = " << matched_v_lines[i].angle;
                std::cout << std::endl;
            } else {
                std::cout << "  Slot[" << i << "] ✗ 未匹配" << std::endl;
            }
        }
        std::cout << "垂直方向有效匹配数: " << v_detected_count << " / 4" << std::endl;

        std::cout << "\n========== 水平方向槽位匹配结果 ==========" << std::endl;
        for (int i = 0; i < 4; ++i) {
            if (h_slot_has_detection[i]) {
                std::cout << "  Slot[" << i << "] ✓ 匹配成功"
                        << "  intercept = " << matched_h_lines[i].intercept;
                // 若有 angle 也可输出
                std::cout << std::endl;
            } else {
                std::cout << "  Slot[" << i << "] ✗ 未匹配" << std::endl;
            }
        }
        std::cout << "水平方向有效匹配数: " << h_detected_count << " / 4" << std::endl;
    }

    // ====================  清洗画布中的杂散孤立线 ====================
    std::vector<bool> v_cand_matched(v_candidates.size(), false);
    std::vector<bool> h_cand_matched(h_candidates.size(), false);
    for(int i = 0; i < 4; ++i) {
        if(v_slot_has_detection[i]) {
            for(size_t j = 0; j < v_candidates.size(); ++j) {
                if(std::abs(v_candidates[j].intercept - matched_v_lines[i].intercept) < 1.0) v_cand_matched[j] = true;
            }
        }
        if(h_slot_has_detection[i]) {
            for(size_t j = 0; j < h_candidates.size(); ++j) {
                if(std::abs(h_candidates[j].intercept - matched_h_lines[i].intercept) < 1.0) h_cand_matched[j] = true;
            }
        }
    }
    for(size_t j = 0; j < v_candidates.size(); ++j) {
        if(!v_cand_matched[j]) { for(const auto& pt : v_candidates[j].points) merged_mask_.at<uchar>(pt) = 0; }
    }
    for(size_t j = 0; j < h_candidates.size(); ++j) {
        if(!h_cand_matched[j]) { for(const auto& pt : h_candidates[j].points) merged_mask_.at<uchar>(pt) = 0; }
    }

    // ==================== 纯检测线间距解算（全依赖真实数据与索引差） ====================
    float avg_step_v = 0.0f; // 竖线之间的实际单格像素步长
    float avg_step_h = 0.0f; // 横线之间的实际单格像素步长
    bool v_step_valid = false;
    bool h_step_valid = false;

    // 解算竖线像素间距（处理非连续：如只有0号和3号线，index_diff = 3，间距除以3即得单格步长）
    if (v_detected_count >= 2) {
        float sum_step_v = 0.0f; int v_pairs = 0;
        for (int i = 0; i < 4; ++i) {
            for (int k = i + 1; k < 4; ++k) {
                if (v_slot_has_detection[i] && v_slot_has_detection[k]) {
                    float pixel_diff = matched_v_lines[k].intercept - matched_v_lines[i].intercept;
                    int index_diff = k - i; // 利用判定好的归属索引差作为轴线跨度
                    if (index_diff > 0) {
                        sum_step_v += pixel_diff / static_cast<float>(index_diff);
                        v_pairs++;
                    }
                }
            }
        }
        if (v_pairs > 0) { avg_step_v = sum_step_v / v_pairs; v_step_valid = true; }
    }

    // 解算横线像素间距
    if (h_detected_count >= 2) {
        float sum_step_h = 0.0f; int h_pairs = 0;
        for (int i = 0; i < 4; ++i) {
            for (int k = i + 1; k < 4; ++k) {
                if (h_slot_has_detection[i] && h_slot_has_detection[k]) {
                    float pixel_diff = matched_h_lines[k].intercept - matched_h_lines[i].intercept;
                    int index_diff = k - i;
                    if (index_diff > 0) {
                        sum_step_h += pixel_diff / static_cast<float>(index_diff);
                        h_pairs++;
                    }
                }
            }
        }
        if (h_pairs > 0) { avg_step_h = sum_step_h / h_pairs; h_step_valid = true; }
    }

    if(test_) std::cout << "\033[1;32m竖线之间的实际单格像素步长: " << avg_step_v << "||" << "横线之间的实际单格像素步长: " << avg_step_h << "\033[0m" << std::endl;
    
    // ====================  跨方向间距互补 ====================
    if (v_step_valid && !h_step_valid) {
        avg_step_h = avg_step_v; // 横线不足2条，直接共享竖线测出的高精度单格步长
        h_step_valid = true;
        if(test_) std::cout << "\033[1;35m[拓扑互补] 横线密集度不足：直接共享竖线绝对像素步长 Step=" << avg_step_h << "\033[0m" << std::endl;
    } 
    else if (h_step_valid && !v_step_valid) {
        avg_step_v = avg_step_h; // 竖线不足2条，直接共享横线测出的高精度单格步长
        v_step_valid = true;
        if(test_) std::cout << "\033[1;35m[拓扑互补] 竖线密集度不足：直接共享横线绝对像素步长 Step=" << avg_step_v << "\033[0m" << std::endl;
    }
    else if (!v_step_valid && !h_step_valid) {
        // 极其罕见的全盲兜底：若双向检出均小于2条，迫不得已利用预测模板线距计算一个初始步长
        float total_sim_v = 0.0f; int sim_v_pairs = 0;
        for(int i = 0; i < 3; ++i) {
            if(sim_v_slots_[i].valid && sim_v_slots_[i+1].valid) {
                total_sim_v += std::abs(sim_v_slots_[i+1].line.intercept - sim_v_slots_[i].line.intercept);
                sim_v_pairs++;
            }
        }
        avg_step_v = (sim_v_pairs > 0) ? (total_sim_v / sim_v_pairs) : 300.0f;
        avg_step_h = avg_step_v;
    }
    if(test_) std::cout << "\033[1;32m竖线单格像素后步长: " << avg_step_v << "||" << "横线单格像素后步长: " << avg_step_h << "\033[0m" << std::endl;

    // ====================  提取观测角度（无观测则用预测线角度兜底） ====================
    float avg_v_angle = 1.5708f; int v_ang_cnt = 0; float sum_v_ang = 0.0f;
    for (int i = 0; i < 4; ++i) { 
        if (v_slot_has_detection[i]) { 
            if(matched_v_lines[i].angle < 0) matched_v_lines[i].angle = -matched_v_lines[i].angle;
            sum_v_ang += matched_v_lines[i].angle; v_ang_cnt++; 
        }
    }
    if (v_ang_cnt > 0) avg_v_angle = sum_v_ang / v_ang_cnt;
    else { 
        for(int i=0; i<4; ++i) { 
            if(sim_v_slots_[i].valid) { avg_v_angle = sim_v_slots_[i].line.angle; break; } 
        } 
    }

    float avg_h_angle = 0.0f; int h_ang_cnt = 0; float sum_h_ang = 0.0f;
    for (int i = 0; i < 4; ++i) { 
        if (h_slot_has_detection[i]) { sum_h_ang += matched_h_lines[i].angle; h_ang_cnt++; } 
    }
    if (h_ang_cnt > 0) avg_h_angle = sum_h_ang / h_ang_cnt;
    else { 
        for(int i=0; i<4; ++i) { 
            if(sim_h_slots_[i].valid) { avg_h_angle = sim_h_slots_[i].line.angle; break; } 
        } 
    }

    // ====================  纯检测基准拓扑外推 ====================
    
    // --- 竖线网格闭合 ---
    int v_anchor_idx = -1; // 寻找第一个有真实检测的线作为纯图像空间的拓扑基点
    for (int i = 0; i < 4; ++i) { if (v_slot_has_detection[i]) { v_anchor_idx = i; break; } }

    for (int i = 0; i < 4; ++i) {
        tracked_v_[i].is_visible = v_slot_has_detection[i];
        tracked_v_[i].angle = avg_v_angle;

        if (v_slot_has_detection[i]) {
            tracked_v_[i].intercept = matched_v_lines[i].intercept;
        } else if (v_anchor_idx != -1) {
            tracked_v_[i].intercept = matched_v_lines[v_anchor_idx].intercept + static_cast<float>(i - v_anchor_idx) * avg_step_v;
        } else {
            // 双向全盲边缘特殊降级
            tracked_v_[i].intercept = sim_v_slots_[i].valid ? sim_v_slots_[i].line.intercept : (W_ / 2.0f + (i - 1.5f) * avg_step_v);
        }
    }

    // --- 横线网格闭合 ---
    int h_anchor_idx = -1; 
    for (int i = 0; i < 4; ++i) { if (h_slot_has_detection[i]) { h_anchor_idx = i; break; } }

    for (int i = 0; i < 4; ++i) {
        tracked_h_[i].is_visible = h_slot_has_detection[i];
        tracked_h_[i].angle = avg_h_angle;

        if (h_slot_has_detection[i]) {
            // 只要匹配成功，直接无条件采用真实检测线截距！
            tracked_h_[i].intercept = matched_h_lines[i].intercept;
        } else if (h_anchor_idx != -1) {
            // 纯粹利用拓扑基点在图像坐标系下内插或者向外平移外推
            tracked_h_[i].intercept = matched_h_lines[h_anchor_idx].intercept + static_cast<float>(i - h_anchor_idx) * avg_step_h;
        } else {
            tracked_h_[i].intercept = sim_h_slots_[i].valid ? sim_h_slots_[i].line.intercept : (H_ / 2.0f + (i - 1.5f) * avg_step_h);
        }
    }

    // ==================== 【构建输出九宫格 quads_】 ====================
    computeGridQuads(640, 480, 960, 720);

    auto t3 = std::chrono::high_resolution_clock::now();

    // std::cout << "ALL TIME = " << std::chrono::duration<double, std::milli>(t3 - t2).count() << "\n";
    
        cv::resize(frame, canvas_, cv::Size(W_, H_), 0, 0, cv::INTER_LINEAR);

        for (int i = 0; i < 4; ++i) {
            // 绘制竖向先验线 (用青色 Cyan)
            if (sim_v_slots_[i].valid) {
                cv::line(canvas_, sim_v_slots_[i].line.pt1, sim_v_slots_[i].line.pt2, 
                            cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
            }
            // 绘制横向先验线 (用洋红 Magenta)
            if (sim_h_slots_[i].valid) {
                cv::line(canvas_, sim_h_slots_[i].line.pt1, sim_h_slots_[i].line.pt2, 
                            cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
            }
        }

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                cv::Vec4f line_h = slotToVec4f(tracked_h_[i], false);
                cv::Vec4f line_v = slotToVec4f(tracked_v_[j], true);
                cv::Point2f pt = computeIntersection(line_h, line_v);
                if (pt.x >= 0 && pt.x < canvas_.cols && pt.y >= 0 && pt.y < canvas_.rows) {
                    cv::Scalar color = (tracked_h_[i].is_visible && tracked_v_[j].is_visible) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                    cv::circle(canvas_, pt, 6, color, -1);
                }
            }
        }
    // std::cout <<"[Info] is carrying"<<std::endl;

    {
        std::lock_guard<std::mutex> lock(mat_mutex_);
        
        mask_show_ = mask_.clone(); 
        merged_mask_show_ = merged_mask_.clone(); 
        canvas_show_ = canvas_.clone(); 
    }

    return quads_;
}
