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

        cv::Mat component_mask = (labels == i);//重载为 “逐元素比较操作符”
        merged_mask_.setTo(255, component_mask);

        std::vector<cv::Point> points;
        cv::findNonZero(component_mask, points);
        if (points.size() < 2) continue;

        cv::Vec4f lp;
        cv::fitLine(points, lp, cv::DIST_L2, 0, 0.01, 0.01);
        
        float intercept;
        if (is_vertical) {
            float dy = (lp[1] == 0.0f) ? 1e-5f : lp[1];
            intercept = lp[2] + (lp[0] / dy) * (H_ / 2.0f - lp[3]); 
        } else {
            float dx = (lp[0] == 0.0f) ? 1e-5f : lp[0];
            intercept = lp[3] + (lp[1] / dx) * (W_ / 2.0f - lp[2]);
        }
        float angle = std::atan2(lp[1], lp[0]);
        // if (is_vertical)
        // {
        //     if(angle < 0) angle = -angle;
        // }
        raw_candidates.push_back({points, intercept, angle});
    }

    // ==================== 2. 贪心聚类融合 ====================
    std::vector<LineCandidate> final_candidates; 
    std::vector<bool> visited(raw_candidates.size(), false);
    
    for (size_t i = 0; i < raw_candidates.size(); ++i) {
        if (visited[i]) continue;
        std::vector<cv::Point> merged_points = raw_candidates[i].points;
 
        // ==========================================================
        
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

        float final_angle = 0.0f;
 
        final_angle = std::atan2(final_lp[1], final_lp[0]); 
        
        // ==================== 3：基于矢量投影的长度筛选 ====================
        float min_t = std::numeric_limits<float>::max();
        float max_t = -std::numeric_limits<float>::max();
        float min_w = std::numeric_limits<float>::max();
        float max_w = -std::numeric_limits<float>::max();

        float vx = final_lp[0]; 
        float vy = final_lp[1]; 
        float x0 = final_lp[2]; 
        float y0 = final_lp[3]; 

        float nx = -vy;         // 法线向量
        float ny = vx;
        for (const auto& pt : merged_points) {
            float t = (pt.x - x0) * vx + (pt.y - y0) * vy;
            if (t < min_t) min_t = t;
            if (t > max_t) max_t = t;
            // 宽度投影 (法线方向)
            float w = (pt.x - x0) * nx + (pt.y - y0) * ny;
            if (w < min_w) min_w = w;
            if (w > max_w) max_w = w;
        }
        float line_length = max_t - min_t; 

        if (line_length < 100) {
            if(test_){
                // std::cout << "\033[1;31m[LINE FILTERED] 剔除不合理短线 -> 方向: " 
                //       << (is_vertical ? "竖线" : "横线") 
                //       << " | 截距: " << final_intercept 
                //       << " | 物理长度: " << line_length 
                //       << " (低于阈值 " << 100 << ")\033[0m" << std::endl;
            }
            for (const auto& pt : merged_points){
                merged_mask_.at<uchar>(pt) = 0;
            }
            continue; 
        }
        // =====================================================================

        // if(is_vertical) std::cout << "竖线宽度：" << max_w - min_w << std::endl;

        // if (max_w - min_w < 5) continue;
        final_candidates.push_back({merged_points, final_intercept, final_angle, min_w, max_w});
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
            // ==================== 【新增：输出每个格子的四点坐标】 ====================
            // std::cout << "[格子 " << r << "-" << c << "] "
            //           << (quad.valid ? "\033[1;32m[有效 Valid]\033[0m" : "\033[1;31m[无效 Invalid]\033[0m") << "\n"
            //           << "  左上 (TL): (" << quad.pts[0].x << ", " << quad.pts[0].y << ")\n"
            //           << "  左下 (BL): (" << quad.pts[1].x << ", " << quad.pts[1].y << ")\n"
            //           << "  右下 (BR): (" << quad.pts[2].x << ", " << quad.pts[2].y << ")\n"
            //           << "  右上 (TR): (" << quad.pts[3].x << ", " << quad.pts[3].y << ")\n"
            //           << "-------------------------------------------" << std::endl;
            // =====================================================================
            
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
    else{
        grid_depth_ = grid_depth_rec_;
        return;
    }
    grid_depth_rec_ = grid_depth_;
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

void TRTNode::filterLinesByGridConsistency(std::vector<LineCandidate>& candidates, float est_grid_w, cv::Mat& mask_to_clear) 
{
    // std::cout << "\ncandidates.size(): " << candidates.size() << " est_grid_w: " << est_grid_w << std::endl;
    if (candidates.size() < 3) return; // 小于3条线无法通过相对间距判定

    std::sort(candidates.begin(), candidates.end(), [](const LineCandidate& a, const LineCandidate& b) {
        return a.intercept < b.intercept;
    });

    std::vector<bool> to_remove(candidates.size(), false);
    const float EPS = 0.15f; // 允许 的间距误差

    for (size_t i = 0; i < candidates.size(); ++i) {
        int valid_gaps = 0;
        int invalid_gaps = 0;
        // std::cout << candidates[i].intercept << std::endl;
        for (size_t j = 0; j < candidates.size(); ++j) {
            if (i == j) continue;
            float dist = std::abs(candidates[i].intercept - candidates[j].intercept);
            // 检查间距是否近似为 1*w, 2*w, 3*w
            float ratio = dist / est_grid_w;
            float closest_multiple = std::round(ratio);
            
            // if(i == 0) std::cout << "dist: " << dist << " ratio: " << ratio << " closest_multiple: " << closest_multiple << std::endl;

            if (closest_multiple >= 1 && closest_multiple <= 3) {
                if (std::abs(ratio - closest_multiple) < EPS) {
                    valid_gaps++;
                } else {
                    invalid_gaps++;
                }
            } else {
                invalid_gaps++;
            }
        }
        // 如果错误的间隔 >= 2，判定为误检线
        if (invalid_gaps >= 2) {
            to_remove[i] = true;
            // std::cout << "误检线: " << i << " 条" << std::endl;
        }
    }
    // 执行剔除
    auto it = candidates.begin();
    for (size_t i = 0; i < to_remove.size(); ++i) {
        if (to_remove[i]) {
            // 在 mask 中抹除
            for (const auto& pt : it->points) {
                mask_to_clear.at<uchar>(pt) = 0;
            }
            it = candidates.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<Unet::Quad2D> TRTNode::getgridquads(const cv::Mat& frame, float& pos_z)
{
    // 声明聚类与追踪所需的阈值
    static const float DIST_THRESH = 50.0f;     
    static const float ANGLE_THRESH = 0.174f;   

    // if (W_ == 0 || H_ == 0) {
    //     W_ = frame.cols;
    //     H_ = frame.rows;
    // }

    // ==================== 图像形态学后处理 ====================
    cv::morphologyEx(mask_, mask_, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    cv::Mat v_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 150));
    cv::Mat h_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(150, 1));
    cv::Mat vertical_lines, horizontal_lines;

    cv::morphologyEx(mask_, vertical_lines, cv::MORPH_OPEN, v_kernel);
    cv::morphologyEx(mask_, horizontal_lines, cv::MORPH_OPEN, h_kernel);

    merged_mask_ = cv::Mat::zeros(mask_.size(), CV_8UC1);

    std::vector<LineCandidate> v_candidates = extractAndClusterLines(vertical_lines, merged_mask_, true, DIST_THRESH, ANGLE_THRESH);
    std::vector<LineCandidate> h_candidates = extractAndClusterLines(horizontal_lines, merged_mask_, false, DIST_THRESH, ANGLE_THRESH);

    cv::imshow("merged_mask_src", merged_mask_);
    
    // std::cout << "原始竖线条数: " << v_candidates.size() << " 原始横线条数: " << h_candidates.size() << std::endl;
    // ==================== 【0. 新增：计算 best_d 前，基于现有线自适应估算格子像素宽度】 ====================
    float est_grid_w = tracked_grid_size_;
    {
        std::vector<float> diffs;
        if (v_candidates.size() >= 2) {
            std::vector<float> x_vals;
            for (const auto& vc : v_candidates) x_vals.push_back(vc.intercept);
            std::sort(x_vals.begin(), x_vals.end()); // 排序确保 x_vals[j] > x_vals[i]
            
            for (size_t i = 0; i < x_vals.size(); ++i) {
                for (size_t j = i + 1; j < x_vals.size(); ++j) {
                    float d = x_vals[j] - x_vals[i];
                    // std::cout << "竖线相互间隔 (线" << i << " -> 线" << j << ")：" << d << std::endl;
                    if (d > 30.0f && d < 700.0f) {
                        diffs.push_back(d);
                    }
                }
            }
        }
        if (h_candidates.size() >= 2) {
            std::vector<float> y_vals;
            for (const auto& hc : h_candidates) y_vals.push_back(hc.intercept);
            std::sort(y_vals.begin(), y_vals.end()); // 排序确保 y_vals[j] > y_vals[i]
            for (size_t i = 0; i < y_vals.size(); ++i) {
                for (size_t j = i + 1; j < y_vals.size(); ++j) {
                    float d = y_vals[j] - y_vals[i];
                    // std::cout << "横线相互间隔 (线" << i << " -> 线" << j << ")：" << d << std::endl;
                    if (d > 30.0f && d < 700.0f) {
                        diffs.push_back(d);
                    }
                }
            }
        }
        if (diffs.size() >= 3) {
            // 1. 必须先排序，才能进行高效的临近聚类
            std::sort(diffs.begin(), diffs.end());
            std::vector<std::vector<float>> clusters;
            std::vector<float> current_cluster = {diffs[0]};
            // 聚类阈值比例：邻近的间距值差异在 15% 以内则划分为同一种网格大小
            // 这比固定像素阈值更优秀，能完美适应相机视野中“近大远小”的透视形变
            const float CLUSTER_THRESH_RATIO = 0.1f; 
            for (size_t i = 1; i < diffs.size(); ++i) {
                if ((diffs[i] - diffs[i-1]) <= (diffs[i-1] * CLUSTER_THRESH_RATIO)) {
                    current_cluster.push_back(diffs[i]);
                } else {
                    clusters.push_back(current_cluster);
                    current_cluster = {diffs[i]}; // 开启新的一簇
                }
            }
            clusters.push_back(current_cluster); // 压入最后一簇
            // 2. 寻找包含元素数量最多（最具有代表性）的一簇
            size_t max_size = 0;
            size_t best_cluster_idx = 0;
            for (size_t i = 0; i < clusters.size(); ++i) {
                if (clusters[i].size() > max_size) {
                    max_size = clusters[i].size();
                    best_cluster_idx = i;
                }
            }
            // 3. 计算该最大簇内所有有效间距的平均值
            float sum = 0.0f;
            for (float val : clusters[best_cluster_idx]) {
                sum += val;
            }
            est_grid_w = sum / clusters[best_cluster_idx].size();
            // std::cout << "\033[1;32m[GRID CLUSTER] 聚类成功 -> 独立簇数: " << clusters.size() 
            //             << " | 最大簇样本数: " << max_size << "\033[0m" << std::endl;
        } 
        else if (!diffs.empty()) {
            // 找到 diffs 中与 tracked_grid_size_ 最接近的元素
            auto it = std::min_element(diffs.begin(), diffs.end(),
                [this](float a, float b) {
                    return std::abs(a - tracked_grid_size_) < std::abs(b - tracked_grid_size_);
                });
            est_grid_w = *it;
        }
    }
    //至少在一个方向上有两个线时， est_grid_w != 0
    std::cout << "格子初步宽度est_grid_w: " << est_grid_w << std::endl; 
    test_grid_size_ = est_grid_w;


    // ====================  根据格子初步宽度 去除杂线  ====================

    if (est_grid_w > 10.0f && (v_candidates.size() >= 3 || h_candidates.size() >= 3)) {
        filterLinesByGridConsistency(v_candidates, est_grid_w, merged_mask_);
        filterLinesByGridConsistency(h_candidates, est_grid_w, merged_mask_);
    }


    // ==================== 【1. 基于自适应像素宽度直接解析计算最佳深度 best_d】 ====================
    const float Z_w_vals[4] = {2410.0f, 1880.0f, 1340.0f, 810.0f};
    float best_d = 4000.0f; // 默认兜底值

    if (est_grid_w > 10.0f) {
        const float L_phys = 535.0f; // 九宫格单格的标准物理物理尺寸（毫米）
        best_d = (fy_ * L_phys) / est_grid_w;
        
        // 增加物理合理性区间裁剪，防止极端畸变或错误估算导致崩溃
        if (best_d < 400.0f) best_d = 400.0f;
        if (best_d > 4000.0f) best_d = 4000.0f;

        // std::cout << "[解析深度估计] 基于像素单格宽度: " << est_grid_w << " 像素 | 逆投影解出 best_d: " << best_d << " mm" << std::endl;
        
    } else {
        if (test_) {
            // std::cout << "[解析深度估计] 未提取到有效单格像素宽度，启用默认兜底 best_d: " << best_d << " mm" << std::endl;
        }
    }
    // std::cout << "best_d: " << best_d << " pos_z: " << pos_z << std::endl;

    //====================  匹配横线 ====================
    for (int s = 0; s < 4; ++s) {
        ideal_h_v_[s] = (fy_ * (pos_z - Z_w_vals[s])) / best_d + ppy_;
        // std::cout << "ideal_h_v_[" << s << "]: " << ideal_h_v_[s] << std::endl;
    }
    for(int i = 0; i < h_candidates.size(); ++i)
    {
        // std::cout << "h_candidates[" << i << "]: " << h_candidates[i].intercept << " " <<  180 * (h_candidates[i].angle / 3.1415) << std::endl;
    }
    float avg_h_angle = 0.0f;
    int h_match_cnt = 0;
    const float MATCH_THRESH_V = est_grid_w / 2; 
    for (int s = 0; s < 4; ++s) {
        float ideal_v = ideal_h_v_[s];
        int best_match_idx = -1;
        float min_diff = MATCH_THRESH_V;
        for (size_t i = 0; i < h_candidates.size(); ++i) {
            float diff = std::abs(h_candidates[i].intercept - ideal_v);
            if (diff < min_diff) {
                min_diff = diff;
                best_match_idx = i;
            }
        }
        if (best_match_idx != -1) {
            tracked_h_[s].intercept = h_candidates[best_match_idx].intercept;
            tracked_h_[s].angle = h_candidates[best_match_idx].angle;
            avg_h_angle += tracked_h_[s].angle;
            tracked_h_[s].is_visible = true; 
            h_match_cnt++;
        } else {
            tracked_h_[s].intercept = ideal_v;
            tracked_h_[s].angle = 0.0f; 
            tracked_h_[s].is_visible = false; 
        }
    }
    avg_h_angle = (h_match_cnt > 0) ? (avg_h_angle / h_match_cnt) : 0.0f;
    // std::cout << "avg_h_angle: " << 180 * (avg_h_angle / 3.1415) << " 度" << std::endl;
    for (int s = 0; s < 4; ++s) {
        // std::cout << "tracked_h_[" << s << "].intercept: " << tracked_h_[s].intercept << std::endl;
        if (tracked_h_[s].angle == 0.0f) {
            tracked_h_[s].angle = avg_h_angle;
        }
    }

    start_push_ = 0;
    bool start_1 = tracked_h_[1].intercept > est_grid_w*0.5;
    bool start_2 = tracked_h_[1].is_visible;
    if(start_1 && start_2) start_push_ = 1;

    // ==================== 拓扑时序槽位追踪与节点输出 ====================
    return trackGridAndGetNodes(v_candidates, est_grid_w, frame, merged_mask_);
}



bool TRTNode::reconstructGridLines(std::vector<LineCandidate>& v_candidates, 
                                    int max_dim, 
                                    const float& est_grid_w, 
                                    const cv::Mat& mask,
                                    std::vector<float>& final_intercepts, 
                                    std::vector<float>& final_angles)
{

    auto count_mask_pixels = [&](float center_x, float strip_width) {
        if (std::isnan(center_x) || center_x < 0 || center_x >= mask.cols) return 0;
        
        int count = 0;
        int col = static_cast<int>(center_x); // 显式转为整型列坐标
        for (int r = 0; r < mask.rows; r += 2) {
            if (mask.at<uchar>(r, col) > 0) {
                count++;
            }
        }
        return count;
    };
    
    float strip_w = std::max(10.0f, est_grid_w * 0.1f); // 动态自适应探测宽度（给目标线左右留出适当容差）

    int num_v = v_candidates.size();


    if (num_v >= 4) {
        std::sort(v_candidates.begin(), v_candidates.end(), [](const LineCandidate& a, const LineCandidate& b) {
            return a.points.size() > b.points.size();
        });
        std::vector<LineCandidate> top4 = {v_candidates[0], v_candidates[1], v_candidates[2], v_candidates[3]};
        std::sort(top4.begin(), top4.end(), [](const LineCandidate& a, const LineCandidate& b) {
            return a.intercept < b.intercept;
        });
        for (int i = 0; i < 4; ++i) {
            final_intercepts[i] = top4[i].intercept;
            final_angles[i] = top4[i].angle;
        }
        return true;
    }
    
    else if (num_v == 3) {
        std::sort(v_candidates.begin(), v_candidates.end(), [](const LineCandidate& a, const LineCandidate& b) {
            return a.intercept < b.intercept;
        });
        float x0 = v_candidates[0].intercept;
        float x1 = v_candidates[1].intercept;
        float x2 = v_candidates[2].intercept;

        float a0 = v_candidates[0].angle;
        float a1 = v_candidates[1].angle;
        float a2 = v_candidates[2].angle;

        float x_min = v_candidates[0].min_w;
        float x_max = v_candidates[2].max_w;

        float g1 = x1 - x0; float g2 = x2 - x1;

        std::cout << "g1: " << g1 << " g2: " << g2 << " est_grid_w: " << est_grid_w << std::endl;
        if (g1 > 1.45f * est_grid_w) { // 槽位 1 缺失 (断开的是左间距)
            final_intercepts[0] = x0;
            final_intercepts[1] = x0 + est_grid_w;
            final_intercepts[2] = x1;
            final_intercepts[3] = x2;

            final_angles[0] = a0;
            final_angles[1] = a0;
            final_angles[2] = a1;
            final_angles[3] = a2;

            return true;

        } else if (g2 > 1.45f * est_grid_w) { // 槽位 2 缺失 (断开的是右间距)
            final_intercepts[0] = x0;
            final_intercepts[1] = x1;
            final_intercepts[2] = x1 + est_grid_w;
            final_intercepts[3] = x2;

            final_angles[0] = a0;
            final_angles[1] = a1;
            final_angles[2] = a1;
            final_angles[3] = a2;

            return true;
            
        } else {  
            std::cout << "left_x: " << x0 + x_min - 0.1f * est_grid_w << " right_x: " << x2 + x_max + 0.1f * est_grid_w << " strip_w: " << strip_w << std::endl;

            int left_pixels = count_mask_pixels(x0 + x_min - 0.1f * est_grid_w, strip_w);
            int right_pixels = count_mask_pixels(x2 + x_max + 0.1f * est_grid_w, strip_w);

            std::cout << "left_pixels: " << left_pixels << " right_pixels: " << right_pixels << std::endl;

            if (left_pixels > right_pixels && left_pixels > 5) { 
                final_intercepts[0] = x0 - est_grid_w;
                final_intercepts[1] = x0; final_intercepts[2] = x1; final_intercepts[3] = x2;

                final_angles[0] = a0;
                final_angles[1] = a0;
                final_angles[2] = a1;
                final_angles[3] = a2;

                return true;

            } 
            else if (left_pixels < right_pixels && right_pixels > 5) { 
                final_intercepts[0] = x0; final_intercepts[1] = x1; final_intercepts[2] = x2;
                final_intercepts[3] = x2 + est_grid_w;

                final_angles[0] = a0;
                final_angles[1] = a1;
                final_angles[2] = a2;
                final_angles[3] = a2;

                return true;

            }
            else if (left_pixels < 20 && right_pixels < 20) { 
                
                if(est_grid_w < 200) return false;

                float len_l = std::abs(x0 - 0);
                float len_r = std::abs(x2 - merged_mask_.cols - 1);

                if(std::abs(len_l - len_r) < 50) return false; // 三条线占满画面，无法确定是那三条

                if (len_l > len_r){
                    final_intercepts[0] = x0; final_intercepts[1] = x1; final_intercepts[2] = x2;
                    final_intercepts[3] = x2 + est_grid_w;

                    final_angles[0] = a0;
                    final_angles[1] = a1;
                    final_angles[2] = a2;
                    final_angles[3] = a2;
                }
                else{
                    final_intercepts[0] = x0 - est_grid_w;
                    final_intercepts[1] = x0; final_intercepts[2] = x1; final_intercepts[3] = x2;

                    final_angles[0] = a0;
                    final_angles[1] = a0;
                    final_angles[2] = a1;
                    final_angles[3] = a2;
                }

                return true;

            }
        }
    }

    else if (num_v == 2) {
        std::sort(v_candidates.begin(), v_candidates.end(), [](const LineCandidate& a, const LineCandidate& b) {
            return a.intercept < b.intercept;
        });
        float x0 = v_candidates[0].intercept;
        float x1 = v_candidates[1].intercept;

        float a0 = v_candidates[0].angle;
        float a1 = v_candidates[1].angle;

        float x_min = v_candidates[0].min_w;
        float x_max = v_candidates[1].max_w;

        float g = x1 - x0;

        // std::cout << "x_min: " << x_min << " x0: " << x0 << " x1: " << x1 << " x_max: " << x_max << std::endl;

        int left_pixels = count_mask_pixels(x0 + x_min - 0.1f * est_grid_w, strip_w);
        int right_pixels = count_mask_pixels(x1 + x_max + 0.1f * est_grid_w, strip_w);

        // std::cout << "left_pixels: " << left_pixels << " right_pixels: " << right_pixels << std::endl;

        if (left_pixels < right_pixels*0.2 && right_pixels > 5) {  

            if(x0 <= 200) return false;

            final_intercepts[0] = x0;
            final_intercepts[1] = x1;
            final_intercepts[2] = x1 + est_grid_w;
            final_intercepts[3] = x1 + 2.0f * est_grid_w;

            final_angles[0] = a0;
            final_angles[1] = a1;
            final_angles[2] = a1;
            final_angles[3] = a1;

            return true;

        } else if (left_pixels*0.2 > right_pixels && left_pixels > 5) {  

            if(x1 > mask.cols - 200) return false;

            final_intercepts[0] = x0 - 2.0f * est_grid_w;
            final_intercepts[1] = x0 - est_grid_w;
            final_intercepts[2] = x0;
            final_intercepts[3] = x1;

            final_angles[0] = a0;
            final_angles[1] = a0;
            final_angles[2] = a0;
            final_angles[3] = a1;

            return true;

        } else if (right_pixels > 5 && left_pixels > 5) {  
            final_intercepts[0] = x0 - est_grid_w;
            final_intercepts[1] = x0;
            final_intercepts[2] = x1;
            final_intercepts[3] = x1 + est_grid_w;

            final_angles[0] = a0;
            final_angles[1] = a0;
            final_angles[2] = a1;
            final_angles[3] = a1;

            return true;

        } else {
            if(est_grid_w < 300) return false;

            float len_l = std::abs(x0 - 0);
            float len_r = std::abs(x1 - merged_mask_.cols - 1);

            if(std::abs(len_l - len_r) < 50) return false; // 两条线占满画面，无法确定是那两条

            if (len_l < len_r) {
                final_intercepts[0] = x0 - 2.0f * est_grid_w;
                final_intercepts[1] = x0 - est_grid_w;
                final_intercepts[2] = x0;
                final_intercepts[3] = x1;

                final_angles[0] = a0;
                final_angles[1] = a0;
                final_angles[2] = a0;
                final_angles[3] = a1;

            } else {
                final_intercepts[0] = x0;
                final_intercepts[1] = x1;
                final_intercepts[2] = x1 + est_grid_w;
                final_intercepts[3] = x1 + 2.0f * est_grid_w;

                final_angles[0] = a0;
                final_angles[1] = a1;
                final_angles[2] = a1;
                final_angles[3] = a1;
            }

            return true;

        }
    }
    else {
        // std::cout << "检测出的竖线条数: " << num_v << std::endl;
        return false;
    }

}

void TRTNode::trackSingleDirection(std::vector<LineCandidate>& candidates, std::vector<TrackedLine>& tracked_lines) 
{
    std::vector<bool> det_matched(candidates.size(), false);
    std::vector<int> track_to_det(4, -1);
    int match_count = 0;  

    // 1. 基础数据关联（贪心匹配）
    for (int k = 0; k < 4; ++k) {
        int best_idx = -1;
        // float min_dist = 150.0f;  
        float min_dist = test_grid_size_ * 0.5;
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
            if (scale_a < 0.8f || scale_a > 1.2f) {
                // std::cout << "\033[1;33m[WARN] 估计的缩放率异常 (" << scale_a << ")，退化为纯平移模型\033[0m" << std::endl;
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
        // std::cout << "\033[1;36m[MOTION ESTIMATE] 匹配线数: " << match_count 
        //           << " | 缩放系数(a): " << scale_a 
        //           << " (" << (scale_a > 1.0f ? "网格扩大/靠近" : "网格缩小/远离") << ")"
        //           << " | 偏移基准(b): " << shift_b << "\033[0m" << std::endl;
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
}



std::vector<Unet::Quad2D>  TRTNode::trackGridAndGetNodes(std::vector<LineCandidate>& v_candidates, 
                                    const float& est_grid_w, 
                                    const cv::Mat& current_frame,
                                    const cv::Mat& mask)
{
    std::vector<float> final_v_intercepts, final_v_angles;
    final_v_intercepts.resize(4, 0.0f);
    final_v_angles.resize(4, 0.0f);
    bool v_ok = reconstructGridLines(v_candidates, W_, est_grid_w, mask, final_v_intercepts, final_v_angles);

    bool global_correction_applied = false;

    if(v_ok){
        global_correction_applied = true;
        for (int i = 0; i < 4; ++i) {
            tracked_v_[i] = {i, final_v_intercepts[i], final_v_angles[i], true};
        }
        if (!grid_initialized_) {
            grid_initialized_ = true;
            // std::cout << "\033[1;32m>>> [INIT SUCCESS] Grid Slots Initialized Successfully via Reasonable Geometry! <<<\033[0m" << std::endl;
        } else {
            // std::cout << "\033[1;36m>>> [DYNAMIC CALIBRATION] Tracked slots corrected successfully via global reconstruction. <<<\033[0m" << std::endl;
        }
    }

    // 如果系统至今连一次完美的初始化都没成功过，且本帧重建也失败/不合理，直接退场，不盲画交点
    if (!grid_initialized_) {
        // std::cout << "\n一次初始化都没有成功\n" << std::endl;
        return {};
    }

    // 没有应用全局矫正，进行跟踪
    if (!global_correction_applied) {
        trackSingleDirection(v_candidates, tracked_v_);
    }

    if (grid_initialized_) {
        std::vector<float> visible_gaps;
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
        if (!visible_gaps.empty()) {
            float sum_gap = 0;
            for (float g : visible_gaps) sum_gap += g;
            float current_measured_grid = sum_gap / visible_gaps.size();

            if (tracked_grid_size_ > 0) {
                last_scale_a_ = current_measured_grid / tracked_grid_size_;
                if (last_scale_a_ < 0.8f || last_scale_a_ > 1.25f) last_scale_a_ = 1.0f;
            }
            tracked_grid_size_ = current_measured_grid;
        } 
        else {
            last_scale_a_ = 1.0f;
        }
    }

    computeGridQuads(640, 480, 960, 720);

    cv::resize(current_frame, canvas_, cv::Size(W_, H_), 0, 0, cv::INTER_LINEAR);

    cv::Mat frame_src = current_frame.clone();
    cv::Mat frame_src_first= current_frame.clone();


    // 3. 在 canvas_ 上绘制 quads_ 中的顶点
    for (const auto& quad : quads_) {
        if (!quad.valid) continue;
        for (size_t i = 0; i < quad.pts.size(); ++i) {
            // 画点
            cv::circle(frame_src, quad.pts[i], 5, cv::Scalar(0, 0, 255), -1);
            // 连线形成网格轮廓
            cv::line(frame_src, quad.pts[i], quad.pts[(i + 1) % 4], cv::Scalar(255, 255, 0), 2);
        }
    }

    // 可选：显示当前深度信息
    std::string depth_str = "est_grid_w: " + std::to_string(test_grid_size_) + " mm";
    cv::putText(frame_src, depth_str, cv::Point(200, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

    int status_y = 30; // 初始Y坐标
    std::vector<std::pair<std::string, bool>> status_list = {
        {"v_ok", v_ok},
        {"global_correction_applied", global_correction_applied},
        {"grid_initialized_", grid_initialized_}
    };

    for (const auto& item : status_list) {
        std::string status_str = item.first + ": " + (item.second ? "true" : "false");
        // 根据真假自适应颜色：真用青绿(Cyan)，假用亮红
        cv::Scalar text_color = item.second ? cv::Scalar(255, 255, 0) : cv::Scalar(0, 0, 255); 
        
        // 1. 绘制黑色阴影层（厚度为2，稍微偏移，增强复杂背景下的鲁棒可见度）
        cv::putText(frame_src, status_str, cv::Point(16, status_y + 1), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        // 2. 绘制彩色前景层
        cv::putText(frame_src, status_str, cv::Point(15, status_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
        
        status_y += 25; // 换行下移
    }

    // ==================== 新增：在 frame_src 绘制 tracked_h_ 信息并保存 ====================
    int text_y = 130; // 初始写入的Y坐标
    for (int i = 0; i < tracked_v_.size(); ++i) {
        
        // 格式化当前横线的信息字符串
        std::string info_str = "V[" + std::to_string(i) + 
                               "]: Intercept=" + std::to_string(static_cast<int>(tracked_v_[i].intercept));
        
        // 绘制一层黑色阴影避免白背景导致看不清
        cv::putText(frame_src, info_str, cv::Point(11, text_y + 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        // 绘制绿色的文本文字
        cv::putText(frame_src, info_str, cv::Point(10, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        text_y += 25; // 换行下移
    }

    int text_y_2 = 250;
    for (int i = 0; i < tracked_h_.size(); ++i) {

        int ideal_v = static_cast<int>(tracked_h_[i].intercept / 1.5);
        cv::line(frame_src, cv::Point(0, ideal_v), cv::Point(W_, ideal_v), cv::Scalar(0, 255, 0), 1);
        
        // 格式化当前横线的信息字符串
        std::string info_str = "H[" + std::to_string(i) + 
                               "]: Intercept=" + std::to_string(static_cast<int>(tracked_h_[i].intercept)) +
                                " | " + std::to_string(ideal_v);

        cv::Scalar text_color = tracked_h_[i].is_visible ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        // 绘制一层黑色阴影避免白背景导致看不清
        cv::putText(frame_src, info_str, cv::Point(11, text_y_2 + 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        // 绘制绿色的文本文字
        cv::putText(frame_src, info_str, cv::Point(10, text_y_2), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1);
        text_y_2 += 25; // 换行下移
    }

    // 自动按帧序列号保存至指定路径
    static int frame_counter = 0;
    std::string save_dir = "grid/";
    
    // 生成格式化的文件名，如 frame_00042.jpg
    char filename[64];
    char filename_first[64];
    char filename_mask[64];
    std::snprintf(filename, sizeof(filename), "frame_%05d.jpg", frame_counter++);
    std::snprintf(filename_first, sizeof(filename_first), "frame_%05d.jpg", frame_counter++);
    std::snprintf(filename_mask, sizeof(filename_mask), "frame_%05d.jpg", frame_counter++);

    std::string full_save_path = save_dir + filename;
    std::string full_save_fist_path = save_dir + filename_first;
    std::string full_save_mask_path = save_dir + filename_mask;

    // 检查目录是否存在或者直接写入（需确保文件夹本身已手动创建好）
    try {
        cv::imwrite(full_save_path, frame_src);
        cv::imwrite(full_save_fist_path, frame_src_first);
        cv::imwrite(full_save_mask_path, mask);
    } catch (const cv::Exception& ex) {
        std::cerr << "\033[1;31m[imwrite error]: 无法保存图像到 " << full_save_path 
                  << "，请确认文件夹路径是否存在！ 错误原因: " << ex.what() << "\033[0m" << std::endl;
    }
    // ===================================================================================

    // 如果需要查看效果，可以取消注释下面这行（需确保所在环境有 GUI）
    cv::namedWindow("merged_mask_", cv::WINDOW_NORMAL);
    cv::namedWindow("frame_src", cv::WINDOW_NORMAL);
    cv::namedWindow("canvas_", cv::WINDOW_NORMAL);

    cv::imshow("merged_mask_", merged_mask_);
    cv::imshow("frame_src", frame_src);
    cv::imshow("canvas_", canvas_); 
    cv::waitKey(1);
    // ====================================================
    return quads_;
}