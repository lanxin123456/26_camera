#include "act_d455/kun.hpp"

DealImg::DealImg(const std::string& config_path) : node_(std::make_shared<GridPublish>())
{
    // 读取配置文件
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "无法打开配置文件: " << config_path << std::endl;
        return;
    }
    // 读取配置参数
    int mode;
    fs["MODE"] >> mode;
    if(mode == 1)
    {
        R1_ = 2;
        R2_ = 2;
        F_ = 1;
    }
    else if(mode == 2)
    {
        R1_ = 3;
        R2_ = 4;
        F_ = 1;
    }
    else if(mode == 3)
    {
        fs["R1"] >> R1_;
        fs["R2"] >> R2_;
        fs["F"] >> F_;
    }
    fit_size_ = R1_ + R2_ + F_;

    fs["LAG_FRAMES"] >> lag_frames_;
    fs["Brightness"] >> brightness_thresh_;
    fs["MinArea"] >> min_area_;
    fs["TargetWidth"] >> target_width_;
    fs["TargetHeight"] >> target_height_;
    fs["AvgBrightness"] >> avg_brightness_thresh_;
    fs["DIST_X"] >> dist_x_;
    fs["DIST_Y"] >> dist_y_;
    fs["GridCollectionFrames"] >> grid_collection_frames_;
    fs["AreaMinThreshold"] >> area_min_threshold_;
    fs["ContinueWrongFrames"] >> continue_wrong_frames_;

    fs["BLUE"]["HU"] >> blue_hu_;
    fs["BLUE"]["HL"] >> blue_hl_;
    fs["BLUE"]["DBR"] >> blue_dbr_;
    fs["GREEN"]["HU"] >> green_hu_;
    fs["GREEN"]["HL"] >> green_hl_;
    fs["GREEN"]["GL"] >> green_gl_;
    fs["GREEN"]["DGR"] >> green_dgr_;
    fs["GREEN"]["DGB"] >> green_dgb_;
    fs["RED"]["HU"]["U"] >> red_hu_u_;
    fs["RED"]["HU"]["L"] >> red_hu_l_;
    fs["RED"]["HL"]["U"] >> red_hl_u_;
    fs["RED"]["HL"]["L"] >> red_hl_l_;
    fs["RED"]["RL"] >> red_rl_;
    fs["RED"]["DRB"] >> red_drb_;
    fs["RED"]["DRG"] >> red_drg_;
    fs["DIRECT"]["RGU"] >> derect_rgu_;
    fs["DIRECT"]["BU"] >> derect_bu_;
    fs["DIRECT"]["RGL"] >> derect_rgl_;
    fs["DIRECT"]["BL"] >> derect_bl_;
    fs["DIRECT"]["VU"] >> derect_vu_;
    fs["DIRECT"]["VL"] >> derect_vl_;

    fs.release();
}


bool DealImg::Deal(const cv::Mat &img)
{
    if(result_ > 1)
    {
        return false;
    }
    src_img_.release();
    perspective_img_.release();
    // 如果滞后中，直接返回
    if(is_lagging_)
    {
        current_lag_frame_++;
        if(current_lag_frame_ >= lag_frames_)
        {
            // 数据稳定，关闭滞后，开始采集
            std::cout << "数据稳定，关闭滞后，开始采集" << std::endl;
            is_lagging_ = false;
            current_lag_frame_ = 0;
            current_collection_frame_ = 0;
            is_collecting_ = true;
            collected_grids_.clear();
        }
        else
        {
            return true;
        }
    }
    src_img_ = img.clone();
    cv::Mat hsv;
    // std::cerr << "Converting image to HSV color space..." << std::endl;
    cvtColor(src_img_, hsv, cv::COLOR_BGR2HSV);

    // std::cerr << "DealImg::Deal called with image of size: " << src_img_.cols << "x" << src_img_.rows << std::endl;
    
    // 分离HSV
    std::vector<cv::Mat> channels;
    split(hsv, channels);
    cv::Mat h = channels[0];
    cv::Mat s = channels[1];
    cv::Mat v = channels[2];

    cv::Mat mask;
    cv::inRange(v, cv::Scalar(brightness_thresh_), cv::Scalar(255), mask);

    // std::cerr << "Initial mask created with brightness threshold: " << brightness_thresh_ << std::endl;

    deal_mask_(mask);
#ifdef IMSHOW
    cv::namedWindow("dealed_mask", cv::WINDOW_NORMAL);
    cv::imshow("dealed_mask", mask);
#endif

    cv::Point2f corners[4];
    if(find_largest_counter_(mask, corners))
    {
        perspective_transform_(src_img_, corners);
        static int count__ = 0;
        if(!perspective_img_.empty())
        {
            count__++;
            std::string filename = "/home/action/code/ROBOCON2026_base/src/cv/camera/data/" + std::to_string(count__) + ".jpg";
            cv::imwrite(filename, perspective_img_);
        }
        if(all_in_normal_range_handle_(perspective_img_))
        {
            if(need_turn_)
            {
                cv::rotate(perspective_img_, perspective_img_, cv::ROTATE_180);
            }
            if(analyze_colored_(perspective_img_))
            {
                // 如果不滞后，处理
                if(!is_lagging_)
                {
                    std::cout << "检测到的中心点信息:" << std::endl;
                    for(auto& p : detected_centers_)
                    {
                        std::cout << "x: " << p.x << ", y: " << p.y << ", color_code: " << p.color_code << std::endl;
                    }
                    std::cout << "=================" << std::endl;
                }
                // 如果检测，设置为滞后模式（第一次合格）
                if(detected_centers_.size() == fit_size_ && is_fit_rule_(detected_centers_))
                {
                    result_ = 0;
                    // 在非采集模式下，进行滞后设置
                    if(!is_collecting_)
                    {
                        is_lagging_ = true;
                        current_lag_frame_ = 0;
                    }
                    // 采集模式，进行采集信息的处理
                    // 第一次开始于滞后模式结束时
                    else
                    {
                        std::cout << "采集模式，进行采集信息的处理" << std::endl;
                        current_collection_frame_++;
                        collected_grids_.push_back(detected_centers_);
                        if(current_collection_frame_ >= grid_collection_frames_)
                        {
                            // 采集完成，关闭采集模式
                            is_collecting_ = false;
                            // 开始处理
                            std::cout << "采集完成，开始处理" << std::endl;
                            auto final_grid = analyze_all_grids_();
                            std::cout << "\n[INFO] 采集完成，已生成最终网格结果！" << std::endl;
                            std::cout << "=== 最终网格 ===" << std::endl;
                            for(int r = 0; r < 4; ++r)
                            {
                                for(int c = 0; c < 3; ++c)
                                {
                                    std::cout << final_grid[r][c] << " ";
                                }
                                std::cout << std::endl;
                            }
                            std::cout << "=================" << std::endl;
                            current_wrong_frame_ = 0;
                            result_ = encode_grid_(final_grid);
                        }
                    }
                }
                else
                {
                    current_wrong_frame_++;
                    if(current_wrong_frame_ >= continue_wrong_frames_)
                    {
                        result_ = -1;
                    }
                }
                node_->publish_grid(result_);
                detected_centers_.clear();
            }
        }
    }
#ifdef IMSHOW
    cv::namedWindow("mask", cv::WINDOW_NORMAL);
    cv::imshow("mask", mask);
#endif
    return true;
}

bool DealImg::is_empty_(const std::array<std::array<int, 3>, 4>& grid)
{
    for(const auto& row : grid)
    {
        for(const auto& cell : row)
        {
            if(cell != 0)
            {
                return false;
            }
        }
    }
    return true;
}

int32_t DealImg::encode_grid_(const std::array<std::array<int, 3>, 4>& grid)
{
    int32_t encoded_value = 0;
    int bit_position = 0;
    if(!is_empty_(grid))
    {
        for(int i = 3; i >= 0; --i)
        {
            for(int j = 0; j < 3; ++j)
            {
                int value = grid[i][j];
                if(value < 0 || value > 3)
                {
                    value = std::max(0, std::min(3, value));
                }
                encoded_value |= (static_cast<int32_t>(value) & 0x3) << bit_position;
                bit_position += 2;
            }
        }
    }
    std::cout << "编码值: " << encoded_value << std::endl;
    return encoded_value;
}

void DealImg::deal_mask_(cv::Mat &mask, int close_kernel, int open_kernel)
{
    cv::Mat close_kernel_mat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(close_kernel, close_kernel));
    cv::Mat open_kernel_mat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(open_kernel, open_kernel));   
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, close_kernel_mat);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, open_kernel_mat);
}

std::vector<cv::Point2f> DealImg::order_points(const std::vector<cv::Point2f>& pts)
{
    std::vector<cv::Point2f> rect(4);
    std::vector<cv::Point2f> pts_list(pts);

    std::vector<int> sums, diffs;
    for(const auto& pt : pts_list)
    {
        sums.push_back(pt.x + pt.y);
        diffs.push_back(pt.x - pt.y);
    }
    rect[0] = pts_list[std::min_element(sums.begin(), sums.end()) - sums.begin()];
    rect[1] = pts_list[std::max_element(diffs.begin(), diffs.end()) - diffs.begin()];
    rect[2] = pts_list[std::max_element(sums.begin(), sums.end()) - sums.begin()];
    rect[3] = pts_list[std::min_element(diffs.begin(), diffs.end()) - diffs.begin()];
    return rect;
}

bool DealImg::find_largest_counter_(const cv::Mat &mask, cv::Point2f* corners)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty())
    {
        std::cout << "没有找到轮廓" << std::endl;
        return false;
    }
    // 找到最大的轮廓
    int max_index = -1;
    double max_area = 0.0;
    for (int i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        if (area > max_area && area > min_area_)
        {
            max_area = area;
            max_index = i;
        }
    }
    if(max_index == -1)
    {
        std::cout << "没有找到最大的轮廓" << std::endl;
        return false;
    }

    // 极值四点
    std::vector<cv::Point> largest_contour = contours[max_index];

    float min_sum = std::numeric_limits<float>::max();
    float max_sum = std::numeric_limits<float>::lowest();
    float min_diff = std::numeric_limits<float>::max();
    float max_diff = std::numeric_limits<float>::lowest();
    cv::Point2f point_min_sum, point_max_sum, point_min_diff, point_max_diff;
    for(const auto& point : largest_contour)
    {
        float x = point.x;
        float y = point.y;
        float sum = x + y;
        float diff = x - y;

        if(sum < min_sum)
        {
            min_sum = sum;
            point_min_sum = cv::Point2f(x, y);
        }
        if(sum > max_sum)
        {
            max_sum = sum;
            point_max_sum = cv::Point2f(x, y);
        }
        if(diff < min_diff)
        {
            min_diff = diff;
            point_min_diff = cv::Point2f(x, y);
        }
        if(diff > max_diff)
        {
            max_diff = diff;
            point_max_diff = cv::Point2f(x, y);
        }
    }

    std::vector<cv::Point2f> raw_corners;
    raw_corners.push_back(point_min_sum);
    raw_corners.push_back(point_max_sum);
    raw_corners.push_back(point_min_diff);
    raw_corners.push_back(point_max_diff);
    std::vector<cv::Point2f> ordered_corners = order_points(raw_corners);
    for(int i = 0; i < 4; ++i)
    {
        corners[i] = ordered_corners[i];
    }

    // 绘制结果
    cv::Mat img_with_contour = src_img_.clone();
    cv::drawContours(img_with_contour, contours, max_index, cv::Scalar(0, 255, 0), 2);
    for (int j = 0; j < 4; j++)
    {
        cv::line(img_with_contour, corners[j], corners[(j+1)%4], cv::Scalar(0, 0, 255), 2);
    }



#ifdef IMSHOW
    cv::namedWindow("LCO", cv::WINDOW_NORMAL);
    cv::imshow("LCO", img_with_contour);
#endif
    return true;
}

void DealImg::perspective_transform_(cv::Mat &img, const cv::Point2f* corners)
{
    std::vector<cv::Point2f> ordered_corners = order_points(std::vector<cv::Point2f>(corners, corners + 4));
    cv::Point2f dst_corners[4];
    dst_corners[0] = cv::Point2f(0, 0);
    dst_corners[1] = cv::Point2f(target_width_, 0);
    dst_corners[2] = cv::Point2f(target_width_, target_height_);
    dst_corners[3] = cv::Point2f(0, target_height_);
    cv::Mat transform = cv::getPerspectiveTransform(ordered_corners.data(), dst_corners);
    perspective_img_ = img.clone();
    cv::warpPerspective(img, perspective_img_, transform, cv::Size(target_width_, target_height_));
#ifdef IMSHOW
    cv::namedWindow("correct_img", cv::WINDOW_NORMAL);
    cv::imshow("correct_img", perspective_img_);
#endif
}

bool DealImg::all_in_normal_range_handle_(cv::Mat& img)
{
    cv::Mat hsv;
    cv::Mat direct_mask = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels, bgr_channels;
    cv::split(img, bgr_channels); // [B, G, R]
    cv::split(hsv, hsv_channels); // [H, S, V]

    const cv::Mat& h = hsv_channels[0];
    const cv::Mat& s = hsv_channels[1];
    const cv::Mat& v = hsv_channels[2];
    const cv::Mat& g = bgr_channels[1];
    const cv::Mat& b = bgr_channels[0];
    const cv::Mat& r = bgr_channels[2];

    bool found_r = false, found_g = false, found_b = false;
    int count_r = 0, count_g = 0, count_b = 0;
    double direct_x_ = 0;
    int direct_count_ = 0;

    for(int y = 0; y < img.rows; y++)
    {
        for(int x = 0; x < img.cols; x++)
        {
            uchar pixel_h = h.at<uchar>(y, x);
            uchar pixel_v = v.at<uchar>(y, x);

            uchar pixel_r = r.at<uchar>(y, x);
            uchar pixel_g = g.at<uchar>(y, x);
            uchar pixel_b = b.at<uchar>(y, x);

            if(remove_bright_or_dark_(pixel_r, pixel_g, pixel_b))
            {
                if(find_r_(pixel_h, pixel_r, pixel_g, pixel_b))
                {
                    count_r++;
                    found_r = true;
                    img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                }
                if(find_g_(pixel_h, pixel_g, pixel_r, pixel_b))
                {
                    count_g++;
                    found_g = true;
                    img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
                }
                if(find_b_(pixel_h, pixel_r, pixel_b))
                {
                    count_b++;
                    found_b = true;
                    img.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0);
                }
                if(find_direct_(pixel_v, pixel_r, pixel_g, pixel_b))
                {
                    direct_mask.at<uchar>(y, x) = 255;
                }
            }
        }
    }
    
    // cv::namedWindow("DIRECT", cv::WINDOW_NORMAL);
    // cv::imshow("DIRECT", direct_mask);

    cv::Mat labels, stats, centroids;
    int num_comp = cv::connectedComponentsWithStats(direct_mask, labels, stats, centroids, 8, CV_32S);
    if(num_comp > 1)
    {
        for(int i = 1; i < num_comp; ++i)
        {
            double center_x = centroids.at<double>(i, 0);
            direct_x_ += center_x;
            direct_count_++;
        }
    }

    if(direct_count_ > 0)
    {
        direct_x_ = direct_x_ / direct_count_;
        // std::cout << "direct_x 连通域中心平均: " << direct_x_ << std::endl;
        // std::cout << "direct_count_连通域个数: " << direct_count_ << std::endl;
    }
    if(direct_x_ >= target_width_ * 0.5)
    {
        need_turn_ = false;
    }
    else
    {
        need_turn_ = true;
    }
    
    // std::cout << "count_r: " << count_r << ", count_g: " << count_g << ", count_b: " << count_b << std::endl;
    return found_r && found_g && found_b;
}

bool DealImg::analyze_colored_(cv::Mat& img)
{
    cv::Mat mask_r, mask_g, mask_b;

    cv::Scalar color_r(0.0, 0.0, 0.0);
    cv::Scalar color_g(0.0, 255.0, 0.0);
    cv::Scalar color_b(255.0, 0.0, 0.0);

    cv::inRange(img, color_r, color_r, mask_r);
    cv::inRange(img, color_g, color_g, mask_g);
    cv::inRange(img, color_b, color_b, mask_b);


    deal_mask_(mask_r, 1, 9);
    deal_mask_(mask_g, 1, 9);
    deal_mask_(mask_b, 1, 9);

    bool r_ok = (analyze_single_color_(mask_r, "r") > 0);
    bool g_ok = (analyze_single_color_(mask_g, "g") > 0);
    bool b_ok = (analyze_single_color_(mask_b, "b") > 0);
    filter_centers_();

    return r_ok && g_ok && b_ok;
}

int DealImg::analyze_single_color_(const cv::Mat& mask, const std::string color_name)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if(contours.empty())
    {
        return 0;
    }

    int processes_count = 0;
#ifdef IMSHOW
    cv::Mat debug_img = mask.clone();
#endif

    for(size_t i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        if(area < area_min_threshold_)
        {
            continue;
        }

        cv::Moments m = cv::moments(contours[i]);
        if(m.m00 != 0)
        {
            int cx = static_cast<int>(m.m10 / m.m00);
            int cy = static_cast<int>(m.m01 / m.m00);
            if(color_name == "r")
            {
                detected_centers_.push_back({cx, cy, 3});
            }
            else if(color_name == "g")
            {
                detected_centers_.push_back({cx, cy, 1});
            }
            else if(color_name == "b")
            {
                detected_centers_.push_back({cx, cy, 2});
            }
#ifdef IMSHOW
            cv::circle(debug_img, cv::Point(cx, cy), 5, cv::Scalar(0, 0, 255), -1);
#endif
            processes_count++;
        }
    }
#ifdef IMSHOW
    if(processes_count > 0)
    {
        cv::namedWindow("Debug_" + color_name, cv::WINDOW_NORMAL);
        cv::imshow("Debug_" + color_name, debug_img);
    }
#endif
    return processes_count;
}

void DealImg::filter_centers_()
{
    if(detected_centers_.size() < 6)
    {
        return;
    }

    std::vector<bool> processed(detected_centers_.size(), false);
    std::vector<CenterInfo> filtered_centers;

    for(size_t i = 0; i < detected_centers_.size(); i++)
    {
        if(processed[i])
        {
            continue;
        }

        const CenterInfo& center = detected_centers_[i];
        processed[i] = true;
        std::vector<CenterInfo> group;
        group.push_back(center);

        for(size_t j = i + 1; j < detected_centers_.size(); j++)
        {
            if(processed[j] || detected_centers_[j].color_code != center.color_code)
            {
                continue;
            }
            double distance_x = std::abs(detected_centers_[i].x - detected_centers_[j].x);
            double distance_y = std::abs(detected_centers_[i].y - detected_centers_[j].y);
            if(distance_x < dist_x_ && distance_y < dist_y_)
            {
                group.push_back(detected_centers_[j]);
                processed[j] = true;
            }
        }

        if(group.size() > 1)
        {
            CenterInfo merged_center;
            merged_center.color_code = center.color_code;
            long long sum_x = 0, sum_y = 0;
            for(const auto& center : group)
            {
                sum_x += center.x;
                sum_y += center.y;
            }
            merged_center.x = static_cast<int>(sum_x / group.size());
            merged_center.y = static_cast<int>(sum_y / group.size());
            filtered_centers.push_back(merged_center);
        }
        else
        {
            filtered_centers.push_back(center);
        }
    }
    detected_centers_ = std::move(filtered_centers);
}

std::array<std::array<int, 3>, 4> DealImg::analyze_all_grids_()
{
    if(collected_grids_.empty())
    {
        std::cout << "错误： 没有收集到任何数据" << std::endl; 
        std::array<std::array<int, 3>, 4> empty_grid{};
        for(auto& row : empty_grid) row.fill(0);
        return empty_grid;
    }
    std::map<std::string, int> grid_frequency_map;
    for(const auto& center_list : collected_grids_)
    {
        std::string key = serialize_center_info_list_(center_list);
        grid_frequency_map[key]++;
    }
    std::string most_frequent_key = "";
    int max_count = 0;
    for(const auto& pair : grid_frequency_map)
    {
        if(pair.second > max_count)
        {
            max_count = pair.second;
            most_frequent_key = pair.first;
        }
    }
    std::vector<DealImg::CenterInfo> most_frequent_centers = deserialize_center_info_list_(most_frequent_key);
    return gridify_centers_(most_frequent_centers);
}

std::string DealImg::serialize_center_info_list_(const std::vector<DealImg::CenterInfo>& centers)
{
    std::ostringstream oss;
    std::vector<DealImg::CenterInfo> sorted_centers = centers;
    std::sort(sorted_centers.begin(), sorted_centers.end(), [](const DealImg::CenterInfo& a, const DealImg::CenterInfo& b){
        if(a.x != b.x) return a.x < b.x;
        if(a.y != b.y) return a.y < b.y;
        return a.color_code < b.color_code;
    });
    for(size_t i = 0; i < sorted_centers.size(); ++i)
    {
        oss << sorted_centers[i].x << "," << sorted_centers[i].y << "," << sorted_centers[i].color_code;
        if(i < sorted_centers.size() - 1) oss << ";"; // 用分号分隔每个点
    }
    return oss.str();
}

std::vector<DealImg::CenterInfo> DealImg::deserialize_center_info_list_(const std::string& serialized)
{
    std::vector<DealImg::CenterInfo> centers;
    std::istringstream iss(serialized);
    std::string token;
    while(std::getline(iss, token, ';'))
    {
        std::istringstream point_ss(token);
        std::string coord_token;
        DealImg::CenterInfo center;

        std::getline(point_ss, coord_token, ',');
        center.x = std::stoi(coord_token);

        std::getline(point_ss, coord_token, ',');
        center.y = std::stoi(coord_token);

        std::getline(point_ss, coord_token, ',');
        center.color_code = std::stoi(coord_token);

        centers.push_back(center);
    }
    return centers;
}

std::array<std::array<int, 3>, 4> DealImg::gridify_centers_(const std::vector<DealImg::CenterInfo>& centers)
{
    if(centers.empty())
    {
        std::cout << "未检测到任何颜色块" << std::endl;
        std::array<std::array<int, 3>, 4> empty_grid{};
        for(auto& row : empty_grid) row.fill(0);
        return empty_grid;
    }

    std::vector<int> x_coords, y_coords;
    for(const auto& center : centers)
    {
        x_coords.push_back(center.x);
        y_coords.push_back(center.y);
    }

    auto [min_x_it, max_x_it] = std::minmax_element(x_coords.begin(), x_coords.end());
    auto [min_y_it, max_y_it] = std::minmax_element(y_coords.begin(), y_coords.end());

    int min_x = *min_x_it;
    int max_x = *max_x_it;
    int min_y = *min_y_it;
    int max_y = *max_y_it;
    max_y = std::max(max_y, 1045);
    min_y = std::min(min_y, 275);
    max_x = std::max(max_x, 1650);
    min_x = std::min(min_x, 200);

    double row_spacing = (y_coords.size() > 1) ? (static_cast<double>(max_y - min_y) / 3.0) : 0.0;
    double col_spacing = (x_coords.size() > 1) ? (static_cast<double>(max_x - min_x) / 2.0) : 0.0;

    std::vector<double> theoretical_row_centers = { min_y, min_y + row_spacing, min_y + 2 * row_spacing, max_y };
    std::vector<double> theoretical_col_centers = { min_x, min_x + col_spacing, max_x };
    std::array<std::array<int, 3>, 4> grid{};
    for(auto& row : grid) row.fill(0);

    for(const auto& center : centers)
    {
        int px = center.x;
        int py = center.y;
        int color_code = center.color_code;
        auto best_row_it = std::min_element(theoretical_row_centers.begin(), theoretical_row_centers.end(),
                                            [py](double center, double val) { return std::abs(py - center) < std::abs(py - val); });
        int row = std::distance(theoretical_row_centers.begin(), best_row_it);
        auto best_col_it = std::min_element(theoretical_col_centers.begin(), theoretical_col_centers.end(),
                                            [px](double center, double val) { return std::abs(px - center) < std::abs(px - val); });
        int col = std::distance(theoretical_col_centers.begin(), best_col_it);

        row = std::max(0, std::min(3, row));
        col = std::max(0, std::min(2, col));

        grid[row][col] = color_code;
    }

    return grid;
}

bool DealImg::is_fit_rule_(std::vector<DealImg::CenterInfo>& centers)
{
    int r = 0, g = 0, b = 0;
    for(const auto& center : centers)
    {
        if(center.color_code == 1)
        {
            g++;
        }
        else if(center.color_code == 2)
        {
            b++;
        }
        else if(center.color_code == 3)
        {
            r++;
        }
    }
    return r == F_ && g == R1_ && b == R2_;
}


