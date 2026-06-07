#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <sstream>
#include <tuple>
#include <vector>

/*
 * 编码设置：从入口开始，从左到右，从下到上，每个单元格两位
*/

// #define IMSHOW


class GridPublish : public rclcpp::Node
{
public:
    GridPublish() : Node("grid_publish_node")
    {
        publisher_ = this->create_publisher<std_msgs::msg::Int32>("/encoded_grid", 10);
    }
    void publish_grid(int32_t encoded_value)
    {
        auto msg = std_msgs::msg::Int32();
        msg.data = encoded_value;
        publisher_->publish(msg);
        RCLCPP_INFO(this->get_logger(), "Published grid: %d", encoded_value);
    }
private:
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr publisher_;
};


class DealImg
{
public:
    DealImg() = default;
    DealImg(const std::string& config_path);
    ~DealImg() = default;

    // 如果没得到最终结果，返回空指针
    bool Deal(const cv::Mat &img);

    void cleanup()
    {
        if(node_)
        {
            node_.reset();
        }
        src_img_.release();
        perspective_img_.release();
        detected_centers_.clear();
        collected_grids_.clear();
    }


private:
    std::shared_ptr<GridPublish> node_;
    int result_;
    int lag_frames_;
    int current_lag_frame_ = 0;
    bool is_lagging_ = false;
    double brightness_thresh_;
    double avg_brightness_thresh_;
    int min_area_;
    float target_width_;
    float target_height_;
    cv::Mat src_img_;
    cv::Mat perspective_img_;
    int dist_x_;
    int dist_y_;
    int grid_collection_frames_;
    int area_min_threshold_;
    int continue_wrong_frames_;
    int current_collection_frame_ = 0;
    int current_wrong_frame_ = 0;
    bool is_collecting_ = false;
    bool need_turn_ = false;

    int fit_size_;
    int R1_;
    int R2_;
    int F_;

    struct CenterInfo
    {
        int x, y;
        int color_code;
    };
    struct CandidateContour
    {
        std::vector<cv::Point> contour;
        double area;
        cv::Moments moments;
    };
    std::vector<CenterInfo> detected_centers_;
    std::vector<std::vector<CenterInfo>> collected_grids_;

    int blue_hu_;
    int blue_hl_;
    int blue_dbr_;
    int green_hu_;
    int green_hl_;
    int green_gl_;
    int green_dgr_;
    int green_dgb_;
    int red_hu_u_;
    int red_hl_u_;
    int red_hu_l_;
    int red_hl_l_;
    int red_rl_;
    int red_drb_;
    int red_drg_;

    int derect_rgu_;
    int derect_bu_;
    int derect_rgl_;
    int derect_bl_;
    int derect_vu_;
    int derect_vl_;



    void deal_mask_(cv::Mat &mask, int close_kernel = 5, int open_kernel = 5);
    std::vector<cv::Point2f> order_points(const std::vector<cv::Point2f>& pts);
    bool find_largest_counter_(const cv::Mat &mask, cv::Point2f* corners);
    void perspective_transform_(cv::Mat &img, const cv::Point2f* corners);
    bool all_in_normal_range_handle_(cv::Mat& img);
    bool analyze_colored_(cv::Mat& img);
    int analyze_single_color_(const cv::Mat& mask, const std::string color_name);
    void filter_centers_();
    std::array<std::array<int, 3>, 4> analyze_all_grids_();
    std::string serialize_center_info_list_(const std::vector<DealImg::CenterInfo>& centers);
    std::vector<DealImg::CenterInfo> deserialize_center_info_list_(const std::string& serialized);
    std::array<std::array<int, 3>, 4> gridify_centers_(const std::vector<DealImg::CenterInfo>& centers);
    bool is_fit_rule_(std::vector<DealImg::CenterInfo>& centers);
    bool is_empty_(const std::array<std::array<int, 3>, 4>& grid);
    int32_t encode_grid_(const std::array<std::array<int, 3>, 4>& grid);

    bool find_r_(uchar pixel_h, uchar pixel_r, uchar pixel_g, uchar pixel_b)
    {
        bool con_1 = pixel_h <= red_hu_u_ && pixel_h >= red_hu_l_;
        bool con_2 = pixel_h <= red_hl_u_ && pixel_h >= red_hl_l_;
        bool con_3 = pixel_r >= red_rl_ && ((pixel_r - pixel_g) >= red_drg_) && ((pixel_r - pixel_b) >= red_drb_);
        return (con_1 || con_2) && con_3;
    }
    bool find_g_(uchar pixel_h, uchar pixel_g, uchar pixel_r, uchar pixel_b)
    {
        bool con_1 = pixel_h <= green_hu_ && pixel_h >= green_hl_;
        bool con_2 = pixel_g >= green_gl_ && ((pixel_g - pixel_r) >= green_dgr_) && ((pixel_g - pixel_b) >= green_dgb_);
        return con_1 && con_2;
    }
    bool find_b_(uchar pixel_h, uchar pixel_r, uchar pixel_b)
    {
        return (pixel_h <= blue_hu_) && (pixel_h >= blue_hl_) && ((pixel_b - pixel_r) >= blue_dbr_);
    }
    bool find_direct_(uchar pixel_v, uchar pixel_r, uchar pixel_g, uchar pixel_b)
    {
        bool con_1 = pixel_v >= derect_vl_ && pixel_v <= derect_vu_;
        bool con_2 = pixel_r >= derect_rgl_ && pixel_r <= derect_rgu_;
        bool con_3 = pixel_g >= derect_rgl_ && pixel_g <= derect_rgu_;
        bool con_4 = pixel_b >= derect_bl_ && pixel_b <= derect_bu_;
        return con_1 && con_2 && con_3 && con_4;
    }
    bool remove_bright_or_dark_(uchar pixel_r, uchar pixel_g, uchar pixel_b)
    {
        if(pixel_r > 230 && pixel_g > 230 && pixel_b > 230)
        {
            return false;
        }
        if(pixel_r < 80 && pixel_g < 80 && pixel_b < 80)
        {
            return false;
        }
        return true;
    }

};


