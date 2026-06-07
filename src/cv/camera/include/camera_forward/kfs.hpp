#pragma once

#include <chrono>
#include <deque>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <fstream>
#include <std_msgs/msg/float32.hpp>
#include "base_interfaces/msg/camera.hpp"
#include "base_interfaces/msg/camera_kfs.hpp"

#include "camera_forward/save_video.hpp"
#include "camera_forward/actCamera_for.hpp"

#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <rclcpp/rclcpp.hpp>
#include <unordered_set>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <random>
#include <thread>
#include <memory>
#include <trt_yolo/yolo.hpp>

extern std::ofstream kfs_log;
using namespace std::chrono_literals;
using namespace cv;
using namespace std;

// #define SAVE_IMAGE
// #define SAVE_VEDIO
// #define IF_CAMERA

#define VEDIO

struct LineKB
{
    // x = ky + b
    double k {0.0};
    double b {0.0};
    bool valid {false};
    std::vector<cv::Point> inliers;
};

class KFS : public rclcpp :: Node
{
public:
    KFS();
    ~KFS() override;
    void start();
    void stop();

private:
    #ifdef IF_CAMERA
    camera_forward::ActCamera ACam;
    #endif
    std::atomic<int> shared_key_{255}; // 255 代表没有按键按下
    std::ofstream kfs_log_;
    SaveVideoF video_saver_;

    std::mutex global_yaw_;

    std::mutex global_mtx_;
    std::condition_variable cv_frame_;
    bool has_frame_;
    bool visualize_;
    bool over_{false};//kfs可能超过边界
    int l_dist_{0};
    int r_dist_{0};

    std::atomic<bool> running_{true};
    cv::Mat global_frame_;

    cv::Mat dst_;
    std::ostringstream oss_;
    int lenth_;
    int iterations_;
    int min_inliers_;
    int ul_{0};
    int ur_{0};
    int count_l_{0};
    int count_r_{0};
    double cx_{310.87976};
    double fx_{379.51236};

    double dist_thresh_;
    //非 static 成员不能是 constexpr
    static constexpr double L_ = 350.0;
    float dist_;
    float center_x_;//主心x坐标
        
    float lidar_x_{0.};
    float lidar_y_{0.};
    float lidar_z_{0.};
    float lidar_yaw_{0.0};
    float lidar_roll_{0.};
    float lidar_pitch_{0.};

    cv::Mat image_;
    cv::Mat binary_;
    cv::Point center_p_;

    trt_yolo::YOLOv8Config config_;
    trt_yolo::YOLOv8 *yolo_detector_;
    
    vector<Point> points_left;
    vector<Point> points_right;
    std::thread th_camera_;
    std::thread th_process_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
private:
    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
    
    void showdst(const Mat &temp);
    double computeMidX();

    cv::Mat Kfs(const Mat &frame);
    cv::Mat Max(const cv::Mat& src);
    void applyPrewitt_0(const Mat& src, Mat& grad_y);
    void boudary(const Mat& temp);
    void samey(vector<Point>& points_left, vector<Point>& points_right);
    cv::Point center(const vector<Point>& points_left, const vector<Point>& points_right);
    void draw(Mat& dst);
    void publishdist(int p);
    LineKB ransacLineFit( const std::vector<cv::Point>& pts ); //x = k*y + b
    rclcpp::Publisher<base_interfaces::msg::CameraKfs>::SharedPtr pub_kfs_;
    void filterPoints(std::vector<cv::Point>& points, int& dist);
    void camera_thread();
    void process_thread();
};  