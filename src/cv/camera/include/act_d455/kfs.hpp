#pragma once

#include <chrono>
#include <deque>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <fstream>
#include <std_msgs/msg/float32.hpp>

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

    cv::Mat Kfs(const Mat &frame, cv::Rect& rect, const Mat &depth);
    void boudary(const Mat& temp, const cv::Rect& rect);

    const vector<Point>& getPointsLeft() const { return points_left_; }
    const vector<Point>& getPointsRight() const { return points_right_; }

    std::ofstream kfs_log_;
private:
    int iterations_{200};
    double dist_thresh_{3.0};
    int min_inliers_{20};
    struct LineKB
    {
        // x = ky + b
        double k {0.0};
        double b {0.0};
        bool valid {false};
        std::vector<cv::Point> inliers;
    };
    LineKB ransacLineFit( const std::vector<cv::Point>& pts ); //x = k*y + b

    vector<Point> points_left_;
    vector<Point> points_right_;

    cv::Mat binary_;

private:

};  