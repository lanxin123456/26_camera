/*
 * @Author: 於悦洋 yuyueyang2468@163.com
 * @Date: 2025-12-23 23:16:46
 * @LastEditors: 於悦洋 yuyueyang2468@163.com
 * @LastEditTime: 2026-01-29 10:10:35
 * @FilePath: /ROBOCON2026_base/src/cv/camera_backward/include/camera_backward/weapon.hpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <opencv2/opencv.hpp>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <fstream>
#include <chrono>
#include <deque>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <string>
#include <condition_variable>
#include <mutex>

#include <base_interfaces/msg/camera.hpp>
#include <base_interfaces/msg/camera_task.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "trt_yolo/yolo.hpp"
#include "camera_backward/save_video.hpp"
#include <camera_backward/actCamera_back.hpp>


using namespace std::chrono_literals;
using namespace cv;
using namespace std;

// #define SAVE_IMAGE
// #define SAVE_VEDIO
// #define IF_CAMERA

#define VEDIO

class CalFPS
{
public:
    CalFPS();
    ~CalFPS();
    void updateFrameTimer();
    double calculateFPS();
    void drawFPSOnImage(cv::Mat& image, 
                   cv::Point position = cv::Point(20, 40), 
                   double fontScale = 1.0, 
                   cv::Scalar color = cv::Scalar(0, 255, 0), 
                   int thickness = 2);
    void resetFPSCalculator();
private:
    std::deque<long> frameTimes;
    int bufferSize;
    std::chrono::steady_clock::time_point lastFrameTime;
};
 
class Deal : public rclcpp::Node
{
public:

    Deal(const std::string& config_path);
    ~Deal() override;
    void start();
    void stop();
    void calculate_dist(const cv::Mat& dealImage, 
    const std::vector<cv::Rect>& rects); 
    cv::Rect get_rect1() const { return rect_; }
    void deal(const int& count, const int& count_tai, const cv::Rect2f& rect, const Mat &frame);

private:

    int field_; // 0 红场  1 蓝场

    void loadYamlConfig(const std::string& path); 
    #ifdef IF_CAMERA
    camera_backward::ActCamera ACam;
    #endif

    struct FilteredResult {
        cv::Rect rect;
        float iou;
        float prob;
    };

    SaveVideoB video_saver_;

    Mat global_frame_;
    Mat srcframe_;
    Mat binary_;

    bool show_params_;
    bool visualize_;
    Rect rect_;
    Rect rect_fixed_;

    int interval_y_;       //rect 与 yolo框的间隔
    float iou_thresh_;
    std::string image_path_;
    std::string video_path_;

    std::vector<trt_yolo::det::Object> objs_weapon_;
    std::vector<trt_yolo::det::Object> objs_tai_;
    std::vector<cv::Rect>    rects_;               // weapon

    std::mutex task_mutex_;
    std::mutex global_mtx_;
    std::condition_variable cv_frame_;
    std::atomic<bool> running_{true};
    
    bool has_frame_;
    bool start_;

    int task_;
    int current_task_;
    int consistent_frame_count_;                 // 连续满足条件的帧数
    int total_pixels_;
    int changed_pixels_;
    int a_;                 //没有武器计数器
    int b_;                 //有武器计数器
    int c_;                 //没有台计数器

    int b_threshold_;
    int g_threshold_;
    int r_threshold_;
    int h_threshold_up_;
    int h_threshold_down_;
    int s_threshold_down_;

    float dist_;

    trt_yolo::YOLOv8Config config_;
    unique_ptr<trt_yolo::YOLOv8> yolo_all_;

    rclcpp::Publisher<base_interfaces::msg::Camera>::SharedPtr pub_weapon_;
    rclcpp::Subscription<base_interfaces::msg::CameraTask>::SharedPtr sub_task_;
    void task_callback(const base_interfaces::msg::CameraTask::SharedPtr msg);
    void process_rect(const Rect& rect, int& changed_pixels);
    bool cal_ch_rate(const int& changed_pixels_, std::ostringstream& oss);
    bool deal_frame(const Mat &frame, const cv::Rect_<float>& rect, std::ostringstream& oss);
    float calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2);
    void filterAndSortWeapons(
        std::vector<cv::Rect>& rects,
        const std::vector<trt_yolo::det::Object>& objs_weapon,
        float iou_thresh);
    std::ofstream camera_backward;

    void camera_thread_func();
    void process_thread_func();
    std::thread th_camera_;
    std::thread th_process_;
};