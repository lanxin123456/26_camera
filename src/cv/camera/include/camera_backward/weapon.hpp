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
    void deal(const int& count, const int& count_tai, const cv::Rect2f& rect, const cv::Rect2f& rect_gan, const Mat &frame);

private:
    // --- 武器框追踪与推算相关变量 ---
    bool last_frame_has_target_ = false; // 上一帧是否成功追踪到目标
    float last_target_center_x_ = -1.0f; // 目标武器框上一次的中心 x 坐标
    float last_target_w_ = 0.0f;        // 目标武器框上一次的宽度
    float last_target_h_ = 0.0f;     // 记录历史高度
    float target_ratio_ = 0.0f;     // 记录滤波平滑后的高宽比
    float target_dx_ = 0.0f;            // 目标武器框在 x 方向上的每帧像素位移量(速度)
    int missing_frames_ = 0;            // 目标连续消失的帧数
    const int MAX_MISSING_FRAMES = 20;  // 允许最大遮挡外推的帧数（按60fps算大约0.33秒）

private:
    std::vector<cv::Rect>    rects_both_;               // weapon and gun
    int max_assigned_index_ = 0;        // 当前记录的最高序号 (0, 1, 2, 3, 4)
    bool index4_captured_ = false;      // 是否成功捕获并锁定了序号4

    // 用于检测区域进入的“上帧状态历史”（边缘触发器）
    bool last_zone_A_occupied_ = false; 
    bool last_occlusion_state_ = false; // 上帧是否处于“B有A无”的遮挡状态

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

    float record_y_{0.f};//记录历史最大y坐标
    float now_y_{1000.f};   //记录 ok 时的变小后的y坐标
    bool upstate_{true};


    Rect rect_;
    Rect rect_fixed_;

    int interval_y_;       //rect 与 yolo框的间隔
    float iou_thresh_;
    std::string image_path_;
    std::string video_path_;

    std::vector<trt_yolo::det::Object> objs_weapon_;
    std::vector<trt_yolo::det::Object> objs_tai_;
    std::vector<trt_yolo::det::Object> objs_gan_;

    std::vector<cv::Rect>    rects_;               // weapon
    std::vector<cv::Rect>    rects_gan_;               

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
    bool deal_frame(const Mat &frame, const cv::Rect_<float>& rect, const cv::Rect2f& rect_gan, std::ostringstream& oss);
    float calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2);
    void filterAndSortWeapons(
        std::vector<cv::Rect>& rects,
        const std::vector<trt_yolo::det::Object>& objs_weapon,
        float iou_thresh);

    std::ofstream camera_backward;
    std::ofstream test_rect_y;

    void camera_thread_func();
    void process_thread_func();
    std::thread th_camera_;
    std::thread th_process_;
};