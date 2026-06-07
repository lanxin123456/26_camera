// /*
//  * @Author: 於悦洋 yuyueyang2468@163.com
//  * @Date: 2026-01-15 15:15:00
//  * @LastEditors: 於悦洋 yuyueyang2468@163.com
//  * @LastEditTime: 2026-02-02 23:02:21
//  * @FilePath: /ROBOCON2026_base/src/cv/camera_backward/include/camera_backward/assemble.hpp
//  * @Description: 
//  * 
//  * Copyright (c) 2026 by Action, All Rights Reserved. 
//  */
// #pragma once

// #include <chrono>
// #include <deque>
// #include <iostream>
// #include <iomanip>
// #include <ctime>
// #include <fstream>
// #include <string>
// #include <vector>

// #include <opencv2/opencv.hpp>
// #include <rclcpp/rclcpp.hpp>

// #include <base_interfaces/msg/camera.hpp>
// #include <base_interfaces/msg/camera_task.hpp>

// using namespace cv;
// using namespace std;
// extern ofstream camera_backward;

// class Assemble : public rclcpp::Node
// {
// public:
//     Assemble() : Node("assemble")
//     {
//         rect_ = Rect(314, 214, 39, 11);

//         total_pixels_ = rect_.width * rect_.height;

//         pub_assemble_ = this->create_publisher<base_interfaces::msg::Camera>("/weapon_camera_miao", 10);
//         sub_task_2_ = this->create_subscription<base_interfaces::msg::CameraTask>("/lx_task", 10,
//         std::bind(&Assemble::task_callback, this, std::placeholders::_1));
//     }

//     Mat deal_frame(const Mat &frame, const cv::Rect_<float>& rect);
//     cv::Rect get_rect1() const { return rect_; }

// private:
//     int task_;
//     int consistent_frame_count_ = 0;                 // 连续满足条件的帧数
//     int total_pixels_;
//     int changed_pixels_ = 0;
//     int changed_pixels_2 = 0;

//     Mat srcframe_;
//     Mat binary;
//     Rect rect_;
//     Rect rect2;

//     rclcpp::Publisher<base_interfaces::msg::Camera>::SharedPtr pub_assemble_;
//     rclcpp::Subscription<base_interfaces::msg::CameraTask>::SharedPtr sub_task_2_;
// private:
//     void task_callback(const base_interfaces::msg::CameraTask::SharedPtr msg);
//     void process_rect(const Rect& rect, int& changed_pixels);
//     void cal_ch_rate(const int& changed_pixels_, std::ostringstream& oss);
//     void publish_result(std::ostringstream& oss);
// };