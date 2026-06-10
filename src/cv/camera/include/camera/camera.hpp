#pragma once

#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <mutex>
#include <fstream>
#include <atomic>
#include <chrono>

// === 直接使用你原来的类（不改！）===
#include "act_d455/d455_node.hpp"
#include "camera_backward/weapon.hpp"    
// #include "camera_forward/kfs.hpp"               
#include <base_interfaces/msg/camera_choose.hpp>

// 注意：这里必须继承 rclcpp::Node 才能使用订阅和定时器
class MultiCameraNode : public rclcpp::Node
{
public:
    MultiCameraNode(const std::string& config_path);
    ~MultiCameraNode();

    std::shared_ptr<Deal> getDealNode() { return deal_; }
    // std::shared_ptr<KFS> getKfsNode()  { return kfs_; }
    std::shared_ptr<D455Node> getD455Node(){ return d455_; }

private:
    std::mutex mut_camera_;
    int target_camera_ = 0;   // 期望开启的相机编号
    int current_camera_ = 0;  // 当前正在运行的相机编号 (0表示全部关闭)

    std::ofstream camera_error_;
    std::shared_ptr<D455Node> d455_;
    std::shared_ptr<Deal> deal_;
    // std::shared_ptr<KFS> kfs_;
    bool run_d455_{false};
    bool run_kfs_{false};
    bool run_deal_{false};

    rclcpp::Subscription<base_interfaces::msg::CameraChoose>::SharedPtr sub_camera_;
    rclcpp::TimerBase::SharedPtr switch_timer_; // 用于监控状态切换的定时器

    void task_callback(const base_interfaces::msg::CameraChoose::SharedPtr msg);
    void logstart(const std::string str);
    void check_and_switch_camera(); // 核心切换逻辑
    void stop_all_cameras();        // 辅助停止函数
};