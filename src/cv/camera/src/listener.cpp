/*
 * @Author: 於悦洋 yuyueyang2468@163.com
 * @Date: 2026-01-04 15:54:23
 * @LastEditors: 於悦洋 yuyueyang2468@163.com
 * @LastEditTime: 2026-01-23 14:41:10
 * @FilePath: /ROBOCON2026_base/src/cv/camera_forward/src/listener.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <base_interfaces/msg/grid_state.hpp>
#include <chrono>

class Listener : public rclcpp::Node
{
public:
    Listener() : Node("listener_node_cpp")
    {
        RCLCPP_INFO(this->get_logger(), "Hello, world! I am a listener."); 
        subscription_ = this->create_subscription<std_msgs::msg::Float32>("dist_kfs", 10,
                                        std::bind(&Listener::do_cb, this, std::placeholders::_1));
        sub_base_ = this->create_subscription<base_interfaces::msg::GridState>("d455/gridstate", 10,
                                        std::bind(&Listener::do_base, this, std::placeholders::_1));
        // 初始化上一次接收时间
        last_time_ = std::chrono::steady_clock::now();
    }

private:
    void do_cb(const std_msgs::msg::Float32 &msg)
    {
        // 获取当前时间
        auto current_time = std::chrono::steady_clock::now();
        
        // 计算时间间隔（毫秒）
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time_);
        double time_interval_ms = duration.count();
        
        RCLCPP_INFO(this->get_logger(), "订阅到的消息: %f, 时间间隔: %.3f 毫秒", msg.data, time_interval_ms);
        
        // 更新上一次接收时间
        last_time_ = current_time;
    }

    void do_base(const base_interfaces::msg::GridState &msg)
    {
        // 获取当前时间
        auto current_time = std::chrono::steady_clock::now();
        
        // 计算时间间隔（毫秒）
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time_);
        double time_interval_ms = duration.count();
        
        std::stringstream ss;
        for (int i = 0; i < 9; i++)
        {
            ss << msg.grid_state[i] << " ";
        }
        RCLCPP_INFO(this->get_logger(),
                    "grid_state: %s, dt: %.3f ms",
                    ss.str().c_str(),
                    time_interval_ms);

        // 更新上一次接收时间
        last_time_ = current_time;
    }
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr subscription_;
    rclcpp::Subscription<base_interfaces::msg::GridState>::SharedPtr sub_base_;

    std::chrono::steady_clock::time_point last_time_;
};

// int main(int argc, char **argv)
// {
//     rclcpp::init(argc, argv);
//     rclcpp::spin(std::make_shared<Listener>());
//     rclcpp::shutdown();
//     return 0;
// }