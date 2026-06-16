
// #include <rclcpp/rclcpp.hpp>
// #include <std_msgs/msg/float32.hpp>
// #include <base_interfaces/msg/grid_state.hpp>
// #include <chrono>

// int a[9];

// class Listener : public rclcpp::Node
// {
// public:
//     Listener() : Node("listener_node_cpp")
//     {
//         RCLCPP_INFO(this->get_logger(), "Hello, world! I am a listener."); 
//         subscription_ = this->create_subscription<std_msgs::msg::Float32>("dist_kfs", 10,
//                                         std::bind(&Listener::do_cb, this, std::placeholders::_1));
//         sub_base_ = this->create_subscription<base_interfaces::msg::GridState>("d455/gridstate", 10,
//                                         std::bind(&Listener::do_base, this, std::placeholders::_1));
//         // 初始化上一次接收时间
//         last_time_ = std::chrono::steady_clock::now();
//     }

// private:
//     void do_cb(const std_msgs::msg::Float32 &msg)
//     {
//         // 获取当前时间
//         auto current_time = std::chrono::steady_clock::now();
        
//         // 计算时间间隔（毫秒）
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time_);
//         double time_interval_ms = duration.count();
        
//         RCLCPP_INFO(this->get_logger(), "订阅到的消息: %f, 时间间隔: %.3f 毫秒", msg.data, time_interval_ms);
        
//         // 更新上一次接收时间
//         last_time_ = current_time;
//     }

//     void do_base(const base_interfaces::msg::GridState &msg)
//     {
//         // 获取当前时间
//         auto current_time = std::chrono::steady_clock::now();
        
//         // 计算时间间隔（毫秒）
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time_);
//         double time_interval_ms = duration.count();
        
//         std::stringstream ss;
//         for (int i = 0; i < 9; i++)
//         {
//             ss << msg.grid_state[i] << " ";
//             a[i] = msg.grid_state[i];
//         }
//         for (int i = 0; i < 9; i++)
//         {
//             std::cout << a[i] << " " ;
//         }
//         std::cout << std::endl;
//         RCLCPP_INFO(this->get_logger(),
//                     "grid_state: %s, dt: %.3f ms",
//                     ss.str().c_str(),
//                     time_interval_ms);

//         // 更新上一次接收时间
//         last_time_ = current_time;
//     }
//     rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr subscription_;
//     rclcpp::Subscription<base_interfaces::msg::GridState>::SharedPtr sub_base_;

//     std::chrono::steady_clock::time_point last_time_;
// };

// int main(int argc, char **argv)
// {
    
//     rclcpp::init(argc, argv);
//     rclcpp::spin(std::make_shared<Listener>());
//     rclcpp::shutdown();
//     return 0;
// }

//=================== 测试 ===================
#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <base_interfaces/msg/camera_choose.hpp>

class CameraModeSubscriber : public rclcpp::Node
{
public:
    CameraModeSubscriber() 
    : Node("camera_mode_subscriber")
    {
        // 创建订阅者，订阅 /camera_mode 话题，队列大小为 10
        subscription_ = this->create_subscription<base_interfaces::msg::CameraChoose>(
            "/camera_mode", 
            10,
            std::bind(&CameraModeSubscriber::topic_callback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "Camera Mode Subscriber started. Listening for latency...");
    }

private:
    void topic_callback(const base_interfaces::msg::CameraChoose::SharedPtr msg)
    {
        // 1. 获取当前节点时间
        rclcpp::Time now_time = this->now();

        // 2. 将消息中的时间戳转换为 rclcpp::Time
        rclcpp::Time pub_time(msg->stamp);

        // 3. 计算时间差（计算出的 Duration 单位默认是纳秒）
        rclcpp::Duration latency = now_time - pub_time;

        
        double latency_ms = latency.seconds() * 1000.0;

        // 5. 打印接收到的数据和延迟
        RCLCPP_INFO(this->get_logger(), 
                    "Received Camera Mode: %d | Latency: %.3f ms", 
                    msg->camera, latency_ms);
    }

private:
    rclcpp::Subscription<base_interfaces::msg::CameraChoose>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<CameraModeSubscriber>();
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}