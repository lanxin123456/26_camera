#include <chrono>
#include <memory>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <base_interfaces/msg/camera_choose.hpp>

using namespace std::chrono_literals;

class CameraModePublisher : public rclcpp::Node
{
public:
    CameraModePublisher() 
    : Node("camera_mode_publisher"), publish_count_(0)
    {
        // 创建发布者
        publisher_ = this->create_publisher<base_interfaces::msg::CameraChoose>("/camera_mode", 10);

        sequence_ = {1, 2, 1};

        // 定时器（5秒）
        timer_ = this->create_wall_timer(
            200ms,
            std::bind(&CameraModePublisher::timer_callback, this)
        );

        RCLCPP_INFO(this->get_logger(), "Camera Mode Publisher started.");
    }

private:
    void timer_callback()
    {
        auto msg = base_interfaces::msg::CameraChoose();

        // 根据发送次数决定当前值
        if (publish_count_ < sequence_.size())
        {
            msg.camera = sequence_[publish_count_];
        }
        else
        {
            msg.camera = 1;  // 后续固定为1
        }

        msg.stamp = this->now();

        RCLCPP_INFO(this->get_logger(), 
                    "Publishing Camera Mode: %d (count=%d)", 
                    msg.camera, publish_count_);

        publisher_->publish(msg);

        publish_count_++;
    }

private:
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<base_interfaces::msg::CameraChoose>::SharedPtr publisher_;

    std::vector<int> sequence_;  // 初始序列
    size_t publish_count_;       // 发送计数器
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<CameraModePublisher>();
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}
