#include "camera/camera.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto multi_node = std::make_shared<MultiCameraNode>("src/cv/camera/config/camera.yaml");

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(multi_node);
    executor.add_node(multi_node->getD455Node());
    executor.add_node(multi_node->getDealNode());
    // executor.add_node(multi_node->getKfsNode());
    std::thread spin_thread([&](){
        executor.spin();
    });
    while (rclcpp::ok())
    {
        // multi_node->getD455Node()->getViewer().viewer_run();
        // multi_node->getD455Node()->getWallViewer().viewer_run();
        // multi_node->getD455Node()->getGanViewer().viewer_run();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    executor.cancel();
    spin_thread.join();
    rclcpp::shutdown();
    return 0;
}