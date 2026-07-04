#include "camera/camera.hpp"

MultiCameraNode::MultiCameraNode(const std::string& config_path) : Node("multi_camera_manager_node") // 初始化父类 Node
{
    try 
    {         
        sub_camera_ = this->create_subscription<base_interfaces::msg::CameraChoose>(
            "/camera_mode", 10,
            std::bind(&MultiCameraNode::task_callback, this, std::placeholders::_1));

        logstart("camera_error.csv");
        
        d455_ = std::make_shared<D455Node>(config_path);
        deal_ = std::make_shared<Deal>(config_path);
        // kfs_  = std::make_shared<KFS>();

        switch_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(5),
            std::bind(&MultiCameraNode::check_and_switch_camera, this));
    }
    catch (const rclcpp::exceptions::RCLError &e) { 
        std::string msg = std::string("[MultiCameraNode][RCLError] ") + e.what(); 
        std::cerr << msg << std::endl; 
        if (camera_error_.is_open()) camera_error_ << msg << std::endl; 
        throw; 
    } 
    catch (const cv::Exception &e) { 
        std::ostringstream oss; 
        oss << "[MultiCameraNode][OpenCV Exception]\n" 
            << "what: " << e.what() << "\n" 
            << "code: " << e.code << "\n" 
            << "func: " << e.func << "\n" 
            << "file: " << e.file << "\n" 
            << "line: " << e.line; 
        std::string msg = oss.str(); std::cerr << msg << std::endl; 
        if (camera_error_.is_open()) camera_error_ << msg << std::endl; 
        throw; 
    } 
    catch (const std::exception &e) { 
        std::string msg = std::string("[MultiCameraNode][std::exception] ") + e.what(); 
        std::cerr << msg << std::endl; 
        if (camera_error_.is_open()) camera_error_ << msg << std::endl; 
        throw; 
    } 
    catch (...) { 
        std::string msg = "[MultiCameraNode][Unknown Exception]"; 
        std::cerr << msg << std::endl; 
        if (camera_error_.is_open()) camera_error_ << msg << std::endl; 
        throw; 
    } 
}

void MultiCameraNode::task_callback(const base_interfaces::msg::CameraChoose::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(mut_camera_);
    target_camera_ = msg->camera;  
}

void MultiCameraNode::check_and_switch_camera()
{
    int desired_camera{0};
    {
        std::lock_guard<std::mutex> lock(mut_camera_);
        desired_camera = target_camera_;
        desired_camera = 2;
    }
    if (desired_camera == current_camera_) {
        return;
    }
    current_camera_ = desired_camera;

    //先停止所有当前正在运行的相机
    // stop_all_cameras();

    // 假设：1 代表 D455, 2 代表 Deal, 3 代表 KFS
    cout << "当前模式： " << desired_camera << endl;
    if(camera_error_.is_open()) camera_error_ << "当前模式： " << desired_camera << endl;

    
    switch (desired_camera) {
        case 1:
            if(d455_ != nullptr && !run_d455_) 
            {
                d455_->start();
                run_d455_ = true;
            }
            if(deal_ != nullptr && run_deal_) 
            {
                deal_->stop();
                run_deal_ = false;
            }
            break;
        case 2:
            if(deal_ != nullptr && !run_deal_) 
            {
                deal_->start();
                run_deal_ = true;
            }
            if(d455_ != nullptr && run_d455_) 
            {
                d455_->stop(); 
                run_d455_ = false;
            }            
            break;
        case 3:
            if(d455_ != nullptr && !run_d455_) 
            {
                d455_->start();
                run_d455_ = true;
                if(deal_ != nullptr && !run_deal_) 
                {
                    deal_->start();
                    run_deal_ = true;
                }
            }
            break;
        case 0:
            cout << "未开启相机" << endl;
            break;
        default:
            break;
    }
}

void MultiCameraNode::stop_all_cameras()
{
    if(current_camera_ == 3) 
    {
        if(d455_ != nullptr && run_d455_) 
        {
            d455_->stop(); 
            run_d455_ = false;
            return;
        }
    }
    if(d455_ != nullptr && run_d455_) 
    {
        d455_->stop(); 
        run_d455_ = false;
    }
    if(deal_ != nullptr && run_deal_) 
    {
        deal_->stop();
        run_deal_ = false;
    }
    // if(kfs_ != nullptr && run_kfs_ )  
    // {
    //     kfs_->stop();
    //     run_kfs_ = false;
    // }
}

void MultiCameraNode::logstart(const std::string str)
{
    camera_error_.open("camera_error.csv", std::ios::app);
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);
    if(camera_error_.is_open()) camera_error_ << "\n\n\n\n\n"
                                    << "时间, " 
                                    << put_time(localTime, "%Y-%m-%d %H:%M:%S") 
                                    << " ===================================================================================================================================" 
                                    << endl;
}

MultiCameraNode::~MultiCameraNode()
{
    stop_all_cameras();
    if (camera_error_.is_open()) {
        camera_error_.close();
    }
}

