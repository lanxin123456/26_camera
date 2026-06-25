#include <act_d455/d455_node.hpp>

#include <chrono>
#include <cstring>
#include "act_d455/kun.hpp"
#include <filesystem>

#define COLOR 1 //1是红， 0是蓝

D455Node::D455Node(const std::string& config_path, const rclcpp::NodeOptions& options)
: rclcpp::Node("d455_node", options), running_(false), 
config_(7, {"red_r1", "red_r2", "blue_r1", "blue_r2", "gw_gan1", "gy_gan2", "wall"}, 
        {{0, 0, 255}, {0, 0, 200}, {255, 0, 0}, {200, 0, 0}, {0, 255, 255}, {0, 200, 200}, {128, 128, 128}}, 
        cv::Size(640, 640), 100, 0.6f, 0.65f),
config_grid_(7, {"red_r1", "red_r2", "blue_r1", "blue_r2", "gw_gan1", "gy_gan2", "wall"}, 
        {{0, 0, 255}, {0, 0, 200}, {255, 0, 0}, {200, 0, 0}, {0, 255, 255}, {0, 200, 200}, {128, 128, 128}}, 
        cv::Size(640, 640), 100, 0.6f, 0.65f),
config_part2_(1, {"kfs"}, {{0,255,0}}, cv::Size(640, 640), 100, 0.6f, 0.65f),
nine_square_depth_value_{0.f},dealImg_("src/cv/camera/config/deal.yaml")
{
    loadYamlConfig(config_path);

	d455_log_.open("d455_log_.csv", std::ios::app);
	std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);
	if(d455_log_.is_open()) d455_log_ <<"  \n\n\n\n\n时间, " << put_time(localTime, "%Y-%m-%d %H:%M:%S") << " ========================" << endl;

	display_ = this->declare_parameter<bool>("display", true);

	save_frames_ = this->declare_parameter<bool>("save_frames", false);
	frames_dir_ = this->declare_parameter<std::string>("frames_dir", std::string("frames"));
	frame_ext_ = this->declare_parameter<std::string>("frame_ext", std::string("png"));
	if (!frame_ext_.empty() && frame_ext_.front() == '.')
	{
		frame_ext_.erase(frame_ext_.begin());
	}

	if (save_frames_)
	{
		try
		{
			std::filesystem::create_directories(frames_dir_);
		}
		catch (const std::exception& e) 
		{
			RCLCPP_ERROR(this->get_logger(), "Failed to create frames dir '%s': %s", frames_dir_.c_str(), e.what());
			save_frames_ = false;
		}
	}
	odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/map_to_base_link", rclcpp::SensorDataQoS(),
            std::bind(&D455Node::odom_callback, this, std::placeholders::_1));

    R_bw_ <<
    -0.00763784,   0.999536,    0.0295001,
    -0.999861,   -0.00719551,  -0.0150713,
    -0.0148521, -0.0296111,   0.999451;

    cam_pos_ <<
    2084.47,
    10232.9,
    948.578;

    // rclcpp::QoS qos(50);
    // qos.best_effort();

    // odom_highfreq_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
    //     "odin1/odometry_highfreq",
    //     qos,
    //     std::bind(&D455Node::odom_callback, this, std::placeholders::_1));
            
    // d455_sub_ = this->creat_subscription<base_interfaces::msg::AlignStart>
    // ("",rclcpp::SensorDataQoS(),std::bind(&D455Node::d455_callback, this, std::placeholders::_1))
    
    yolo_detector_grid_  = new trt_yolo::YOLOv8("src/cv/best_59.engine",config_grid_);
    yolo_detector_  = new trt_yolo::YOLOv8("src/cv/best_59.engine",config_);

    trt_yolo_part2_ = new trt_yolo::YOLOv8("/home/lx/runs/detect/train28/weights/kfs_69.engine",config_part2_);

    yolo_detector_grid_->make_pipe(true);
    yolo_detector_->make_pipe(true);

    trt_yolo_part2_->make_pipe(true);

	kfs_show_cloud = std::make_shared<PointCloud>();
    gan_show_cloud = std::make_shared<PointCloud>();

	d455_ = std::make_shared<ActD455>(act_d455_params_);
	pcl_ = std::make_shared<Pclprocess>(pcl_params_);

	gstate_ = std::make_shared<Grid_State>(grid_params_);

	align_ = std::make_shared<Align>();
    pick_ = std::make_shared<Pick>();
    kfs_ = std::make_shared<KFS>();
    trt_seg_  = std::make_shared<TRTNode>("/home/lx/兰欣20241872/python/UNet++/output_960x720_620_2/best_621.engine");


    sub_start_test_ = this->create_subscription<base_interfaces::msg::GridStart>("/gridstart", 10,
        std::bind(&D455Node::task_callback, this, std::placeholders::_1));


	std::string base = "zhutianxiang/avt_d455";
	std::string ext = ".avi";
	int counter = 0;
	std::string video_path = base + ext;

	while (std::filesystem::exists(video_path)) {
		video_path = base + "_" + std::to_string(++counter) + ext;
	}

	video_writer.open(video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 60, frame_size,true);
	if (!video_writer.isOpened()) {
		std::cerr << "无法创建视频文件！\n";
	}

	float distance = 0.0f;
	float angle_deg = 0.0f;

    pub_kfs_ = this->create_publisher<base_interfaces::msg::CameraKfs>("/kfs_distance", 10);

	publisher_1 = this->create_publisher<base_interfaces::msg::Align>("d455/align", 10);
    // RCLCPP_INFO(this->get_logger(), "D455自定义消息发布者已初始化");
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());

	running_.store(true);

    cout << "Create D455Node" << endl;
}

void D455Node::publishdist(float dist)
{
    base_interfaces::msg::CameraKfs msg;
    if(abs(dist) <= 200)
    {
        msg.kfs_dist = dist;
        pub_kfs_->publish(msg);
    }
}

void D455Node::task_callback(const base_interfaces::msg::GridStart::SharedPtr msg)
{
    if(msg->start != -1) 
    { 
        std::unique_lock<std::mutex> lock(start_mutex_);
        start_test_ = msg->start;
    } 
}

void D455Node::loadYamlConfig(const std::string& path) 
{
    try {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        
        if (!fs.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "无法打开配置文件: %s", path.c_str());
            return;
        }

        // --- 读取参数 ---
        fs["Field"] >> field_;
        fs["Field"] >> grid_params_.field;
        fs["Kfs_path"] >> kfs_path_;
        
        fs["ShowParams"] >> show_params_;
        fs["Showtest"] >> show_test_;

        fs["D455Node"]["Display"] >> default_display_;
        fs["D455Node"]["DispersionThreshold"] >> dispersion_threshold_;
        fs["D455Node"]["Pcl_params"]["white_gray_min"] >> pcl_params_.white_gray_min;
        fs["D455Node"]["Pcl_params"]["white_gray_max"] >> pcl_params_.white_gray_max;
        fs["D455Node"]["Pcl_params"]["white_hsv_min"] >> pcl_params_.white_hsv_min;
        fs["D455Node"]["Pcl_params"]["white_hsv_max"] >> pcl_params_.white_hsv_max;

        fs["D455Node"]["ActD_params"]["bag_path"] >> act_d455_params_.bag_path;

        fs.release();

        // --- 参数打印输出 ---
        if (show_params_) {
            RCLCPP_INFO(this->get_logger(), "----------- D455Node 参数加载成功 -----------");
            RCLCPP_INFO(this->get_logger(), "场馆设置 (Field): %d", field_);
            RCLCPP_INFO(this->get_logger(), "显示开关 (Display): %s", default_display_ ? "ON" : "OFF");
            
            RCLCPP_INFO(this->get_logger(), "Pcl_params -> Gray 范围: [%d, %d]", 
                        pcl_params_.white_gray_min, pcl_params_.white_gray_max);
            
            // cv::Scalar 打印 H, S, V 分量
            RCLCPP_INFO(this->get_logger(), "Pcl_params -> HSV Min: [%.1f, %.1f, %.1f]", 
                        pcl_params_.white_hsv_min[0], pcl_params_.white_hsv_min[1], pcl_params_.white_hsv_min[2]);
            
            RCLCPP_INFO(this->get_logger(), "Pcl_params -> HSV Max: [%.1f, %.1f, %.1f]", 
                        pcl_params_.white_hsv_max[0], pcl_params_.white_hsv_max[1], pcl_params_.white_hsv_max[2]);

            
            RCLCPP_INFO(this->get_logger(), "--------------------------------------------");
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "----------- D455Node 参数不显示 -----------");
        }

    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "D455Node 解析 YAML 时出错: %s", e.what());
    }
}

void D455Node::start()
{
    cout << "Start D455Node" << endl;
    if(d455_log_.is_open()) d455_log_ << " Start D455Node "  << endl;
    running_.store(true);
    has_frame_ = false;
    retain_.store(true);
    //成员函数指针必须绑定对象
	capture_thread_ = std::thread(&D455Node::captureLoop, this);
    process_part2_thread_ = std::thread(&D455Node::process_part2_Loop, this);
    process_unet_grid_thread_ = std::thread(&D455Node::process_unet_Loop, this);
	process_yolo_kfs_thread_ = std::thread(&D455Node::process_yolo_Loop, this);

    process_state_thread_ = std::thread(&D455Node::process_state_Loop, this);
    display_thread_ = std::thread(&D455Node::displayLoop, this);

    if(d455_log_.is_open()) d455_log_ << " Start D455Node 完毕"  << endl;
 
	// align_thread_ = std::thread(&D455Node::alignLoop,this); 
    // pick_thread_ = std::thread(&D455Node::pickLoop,this); 

    // 2. 给线程命名 (千万注意：名字不能超过15个字符！)
    // pthread_setname_np(capture_thread_.native_handle(), "capture");
    // pthread_setname_np(process_yolo_kfs_thread_.native_handle(), "yolo");
    // pthread_setname_np(process_yolo_wall_thread_.native_handle(), "wall");
    // pthread_setname_np(process_state_thread_.native_handle(), "state_proc");
    // pthread_setname_np(display_thread_.native_handle(), "display");
    // pthread_setname_np(align_thread_.native_handle(), "align");
    // pthread_setname_np(pick_thread_.native_handle(), "pick");
}

D455Node::~D455Node()
{
	stop();
	if (video_writer.isOpened()) {
        video_writer.release(); 
    }
    if (d455_log_.is_open()) {
        d455_log_.close();
    }
}

D455Node::Dataframe::Dataframe(const Dataframe& other)
	: src(other.src.clone()), depth(other.depth.clone())
{}

D455Node::Dataframe& D455Node::Dataframe::operator=(const Dataframe& other)
{
	if (this != &other)
	{
		src = other.src.clone();
		depth = other.depth.clone();
	}
	return *this;
}

D455Node::Dataframe::Dataframe(Dataframe&& other)
	: src(std::move(other.src)), depth(std::move(other.depth))
{}

D455Node::Dataframe& D455Node::Dataframe::operator=(Dataframe&& other)
{
	if (this != &other)
	{
		src = std::move(other.src);
		depth = std::move(other.depth);
	}
	return *this;
}

void D455Node::stop()
{
    cout << "Stop D455" << endl;
    if(d455_log_.is_open()) d455_log_ << " Stop D455 "  << endl;
    running_ = false;
    has_frame_part2_ = true;
    has_frame_ = true;
    detect_obj_wall_ = true;
    has_obj_kfs_ = true;
    has_frame_unet_ = true;
    is_identify_KFS_ = false;

    try {
        d455_->pipe.stop();
        cout << "[ActD455] pipeline stopped successfully." << endl;
    } catch (...) {cout << "d455关闭失败" << endl;}

    get_frame_.notify_all();
    get_wall_.notify_all();
    get_frame_unet_.notify_all();
    get_frame_part2_.notify_all();
    get_wall_or_grid_cloud_.notify_all();
    final_state_.notify_all();

    if (capture_thread_.joinable()) capture_thread_.join();

    if (process_part2_thread_.joinable()) process_part2_thread_.join();

    if (process_unet_grid_thread_.joinable()) process_unet_grid_thread_.join();

    if (process_yolo_kfs_thread_.joinable()) process_yolo_kfs_thread_.join(); 

    if (process_state_thread_.joinable()) process_state_thread_.join(); 

    if (display_thread_.joinable()) display_thread_.join();

    if (pick_thread_.joinable()) pick_thread_.join();

    if (align_thread_.joinable()) align_thread_.join();

    cv::destroyAllWindows();
    if(d455_log_.is_open()) d455_log_ << "d455 stop 结束" << endl;
}

// 新增：里程计回调
void D455Node::odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
	// 读取 /base_link_to_init 的位姿，单位转换与原逻辑保持一致
	const auto &p = msg->pose.pose.position;
	const auto &o = msg->pose.pose.orientation;

    // float x1;
	lidar_x_ = static_cast<float>(p.x * 1000.0);  // m -> mm
	lidar_y_ = static_cast<float>(p.y * 1000.0);
	lidar_z_ = static_cast<float>(p.z * 1000.0);
    lidar_x_pick_= static_cast<float>(p.x * 1000.0);
    lidar_y_pick_= static_cast<float>(p.y * 1000.0);

	tf2::Quaternion q(o.x, o.y, o.z, o.w);
	double roll, pitch, yaw;
	tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
	lidar_roll_  = static_cast<float>(roll  / M_PI * 180.0);
	lidar_pitch_ = static_cast<float>(pitch / M_PI * 180.0);
	lidar_yaw_   = static_cast<float>(yaw   / M_PI * 180.0);
    pick_->getCoordinate(lidar_x_pick_, lidar_y_pick_, -yaw);
    Eigen::Vector3d p_car_w(lidar_x_,lidar_y_,lidar_z_);
    R_wb_ = eulerZYX(lidar_yaw_,lidar_pitch_,lidar_roll_);

    Eigen::Vector3d t_car_cam(-11.0, 199+57-8, 431+80);
    
    {
        std::lock_guard<std::mutex> lock(mut_pos_);
        cam_pos_ = p_car_w + R_wb_ * t_car_cam;
        R_bw_ = R_wb_.transpose();
    }

    // cout << "cam_pos_:\n" << cam_pos_ << "\nR_bw_\n" << R_bw_ << std::endl;

	// gstate_->getlidar(lidar_x_, lidar_y_, lidar_z_, lidar_yaw_, lidar_roll_, lidar_pitch_);
}

float D455Node::rectangle_depth(const cv::Rect &roiRect , const cv::Mat &depthimg, std::ostringstream& oss, int& row, int& col)
{
	pPointCloud srcCloud = std::make_shared<PointCloud>();
    pPointCloud src_Cloud = std::make_shared<PointCloud>();
    
	srcCloud = d455_->PointCloudGenerateRect(roiRect,depthimg,2,1);
	
    // cout << " (row:" << row <<  ",col:" << col << ") " << "【原始】点云数：" << srcCloud->points.size() << endl;

	if (srcCloud->points.size() < 500) 
	{
        if(d455_log_.is_open()) oss << " (row:" << row <<  ",col:" << col << ") " << "【原始】点云数：" << srcCloud->points.size() << " 小于 " << 500 << " ";
		return -1.f;
	}

    // ================== 抽样计算点云离散程度 ================== 0.005ms
    // Eigen::Vector4f centroid;
    // pcl::compute3DCentroid(*srcCloud, centroid);
    // // 抽样计算方差
    // float variance = 0.0f;
    // int sample_count = 0;
    // size_t step = 10;
    // for (size_t i = 0; i < srcCloud->points.size(); i += step)
    // {
    //     const auto& point = srcCloud->points[i];
    //     if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
    //     float dx = point.x - centroid[0];
    //     float dy = point.y - centroid[1];
    //     float dz = point.z - centroid[2];
    //     variance += (dx * dx + dy * dy + dz * dz);
    //     sample_count++;
    // }
    // if (sample_count > 0) variance /= static_cast<float>(sample_count);
    // float dispersion_std_dev = std::sqrt(variance);
    // // cout << " (row:" << row <<  ",col:" << col << ") " << " 点云标准差：" << dispersion_std_dev << endl;
    // if (dispersion_std_dev > dispersion_threshold_)
    // {
    //     if(d455_log_.is_open()) oss << " (row:" << row <<  ",col:" << col << ") " << "【原始】点云标准差：" << dispersion_std_dev << " 小于 " << dispersion_threshold_ << " ";
    //     return -1.f;
    // }
    // ==================

	float distance = pcl_->PlaneSegmentation(srcCloud, src_Cloud, 10);

    // cout << " (row:" << row <<  ",col:" << col << ") " << "【处理后】点云数：" << src_Cloud->points.size() << " 深度：" << distance << endl;
    
	*kfs_show_cloud += *srcCloud;

    return distance;   
}

float D455Node::computeDepthByMode(int current_mode, float lidar_x, pPointCloud& wall_cloud_out)
{
    float depth_value = 0.0f;

    // RCLCPP_INFO(this->get_logger(), "进入模式：%d", current_mode);

    switch (current_mode)
    {
        case 0: 
        {
            // std::lock_guard<std::mutex> lock(disp_trt_mutex_);
            
            depth_value = trt_seg_->Getdep();
            
            break;
        }
        case 1: // wallCloud
        {
            pPointCloud wall_cloud = d455_->GetWallCloud();
            float distance_wall = pcl_->PlaneSegmentation(wall_cloud, wall_cloud_out, 10); // 0.8ms
            depth_value = distance_wall + 10.0f;
            // wall_cloud_viewer_.add_cloud(wall_cloud_out, "wall");
            // pcl_->add_points(wall_cloud, "wall_raw");
            // pcl_->add_points(wall_cloud_out, "wall_out"); // 3ms 5000点
            break;
        } 
        default:
            break;
    }
    return depth_value;
}

// 传入你当前需要匹配的时间戳，返回离它最近的 quads 向量
// 修改返回值类型为 QuadsData，以便同时返回 quads 向量和对应的时间戳
D455Node::QuadsData D455Node::getNearestQuads(std::chrono::steady_clock::time_point target_time)
{
    std::lock_guard<std::mutex> lock(buffer_mutex_); 

    if (quads_buffer_.empty()) {
        return QuadsData(); // 返回一个空的 QuadsData 结构体
    }

    size_t best_index = 0;
    auto min_diff = std::abs(std::chrono::duration_cast<std::chrono::microseconds>(
        quads_buffer_[0].timestamp - target_time).count());

    for (size_t i = 1; i < quads_buffer_.size(); ++i) {
        auto current_diff = std::abs(std::chrono::duration_cast<std::chrono::microseconds>(
            quads_buffer_[i].timestamp - target_time).count());
        
        if (current_diff < min_diff) {
            min_diff = current_diff;
            best_index = i;
        }
    }

    auto max_allow_diff = std::chrono::milliseconds(200);
    if (std::chrono::microseconds(min_diff) > max_allow_diff) { 
        return QuadsData(); // 超时则返回空结构体
    }

    // 成功匹配，直接返回整个结构体对象
    return quads_buffer_[best_index];
}

//void D455Node::d455_callback(const base_interfaces::msg::AlignStart::ConstSharedPtr msg)
//{
//  this->kfs_mode = msg->;
//  this_>gan_mode = msg->;
//}
void D455Node::captureLoop()
{
	using namespace std::chrono_literals;

    if(!running_) 
    {
        cout << "d455无法run" << endl;
        if(d455_log_.is_open()) d455_log_ << " d455无法run "  << endl;
    }
    if(running_.load() && rclcpp::ok())
    {
        if(d455_log_.is_open()) d455_log_ << " start captureLoop "  << endl;
    }

	while (running_.load() && rclcpp::ok())
	{
		while (running_.load() && rclcpp::ok() && d455_->Init() == false)
		{
			RCLCPP_WARN(this->get_logger(), "ActD455 Init failed, retrying...");
			std::this_thread::sleep_for(300ms);
		}
		if (!running_.load() || !rclcpp::ok())
		{
			break;
		}
		while (running_.load() && rclcpp::ok())
		{
			auto now = buffer::Timestamp::now();
			if (!d455_->Update())
			{
				RCLCPP_WARN(this->get_logger(), "ActD455 Update failed, re-initializing...");
				d455_->release();
				std::this_thread::sleep_for(300ms);
				break;  // 回到外层重新 Init
			}    
            cv::Mat frame;
            frame = d455_->GetSrcImage().clone();

            // if(is_identify_KFS_)
            // {
            //     cout << "进入一次 is_identify_KFS_" << endl;
            //     is_identify_KFS_ = false;
            // }

            // if(is_identify_KFS_)
            // {
            //     if(d455_log_.is_open()) d455_log_ << "看手机" << endl;
            //     if(dealImg_.Deal(frame)) 
            //     {
            //         cv::waitKey(1);
            //     }
            //     else
            //     {
            //         is_identify_KFS_ = false;
            //         std::this_thread::sleep_for(std::chrono::milliseconds(10));
            //         dealImg_.cleanup();
            //     }
            // }
            
            if (!frame.empty()) video_writer.write(frame);
            
            // video_writer.write(dataframe_pick_.src);
            if(start_test_ == 1) 
            {
                std::unique_lock<std::mutex> lock(frame_mutex);
                depth_part2_ = d455_ ->GetDepthImage().clone();
                src_part2_ = d455_->GetSrcImage().clone();
                // src_part2_ = cv::imread("grid/frame_02224.jpg");
                has_frame_part2_ = true;
            } 
            get_frame_part2_.notify_all();

 
            {
                std::unique_lock<std::mutex> lock(frame_mutex);
                depth_ = d455_ ->GetDepthImage().clone();
                src_ = d455_->GetSrcImage().clone();
                // src_ = cv::imread("grid/frame_02272.jpg");
                has_frame_ = true;
            } 
            get_frame_.notify_all();

            {
                std::unique_lock<std::mutex> lock(frame_unet_mutex);
                src_unet_ = d455_->GetSrcImage().clone();
                src_unet_ = cv::imread("grid/frame_00937.jpg");
                depth_unet_ = d455_ ->GetDepthImage().clone();
                has_frame_unet_ = true;
            }
            get_frame_unet_.notify_all();

            // if(save_video_)
            // {
            //     if(running_)
            //     {
            //         video_saver_.start(frame);
            //         video_saver_.write(frame);
            //     }
            //     else if(!running_)
            //     {
            //         cout << "结束d455录像" << endl;
            //         video_saver_.stop();                    
            //     }
            // }

			auto finish = buffer::Timestamp::now();
			auto diff = buffer::Timestamp::diff(finish, now);
			// std::cout << "采集循环时间间隔: " << diff/1000.0 << " ms" << endl;
            // if(d455_log_.is_open()) d455_log_ << "采集循环时间间隔: " << diff/1000.0  << endl;
		}
	}
    cout << -1 << endl;
}

void D455Node::process_part2_Loop()
{
    while (running_.load() && rclcpp::ok())
    {
        auto now_part2 = buffer::Timestamp::now();

        cv::Mat img, depth;
        {
            std::unique_lock<std::mutex> lock(frame_mutex);
            get_frame_part2_.wait(lock,[&]{
                return (has_frame_part2_ || !rclcpp::ok() || !running_);
            });
            if(!rclcpp::ok() || !running_) break;
            img = src_part2_;
            depth = depth_part2_;
            has_frame_part2_ = false;
        }

        // img = cv::imread(kfs_path_);

        if(img.empty())
        {
            cout << "传入图像为空" << endl;
            continue;
        }
        if(trt_yolo_part2_ == nullptr)
        {
            cout << "trt_yolo_part2_指针为空" << endl;
            continue;
        }
        // cv::imshow("img", img);
        // cv::waitKey(1);
        auto finish = buffer::Timestamp::now();
        auto diff = buffer::Timestamp::diff(finish, now_part2);
        // std::cout << "part2_更新图像时间间隔: " << diff/1000.0 << " ms" << endl;

        if(start_test_ == 1)  
        {

            trt_yolo_part2_->detect(img);

            // cout << "trt_yolo_part2_->objs.size(): " << trt_yolo_part2_->objs.size() << endl;
            if(trt_yolo_part2_->objs.size() == 0) 
            {
                // cout << "没有kfs" << endl;
                continue;
            }

            cv::Rect best_rect;
            float max_area = 0.0f;
            for (const auto& obj : trt_yolo_part2_->objs)
            {
                if (obj.label == 0)
                {
                    float area = obj.rect.width * obj.rect.height;
                    if (area > max_area)
                    {
                        max_area = area;
                        // best_rect = cv::Rect(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
                    }
                }
            }

            if (max_area == 0.0f) continue;

            best_rect = trt_yolo_part2_->objs[0].rect;

            cv::Mat gray = kfs_->Kfs(img, best_rect, depth);
            int whitePixelCount = cv::countNonZero(gray);
            if(whitePixelCount <= 100)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                
                continue;            
            }

            kfs_->boudary(gray, best_rect);

            cv::Mat vis_img = cv::Mat::zeros(img.size(), CV_8UC3);
            // cv::Mat white_mask(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
            // white_mask.copyTo(vis_img, gray);
            cv::rectangle(vis_img, best_rect, cv::Scalar(0, 0, 255), 2);
            const auto& pts_left = kfs_->getPointsLeft();
            const auto& pts_right = kfs_->getPointsRight();
            Scalar green_color(0, 255, 0);
            Scalar red_color(0, 0, 255);

            // cout << "pts_left.size(): " << pts_left.size() << " pts_right.size(): " << pts_right.size() << endl;

            float dist{0.0};
            double mean_left{0.0};
            double mean_right{0.0};
            if (!pts_left.empty()) 
            {
 
                double sum_left = 0;
                for (const auto& pt : pts_left) sum_left += pt.x;
                mean_left = sum_left / pts_left.size();
            }
            if (!pts_right.empty()) 
            {
                double sum_right = 0;
                for (const auto& pt : pts_right) sum_right += pt.x;
                mean_right = sum_right / pts_right.size();
            }
            // cout << "mean_left: " << mean_left << " mean_right: " << mean_right << endl;

            if(mean_left != 0 && mean_right != 0) 
            {
                dist = 350 / (mean_right - mean_left) * ((mean_right + mean_left) / 2 - 317.011) - 11.0;
                if(dist >= 200) dist = 999;
                if(dist <= -200) dist = -999;
            }
            else if (mean_left == 0) dist = 999;
            else if (mean_right == 0) dist = -999;

            if(kfs_->kfs_log_.is_open()) kfs_->kfs_log_  << dist << endl;

            publishdist(dist);

            // ========== 保存图像并标注 dist 值 ==========
            auto now1 = buffer::Timestamp::now();

// ========== 保存图像并标注 dist 值 ==========
if (!img.empty())  // 仅当有效时才保存
{
    // 复制原图，避免修改显示用的图像
    cv::Mat img_with_text = img.clone();
    cv::Mat gray_text = gray.clone();

    for (const auto& pt : pts_left) cv::circle(img_with_text, pt, 1, green_color, -1);
    for (const auto& pt : pts_right) cv::circle(img_with_text, pt, 1, green_color, -1);

    cv::rectangle(img_with_text, best_rect, cv::Scalar(255, 255, 255), 1);

    // 格式化 dist 字符串，保留适当精度
    std::string dist_text = "dist: " + std::to_string(dist);
    // 左上角第一行：dist 值 (y=30)
    cv::putText(img_with_text, dist_text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    // 新增：显示左右点数 (y=60)
    std::string count_text = "Left count: " + std::to_string(pts_left.size()) +
                             ", Right count: " + std::to_string(pts_right.size());
    cv::putText(img_with_text, count_text, cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    // 新增：显示左右均值 (y=90)
    std::string mean_text = "Mean left: " + std::to_string(mean_left) +
                            ", Mean right: " + std::to_string(mean_right);
    cv::putText(img_with_text, mean_text, cv::Point(10, 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    // 保存路径（确保目录存在）
    std::string save_dir = "/home/lx/code/aaa/2.6_oneYolo_CHANLLENGE_960_720_git/kfs/";
    // 使用时间戳或计数器生成唯一文件名
    static int save_counter = 0;
    std::string filename = save_dir + "kfs_" + std::to_string(save_counter++) + ".jpg";
    std::string grayname = save_dir + "kfs_" + std::to_string(save_counter++) + ".jpg";
    cv::imwrite(filename, img_with_text);
    // cv::imwrite(grayname, gray_text);
    // 可选：将保存的文件名写入日志
    // if(kfs_->kfs_log_.is_open()) kfs_->kfs_log_ << "Saved: " << filename << std::endl;
}
// ==========================================

//             for (const auto& pt : pts_left) cv::circle(vis_img, pt, 1, green_color, -1);
//             for (const auto& pt : pts_right) cv::circle(vis_img, pt, 1, red_color, -1);
            // cv::Mat res = trt_yolo_part2_->res;
            // cv::namedWindow("gray", cv::WINDOW_NORMAL);
            // cv::imshow("gray", gray);
            // cv::namedWindow("res", cv::WINDOW_NORMAL);
            // cv::imshow("res", res);
            // cv::imshow("KFS_Visualization", vis_img);
            // cv::waitKey(1);

            auto finish1 = buffer::Timestamp::now();
            auto diff1 = buffer::Timestamp::diff(finish1, now1);
            // std::cout << "part2_记录图像时间间隔: " << diff1/1000.0 << " ms" << endl;
        }
    }
}


void D455Node::process_unet_Loop()
{
    while (running_.load() && rclcpp::ok())
    {
        cv::Mat img;
        cv::Mat dep;
        {
            std::unique_lock<std::mutex> lock(frame_unet_mutex);
            get_frame_unet_.wait(lock,[&]{
                return (has_frame_unet_ || !rclcpp::ok() || !running_);
            });
            if(!rclcpp::ok() || !running_) break;
            img = src_unet_;
            dep = depth_unet_; 
            has_frame_unet_ = false;
        }
        if(img.empty() || dep.empty()) continue;

        int start_test{0};
        {
            std::unique_lock<std::mutex> lock(start_mutex_);
            start_test = start_test_;
        }
        if(start_test != 2) continue;

        trt_seg_->detect(img);//--12ms


        float pos_z{910.0};
        { 
            std::unique_lock<std::mutex> lock(mut_pos_);
            // pos_z = lidar_z_ + 400 + 431 + 80;
            pos_z = 0 + 400 + 431 + 80;
            pos_z += 0;
            // if(d455_log_.is_open()) d455_log_  << " pos_z: " << pos_z << endl;
        } 

        std::vector<Unet::Quad2D> quads = trt_seg_->getgridquads(img, pos_z); //--4ms
        
        if(quads.empty()){
            // cout << "没有匹配出九宫格" << endl;
            continue;
        }
        distances_ = fitPixelScale(quads);

        float nine_square_depth_value{0.f};
        std::ostringstream oss;
        nine_square_depth_value = trt_seg_->Getdep();

        gstate_ -> publish_dist(distances_, nine_square_depth_value, pos_z);

        if(d455_log_.is_open()) d455_log_ << std::fixed << std::setprecision(1) << " yolo九宫格深度: " << nine_square_depth_value << " 车x: " << nine_square_depth_value + (199+57-8) + 150;



        yolo_detector_grid_->detect(img);
        if(yolo_detector_grid_->objs.empty()) 
        {
            std::this_thread::sleep_for(10ms);
            continue;
        }
        
        std::vector<trt_yolo::det::Object> objs_kfs_red_r1;
        std::vector<trt_yolo::det::Object> objs_kfs_red_r2;
        std::vector<trt_yolo::det::Object> objs_kfs_blue_r1;
        std::vector<trt_yolo::det::Object> objs_kfs_blue_r2;
        std::vector<trt_yolo::det::Object> objs_gw_gan1;
        std::vector<trt_yolo::det::Object> objs_gy_gan2;
        std::vector<trt_yolo::det::Object> objs_wall;

        for (const auto& obj : yolo_detector_grid_->objs)
        {
            int lab = obj.label;
            switch (lab)
            {
                case 0: objs_kfs_red_r1.push_back(obj); break;
                case 1: objs_kfs_red_r2.push_back(obj); break;
                case 2: objs_kfs_blue_r1.push_back(obj); break;
                case 3: objs_kfs_blue_r2.push_back(obj); break;
                case 4: objs_gw_gan1.push_back(obj); break;
                case 5: objs_gy_gan2.push_back(obj); break;
                case 6: objs_wall.push_back(obj); break;        
            default:
                break;
            }
        }

        {
            std::unique_lock<std::mutex> lock(compute_state_);
            // if(retain_ )
            // {
                objs_kfs_red_state_.clear();
                objs_kfs_red_state_.reserve(objs_kfs_red_r1.size() + objs_kfs_red_r2.size());
                objs_kfs_red_state_.insert(objs_kfs_red_state_.end(), objs_kfs_red_r1.begin(), objs_kfs_red_r1.end());
                objs_kfs_red_state_.insert(objs_kfs_red_state_.end(), objs_kfs_red_r2.begin(), objs_kfs_red_r2.end());

                objs_kfs_blue_state_.clear();
                objs_kfs_blue_state_.reserve(objs_kfs_blue_r1.size() + objs_kfs_blue_r2.size());
                objs_kfs_blue_state_.insert(objs_kfs_blue_state_.end(), objs_kfs_blue_r1.begin(), objs_kfs_blue_r1.end());
                objs_kfs_blue_state_.insert(objs_kfs_blue_state_.end(), objs_kfs_blue_r2.begin(), objs_kfs_blue_r2.end());
                
                latest_img_state_ = img;
                latest_depth_state_ = dep;
                objs_wall_ = objs_wall;
                detect_obj_wall_ = true;
                has_obj_kfs_ = true;

                nine_square_depth_value_ = nine_square_depth_value;  
                quads_ =  quads;
                has_float_ = true;
                // quads_timestamp_ = std::chrono::steady_clock::now();
            // }
        }
        // retain_ = false;  
        final_state_.notify_one();
    }
}

void D455Node::process_yolo_Loop()
{ 
    if(d455_log_.is_open()) d455_log_ << " start process_yolo_Loop "  << endl;

    while (running_.load() && rclcpp::ok())
	{ 
        auto now = buffer::Timestamp::now();

        cv::Mat img;
        cv::Mat dep;  
        cv::Mat white_mask; 
        std::vector<trt_yolo::det::Object> objs_kfs_red_r1;
        std::vector<trt_yolo::det::Object> objs_kfs_red_r2;
        std::vector<trt_yolo::det::Object> objs_kfs_blue_r1;
        std::vector<trt_yolo::det::Object> objs_kfs_blue_r2;
        std::vector<trt_yolo::det::Object> objs_gw_gan1;
        std::vector<trt_yolo::det::Object> objs_gy_gan2;
        std::vector<trt_yolo::det::Object> objs_wall;

        {
            std::unique_lock<std::mutex> lock(frame_mutex);
            get_frame_.wait(lock,[&]{
                return (has_frame_ || !rclcpp::ok() || !running_);
            }); 
            if(!rclcpp::ok() || !running_) 
            {
                break;
            }
            img = src_;  
            dep = depth_; 
            has_frame_ = false;   
        }  

        //-----
        dataframe_align_.depth = dep;
        dataframe_align_.src = img;
        if (!dataframe_align_.src.empty() && !dataframe_align_.depth.empty())
        {
            align_buffer_.write(std::move(dataframe_align_));
        }
        //-----
        dataframe_pick_.depth = dep;
        dataframe_pick_.src = img;
        if (!dataframe_pick_.src.empty() && !dataframe_pick_.depth.empty())
        {
            pick_buffer_.write(std::move(dataframe_pick_));
        }

        yolo_detector_->detect(img);
        
        if (display_) 
        {
            std::lock_guard<std::mutex> lock(disp_mutex_);
            disp_ = yolo_detector_->res;
            disp_src_ = img; 
        }
        if(yolo_detector_->objs.empty()) 
        {
            std::this_thread::sleep_for(10ms);
            continue;
        }

        for (const auto& obj : yolo_detector_->objs)
        {
            int lab = obj.label;
            switch (lab)
            {
                case 0: objs_kfs_red_r1.push_back(obj); break;
                case 1: objs_kfs_red_r2.push_back(obj); break;
                case 2: objs_kfs_blue_r1.push_back(obj); break;
                case 3: objs_kfs_blue_r2.push_back(obj); break;
                case 4: objs_gw_gan1.push_back(obj); break;
                case 5: objs_gy_gan2.push_back(obj); break;
                case 6: objs_wall.push_back(obj); break;        
            default:
                break;
            }
        }
        // if(d455_log_.is_open()) d455_log_ << "\n--【 墙数量：" << objs_wall.size() 
        //                             << " 红块r1数量：" << objs_kfs_red_r1.size() 
        //                             << " 红块r2数量：" << objs_kfs_red_r2.size() 
        //                             << " 蓝块r1数量：" << objs_kfs_blue_r1.size() 
        //                             << " 蓝块r2数量：" <<  objs_kfs_blue_r2.size() << "】--\n";

        //-----
        {
            std::unique_lock<std::mutex> lock(gan_mut_);
            objs_gw_gan1_ = objs_gw_gan1;
            objs_gy_gan2_ = objs_gy_gan2;
        }
        
        //-----给pick线程
        {
            std::unique_lock<std::mutex> lock(kfs_mut_);
            objs_kfs_red_r1_ = objs_kfs_red_r1;
            objs_kfs_red_r2_ = objs_kfs_red_r2;
            objs_kfs_blue_r1_ = objs_kfs_blue_r1;
            objs_kfs_blue_r2_ = objs_kfs_blue_r2;
        }
        

        auto finish = buffer::Timestamp::now();
        auto diff = buffer::Timestamp::diff(finish, now);
        // std::cout << "整个模型时间间隔: " << diff/1000.0 << " ms" << endl;
    } 
    cout << "process_yolo_Loop 线程退出" << endl;
}

void D455Node::process_state_Loop()
{
    if(d455_log_.is_open()) d455_log_ << " start process_state_Loop "  << endl;

    while (running_.load() && rclcpp::ok())
    {
        auto now = buffer::Timestamp::now();

        std::ostringstream oss;
        static int no;
        std::vector<trt_yolo::det::Object> objs_kfs_red_state;
        std::vector<trt_yolo::det::Object> objs_kfs_blue_state;
        std::vector<trt_yolo::det::Object> objs_kfs_state;

        cv::Mat depth;
        cv::Mat latest_img_state;
        float nine_square_depth_value{0.0};
        std::vector<Unet::Quad2D> quads;
        // std::chrono::steady_clock::time_point now_time; // 图像数据对应的时间戳

        {   // 18ms ~ 38ms
            std::unique_lock<std::mutex> lock(compute_state_);
            final_state_.wait(lock, [this]{
                return (has_obj_kfs_ && has_float_) || !rclcpp::ok() || !running_;
            });
            if(!rclcpp::ok() || !running_) 
            {
                break;
            }
            nine_square_depth_value = nine_square_depth_value_;
            objs_kfs_red_state = objs_kfs_red_state_;
            objs_kfs_blue_state = objs_kfs_blue_state_;
            depth = latest_depth_state_;
            has_obj_kfs_ = false;
            has_float_ = false;
            // retain_ = true; //防止在等 has_float_ 时 objs_kfs_ 更新
            latest_img_state = latest_img_state_;
            // now_time = quads_timestamp_; // 获取最新图像的状态时间戳
            quads = quads_;
        }

        objs_kfs_state.clear();
        objs_kfs_state.reserve(objs_kfs_red_state.size() + objs_kfs_blue_state.size());
        objs_kfs_state.insert(objs_kfs_state.end(), 
                      objs_kfs_red_state.begin(), 
                      objs_kfs_red_state.end());
        objs_kfs_state.insert(objs_kfs_state.end(), 
                      objs_kfs_blue_state.begin(), 
                      objs_kfs_blue_state.end());
        
                
        kfs_show_cloud -> clear();

        gstate_ -> sortObjectsOrder_Red(objs_kfs_state);

        // 重构处：接收返回的完整结构体，并抽离出 quads 和对应的时间戳
        // D455Node::QuadsData matched_quads_data = getNearestQuads(now_time);
        // const auto& quads = matched_quads_data.quads;

        // std::chrono::steady_clock::time_point quads_match_time = matched_quads_data.timestamp;

        if(quads.empty()) 
        {
            cout << "quads为空" << endl;
            continue;
        }
        for (const auto& quad : quads)
        {
            cv::Scalar color = quad.valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            if (quad.pts.empty()) continue;
            for (const auto& pt : quad.pts)
            {
                if(pt.x >= 0 && pt.x < latest_img_state.cols && pt.y >= 0 && pt.y < latest_img_state.rows){
                    cv::circle(latest_img_state, pt, 3, cv::Scalar(0, 255, 0), -1);
                }
            }
        }

        bool grid_look[3][3]; 
        std::fill(&grid_look[0][0], &grid_look[0][0] + 9, true);

        const int OCCLUSION_PIXEL_THRESH = 600; 
        const float DEPTH_TOLERANCE = 150.0f;   

        for (size_t i = 0; i < quads.size(); ++i)
        {
            const auto& quad = quads[i];
            if (!quad.valid || quad.pts.empty()) continue;

            int row, col;
            if(field_ == 0) {
                row = i / 3;
                col = i % 3;
            } else if(field_ == 1) {
                row = i / 3;
                col = 2 - i % 3;
            }

            std::vector<cv::Point> pts_cv;
            for (const auto& pt : quad.pts) {
                pts_cv.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
            }

            cv::Rect rect = cv::boundingRect(pts_cv);
            int x_min = std::max(0, rect.x);
            int y_min = std::max(0, rect.y);
            int x_max = std::min(depth.cols - 1, rect.x + rect.width);
            int y_max = std::min(depth.rows - 1, rect.y + rect.height);

            int occluded_pixels_count = 0;

            for (int y = y_min; y <= y_max; ++y) {
                for (int x = x_min; x <= x_max; ++x) {
                    if (cv::pointPolygonTest(pts_cv, cv::Point2f(x, y), false) >= 0) {
                        float d_val = (depth.type() == CV_32FC1) ? depth.at<float>(y, x) : static_cast<float>(depth.at<uint16_t>(y, x));
                        if (d_val > 10 && d_val < (nine_square_depth_value - DEPTH_TOLERANCE)) {
                            occluded_pixels_count++;
                        }
                    }
                }
            }

            if (occluded_pixels_count > OCCLUSION_PIXEL_THRESH) {
                grid_look[row][col] = false;
                if(d455_log_.is_open()) {
                    oss << " [格子(" << row << "," << col << ")前方检测到遮挡，跳过更新. 计数值: " << occluded_pixels_count << "] ";
                }
            }
        }

        std::vector<std::string> img_text_lines;

        bool grid_occupied[3][3] = {{false}};
        for (const auto& obj : objs_kfs_state) 
        {
            if(obj.rect.width > 630 || (obj.rect.width / obj.rect.height) > 2.5 || (obj.rect.width / obj.rect.height) < 0.4) 
            {
                continue;
            }
            Eigen::Vector2d center_px(obj.rect.x + obj.rect.width * 0.5, 
                                    obj.rect.y + obj.rect.height * 0.5);
            int matched_grid_id = -1; 
            for (int i = 0; i < 9; ++i)
            {
                if (gstate_->isPointInQuad2D(center_px, quads[i])) 
                {
                    matched_grid_id = i + 1;
                    break;
                }
            }
            if (matched_grid_id == -1)
            {
                continue;
            }
            int row, col;
            if(field_ == 0)
            {
                row = (matched_grid_id - 1) / 3;
                col = (matched_grid_id - 1) % 3;
            }
            else if(field_ == 1)
            {
                row = (matched_grid_id - 1) / 3;
                col = 2 - (matched_grid_id - 1) % 3;
            }

            if (!grid_look[row][col]) 
            {
                if(d455_log_.is_open()) {
                    oss << " 格子【" << row << "," << col << "】跳过更新";
                }
                continue;
            }

            if (grid_occupied[row][col]) continue;
            
            float kfs_depth_value = rectangle_depth(obj.rect, depth, oss, row, col);

            if(kfs_depth_value < 300 || kfs_depth_value > 4000) 
            {
                if(d455_log_.is_open()) oss << " [ERROR: " << matched_grid_id << "号kfs深度: " << kfs_depth_value << "] ";
                continue;
            }

            if (kfs_depth_value > (nine_square_depth_value + 300))
            {
                if(d455_log_.is_open()) oss << "  在【" << matched_grid_id << "】号格的kfs深度: " << std::fixed << std::setprecision(1) << kfs_depth_value <<" 未进入 九宫格【" << nine_square_depth_value << "】" ;
                grid_state_[row][col] = 0;
                // cout << "  在【" << matched_grid_id << "】号格的kfs深度: " << std::fixed << std::setprecision(1) << kfs_depth_value <<" 未进入 九宫格【" << nine_square_depth_value << "】" << endl;
                continue;
            }
            if(d455_log_.is_open()) oss << " " << matched_grid_id << " 号格的kfs" << "深度 " << std::fixed << std::setprecision(1) << kfs_depth_value;
            // cout << " " << matched_grid_id << " 号格的kfs" << "深度 " << std::fixed << std::setprecision(1) << kfs_depth_value << endl;;

            std::ostringstream text_oss;
            text_oss << "Grid " << matched_grid_id << " KFS Depth: " << std::fixed << std::setprecision(1) << kfs_depth_value;
            img_text_lines.push_back(text_oss.str());

            if (obj.label == 0 || obj.label == 1) grid_state_[row][col] = 1;
            else if (obj.label == 2 || obj.label == 3) grid_state_[row][col] = 2;

            grid_occupied[row][col] = true; 
        }


        for (size_t i = 0; i < quads.size(); ++i)
        {
            bool isCenterOutside = true; 
            const auto& quad = quads[i];
            Eigen::Vector2f center = (Eigen::Vector2f(quad.pts[0].x, quad.pts[0].y) +
                                    Eigen::Vector2f(quad.pts[1].x, quad.pts[1].y) +
                                    Eigen::Vector2f(quad.pts[2].x, quad.pts[2].y) +
                                    Eigen::Vector2f(quad.pts[3].x, quad.pts[3].y)) / 4.0f;
            if(center.x() < 0.0 || center.x() >= 640.0 || center.y() < 0.0 || center.y() >= 480.0) continue;
            if (!quad.valid) 
            {
                continue;
            }

            int row, col;
            if(field_ == 0)
            {
                row = i / 3;
                col = i % 3;
            }
            else if(field_ == 1)
            {
                row = i / 3;
                col = 2 - i % 3;
            }

            if (!grid_look[row][col]) continue;

            for (const auto& obj : objs_kfs_state)
            {
                bool isInside = ((quad.pts[0].x + quad.pts[1].x)/2 <= obj.rect.x + obj.rect.width/2 ) && 
                ((quad.pts[2].x + quad.pts[3].x)/2 >= obj.rect.x + obj.rect.width/2 ) && 
                ((quad.pts[0].y + quad.pts[3].y)/2 <= obj.rect.y + obj.rect.height/2) && 
                ((quad.pts[1].y + quad.pts[2].y)/2 >= obj.rect.y + obj.rect.height/2);
                    
                if (isInside)
                {
                    isCenterOutside = false;   
                    break;
                }
            }          
            if(isCenterOutside)
            {
                grid_state_[row][col] = 0;
            }
        }

        if(std::memcmp(grid_state_,grid_pre_state_,sizeof(int[3][3]))) 
        {
            std::memcpy(grid_pre_state_, grid_state_, sizeof(int[3][3]));
            count = 0;
        }
        else 
        {
            count++;
            if(count >= 3)
            {
                std::memcpy(gstate_->grid_state_, grid_state_, sizeof(int[3][3]));
                count = 0;
            }
        }

        for (int i = 0; i < 3; ++i)
        {
            oss << "{";
            for (int j = 0; j < 3; ++j)
            {
                oss << gstate_->grid_state_[i][j] << " ";
            }
            oss << "}";
            if (i != 2) oss << ", ";
        }

        oss << "\n------------------------------------------\n";
         
        gstate_ -> publish_state();
        
        if (d455_log_.is_open()) 
        {
            d455_log_ << oss.str() << "\n"; 
        } 

        // ==================== 图像绘制、时间戳叠加与保存逻辑 ====================
        if (!latest_img_state.empty())
        {
            // 1. 绘制矩形框
            for (const auto& obj : objs_kfs_state)
            {
                cv::Scalar box_color;
                if (obj.label == 0 || obj.label == 1) {
                    box_color = cv::Scalar(0, 0, 255); // 红
                } else if (obj.label == 2 || obj.label == 3) {
                    box_color = cv::Scalar(255, 0, 0); // 蓝
                } else {
                    box_color = cv::Scalar(0, 255, 255); 
                }
                cv::rectangle(latest_img_state, obj.rect, box_color, 2);
            }

            // 2. 将最终的 grid_state_ 矩阵信息追加到绘图文本列表中
            img_text_lines.push_back("Grid State Matrix:");
            for (int i = 0; i < 3; ++i)
            {
                std::ostringstream matrix_oss;
                matrix_oss << "  [ " << grid_state_[i][0] << " " << grid_state_[i][1] << " " << grid_state_[i][2] << " ]";
                img_text_lines.push_back(matrix_oss.str());
            }


            img_text_lines.push_back("------------------------");


            // 4. 在图像左上角用绿色逐行印下所有收集到的信息
            int start_y = 30;            
            const int line_height = 25;  
            for (const auto& line_str : img_text_lines)
            {
                cv::putText(latest_img_state, line_str, cv::Point(15, start_y), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                start_y += line_height;
            }

            // 5. 生成不重名的路径并保存文件
            static int frame_counter = 0;
            std::string save_dir = "/home/lx/code/aaa/2.6_oneYolo_CHANLLENGE_960_720_git/grid_state/";
            
            char filename[64];
            std::snprintf(filename, sizeof(filename), "frame_%05d.jpg", frame_counter++);
            std::string full_save_path = save_dir + filename;

            try {
                cv::imwrite(full_save_path, latest_img_state);
            }
            catch (const cv::Exception& e) {
                std::cerr << "imwrite failed: " << e.what() << std::endl;
            }
        }
        // ======================================================================

        auto finish = buffer::Timestamp::now();
        auto diff = buffer::Timestamp::diff(finish, now);
    } 
    if(show_test_) cout << "process_state_Loop 线程退出" << endl;
}









void D455Node::alignLoop() 
{
    align_->center_x = d455_->GetNewCenterX();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	while (running_.load() && rclcpp::ok())
 	{
        if((align_->center_x = d455_->GetNewCenterX()) == -1)
        {
            std::this_thread::sleep_for(1ms);
            continue;
        } 
        // std::cout << align_->center_x << std::endl; 		
        // std::cout << "进入循环" << std::endl;
		Dataframe df_;
        std::vector<trt_yolo::det::Object> objs_gw_gan1;
        std::vector<trt_yolo::det::Object> objs_gy_gan2;
        align_->R1_is_putting = false;

		if (!align_buffer_.read(df_)) 
		{
			std::this_thread::sleep_for(1ms);
			continue;
		}
		else 
		{
            // std::cout << "读到图df_" << std::endl;
			// auto align_msg = base_interfaces::msg::Align();
            
			if(yolo_detector_)
            {
                if (df_.src.empty() || df_.depth.empty()) 
				{
                    // std::cout << "没图" << std::endl;
                    continue;
				} 
                {
                    std::unique_lock<std::mutex> lock(gan_mut_);
                    objs_gw_gan1 = objs_gw_gan1_;
                    objs_gy_gan2 = objs_gy_gan2_;
                }
                if(objs_gw_gan1.empty() && objs_gy_gan2.empty())
                {
                    // std::cout << "俩都没有" << std::endl;
					align_->align_start = false;
                    align_->publish();
                    continue;
                }
                else
                {
                    // std::cout << "进入主要程序" << std::endl;
                    align_->align_start = true;
                    align_->R1_is_putting = false;
                    if(objs_gy_gan2.empty())
                    {
                        align_->R1_is_putting = true;
                        align_->publish();
                        continue;
                    }
					const auto& fixobj = objs_gy_gan2[0];
					align_->align_rect = fixobj.rect;
                    if(objs_gw_gan1.empty())
                    {
                        align_->should_climb = true;
                    }
                    else
                    {
                        align_->should_climb = false;
                    }

					cloud = d455_->PointCloudGenerateRect(align_->align_rect, df_.depth,2,2);
						
					if(!cloud->empty())
					{
						// cloud = pcl_->removeStatisticalOutliers(cloud);
						// cloud = pcl_->removeRadiusOutliers(cloud);
						align_->getDepth(cloud);
                    }
				
                    if(align_->outDepth() >= 400 && align_->outDepth() <=1200)
					{
                        align_->use_pca = true;
						align_->getVector(cloud);
						align_->calculateWideRodCenter();
					}
                    else
                    {
                        align_->use_pca = false;
                    }
                    align_->publish();
                }
            }
        }
    }
    cout << "alignLoop 线程退出" << endl;
}

void D455Node::pickLoop()
{
    pick_->center_x = d455_->GetNewCenterX();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    while (running_.load() && rclcpp::ok())
    {
        if((pick_->center_x = d455_->GetNewCenterX()) == -1)
        {
            std::this_thread::sleep_for(1ms);
            continue;
        }
        cv::Mat src;
        cv::Mat depth;
        Dataframe df_;
        std::vector<trt_yolo::det::Object> objs_kfs_red_r1;
        std::vector<trt_yolo::det::Object> objs_kfs_red_r2;
        std::vector<trt_yolo::det::Object> objs_kfs_blue_r1;
        std::vector<trt_yolo::det::Object> objs_kfs_blue_r2;

        if(!pick_buffer_.read(df_))
        {
            std::this_thread::sleep_for(1ms);
			continue;
        }
        else
        {
            if (yolo_detector_)
            {
                if (df_.src.empty() || df_.depth.empty()) 
                {
                    continue;
                }
                {
                    std::unique_lock<std::mutex> lock(kfs_mut_);
                    objs_kfs_red_r1 = objs_kfs_red_r1_;
                    objs_kfs_red_r2 = objs_kfs_red_r2_;
                    objs_kfs_blue_r1 = objs_kfs_blue_r1_; 
                    objs_kfs_blue_r2 = objs_kfs_blue_r2_; 
                }
                if(objs_kfs_red_r1.empty() && COLOR == 1)
                {
                    pick_->pick_start = false;
                    pick_->publish();
                    continue;
                }
                if(objs_kfs_blue_r1.empty() && COLOR == 0)
                {
                    pick_->pick_start = false;
                    pick_->publish();
                    continue;
                }
                
            
                pick_->pick_start = true;

                if(COLOR == 1)
                {
                    std::vector<cv::Rect> rects;
                    for(auto object : objs_kfs_red_r1)
                    {
                        rects.push_back(object.rect);
                    }
                    pick_->Input(df_.depth, rects, d455_);
                }
                else
                {
                    std::vector<cv::Rect> rects;
                    for(auto object : objs_kfs_blue_r1)
                    {
                        rects.push_back(object.rect);
                    }
                    pick_->Input(df_.depth, rects, d455_);
                }
                pick_->publish();
            
            }
        }
    }
    cout << "pickLoop 线程退出" << endl;
}


void D455Node::displayLoop()
{
    if(display_)
    {
        cv::namedWindow("yolo_detect", cv::WINDOW_NORMAL);
        cv::namedWindow("src", cv::WINDOW_NORMAL); 
        // cv::namedWindow("grid", cv::WINDOW_NORMAL); 
        cv::namedWindow("grid_mask", cv::WINDOW_NORMAL); 
        cv::namedWindow("merged_mask", cv::WINDOW_NORMAL); 
        cv::namedWindow("canvas", cv::WINDOW_NORMAL); 
    }

    static unsigned int save_counter = 0;

    while (running_.load() && rclcpp::ok())
    {
        cv::Mat disp, src, grid, merged_mask, canvas,mask;

        merged_mask = trt_seg_->Getmerged_mask();
        canvas = trt_seg_->Getcanvas();
        mask = trt_seg_->Getmask();

        // std::cout << "[Display] Canvas size: " << canvas.size() << ", Mask size: " << mask.size() << std::endl;
  
        
        {
            std::lock_guard<std::mutex> lock(disp_mutex_);
            if (!disp_.empty()) { disp = disp_; disp_.release(); } 
            if (!disp_src_.empty()) { src = disp_src_; disp_src_.release(); }
        }

        Eigen::Matrix3d R_bw;
        Eigen::Vector3d cam_pos;
        
        {
            std::lock_guard<std::mutex> lock(mut_pos_);
            cam_pos = cam_pos_;
            R_bw = R_bw_;
        }

        // 耗时的 imshow 放在锁外面
        if (!disp.empty()) cv::imshow("yolo_detect", disp); 
        if (!src.empty()) cv::imshow("src", src);
        // if (!grid.empty()) cv::imshow("grid", grid);
        if (!mask.empty()) cv::imshow("grid_mask", mask);
        if (!merged_mask.empty()) cv::imshow("merged_mask", merged_mask);
        if (!canvas.empty()) cv::imshow("canvas", canvas);
        

        ++save_counter;  
        // 生成4位数字后缀，例如 0001
        std::ostringstream suffix_ss;
        suffix_ss << std::setw(4) << std::setfill('0') << save_counter;
        std::string suffix = suffix_ss.str();

        std::string save_dir = "src/cv/canvas_61/";
        // 确保目录存在（需要 C++17 的 filesystem）
        std::filesystem::create_directories(save_dir);

        if (!mask.empty()) {
            std::string path = save_dir + suffix + "mask_" + ".png";
            if (!cv::imwrite(path, mask))
                std::cerr << "[ERROR] 保存 mask 失败: " << path << std::endl;
        }
        if (!merged_mask.empty()) {
            std::string path = save_dir + suffix  + "merged_mask_"  + ".png";
            if (!cv::imwrite(path, merged_mask))
                std::cerr << "[ERROR] 保存 merged_mask 失败: " << path << std::endl;
        }
        if (!canvas.empty()) {
            std::string path = save_dir + suffix  + "canvas_" + ".png";
            if (!cv::imwrite(path, canvas))
                std::cerr << "[ERROR] 保存 canvas 失败: " << path << std::endl;
        }
        std::string pose_path = save_dir + suffix + "pose_" + ".txt";
        std::ofstream pose_file(pose_path);
        if (pose_file.is_open()) {
            pose_file << "cam_pos: " << cam_pos.x() << " " << cam_pos.y() << " " << cam_pos.z() << "\n";
            pose_file << "R_bw:\n";
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    pose_file << R_bw(i, j) << (j == 2 ? "\n" : " ");
                }
            }
            pose_file.close();
        } else {
            std::cerr << "[ERROR] 无法打开位姿文件: " << pose_path << std::endl;
        }

        cv::waitKey(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    cv::destroyAllWindows();
    cout << "displayLoop 线程退出" << endl;
}