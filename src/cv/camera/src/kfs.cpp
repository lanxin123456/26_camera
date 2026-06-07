#include "camera_forward/kfs.hpp"
/******************************** *****************/
#define CAM_PLACE 2 //1在上，2在下

KFS::KFS():
    Node{"deal"}, 
    lenth_{0},
    iterations_{200},
    min_inliers_{20},
    dist_thresh_{3.0},
    dist_{9999},
    center_x_{346.16135},
    visualize_{true},
    config_(1, {"red"}, {{0, 0, 255}},cv::Size(640, 640), 100, 0.6f, 0.65f)
{
    #ifdef IF_CAMERA
    ACam.Init();
    #endif
    pub_kfs_ = this->create_publisher<base_interfaces::msg::CameraKfs>("/kfs_distance", 10);
	odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/map_to_base_link", rclcpp::SensorDataQoS(),
            std::bind(&KFS::odom_callback, this, std::placeholders::_1));

    yolo_detector_  = new trt_yolo::YOLOv8("/home/lx/runs/detect/train17/weights/best.engine",config_);
    yolo_detector_->make_pipe(true);
    kfs_log_.open("camera_forward.csv", std::ios::trunc);
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);
    if(kfs_log_.is_open()) kfs_log_ << "\n\n\n\n\n"
                                    << "时间, " 
                                    << put_time(localTime, "%Y-%m-%d %H:%M:%S") 
                                    << " ===================================================================================================================================" 
                                    << endl;    
}
void KFS::start()
{
    cout << "Start KFS" << endl;
    running_.store(true);
    //成员函数指针必须绑定对象
    th_camera_ = std::thread(&KFS::camera_thread, this);
    th_process_ = std::thread(&KFS::process_thread, this);
}

KFS::~KFS()
{
    stop();
}

void KFS::stop()
{
    running_ = false;
    cv_frame_.notify_all();
    
    if(th_camera_.joinable()) th_camera_.join();
    if(th_process_.joinable()) th_process_.join();
    if (visualize_) cv::destroyAllWindows();
    cout << "Stop KFS" << endl;
}

// 新增：里程计回调
void KFS::odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
	// 读取 /base_link_to_init 的位姿，单位转换与原逻辑保持一致
	const auto &p = msg->pose.pose.position;
	const auto &o = msg->pose.pose.orientation;

	lidar_x_ = static_cast<float>(p.x * 1000.0);  // m -> mm
	lidar_y_ = static_cast<float>(p.y * 1000.0);
	lidar_z_ = static_cast<float>(p.z * 1000.0);

	tf2::Quaternion q(o.x, o.y, o.z, o.w);
	double roll, pitch, yaw;
	tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
	lidar_roll_  = static_cast<float>(roll  / M_PI * 180.0);
	lidar_pitch_ = static_cast<float>(pitch / M_PI * 180.0);
	float lidar_yaw  = static_cast<float>(yaw   / M_PI * 180.0);
    {
        std::unique_lock<std::mutex> lock(global_yaw_);
        lidar_yaw_ = lidar_yaw;
    }
}

void KFS::camera_thread()
{
    #ifdef VEDIO
    cv::VideoCapture cap("/home/lx/yolo_biao/camera_for_2026-04-13_16-11-49.avi"); 
    if(!cap.isOpened())
    {
        std::cerr << "[ERROR] Cannot open video file\n";
        rclcpp::shutdown();
        return;
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 120.0;
    auto frame_period = std::chrono::microseconds((int)(1e6 / fps));
    
    // --- 状态变量 ---
    bool is_paused = false;
    cv::Mat cached_frame;
    #endif
    
    while(running_.load() && rclcpp::ok())
    {
        auto up = std::chrono::steady_clock::now();
        cv::Mat frame, frame_video;
        
        #ifdef VEDIO
        // --- 从 process_thread 获取键盘指令 ---
        // 使用 exchange 获取按键后立即重置为 255，避免重复触发
        int key = shared_key_.exchange(255);
        bool frame_stepped = false; 

        if (key == ' ') { 
            is_paused = !is_paused;
            std::cout << (is_paused ? ">>> [INFO] 视频已暂停 (按 A/D 逐帧步进)" : ">>> [INFO] 视频恢复读取") << std::endl;
        }

        if (is_paused) 
        {
            if (key == 'a' || key == 'A') {
                double current_pos = cap.get(cv::CAP_PROP_POS_FRAMES);
                double target_pos = std::max(0.0, current_pos - 2.0);
                cap.set(cv::CAP_PROP_POS_FRAMES, target_pos);
                cap >> frame;
                if (frame.empty()) { 
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                    cap >> frame;
                }
                if (!frame.empty()) {
                    cached_frame = frame.clone();
                    frame_stepped = true;
                    std::cout << ">>> [INFO] 上一帧 (Pos: " << cap.get(cv::CAP_PROP_POS_FRAMES) - 1 << ")" << std::endl;
                }
            } 
            else if (key == 'd' || key == 'D') {
                cap >> frame;
                if (frame.empty()) { 
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                    cap >> frame;
                }
                if (!frame.empty()) {
                    cached_frame = frame.clone();
                    frame_stepped = true;
                    std::cout << ">>> [INFO] 下一帧 (Pos: " << cap.get(cv::CAP_PROP_POS_FRAMES) - 1 << ")" << std::endl;
                }
            }
        }

        if (is_paused) {
            // 如果没有发生A/D步进，则持续克隆缓存帧往下发送
            if (!frame_stepped && !cached_frame.empty()) {
                frame = cached_frame.clone();
            }
        } else {
            cap >> frame;
            if(frame.empty())
            {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                continue;
            }
            cached_frame = frame.clone(); 
        }

        #ifdef SAVE_VEDIO
        frame_video = frame.clone();
        video_saver_.start(frame_video);
        video_saver_.write(frame_video);
        #endif
        
        std::this_thread::sleep_for(frame_period);
        
        {
            std::unique_lock<std::mutex> lock(global_mtx_);
            // 这里将 frame 发走，process_thread 就会持续收到当前暂停/步进的帧并进行处理
            global_frame_ = std::move(frame);
            has_frame_ = true;
        }
        cv_frame_.notify_one();
        
        #else // 真实相机模式保持不变
            frame = ACam.Update();
            if(frame.empty()) 
            {
                std::cout << "更新的帧为空！" << std::endl;
                continue;
            }
            #ifdef SAVE_VEDIO
            frame_video = frame.clone();
            video_saver_.start(frame_video);
            video_saver_.write(frame_video);
            #endif
            {
                std::unique_lock<std::mutex> lock(global_mtx_);
                global_frame_ = std::move(frame);
                has_frame_ = true;
            }
            cv_frame_.notify_one();
        #endif
        
        auto up_end = std::chrono::steady_clock::now();
    }
}

void KFS::boudary(const Mat& temp)
{
    for(int i = 0; i < temp.rows; i++)
    {
        int left_x = -1;
        int right_x = -1;
        for(int j = 0; j < temp.cols; j++)
        {
            if(temp.at<uchar>(i,j) >= 200)
            {
                left_x = j;
                break;
            }
        }
        for(int j = temp.cols - 1; j >= 0; j--)
        {
            if(temp.at<uchar>(i,j) >= 200)
            {
                right_x = j;
                break;
            }
        }
        if(left_x!= -1 && right_x!= -1)
        {
            points_left.emplace_back(left_x, i);
            points_right.emplace_back(right_x, i);
        }
    }
}

void KFS::samey(vector<Point>& points_left, vector<Point>& points_right)
{
    vector<cv::Point> filtered_left;
    vector<cv::Point> filtered_right;

    // cout << "samey函数传参里 左边点数：" << points_left.size() << " 右边点数：" << points_right.size() << endl;
    
    std::unordered_set<int> right_y_values; //哈希集合 底层是 哈希表
    for(const auto& p : points_right)
    {
        right_y_values.insert(p.y); // 记录右边界点的y值
    }
    for(const auto& p : points_left)
    {
        if(right_y_values.count(p.y)) filtered_left.push_back(p);       
    }
    std::unordered_set<int> left_y_values;
    for (const auto& p : points_left)
    {
        left_y_values.insert(p.y); // 记录左边界点的y值
    }
    for (const auto& p : points_right)
    {
        if (left_y_values.count(p.y)) filtered_right.push_back(p);           
    }
    points_left.swap(filtered_left);
    points_right.swap(filtered_right);
}

cv::Point KFS::center(const vector<Point>& points_left, const vector<Point>& points_right)
{
    if(points_left.empty() || points_right.empty()) 
    {
        // cout << "无边界点" << endl;
        return cv::Point(-1, -1);
    }

    std::vector<double> center_xs;
    center_xs.reserve(points_left.size());
    for(size_t i = 0; i < points_left.size(); i++)
    {
        double cx = (points_left[i].x + points_right[i].x) / 2.0;
        center_xs.push_back(cx);
    }
    int median_y = points_left[points_left.size() / 2].y;

    ul_ = points_left[points_left.size() / 2].x;
    ur_ = points_right[points_right.size() / 2].x;

    double sum_x = 0.0;
    int count = 0;
    for(size_t i = 0; i < center_xs.size(); i++)
    {
        if(abs(points_left[i].y - median_y) <= 5)
        {
            sum_x += center_xs[i];
            count++;
        }
    }
    if(count > 0)
    {
        double center_x = sum_x / count;
        center_x = round(center_x * 100.0) / 100.0;
        return cv::Point(center_x, median_y);
    }
    else
    {
        // cout << "count is 0" << endl;
        return cv::Point(-1, -1);
    }
}


double KFS::computeMidX()
{
    double theta{0.0};
    double x_mid{0.0};
    {
        std::unique_lock<std::mutex> lock(global_yaw_);
        theta = lidar_yaw_;
    }
    if(theta > -10 && theta < 10) theta = -theta;
    else if(theta > -100 && theta < -80) theta += 90; 
    else if(theta > 80 && theta < 100) theta -= 90; 
    else
    {
        x_mid = 999;
        return x_mid;
    }
    theta = theta * 3.1415926 / 180;
    // cout << "computeMidX: theta " << theta << endl;
    double t1 = (ul_ - cx_) / fx_;
    double t2 = (ur_ - cx_) / fx_;
    // cout << "computeMidX: ul_ ur_ " << ul_ << " " << ur_<< endl;

    // cout << "computeMidX: t1 t2 " << t1 << " " << t2<< endl;
 
    double dx = L_ * cos(theta);
    double dz = L_ * sin(theta);
    // cout << "computeMidX: dx  dz " << dx << " " << dz << endl;
 
    double denom = t2 - t1;
    if (std::abs(denom) < 1e-6)
    {
        return 999;  
    }
    double lambda1 = (dx - t2 * dz) / denom;
 
    double lambda2 = lambda1 + dz;
    // cout << "computeMidX: lambda1 lambda2 " << lambda1 << " " << lambda2 << endl;
 
    x_mid = (t2 * lambda2 + t1 * lambda1) / 2.0;
    // if(kfs_log_.is_open()) oss_ <<  "\ttheta: " << theta << "\tcomputeMidX: ul_ ur_( " << ul_ << " " << ur_ << " )";
    return x_mid;
}

void KFS::draw(Mat& dst)
{
    center_p_ = center(points_left, points_right);
    if(center_p_.x != -1 && center_p_.y != -1)
    {
        cv::circle(dst, center_p_, 2, cv::Scalar(0, 255, 0), -1); 
    
        // 画中心线（红色）
        cv::line(dst, cv::Point(center_x_, 0), cv::Point(center_x_, dst.rows - 1), cv::Scalar(0, 0, 255), 1);
        //画过中兴水平线
        lenth_ = 0;
        for(int i = points_left[points_left.size() / 2].x; i <= points_right[points_right.size() / 2].x; i++)
        {
            cv::circle(dst, cv::Point(i, center_p_.y), 0.5, cv::Scalar(0, 0, 255), -1);
            lenth_++;
        }
        // 画左边界线（白色）
        for (const auto& p : points_left)
        {
            cv::circle(dst, p, 1, cv::Scalar(255, 255, 255), -1);  // 白色点
        }
        // 画右边界线（白色）
        for (const auto& p : points_right)
        {
            cv::circle(dst, p, 1, cv::Scalar(255, 255, 255), -1);  // 白色点
        }
    }
    else
    {
        // cout << "center_p_ is (-1, -1)" << endl;
    }
}

cv::Mat KFS::Kfs(const Mat &frame)
{
    Mat hsv;
    binary_ = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    for (int i = 0; i < frame.rows; i++) 
    {
        for (int j = 0; j < frame.cols; j++) 
        {
            Vec3b hsv_pixel = hsv.at<Vec3b>(i, j);
            Vec3b bgr_pixel = frame.at<Vec3b>(i, j);

            // 红色判断条件：
            bool red_bgr = (bgr_pixel[0] < 235 && bgr_pixel[1] < 235 && bgr_pixel[2] > 30);
            bool red_hsv_h = (hsv_pixel[0] > 160 || hsv_pixel[0] < 15);
            bool red_hsv_s = (hsv_pixel[1] > 20);
            if (red_bgr && red_hsv_h && red_hsv_s) 
            {
                binary_.at<uchar>(i, j) = 255;
            }

            //蓝色判断条件：
            // bool blue_bgr = (bgr_pixel[2] < 235 && bgr_pixel[1] < 235 && bgr_pixel[0] > 35);
            // bool blue_hsv_h = (hsv_pixel[0] > 100 && hsv_pixel[0] < 130);
            // bool blue_hsv_s = (hsv_pixel[1] > 20);
            // if (blue_bgr && blue_hsv_h && blue_hsv_s) 
            // {
            //     binary_.at<uchar>(i, j) = 255;
            // }
        }
    }

    Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    morphologyEx(binary_, binary_, MORPH_OPEN, kernel); // 开运算 去毛刺
    morphologyEx(binary_, binary_, MORPH_CLOSE, kernel); // 闭运算 填空隙
    // // // 中值滤波 - 有效去除颗粒噪点
    // cv::Mat a = binary_.clone();

    // cv::medianBlur(binary_, binary_, 5);

    return binary_;
}

cv::Mat KFS::Max(const cv::Mat& src)
{
    //label: 每个像素点所属的连通域标记（0 ~ n-1）
    //stats: 连通域的统计信息，包括面积、矩形框
    //centroid: 连通域的中心坐标
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::Mat label, stats, centroid;
    int n = cv::connectedComponentsWithStats(binary_, label, stats, centroid, 8, CV_32S);//32位有符号int
    if(n <= 1) return mask;
    int max_area = 0;
    int max_label = -1;
    int second_area = 0;
    int second_label = -1;
    float center_x_connected = -1.0;
    int area;
    for(int i = 1; i < n; i++)
    {
        center_x_connected = centroid.at<double>(i, 0);//第 i 个连通域的 x 坐标
        area = stats.at<int>(i, cv::CC_STAT_AREA);
        // cout << "area" << i << ": " << area << endl;
        if (area > max_area && abs(center_x_connected - center_x_) < 260) 
        {
            max_area = area;
            max_label = i;
        }
    }
    // cout << "max_label: " << max_label << ", " << "max_area: " << max_area << endl;
    for(int i = 1; i < n; i++)
    {
        area = stats.at<int>(i, cv::CC_STAT_AREA);
        if(area > second_area && i != max_label)
        {
            second_area = area;
            second_label = i;
        }
    }
    if(abs(second_area - max_area) > 10000)
    {
        // cout << "second_area is too large" << endl;
        second_area = 0;
        second_label = -1;
    }
    // cout << "max_area: " << max_area << ", " << "second_area: " << second_area << endl;
 
    mask.setTo(255, (label == max_label) | (label == second_label));
    return mask;
}

void KFS::applyPrewitt_0(const Mat& src, Mat& grad_y) 
{
    // Prewitt算子核
    Mat kernel_y = (Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    
    // 应用卷积 CV_32F：用更大的盒子放超过0~255的数值
    filter2D(src, grad_y, CV_32F, kernel_y);
    
    // 转换为8位图像用于显示 常用于把梯度（可能为负/超出 0-255）转换为显示范围。
    convertScaleAbs(grad_y, grad_y);

    int rows = src.rows;
    int cols = src.cols;
    for(int r = 0; r < rows; r++)
    {
        // 左边界
        if(src.at<uchar>(r,0) == 255)
        {
            count_l_++;
            grad_y.at<uchar>(r,0) = 255;
        }

        // 右边界
        if(src.at<uchar>(r,cols-1) == 255)
        {
            count_r_++;
            grad_y.at<uchar>(r,cols-1) = 255;
        }
    }
}

void draw_points(const Mat& img, const vector<Point>& points_left, const vector<Point>& points_right)
{
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC3);
    for(const auto& p : points_left)
    {
        circle(mask, p, 2, Scalar(0, 255, 0), -1);
    }
    for(const auto& p : points_right)
    {
        circle(mask, p, 2, Scalar(0, 0, 255), -1);
    }
    // cv::namedWindow("points", cv::WINDOW_NORMAL);
    // cv::imshow("points", mask);
    // cv::waitKey(1);
}

LineKB KFS::ransacLineFit( const std::vector<cv::Point>& pts )
{
    LineKB best;
    if (pts.size() < 10) return best;

    std::mt19937 rng(static_cast<unsigned>(time(nullptr)));
    std::uniform_int_distribution<int> dist(0, pts.size() - 1);

    int best_inliers = 0;

    for (int it = 0; it < iterations_; it++)
    {
        const cv::Point& p1 = pts[dist(rng)];
        const cv::Point& p2 = pts[dist(rng)];
        if (std::abs(p1.y - p2.y) < 2) continue;

        double k = (p2.x - p1.x) / double(p2.y - p1.y);
        double b = p1.x - k * p1.y;

        std::vector<cv::Point> current_inliers;
        current_inliers.reserve(pts.size()); 

        for (const auto& p : pts)
        {
            double x_est = k * p.y + b;
            double d = std::abs(p.x - x_est);
            if (d < dist_thresh_)
            {
                current_inliers.push_back(p);  
            }
        }

        int inliers = static_cast<int>(current_inliers.size());

        if (inliers > best_inliers && inliers > min_inliers_)
        {
            best_inliers = inliers;
            best.k = k;
            best.b = b;
            best.inliers.swap(current_inliers); 
            best.valid = true;
        }
    }
    // cout << "best.inliers.size(): " << best.inliers.size() << endl;
    return best;
}

void KFS::filterPoints(std::vector<cv::Point>& points, int& dist)
{
    // 用于记录被移除点的最大和最小 y 坐标
    int max_y = INT_MIN;
    int min_y = INT_MAX;
    bool has_removed = false;

    for (const auto& p : points)
    {
        if (p.x <= 2 || p.x >= 637)
        {
            if (p.y > max_y) max_y = p.y;
            if (p.y < min_y) min_y = p.y;
            has_removed = true;
        }
    }
    dist = has_removed ? (max_y - min_y) : 0;

    auto should_remove = [](const cv::Point& p) -> bool {
    return (p.x <= 2) || (p.x >= 637);
    };
    points.erase(
        std::remove_if(points.begin(), points.end(), should_remove),
        points.end()
    );
}

void KFS::showdst(const Mat &temp)
{

    dst_ = cv::Mat::zeros(temp.size(), CV_8UC3);
    samey(points_left, points_right);               //同一行
    // draw_points(temp, points_left, points_right);
    draw(dst_);
    cv::namedWindow("dst_", cv::WINDOW_NORMAL);
    cv::imshow("dst_", dst_);

    float s;
    if(lenth_ > 0) s = 350.0 / lenth_;
    else s = 0.0;
    s = round(s * 100.0) / 100.0;
    dist_ = s * (center_p_.x - center_x_);
    dist_ = round(dist_ * 100.0) / 100.0;


    double dist_new = computeMidX();//dist_new = 600 则是没有转到位
    dist_ = -dist_new;

    if(dist_ > 350) dist_ = 350;
    if(dist_ < -350) dist_ = -350;
   
    publishdist(0);

    return;
}

void KFS::publishdist(int q)
{
    base_interfaces::msg::CameraKfs msg;
    msg.kfs_dist = dist_;
    msg.flag_kfs = q;
    cout << "新距离: " << dist_ << endl;
    if (kfs_log_.is_open()) kfs_log_ << dist_ << ", " << q << " ";
    pub_kfs_->publish(msg);
}

void KFS::process_thread()
{ 
    while (running_.load() && rclcpp::ok())
    { 
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(global_mtx_);
            cv_frame_.wait(lock, [this]{return has_frame_ || !rclcpp::ok() || !running_;});
            if(!rclcpp::ok() || !running_) break;
            frame = global_frame_.clone();
            has_frame_ = false;
        }
        frame = cv::imread("/home/lx/yolo_biao/5_yolo_b/1776070445_420386.jpg");

        if(frame.empty()) 
        {
            std::cout << "没有成功获取图像" << std::endl;
            continue;
        }
        
        // --- 显示图像并统一捕获按键 ---
        cv::namedWindow("frame", cv::WINDOW_NORMAL);
        cv::imshow("frame", frame);
        int key = cv::waitKey(1) & 0xFF;
        
        // 如果有按键按下，将其传递给 shared_key_ 供相机线程使用
        if (key != 0xFF) {
            shared_key_.store(key);
        }

        // ================= 新增：提取 YOLO 中 label=0/1 且面积最大的框 =================
        yolo_detector_->detect(frame);

        if(yolo_detector_->objs.size() == 0)
        {
            cout << "没有kfs" << endl;
            publishdist(2);
            continue;
        }

        cv::namedWindow("res", cv::WINDOW_NORMAL);
        cv::imshow("res", yolo_detector_->res);

        cv::Rect best_rect;
        float max_area = 0.0f;
        for (const auto& obj : yolo_detector_->objs)
        {
            if (obj.label == 0)
            {
                float area = obj.rect.width * obj.rect.height;
                if (area > max_area)
                {
                    max_area = area;
                    best_rect = cv::Rect(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
                }
            }
        }

        #if CAM_PLACE == 1
            cout << "左边界: " << best_rect.x << " 右边界： " << best_rect.x + best_rect.width << endl;
            if(best_rect.x <= 2 && best_rect.x + best_rect.width >= 637)
            {
                publishdist(1);
                if (kfs_log_.is_open()) kfs_log_ << ",左边界, " << best_rect.x << " ,右边界, " << best_rect.x + best_rect.width << endl;
                cv::waitKey(1);
                continue;
            }
            else if(best_rect.x <= 2)
            {
                dist_ = 999;
                publishdist(0);
                if (kfs_log_.is_open()) kfs_log_ << ",左边界, " << best_rect.x << " ,右边界, " << best_rect.x + best_rect.width << endl;
                cv::waitKey(1);
                continue;
            }
            else if(best_rect.x + best_rect.width >= 637)
            {
                dist_ = -999;
                publishdist(0);
                if (kfs_log_.is_open()) kfs_log_ << ",左边界, " << best_rect.x << " ,右边界, " << best_rect.x + best_rect.width << endl;
                cv::waitKey(1);
                continue;
            } 
        #endif

        // 创建一个与 frame 尺寸相同、类型相同的全黑图像
        cv::Mat masked_frame = cv::Mat::zeros(frame.size(), frame.type());
        if (max_area > 0)
        {
            // 与原图边界求交集，防止 YOLO 预测的框超出图像边界导致 copyTo 崩溃
            cv::Rect valid_rect = best_rect & cv::Rect(0, 0, frame.cols, frame.rows);
            if (valid_rect.area() > 0)
            {
                // 将原图中 valid_rect 区域的像素，拷贝到黑图的对应区域
                frame(valid_rect).copyTo(masked_frame(valid_rect));
            }
        }
        // 图像处理
        cv::Mat gray = Kfs(masked_frame);   // 灰度图
        int whitePixelCount = cv::countNonZero(gray);
        if(whitePixelCount <= 100)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            publishdist(2);
            continue;            
        }

        cv::Mat max = Max(masked_frame);    // 提取目标区域
        cv::Mat temp;
        applyPrewitt_0(max, temp);   // 提取左右边界
        cv::namedWindow("max", cv::WINDOW_NORMAL);

        cv::imshow("max", max);
        

        cv::waitKey(1);

        // 偏差判断
        // if(count_l_ > 180 && count_r_ > 180)
        // {
        //     publishdist(1);
        //     count_l_ = 0;
        //     count_r_ = 0;
        //     continue;
        // }
        // else if(count_l_ > 180) 
        // {
        //     dist_ = 999;
        //     publishdist(0);
        //     cout << "dist_: " << dist_ << " count_l_: " << count_l_ << endl;
        //     count_l_ = 0;
        //     continue;
        // }
        // else if(count_r_ > 180) 
        // {
        //     dist_ = -999;
        //     publishdist(0);
        //     cout << "dist_: " << dist_ << " count_r_: " << count_r_ << endl;
        //     count_r_ = 0;
        //     continue;
        // }
        // cout << "count_r_: " << count_r_ << " count_l_: " << count_l_ << endl;
        
        #if CAM_PLACE == 2
            over_ = false;
            if(best_rect.x <= 2 || best_rect.x + best_rect.width >= 637)
            {
                over_ = true;
            }
        #endif

        points_left.clear();
        points_right.clear();
        boudary(temp);                                  //提取边界点

        if(over_)
        {
            filterPoints(points_left, l_dist_);
            filterPoints(points_right, r_dist_);
        }

        points_left = ransacLineFit(points_left).inliers;
        points_right = ransacLineFit(points_right).inliers;

        cout << "左边界点数：" << points_left.size() << " 右边界点数： " << points_right.size() << endl;
        if(over_)
        {
            if(abs(best_rect.height - l_dist_) / best_rect.height > 0.9)
            {
                dist_ = 999;
                publishdist(0);
                continue;
            }
            else if(abs(best_rect.height - r_dist_) / best_rect.height > 0.9)
            {
                dist_ = -999;
                publishdist(0);
                continue;
            }
        }


        showdst(temp); // 计算偏差并发送
        if (kfs_log_.is_open()) kfs_log_ << ",左边界, " << best_rect.x << " ,右边界, " << best_rect.x + best_rect.width << endl;

        #ifdef SAVE_IMAGE
        struct timeval tv;
        gettimeofday(&tv, NULL);
        // 注意：这里去掉了重复的 cv::waitKey(1)，直接复用上面获取的 key
        int disp_flag = 0;
        if(key == 'p')
        {
            std::cout << "保存图片+++++++++++++++++++++++++++++++++++++++" << std::endl;
            std::cout << cv::imwrite(cv::format("/home/lx/frame/%ld_%ld.jpg", tv.tv_sec, tv.tv_usec), frame) << std::endl;
            disp_flag++;
        }
        if(key == 'o') disp_flag--;
        #endif

        #ifdef SAVE_VEDIO
        if (!rclcpp::ok())
        {
            video_saver_.stop();
        }
        #endif
    }
}
