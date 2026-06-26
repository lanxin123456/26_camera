/*
 * @Author: 於悦洋 yuyueyang2468@163.com
 * @Date: 2025-12-24 23:06:38
 * @LastEditors: 於悦洋 yuyueyang2468@163.com
 * @LastEditTime: 2026-01-21 09:40:32
 * @FilePath: /ROBOCON2026_base/src/cv/camera_backward/src/weapon.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "camera_backward/weapon.hpp"

Deal::Deal(const std::string& config_path):
    Node{"deal"},
    rect_{314, 214, 39, 11},
    rect_fixed_{298, 230, 70, 70},
    start_{false},
    has_frame_{false},
    visualize_{true},
    task_{-1},
    current_task_{-1},
    consistent_frame_count_{0},
    total_pixels_{rect_.width * rect_.height},
    changed_pixels_{0},
    a_{0},
    b_{0},
    c_{0},
    dist_{1000.f},
    config_ {2, {"weapon", "tai"}, {{0, 0, 255}, {0,255,0}}, cv::Size(640,640), 100, 0.6f, 0.65f}
{
    loadYamlConfig(config_path);

    #ifdef IF_CAMERA
    ACam.Init();
    #endif

    yolo_all_ = std::make_unique<trt_yolo::YOLOv8>(std::string("/home/lx/runs/detect/train23/weights/best.engine"), config_);

    yolo_all_->make_pipe(true);

    pub_weapon_ = this->create_publisher<base_interfaces::msg::Camera>("/weapon_camera_miao", 10);
    sub_task_ = this->create_subscription<base_interfaces::msg::CameraTask>("/lx_task", 10,
        std::bind(&Deal::task_callback, this, std::placeholders::_1));

    camera_backward.open("camera_backward.csv", std::ios::app);
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);
    if(camera_backward.is_open()) camera_backward << "\n\n\n\n\n"
                                    << "时间, " 
                                    << put_time(localTime, "%Y-%m-%d %H:%M:%S") 
                                    << " ===================================================================================================================================" 
                                    << endl;
}

void Deal::loadYamlConfig(const std::string& path) 
{
    try {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        
        if (!fs.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "无法打开配置文件: %s", path.c_str());
            return;
        }

        // --- 读取参数 ---
        fs["Field"] >> field_;
        fs["ShowParams"] >> show_params_;

        fs["Deal"]["Display"] >> visualize_;
        fs["Deal"]["Rect"] >> rect_;
        fs["Deal"]["RectFixed"] >> rect_fixed_;
        fs["Deal"]["IntervalY"] >> interval_y_;
        fs["Deal"]["IouThresh"] >> iou_thresh_;
        fs["Deal"]["ImagePath"] >> image_path_;
        fs["Deal"]["VideoPath"] >> video_path_;

        fs["Deal"]["BThresh"] >> b_threshold_;
        fs["Deal"]["GThresh"] >> g_threshold_;
        fs["Deal"]["RThresh"] >> r_threshold_;
        fs["Deal"]["HThreshUP"] >> h_threshold_up_;
        fs["Deal"]["HThreshDOWN"] >> h_threshold_down_;
        fs["Deal"]["SThreshDOWN"] >> s_threshold_down_;


        fs.release();

        // --- 参数打印输出 ---
        if (show_params_) {
            RCLCPP_INFO(this->get_logger(), "----------- Deal 参数加载成功 -----------");
            RCLCPP_INFO(this->get_logger(), "场馆设置 (Field): %d", field_);
            RCLCPP_INFO(this->get_logger(), "显示开关 (Display/visualize): %s", visualize_ ? "ON" : "OFF");
            
            RCLCPP_INFO(this->get_logger(), "矩形 Rect: [x:%d, y:%d, w:%d, h:%d]", 
                        rect_.x, rect_.y, rect_.width, rect_.height);
            
            RCLCPP_INFO(this->get_logger(), "固定矩形 RectFixed: [x:%d, y:%d, w:%d, h:%d]", 
                        rect_fixed_.x, rect_fixed_.y, rect_fixed_.width, rect_fixed_.height);
            
            RCLCPP_INFO(this->get_logger(), "Y轴间隔 (IntervalY): %d", interval_y_);
            RCLCPP_INFO(this->get_logger(), "IOU阈值 (IouThresh): %.2f", iou_thresh_);
            
            // 注意：string 必须加上 .c_str() 才能用 %s 打印
            RCLCPP_INFO(this->get_logger(), "图片路径 (ImagePath): %s", image_path_.c_str());
            RCLCPP_INFO(this->get_logger(), "视频路径 (VideoPath): %s", video_path_.c_str());
            
            RCLCPP_INFO(this->get_logger(), "--------------------------------------------");
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "----------- Deal 参数不显示 -----------");
        }

    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Deal 解析 YAML 时出错: %s", e.what());
    }
}

void Deal::start()
{
    cout << "Start Dealweapon" << endl;
    running_.store(true);
    th_camera_ = std::thread(&Deal::camera_thread_func, this);
    th_process_ = std::thread(&Deal::process_thread_func, this);
}

Deal::~Deal()
{
    stop();
}

void Deal::stop()
{
    cout << "Stop DeDealweaponal" << endl;
    running_ = false;
    cv_frame_.notify_all();
    if(th_camera_.joinable()) th_camera_.join();
    if(th_process_.joinable()) th_process_.join();
    if (visualize_) cv::destroyAllWindows();
}

void Deal::task_callback(const base_interfaces::msg::CameraTask::SharedPtr msg)
{
    if(msg->task != -1) 
    {
        std::unique_lock<std::mutex> lock(task_mutex_);
        task_ = msg->task;
    } 
}

void Deal::camera_thread_func()
{
    #ifdef VEDIO
    cv::VideoCapture cap(video_path_);  // ←你的录像
    if(!cap.isOpened())
    {
        std::cerr << "[ERROR] Cannot open video file\n";
        // rclcpp::shutdown();
        return;
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 60.0;
    auto frame_period = std::chrono::microseconds((int)(1e6 / fps));
    #endif
    
    while(rclcpp::ok() && running_)
    {
        auto up = std::chrono::steady_clock::now();
        cv::Mat frame, frame_video;
        #ifdef VEDIO
        cap >> frame;
        if(frame.empty())
        {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        // #ifdef 
        // frame_video = frame.clone();
        // video_saver_.start(frame_video);
        // video_saver_.write(frame_video);
        // #endif
        std::this_thread::sleep_for(frame_period);
        // frame = cv::imread("/home/lx/frame/frame_51/1777550345_816992.jpg");
        {
            std::unique_lock<std::mutex> lock(global_mtx_);
            global_frame_ = std::move(frame);
            has_frame_ = true;
        }
        cv_frame_.notify_one();
        #else
            frame = ACam.Update();
            if(frame.empty()) 
            {
                cout << "更新的帧为空！" << endl;
                continue;
            }
            #ifdef SAVE_VEDIO
            frame_video = frame.clone();
            if(running_)
            {
                video_saver_.start(frame_video);
                video_saver_.write(frame_video);
            }
            else if(!running_)
            {
                cout << "结束录像" << endl;
                video_saver_.stop();
            }
            #endif
            {
                std::unique_lock<std::mutex> lock(global_mtx_);
                global_frame_ = std::move(frame);
                has_frame_ = true;
            }
            cv_frame_.notify_one();
        #endif
        
        auto up_end = std::chrono::steady_clock::now();
        // cout << " up耗时： " << std::chrono::duration<float, std::milli>(up_end - up).count() << " ms" << endl;
    }
}

void Deal::calculate_dist(const cv::Mat& dealImage,const std::vector<cv::Rect>& rects)
{
    int dist_pixel = dealImage.cols / 2;
    int lenth_pixel = 0;
    float current_min_dist = FLT_MAX;
    cv::Point current_closest_rect(0, 0);

    // 找到当前帧中距离中心最近的检测框
    for (const auto& rect : rects)
    {
        // if(rect.area() < 1000 || rect.area() > 7000) continue;
        float dist = (rect.x + rect.width / 2.0f) - 340.49393;
        if (std::abs(dist) < std::abs(dist_pixel))
        {
            dist_pixel = dist;
            lenth_pixel = rect.width;
            current_min_dist = std::abs(dist);
            current_closest_rect = cv::Point(rect.x, rect.y);
        }
    }

    // 防止除 0 / 无目标 << "\n"
    if (lenth_pixel == 0) 
    {
        dist_ = 0.0f;
        return;
    }
    float pd = 28.0f / lenth_pixel;
    pd = std::round(pd * 100) / 100.0f;
    pd *= dist_pixel;

    // 误差拟合修正
    // dist_ =
    //     -7.490163
    //     + 0.589434 * lenth_pixel
    //     + 0.887561 * pd
    //     - 0.011031 * lenth_pixel * lenth_pixel
    //     + 0.006128 * lenth_pixel * pd
    //     + 0.000173 * pd * pd
    //     + 0.000067 * lenth_pixel * lenth_pixel * lenth_pixel
    //     - 0.000068 * lenth_pixel * lenth_pixel * pd
    //     + 0.000004 * lenth_pixel * pd * pd
    //     - 0.000004 * pd * pd * pd 
    //     - 1;

    // dist_ = pd - (dist_ - pd) - 5.5f;

    dist_ = pd + 4.3;
    if (dist_ < -190) dist_ = -190;
    if (dist_ > 190)  dist_ = 190;
}

bool Deal::deal_frame(const Mat &frame, const cv::Rect_<float>& rect, std::ostringstream& oss)
{
    if(frame.empty()) return false;

    srcframe_ = frame.clone();

    changed_pixels_ = 0;
    binary_ = Mat::zeros(480, 640, CV_8UC1);

    rect_.x = rect.x + rect.width/2 - 19;
    rect_.y = rect.y - 21 - 11;

    // cout << "rect_.x: " << rect_.x << " rect_.y: " << rect_.y << "rect_.width: " << rect_.width << "rect_.height: " << rect_.height << endl;
    process_rect(rect_, changed_pixels_);

    return cal_ch_rate(changed_pixels_, oss);
}


void Deal::process_rect(const Rect& rect, int& changed_pixels)
{
    Mat hsv;
    cv::cvtColor(srcframe_, hsv, cv::COLOR_BGR2HSV);
    for (int i = 0; i < rect.height; i++) 
    {
        for (int j = 0; j < rect.width; j++) 
        {
            int global_i = rect.y + i;
            int global_j = rect.x + j;
            
            Vec3b bgr = srcframe_.at<Vec3b>(global_i, global_j);
            Vec3b hsv_pixel = hsv.at<Vec3b>(global_i, global_j);

            bool red_bgr = (bgr[2] > r_threshold_ && bgr[1] < g_threshold_ && bgr[0] < b_threshold_);
            bool red_h = (hsv_pixel[0] > h_threshold_up_ || hsv_pixel[0] < h_threshold_down_);
            bool red_s = (hsv_pixel[1] > s_threshold_down_);

            if(red_h && red_bgr && red_s)
            {
                binary_.at<uchar>(global_i, global_j) = 255;
                changed_pixels++;
            }
        }
    }
}

bool Deal::cal_ch_rate(const int& changed_pixels_, std::ostringstream& oss)
{
    double change_rate = static_cast<double>(changed_pixels_) / total_pixels_;
    change_rate = std::round(change_rate * 1000) / 1000;
    if(change_rate > 0.7)
    {
        consistent_frame_count_++;
    }
    else if(change_rate <= 0.7)
    {
        if(consistent_frame_count_ > 0) consistent_frame_count_--;
    }
    if(consistent_frame_count_ >= 3)
    {
        // consistent_frame_count_  = 0;  // 重置连续计数
        return true;
    }
    if(camera_backward.is_open() && change_rate > 0.3)
    {
    oss  << "  矩形1变化比例, " << change_rate * 100 << "%" 
            << " | 连续帧, " << consistent_frame_count_ 
            << " | 变化像素, " << changed_pixels_ << "/" << total_pixels_  << "  ";
    }      
    return false; 
    // else
    // {
    // }
}
 

void Deal::deal(const int& count, const int& count_tai, const cv::Rect2f& rect, const Mat &frame)
{
    // if(rect.x < 0 || rect.y < 0 || rect.x + rect.width  > frame.cols || rect.y + rect.height > frame.rows) 
    // {
    //     cout << "invalid rect" << endl;
    //     return;
    // }
    std::ostringstream oss;
    base_interfaces::msg::Camera msg;
    msg.weapon_state = -1;
    bool ok = false;
    int task_local = -1;
    {
        std::unique_lock<std::mutex> lock(task_mutex_);
        task_local = task_;
        task_local = 3;
    }
    if(task_local != current_task_)
    {
        a_ = 0; //没有武器计数器
        b_ = 0; //有武器计数器
        c_ = 0; //没有台计数器
        start_ = false;
    }
    switch(task_local)
    {
//判断有没有取到武器头
        case 0:
            msg.weapon_state = -1;
            msg.dist_y = 9999.0;
            msg.ok_assemble = 0;
            break;
        case 1:
            msg.weapon_state = -1;
            msg.dist_y = 9999.0;
            msg.ok_assemble = 0;
            if(count_tai == 0 && c_ < 3) c_++;
            else if( count_tai > 0 && c_ > 0) c_--;
            // start_ = true;
            if(c_ >= 3)
            {
                // c_ = 0;
                start_ = true;
            }
            if(start_ == true)
            {
                if(count == 0)
                {
                    if(a_ < 3) a_++;
                    else if(a_ == 3)
                    {
                        msg.weapon_state = 0;
                    }
                    if(b_  > 0) b_--;
                }
                else if(count > 0)
                {
                    if(b_ < 3) b_++;
                    else if(b_  == 3)
                    {
                        msg.weapon_state = 1;
                    }
                    if(a_ > 0) a_--;
                }
                cout << "武器数：" << count << endl;
            }
            break;
//开始发距离
        case 2:
            msg.weapon_state = -1;
            msg.dist_y = dist_;
            break;
//开始武器头是否拼接成功
        case 3:
            ok = deal_frame(frame, rect, oss);
            msg.weapon_state = 1;
            msg.dist_y = 0;
            if(ok) 
            {
                msg.ok_assemble = 1;
            }
            break;  
        default:
            // std::cout << "unknown\n";
        break;             
    }
    current_task_ = task_local;
    if(camera_backward.is_open()) oss << " 交并比：" << calculateIoU(rect_fixed_, rect) 
    << " 当前任务, " << current_task_ << "  ,a_, " << a_ <<  "  ,b_, " << b_ <<  "  ,c_, " << c_<< "  ,武器头状态, " << msg.weapon_state << "   ,武器头距离, " << msg.dist_y << "  ,是否拼接成功, " << msg.ok_assemble << "\n";
    
    if(camera_backward.is_open())
    {
        camera_backward << oss.str();
        camera_backward.flush();
    }
    if(task_local != -1) pub_weapon_->publish(msg);
    if(ok) 
    {
        cout << "CLOSE CAMERA_BACKWARD" << endl;
        // rclcpp::shutdown();
    }
}

float Deal::calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2) 
{
    cv::Rect rectInter = rect1 & rect2;
    
    float interArea = rectInter.area();
    
    float rect1Area = rect1.area();
    float rect2Area = rect2.area();
    
    float unionArea = rect1Area + rect2Area - interArea;
    
    if (unionArea <= 0) return 0.0;
    
    return interArea / unionArea;
}

void Deal::filterAndSortWeapons(
    std::vector<cv::Rect>& rects,
    const std::vector<trt_yolo::det::Object>& objs_weapon,
    float iou_thresh)
{
    rects.clear();
    std::vector<FilteredResult> temp_results;
    for (const auto& obj : objs_weapon)
    {
        float iou = calculateIoU(rect_fixed_, obj.rect);
        if (iou > iou_thresh)
        {
            FilteredResult res;
            res.rect = static_cast<cv::Rect>(obj.rect);
            res.iou = iou;
            res.prob = obj.prob;
            temp_results.push_back(res);
        }
    }
    std::sort(temp_results.begin(), temp_results.end(), 
        [](const FilteredResult& a, const FilteredResult& b)
    {
        const float epsilon = 1e-6f;
        if (std::abs(a.iou - b.iou) > epsilon) 
        {
            return a.iou > b.iou; // IoU 大的在前
        }
        return a.prob > b.prob;   // IoU 相同，Prob 大的在前
    });
    rects.reserve(temp_results.size());
    for (const auto& item : temp_results) 
    {
        rects.push_back(item.rect);
    }
}

void Deal::process_thread_func()
{
    if(visualize_)
    {
        cv::namedWindow("srcImage", cv::WINDOW_NORMAL);
        cv::namedWindow("yolo", cv::WINDOW_NORMAL);
    }
    while(rclcpp::ok() && running_)
    {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(global_mtx_);
            cv_frame_.wait(lock, [this]{return has_frame_ || !running_ || !rclcpp::ok();
                });
            if(!rclcpp::ok() || !running_) break;
            frame = global_frame_.clone();
            has_frame_ = false;
        }
        std::ostringstream oss;
        // frame = cv::imread("/home/lx/图片/截图/截图 2025-12-22 11-24-38.png");
        if(frame.empty()) 
        {
            cout << "处理线程frame.empty()" << endl;
            continue;
        }
        cv::Mat dealImage;
        // int c = frame.cols/2;
        // cv::line(frame, cv::Point(c, 0), cv::Point(c, 480), cv::Scalar(0, 255, 0), 2);
        
        cv::resize(frame, dealImage, cv::Size(640, 480));
        if(dealImage.empty()) cout << "dealImage.empty()" << endl;
        yolo_all_->detect(dealImage);

        objs_weapon_.clear();
        objs_tai_.clear();
        rects_.clear();
        cv::Rect rect_fixed_(298, 230, 70, 65);
        for (const auto& obj : yolo_all_->objs)
        {
            int lab = obj.label;
            switch (lab)
            {
                case 0: 
                    objs_weapon_.push_back(obj); 
                    // cout << "x: " << obj.rect.x 
                    //     << " y: " << obj.rect.y 
                    //     << " width: " << obj.rect.width 
                    //     << " height: " << obj.rect.height << endl;
                    rects_.push_back(obj.rect);
                    break;
                case 1: objs_tai_.push_back(obj); break;
            default:
                break;
            }
        }

        cv::Rect2f rect(-1, -1, -1, -1);

        if(objs_weapon_.size() != 0)
        {
            calculate_dist(dealImage, rects_);
        }
        else
        {
        } 

        filterAndSortWeapons(rects_, objs_weapon_, 0.4);
        if(rects_.size() != 0) 
        {
            rect = rects_[0];
        }

        deal(rects_.size(), objs_tai_.size(), rect, dealImage);
        
        #ifdef SAVE_IMAGE
        struct timeval tv;
        gettimeofday(&tv, NULL);
        std::cout << cv::imwrite(cv::format("/home/lx/frame/%ld_%ld.jpg", tv.tv_sec, tv.tv_usec), dealImage) << std::endl; //保存图片
        #endif

        #ifdef SAVE_VEDIO
        if (!rclcpp::ok())
        {
            video_saver_.stop();
        }
        #endif
        if(visualize_)
        {
            cv::rectangle(yolo_all_->res, rect_fixed_, cv::Scalar(0, 255, 0), 1);
            cv::rectangle(dealImage, rect_, cv::Scalar(0, 255, 0), 1);

            if(!frame.empty()) cv::imshow("srcImage", dealImage);
            if(!frame.empty()) cv::imshow("binary_", binary_);
            if(!yolo_all_->res.empty()) cv::imshow("yolo", yolo_all_->res);
        }

        static int save_counter = 0;
        std::string save_dir = "/home/lx/code/aaa/2.6_oneYolo_CHANLLENGE_960_720_git/weapon_record/";
        std::string filename = save_dir + "kfs_" + std::to_string(save_counter++) + ".jpg";
        std::string grayname = save_dir + "kfs_" + std::to_string(save_counter++) + ".jpg";
        cv::imwrite(filename, dealImage);
        cv::imwrite(grayname, binary_);

    cv::waitKey(1);
    }
}