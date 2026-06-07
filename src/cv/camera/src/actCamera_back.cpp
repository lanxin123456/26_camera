#include <camera_backward/actCamera_back.hpp>

camera_backward::ActCamera::ActCamera() : 
    width_(640),
    height_(480),
    fps_(60),
    temp_frame(),
    srcframe(),
    cap(),
    buffer_queue_(),
    queue_mutex_(),
    frame_mutex_()
{
    std::string calibration_path = "src/cv/ost_3.yaml";  // 根据实际路径修改
    loadCalibrationParameters(calibration_path);
}

std::string camera_backward::ActCamera::find_lrcp_camera()
{
    glob_t glob_result;
    glob("/dev/video*",GLOB_TILDE,NULL,&glob_result);
    std::string found_device = "";
    for(int i = 0; i < glob_result.gl_pathc; i++)
    {
        int fd = open(glob_result.gl_pathv[i], O_RDWR);
        if(fd < 0) continue;
        v4l2_capability cap;
        // std::cout << "trying " << glob_result.gl_pathv[i] << std::endl;
        if(ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0)
        {
            std::string card_name(reinterpret_cast<const char*>(cap.card));
            if(card_name.find("LRCP 1080P") != std::string::npos)
            {
                found_device = glob_result.gl_pathv[i];
                close(fd);
                break;
            }
        }
        close(fd);
    }
    globfree(&glob_result);
    if(found_device.empty())
    {
        std::cout << "No Camera found" << std::endl;
        return "";
    }
    std::cout << "LRCP 1080P backward: " << found_device << std::endl;
    return found_device;
}

//标定参数加载
bool camera_backward::ActCamera::loadCalibrationParameters(const std::string& calibration_file)
{
    try {
        //cv::FileStorage OpenCV中用于读写XML/YAML文件的类
        cv::FileStorage fs(calibration_file, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cout << "无法打开标定文件: " << calibration_file << std::endl;
            return false;
        }
        
        // 读取相机内参矩阵
        cv::FileNode camera_matrix_node = fs["camera_matrix"]["data"];
        std::vector<double> camera_matrix_data;
        camera_matrix_node >> camera_matrix_data;
        
        // 将数据转换为 3x3 矩阵
        camera_matrix_ = (cv::Mat_<double>(3, 3) << 
            camera_matrix_data[0], camera_matrix_data[1], camera_matrix_data[2],
            camera_matrix_data[3], camera_matrix_data[4], camera_matrix_data[5],
            camera_matrix_data[6], camera_matrix_data[7], camera_matrix_data[8]);
        
        // 读取畸变系数
        cv::FileNode dist_coeffs_node = fs["distortion_coefficients"]["data"];
        std::vector<double> dist_coeffs_data;
        dist_coeffs_node >> dist_coeffs_data;
        
        // 将数据转换为畸变系数矩阵
        distortion_coeffs_ = (cv::Mat_<double>(1, 5) << 
            dist_coeffs_data[0], dist_coeffs_data[1], dist_coeffs_data[2],
            dist_coeffs_data[3], dist_coeffs_data[4]);
        
        // 预计算畸变校正映射（提高实时处理性能）
        //预计算从畸变图像坐标到正常图像坐标的映射关系，避免在每帧处理时重复计算，大大提高性能。
        cv::initUndistortRectifyMap(
            camera_matrix_, distortion_coeffs_, cv::Mat(),
            camera_matrix_, cv::Size(width_, height_), CV_16SC2, map1_, map2_);
        
        calibration_loaded_ = true;
        std::cout << "相机标定参数加载成功" << std::endl;
        std::cout << "相机内参矩阵: " << std::endl << camera_matrix_ << std::endl;
        std::cout << "畸变系数: " << std::endl << distortion_coeffs_ << std::endl;
        
        return true;
    }
    catch (const std::exception& e) 
    {
        std::cout << "加载标定参数失败: " << e.what() << std::endl;
        calibration_loaded_ = false;
        return false;
    }
}

void camera_backward::ActCamera::Init()
{
    while(true)
    {
        cv::TickMeter tm;
        tm.start();
        video_device_ = find_lrcp_camera();
        // video_device_ = "/dev/video6";
        tm.stop();
        std::cout << "查找相机耗时：" << tm.getTimeMilli() << " ms" << std::endl;
        if(!video_device_.empty()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // 开相机
    std::cout << "Init camera " << video_device_ << std::endl;
    cap.open(video_device_, cv::CAP_V4L2);                                        //当使用 cv::CAP_V4L2参数时，OpenCV 会调用 ​​V4L2（Video4Linux2）驱动接口​​ 与摄像头交互
    if (!cap.isOpened())
    {
        std::string error_msg = "Could not open video device " + video_device_;
        std::cout << "Error: " << error_msg << std::endl;
        throw std::runtime_error(error_msg);
    }

    // 设置参数
    
    if(!cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')))
    {
        std::cout << "MJPG无法设置，使用YUYV" << std::endl;
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width_);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    cap.set(cv::CAP_PROP_FPS, fps_);
    std::cout << "Camera " << video_device_ << 
        " initialized with resolution " << width_ << "x" 
        << height_ << " and fps " << fps_ << std::endl;

    if (exposure_mode_ == 1)
    {
    std::cout << "设置曝光模式为手动" << std::endl;
      cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 3);  // 确保手动模式
    //   cap.set(cv::CAP_PROP_GAIN,20);
    //   cap.set(cv::CAP_PROP_EXPOSURE, 100);
    }
    else
    {
      cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);  // 确保手动模式
      cap.set(cv::CAP_PROP_GAIN,30);
      cap.set(cv::CAP_PROP_EXPOSURE, 100);
    }

    // 清空缓存
    for(int i = 0; i < 5; i++)
    {
        cv::Mat temp;
        cap.read(temp);//每次 read()严格获取​​时间上的下一帧​​，不会跳过或重复
    }

    // 监测与输出
    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    double actual_exposure = cap.get(cv::CAP_PROP_EXPOSURE);
    double actual_wb_temperature = cap.get(cv::CAP_PROP_WB_TEMPERATURE);
    std::cout << "相机已启动" << std::endl;
    std::cout << "设备: " << video_device_ << std::endl;
    std::cout << "分辨率: " << width_ << "x" << height_ << std::endl;
}

void camera_backward::ActCamera::getBuffer()
{
    if(cap.read(temp_frame))
    {
        buffer_queue_.push(std::move(temp_frame));  //// 触发移动构造，转移数据所有权，零拷贝
    }
    else
    {
        std::cout << "获取失败" << std::endl;
    }
}


void camera_backward::ActCamera::clearBuffer()
{
    while(!buffer_queue_.empty())
    {
        buffer_queue_.pop();//移除队列的队首元素（最老的元素）
    }
}

cv::Mat camera_backward::ActCamera::Update()
{
    cap.read(srcframe);
    if(!map1_.empty() && !map2_.empty() && !srcframe.empty())
    {
        cv::remap(srcframe, dstframe, map1_, map2_, cv::INTER_LINEAR);
    }
    else 
    {
        dstframe = srcframe.clone();
    }
    return dstframe;
}


void camera_backward::ActCamera::release()
{
    if(cap.isOpened())
    {
        cap.release();
    }
    this->clearBuffer();
}

camera_backward::ActCamera::~ActCamera()
{
    this->release();
}
