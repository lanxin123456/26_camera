#include <act_d455/act_d455.hpp>

using namespace std;

ActD455::ActD455(const ActDParams& act_d455_params): align(RS2_STREAM_COLOR)
{
    bag_path_ = act_d455_params.bag_path;
    new_center = cv::Point2f(-1.0f, -1.0f);
    srcCloud = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    wallCloud = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    depth_data = new uint16_t[DEPTH_WIDTH * DEPTH_HEIGHT];
    cameraStatus = false;
	std::mutex src_cloud_mutex_;
	std::mutex src_grid_mutex_;
}

ActD455::~ActD455()
{
    delete[] depth_data;
}

bool ActD455::Init(void)
{
    try 
    {
        pipe.stop();
    } catch (...) {}
    cfg.disable_all_streams();

#ifdef RECORD_VIDEO
    std::string save_dir ="./realsense_bag";
    std::filesystem::create_directories(save_dir);
    auto now =std::chrono::system_clock::now();
    auto t =std::chrono::system_clock::to_time_t(now);
    std::string bag_name =save_dir+ "/d455_"+ std::to_string(t)+ ".bag";

    std::cout<< "Recording bag: "<< bag_name<< std::endl;

    // cfg.enable_record_to_file(bag_name);
#endif
	
#ifndef IFCAMERA
	cfg.enable_device_from_file(bag_path_);
#else
    auto devices = ctx.query_devices(); // 获取设备列表
    device_count = devices.size();
    cout << "device_count: " << device_count << endl;
    if (device_count == 0)
    {
        std::cerr << "No device connected, please connect a RealSense device." << std::endl;
        return false;
    }

    auto dev = devices[0];
    cout << "Device Name: " << dev.get_info(RS2_CAMERA_INFO_NAME) << endl;
    cout << "Device Serial Number: " << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << endl;
    cout << "Device FW Version: " << dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) << endl;

    cfg.enable_stream(RS2_STREAM_DEPTH, DEPTH_WIDTH, DEPTH_HEIGHT, RS2_FORMAT_Z16, 60);
    cfg.enable_stream(RS2_STREAM_COLOR, DEPTH_WIDTH, DEPTH_HEIGHT, RS2_FORMAT_BGR8, 60);
#endif
    cout << "enable stream finished." << endl;
    
    rs2_sensor *depthSensor;                 // ??
    rs2::pipeline_profile selection;
    try
    {
        selection = pipe.start(cfg);
        cout << "start cfg OK." << endl;
    }
    catch (const rs2::error& e)
    {
        std::cerr << "[ActD455] pipe.start failed: " << e.what() << std::endl;
        return false;
    }

#ifdef RECORD_VIDEO
    KeepLastNBags(save_dir, bag_name, 4);
#endif

#ifdef IFCAMERA
    auto colorSensors = selection.get_device().query_sensors();
    auto colorSensor = colorSensors[1];
    colorSensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);                             // 设置自动曝光
	// colorSensor.set_option(RS2_OPTION_EXPOSURE, 160);                                    // 设置曝光时间
	// colorSensor.set_option(RS2_OPTION_GAIN, 50);    										// 设置增益
	// colorSensor.set_option(RS2_OPTION_WHITE_BALANCE, 1800);		                        // 设置白平衡
    cout << "set exposure OK." << endl;
#endif
    // 等图像
    try
    {
        frameSet = pipe.wait_for_frames(5000);
    }
    catch (const rs2::error& e)
    {
        std::cerr << "[ActD455] wait_for_frames failed: " << e.what() << std::endl;

        // 先停流释放占用，再尝试硬重启设备
        try
        {
            pipe.stop();
        }
        catch (...)
        {
        }

        try
        {
            selection.get_device().hardware_reset();
            std::cerr << "[ActD455] hardware_reset issued, waiting for device to re-enumerate..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        catch (const std::exception& ex)
        {
            std::cerr << "[ActD455] hardware_reset failed: " << ex.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ActD455] hardware_reset failed: unknown error" << std::endl;
        }

        cameraStatus = false;
        return false;
    }
    cout << "Wait for frames OK." << endl;

    rs2::video_frame colorFrame = frameSet.get_color_frame();									//获取颜色帧
	rs2::depth_frame alignedDepthFrame = frameSet.get_depth_frame();							//获取深度帧

    rs2::stream_profile dprofile = alignedDepthFrame.get_profile();								//读取对齐深度帧参数
    rs2::stream_profile cprofile = colorFrame.get_profile();									//读取对齐颜色帧参数
    
    rs2::video_stream_profile cvsprofile(cprofile);
    color_intrin = cvsprofile.get_intrinsics();													//获取彩色相机内参

    rs2::video_stream_profile dvsprofile(dprofile);
    depth_intrin = dvsprofile.get_intrinsics();													//获取深度相机内参

    depth2color_extrin = dprofile.get_extrinsics_to(cprofile);									//获取深度向彩色相机外参
    color2depth_extrin = cprofile.get_extrinsics_to(dprofile);									//获取彩色向深度相机外参
    cout << "Get intrinics ..\n";

    // 初始化相机内参矩阵和畸变系数（用于去畸变）
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = color_intrin.fx;
    cameraMatrix.at<double>(1, 1) = color_intrin.fy;
    cameraMatrix.at<double>(0, 2) = color_intrin.ppx;
    cameraMatrix.at<double>(1, 2) = color_intrin.ppy;

    // RealSense相机畸变系数 [k1, k2, p1, p2, k3]
    distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    distCoeffs.at<double>(0) = color_intrin.coeffs[0];
    distCoeffs.at<double>(1) = color_intrin.coeffs[1];
    distCoeffs.at<double>(2) = color_intrin.coeffs[2];
    distCoeffs.at<double>(3) = color_intrin.coeffs[3];
    distCoeffs.at<double>(4) = color_intrin.coeffs[4];

    std::cout << "fx: " << color_intrin.fx << " fy: " << color_intrin.fx << " ppx: " << color_intrin.ppx << " ppy: " << color_intrin.ppy << std::endl;

    // 预计算去畸变映射表
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix,
                                cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_32FC1, map1, map2);

    // 1. 准备原图的中心点 (ppx, ppy)
    // 注意：这里用 double 保证精度
    std::vector<cv::Point2f> src_pts;
    src_pts.push_back(cv::Point2f((float)color_intrin.ppx, (float)color_intrin.ppy));

    std::vector<cv::Point2f> dst_pts;

    // 2. 调用 undistortPoints
    // 这一步会计算：原图的点 -> 去畸变后的归一化坐标
    cv::undistortPoints(src_pts, dst_pts, cameraMatrix, distCoeffs, cv::noArray(), cv::noArray());

    // 3. 将归一化坐标转换回像素坐标
    // 公式：x_pixel = x_norm * fx + cx
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    float new_x = static_cast<float>(dst_pts[0].x * fx + cx);
    float new_y = static_cast<float>(dst_pts[0].y * fy + cy);

    new_center = cv::Point2f(new_x, new_y);

    cout << "Camera D455 init done ..\n";
	cameraStatus = true;
    return true;
}

bool ActD455::Update(void)
{
    cv::TickMeter tm;
    tm.start();
    
    try
    {
        frameSet = pipe.wait_for_frames(1000);  // 别用默认15s
    }
    catch (const rs2::error& e)
    {
        std::cerr << "[RealSense] Update wait_for_frames timeout: "
                  << e.what() << std::endl;
        return false;  
    }

    frameSet = align.process(frameSet);

    rs2::video_frame colorFrame = frameSet.get_color_frame();									//获取颜色帧
    rs2::depth_frame alignedDepthFrame = frameSet.get_depth_frame();							//获取深度帧

    int w = colorFrame.get_width();
    int h = colorFrame.get_height();
    rs2_format fmt = colorFrame.get_profile().format();
    cv::Mat rawColor;
    switch (fmt) {
    case RS2_FORMAT_BGR8:
        rawColor = cv::Mat(cv::Size(w, h), CV_8UC3,
                           (void*)colorFrame.get_data(), cv::Mat::AUTO_STEP);
        srcImage = rawColor.clone();   // 已经是 BGR，直接复制
        break;
    case RS2_FORMAT_RGB8:
        rawColor = cv::Mat(cv::Size(w, h), CV_8UC3,
                           (void*)colorFrame.get_data(), cv::Mat::AUTO_STEP);
        cv::cvtColor(rawColor, srcImage, cv::COLOR_RGB2BGR);
        break;
    case RS2_FORMAT_BGRA8:
        rawColor = cv::Mat(cv::Size(w, h), CV_8UC4,
                           (void*)colorFrame.get_data(), cv::Mat::AUTO_STEP);
        cv::cvtColor(rawColor, srcImage, cv::COLOR_BGRA2BGR);
        break;
    case RS2_FORMAT_RGBA8:
        rawColor = cv::Mat(cv::Size(w, h), CV_8UC4,
                           (void*)colorFrame.get_data(), cv::Mat::AUTO_STEP);
        cv::cvtColor(rawColor, srcImage, cv::COLOR_RGBA2BGR);
        break;
    default:
        std::cerr << "Unsupported color format: " << fmt << std::endl;
        return false;
    }


    // cv::Mat rawImage = cv::Mat(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3, (void*)colorFrame.get_data(), cv::Mat::AUTO_STEP);
    cv::remap(srcImage, srcImage, map1, map2, cv::INTER_LINEAR);  // 彩色图去畸变

    cv::Mat rawDepth = cv::Mat(cv::Size(DEPTH_WIDTH, DEPTH_HEIGHT), CV_16U, (void*)alignedDepthFrame.get_data(), cv::Mat::AUTO_STEP);
    // cv::remap(rawDepth, depthImage, map1, map2, cv::INTER_NEAREST);  // 深度图使用最近邻插值

    // ===== 新增：180°旋转 =====
    cv::flip(srcImage, srcImage, -1);
    cv::flip(rawDepth, rawDepth, -1);
    memcpy(depth_data, rawDepth.data,
        DEPTH_WIDTH * DEPTH_HEIGHT * sizeof(uint16_t));
    {
        std::unique_lock<std::mutex> lock(dep);
        depthImage = rawDepth.clone();
    }

    // memcpy(depth_data, alignedDepthFrame.get_data(), DEPTH_WIDTH*DEPTH_HEIGHT * sizeof(uint16_t));
    tm.stop();
#ifdef TIME
    // cout << "Update time: " << tm.getTimeMilli() << " ms" << endl;
#endif
    return true;
}

void ActD455::KeepLastNBags(const std::string& dir,const std::string& current_bag,size_t keep_count)
{
    namespace fs = std::filesystem;
    try
    {
        std::vector<fs::directory_entry>  bag_files;

        for(const auto& entry : fs::directory_iterator(dir))
        {
            if(entry.is_regular_file())
            {
                auto path =entry.path();

                if(path.extension()== ".bag")
                {
                    if(fs::absolute(path) != fs::absolute(current_bag))
                    {
                        bag_files.push_back(entry);
                    }
                }
            }
        }

        if(bag_files.size()<= keep_count) return;
        
        // 按修改时间排序
        std::sort(bag_files.begin(),bag_files.end(),
        [](const auto& a,const auto& b){return fs::last_write_time(a) < fs::last_write_time(b);}
        );

        size_t remove_num =bag_files.size()- keep_count;

        for(size_t i = 0;i < remove_num;++i)
        {
            std::cout<< "Remove old bag: "<< bag_files[i].path()<< std::endl;
            fs::remove(bag_files[i].path());
        }
    }
    catch(const std::exception& e)
    {
        std::cerr<< "KeepLastNBags failed:"<< e.what()<< std::endl;
    }
}


//======================================================
// getColorTexture
// - Function is utilized to extract the RGB data from
// a single point return R, G, and B values.
// Normals are stored as RGB components and
// correspond to the specific depth (XYZ) coordinate.
// By taking these normals and converting them to
// texture coordinates, the RGB components can be
// "mapped" to each individual point (XYZ).
//======================================================
std::tuple<uint8_t, uint8_t, uint8_t> ActD455::GetColorTexture(rs2::video_frame texture, rs2::texture_coordinate Texture_XY)
{
	//-- Get Width and Height coordinates of texture
	int width = texture.get_width();  // Frame width in pixels
	int height = texture.get_height(); // Frame height in pixels

	//-- Normals to Texture Coordinates conversion
	int xValue = min(max(int(Texture_XY.u * width + .5f), 0), width - 1);
	int yValue = min(max(int(Texture_XY.v * height + .5f), 0), height - 1);

	int bytes = xValue * texture.get_bytes_per_pixel();   // Get # of bytes per pixel
	int strides = yValue * texture.get_stride_in_bytes(); // Get line width in bytes
	int textIndex = (bytes + strides);

	const auto newTexture = reinterpret_cast<const uint8_t*>(texture.get_data());

	//-- RGB components to save in tuple
	int newText1 = newTexture[textIndex];
	int newText2 = newTexture[textIndex + 1];
	int newText3 = newTexture[textIndex + 2];

	return std::tuple<uint8_t, uint8_t, uint8_t>(newText1, newText2, newText3);
}

void ActD455::release(void)
{
    try {
        pipe.stop();  // 停止数据流
        #ifdef RECORD_VIDEO
        // 等 recorder flush 完成
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        #endif
    } catch (...) {}  // 忽略可能的异常
    pipe = rs2::pipeline();  // 重置为默认构造状态，释放内部资源
    
    cameraStatus = false;
    std::cout << "Camera pipeline fully reset" << std::endl;
}

void ActD455::GetCameraParam(rs2_intrinsics& _color_intrin, rs2_intrinsics& _depth_intrin, 
    rs2_extrinsics& _depth2color_extrin, rs2_extrinsics& _color2depth_extrin, uint16_t* _data) const
{
    _color_intrin = color_intrin;
    _depth_intrin = depth_intrin;
    _depth2color_extrin = depth2color_extrin;
    _color2depth_extrin = color2depth_extrin;
}

cv::Point3f ActD455::getPointFromPixel(cv::Point2f pixel)
{
    cv::Point3f re_point(0.f, 0.f, 0.f);

	cv::Mat depthImg;
    {
        std::unique_lock<std::mutex> lock(dep);
        depthImg = depthImage.clone();
    } 

	if(depthImg.empty()) {
        cerr << "Depth image is empty" << endl;
        return re_point;
    }

    const int x = static_cast<int>(std::round(pixel.x));
    const int y = static_cast<int>(std::round(pixel.y));

    if(x < 0 || y < 0 || x >= depthImg.cols || y >= depthImg.rows) {
        cerr << "Pixel coordinates are out of bounds" << endl;
        return re_point;
    }

    try 
    {
        const ushort depth_value = depthImg.at<ushort>(y, x);

        if(depth_value == 0) {
            cerr << "Depth value is zero" << endl;
            return re_point;
        }

        const float depth_meters = depth_value / 1000.0f;
        float point[3];

        float pixel_arr[2] = {pixel.x, pixel.y};
        rs2_deproject_pixel_to_point(point, &color_intrin, pixel_arr, depth_meters);
        
        if(point[2] >= 0.1f) {
            re_point.x = point[0];
            re_point.y = point[1];
            re_point.z = point[2];
        }
    } 
    catch(...) 
    {
		cerr << "没有深度值" << endl;
	}

	return re_point;
}

cv::Point3f ActD455::getPointFromPixel(cv::Point2f pixel, float depth)
{
    float pixel_arr[2] = {pixel.x, pixel.y};
    float point[3];
    rs2_deproject_pixel_to_point(point, &color_intrin, pixel_arr, depth);
    return cv::Point3f(point[0], point[1], point[2]);
}

pPointCloud ActD455::PointCloudGenerateRect(const cv::Rect &roiRect,const cv::Mat &depthimg,int downpick_y,int downpick_x)
{
    cv::TickMeter tm;
    tm.start();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cv::Point2f source_point2d;
    PointType pclpoint;

    for (int i = roiRect.y; i < (roiRect.y + roiRect.height); i = i + downpick_y) 
    {
        for (int j = roiRect.x; j < (roiRect.x + roiRect.width); j = j + downpick_x) 
        {
            source_point2d.x = j;
            source_point2d.y = i;
            
            // 使用深度图像副本计算点
            const int x = static_cast<int>(std::round(source_point2d.x));
            const int y = static_cast<int>(std::round(source_point2d.y));
            
            if(x < 0 || y < 0 || x >= depthimg.cols || y >= depthimg.rows) 
                continue;
            
            const ushort depth_value = depthimg.at<ushort>(y, x);
            if (depth_value == 0) 
                continue;
            
            const float depth_meters = depth_value ; //如需转换为米加上 /1000.0f
            float point[3];
            
            float pixel[2] = {source_point2d.x, source_point2d.y};
            rs2_deproject_pixel_to_point(point, &color_intrin, pixel, depth_meters);
            
            pclpoint.x = point[0];
            pclpoint.y = point[1];
            pclpoint.z = point[2];
            
            if (pclpoint.x != 0 && pclpoint.y != 0 && pclpoint.z != 0) {
                cloud->points.push_back(pclpoint);
            }
        }
    }
    tm.stop();
#ifdef TIME
    // cout << "PointCloudGenerate time: " << tm.getTimeMilli() << " ms" << endl;
#endif
    // cout << "cloud->points.size(): " << cloud->points.size() << endl;
    return cloud;
}

void ActD455::PointCloudGenerateRectandMask(const cv::Rect &roiRect,const cv::Mat &maskimg, const float &lidar_y)
{
    cv::TickMeter tm;
    tm.start();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cv::Point2f source_point2d;
    PointType pclpoint;
    cv::Mat depth;
    int gap_val_width = int(roiRect.width / CLOUD_GENERATE_THRESHOULD + 1);
    int gap_val_height = int(roiRect.height / CLOUD_GENERATE_THRESHOULD + 1);
    {
        std::unique_lock<std::mutex> lock(dep);
        depth = depthImage.clone();
    }
    for (int i = roiRect.y; i < (roiRect.y + roiRect.height); i = i + gap_val_height) 
    {
        for (int j = roiRect.x; j < (roiRect.x + roiRect.width); j = j + gap_val_width) 
        {
            source_point2d.x = j;
            source_point2d.y = i;
            
            // 使用深度图像副本计算点
            const int x = static_cast<int>(std::round(source_point2d.x));
            const int y = static_cast<int>(std::round(source_point2d.y));
            
            if(x < 0 || y < 0 || x >= depth.cols || y >= depth.rows) 
                continue;
            
            if (maskimg.at<uchar>(y,x) !=255) 
                continue;

            const ushort depth_value = depth.at<ushort>(y, x);
            if (depth_value <= abs(lidar_y) - 600) 
                continue;
            
            const float depth_meters = depth_value;
            float point[3];
            
            float pixel[2] = {source_point2d.x, source_point2d.y};
            rs2_deproject_pixel_to_point(point, &color_intrin, pixel, depth_meters);
            
            pclpoint.x = point[0];
            pclpoint.y = point[1];
            pclpoint.z = point[2];
            
            if (pclpoint.x != 0 && pclpoint.y != 0 && pclpoint.z != 0) {
                cloud->points.push_back(pclpoint);
            }
        }
    }
    wallCloud->clear();
    pcl::copyPointCloud(*cloud, *wallCloud);
    // std::cout << "最初提取出的白墙点云数量： " << wallCloud->points.size() << std::endl;
    tm.stop();
#ifdef TIME
    // cout << "白墙点云生成时间: " << tm.getTimeMilli() << " ms" << endl;
#endif
    return ;
}

