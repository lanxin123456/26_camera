#ifndef ACT_D455_HPP
#define ACT_D455_HPP

// std
#include <iostream>
#include <fstream>
#include <mutex>
#include <thread>
#include <sstream>
#include <cmath>
#include <ctime>
#include <numeric>
#include <cmath>
#include <memory>
#include <atomic>
#include <vector>
#include <string>
#include <queue>
#include <cstdint>
// unistd
#include <signal.h>
#include <unistd.h>
// eigen
#include <Eigen/Dense>
// realsense
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <librealsense2/hpp/rs_sensor.hpp>
#include <librealsense2/rs_advanced_mode.hpp>
#include <librealsense2/rs_advanced_mode.h>
// pcl
#include <pcl/point_types.h>
#include <pcl/memory.h>
#include <pcl/pcl_base.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "pcl_process.hpp"
#include "Buffer.hpp"

#include <filesystem>

#define DEBUG        	//测试员模式
#define RECORD_VIDEO	//过程录像
#define CAPTURE
#define TXT 			//运行数据记录
#define TIME			//程序计时
// #define IFCAMERA		//已连接相机
#define COMMUN			//进程通信

// #define PCL
#define CLOUD_GENERATE_THRESHOULD 300	//点云下采样的阈值

#define IMAGE_WIDTH 	640
#define IMAGE_HEIGHT 	480
#define DEPTH_WIDTH 	640
#define DEPTH_HEIGHT 	480

// #define BAG_DIR "/home/lx/bag_531/d455_1780191820.bag"

typedef pcl::PointXYZ 			    PointType;
typedef pcl::PointCloud<PointType> 	PointCloud;
typedef PointCloud::Ptr 		    pPointCloud;

//-- ROI of an object
typedef struct
{
	double xMin;
	double xMax;

	double yMin;
	double yMax;

	double zMin;
	double zMax;

} ObjectROI;

class ActD455
{
public:
    ActD455(const ActDParams& act_d455_params);
    ActD455(const ActD455&) = delete;
    ActD455& operator=(const ActD455&) = delete;
    ~ActD455();

    bool Init(void);
    bool Update(void);
    const bool GetCameraStatus(void) const { return cameraStatus; };

    const float GetNewCenterX(void) const { return new_center.x; }; 

    const cv::Mat GetSrcImage(void) const { return srcImage; };
    const cv::Mat GetDepthImage(void) const { return depthImage; };
    uint16_t* GetData(void) const { return depth_data; };
    void GetCameraParam(rs2_intrinsics& _color_intrin, rs2_intrinsics& _depth_intrin, rs2_extrinsics& _depth2color_extrin, 
						rs2_extrinsics& _color2depth_extrin, uint16_t* _data) const;
	void release(void);
    cv::Point3f getPointFromPixel(cv::Point2f pixel);
    cv::Point3f getPointFromPixel(cv::Point2f pixel, float depth);

    std::tuple<uint8_t, uint8_t, uint8_t> GetColorTexture(rs2::video_frame texture, rs2::texture_coordinate Texture_XY);

    pPointCloud GetSrcCloud(void) const { return srcCloud; };
	pPointCloud GetWallCloud(void) const { return wallCloud; };
    pPointCloud PointCloudGenerateRect(const cv::Rect &roiRect,const cv::Mat &depthimg,int downpick_y,int downpick_x, const float& nine_square_depth_value);
	void PointCloudGenerateRectandMask(const cv::Rect &roiRect,const cv::Mat &maskimg, const float &lidar_y);
	
	// float depth_rect(const cv::Rect &roiRect, const string &color);
	// float nine_square_depth(cv::Mat &depthImg);
	rs2::pipeline    pipe;					//数据传输管道
private:
	std::string bag_path_;
    bool cameraStatus;
    uint16_t *depth_data;
    rs2::context ctx;                       //用于查询已连接的 RealSense 设备列表
	size_t device_count;	
	rs2_intrinsics color_intrin;			//颜色相机内参
	rs2_intrinsics depth_intrin;			//深度相机内参
	rs2_extrinsics depth2color_extrin;		//深度向颜色外参
	rs2_extrinsics color2depth_extrin;
	rs2::pointcloud  rs2Cloud;				//深度图计算得到的点云 RS2指realsense2系列相机
	rs2::points      rs2Points;				//点云格式的点
	rs2::align       align;					//对齐图像
	
	rs2::config      cfg;					//初始化
	rs2::frameset    frameSet;				//帧设置
	cv::Mat 		 srcImage;				//源图片，彩色图片source image
	cv::Mat          depthImage;		    //深度图
	cv::Mat         cameraMatrix;			//相机内参矩阵
	cv::Mat         distCoeffs;				//畸变系数
	cv::Mat         map1, map2;				//去畸变映射表

	rs2::temporal_filter tem_filter;		//时间过滤器
	rs2::spatial_filter spat_filter;		//空间过滤器

    cv::Point2f new_center;
    pPointCloud srcCloud;
	pPointCloud wallCloud;
	pPointCloud src_Cloud;
	
	std::mutex src_cloud_mutex_;
	std::mutex grid_cloud_mutex_;
	std::mutex dep;

	void KeepLastNBags(const std::string& dir,const std::string& current_bag,size_t keep_count);
};

#endif // ACT_D455_HPP