#include <rclcpp/rclcpp.hpp>

#include <act_d455/act_d455.hpp>

#include "base_interfaces/msg/pick.hpp"

#include <atomic>
#include <condition_variable>
#include <climits>
#include <cmath>
#include <mutex>
#include <thread>
#include <numeric>    // 为了 std::iota
#include <random>     // 为了 std::random_device, std::mt19937
#include <algorithm>  // 为了 std::shuffle

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <Eigen/Eigenvalues>

#include <pcl/segmentation/region_growing.h>
#include <pcl/features/normal_3d.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h> 
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/surface/mls.h>  // 移动最小二乘法
#include <pcl/filters/voxel_grid.h>  // 体素网格滤波器
#include <pcl/filters/radius_outlier_removal.h>  // 半径离群点移除



struct plane
{
    float depth;
    int plane_num;
    cv::Point3f center1;
    cv::Point3f center2;
    float angle1;
    float angle2;
};

class Pick : public rclcpp::Node
{
public:
    Pick();
    ~Pick();

    void Input(cv::Mat depth, std::vector<cv::Rect> rects, const std::shared_ptr<ActD455>& d455_);

    void func1(int x, int j);
    
    int RansacPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    void getCentroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    void getDepth(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    void getCoordinate(float x, float y, float yaw);

    void publish();

    float outDepth(){return this->depth_;}

    bool use_depth;
    bool pick_start;
    cv::Rect pick_rect;
    float center_x;
    
    
private:
    // std::shared_ptr<ActD455> d455_;
    //cv::Mat clode_rect_;
    float x_;
    float y_;
    float yaw_;
    std::mutex mtx;
    int plane_num_;
    int close_;
    int close_depth_;
    float depth_;
    std::vector<plane> planes;
    std::vector<pcl::ModelCoefficients> plane_coeffs;
    std::vector<pcl::PointIndices> plane_indices;
    std::vector<Eigen::Vector3f> centroids;
    std::vector<float> angles;
    rclcpp::Publisher<base_interfaces::msg::Pick>::SharedPtr publisher_;
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::PointIndices::Ptr inliers; 
    pcl::ModelCoefficients::Ptr coeff;

    float dealX(float x, float y);
    float dealY(float x, float y);
};