#include <rclcpp/rclcpp.hpp>

#include <act_d455/act_d455.hpp>

#include <cmath>
#include "base_interfaces/msg/align.hpp"
#include <vector>
#include <filesystem> 
#include <fstream>
#include <atomic>
#include <condition_variable>
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

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/surface/mls.h>  // 移动最小二乘法
#include <pcl/filters/voxel_grid.h>  // 体素网格滤波器
#include <pcl/filters/radius_outlier_removal.h>  // 半径离群点移除
namespace fs = std::filesystem;

class Align : public rclcpp::Node
{
public:
    Align() ;

    ~Align();

    struct Point3D 
    { 
        float x, y, z; 
    };
    
    void getDepth(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    void removeStatisticalOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float stddev_mult);
    void calculateWideRodCenter();
    void getVector(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    void publish();
    float outDepth(){return depth;}

    cv::Rect align_rect;
    bool align_start;
    bool use_pca;
    bool should_climb;
    float center_x;
    bool R1_is_putting;

private:
    float depth; 
    std::vector<Point3D> points;
    Point3D center;
    rclcpp::Publisher<base_interfaces::msg::Align>::SharedPtr publisher_;

    std::ofstream currentLogFile;

    void saveDataToFile(std::ofstream& outFile, float value1, float value2);
    std::string generateLogFilePath(const std::string& logDirName);
};