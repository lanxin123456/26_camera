/*
 * @Author: 於悦洋 yuyueyang2468@163.com
 * @Date: 2026-01-27 17:18:53
 * @LastEditors: 於悦洋 yuyueyang2468@163.com
 * @LastEditTime: 2026-03-07 19:58:49
 * @FilePath: /ROBOCON2026_base/src/cv/act_d455/include/act_d455/angle.hpp
 * @Description: 
 * 
 * Copyright (c) 2026 by Action, All Rights Reserved. 
 */
#include <rclcpp/rclcpp.hpp>

#include <act_d455/act_d455.hpp>

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


class Angle
{
public:
    Angle() 
    {
    }
    ~Angle();

    struct Point3D 
    { 
        float x, y, z; 
    };
    std::vector<Point3D> points;

    float RANSAC_angle(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    float depth_mode(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    void removeStatisticalOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float stddev_mult);
    void calculateWideRodCenter();
    void getVector(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

    Point3D center;
};