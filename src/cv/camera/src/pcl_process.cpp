/*
 * @Author: 於悦洋 yuyueyang2468@163.com
 * @Date: 2026-01-20 16:26:11
 * @LastEditors: 於悦洋 yuyueyang2468@163.com
 * @LastEditTime: 2026-02-02 19:53:50
 * @FilePath: /ROBOCON2026_base/src/cv/act_d455/src/pcl_process.cpp
 * @Description: 
 * 
 * Copyright (c) 2026 by Action, All Rights Reserved. 
 */
#include <pcl/common/centroid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iomanip>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <Eigen/Dense>
#include <deque>

#include "act_d455/pcl_process.hpp"

Pclprocess::Pclprocess(const PclParams& pcl_params) 
{ 
    white_gray_min_ = pcl_params.white_gray_min;
    white_gray_max_ = pcl_params.white_gray_max;
    white_hsv_min_ = pcl_params.white_hsv_min;
    white_hsv_max_ = pcl_params.white_hsv_max;
}
 
cv::Mat Pclprocess::extract_white(cv::Mat& src)
{
	cv::Mat hsv,gray,mask1,mask2;
	cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

	cv::inRange(gray, white_gray_min_, white_gray_max_, mask1);
	cv::inRange(hsv, white_hsv_min_, white_hsv_max_, mask2);

	cv::Mat valid_mask = (mask1 > 0) & (mask2 > 0);  
	
	return valid_mask;
}

pPointCloud Pclprocess::processing_filtered(const pPointCloud& points)
{
    clear_pointcloud(cluster_cloud);
    if (points->points.size() < 100) 
    {
        //std::cout << "OutlierRemoval函数输入点数太少" << endl;
        return points;
    }
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(points);
    sor.setMeanK(30.0);                                //设置用于统计的邻域点数
    sor.setStddevMulThresh(1.0);                       //若某点到其邻居的平均距离 ​​超过全局平均距离 + std_dev_mul × 全局标准差​​，则视为离群点
    sor.setNegative(false);                            //false: 保留主体
    sor.filter(*cluster_cloud);
    return cluster_cloud;
}

pPointCloud Pclprocess::removeRadiusOutliers(pPointCloud& cloud) 
{
    pPointCloud filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
    ror.setInputCloud(cloud);
    ror.setRadiusSearch(0.05);      // 搜索半径
    ror.setMinNeighborsInRadius(5); // 半径内最少邻居数
    ror.filter(*filtered_cloud);
    
    return filtered_cloud;
}


pPointCloud Pclprocess::removeStatisticalOutliers(pPointCloud& cloud) 
{
    pPointCloud filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);           // 每个点分析的邻近点数量
    sor.setStddevMulThresh(1.0); // 标准差倍数阈值
    sor.filter(*filtered_cloud);
    
    return filtered_cloud;
}

pPointCloud Pclprocess::voxelGridDownsample(const pPointCloud& cloud) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(5.f, 5.f, 5.f);
    
    pPointCloud downsampled(new PointCloud);
    vg.filter(*downsampled);
    return downsampled;
}

float Pclprocess::PlaneSegmentation(pPointCloud& cloud_in, pPointCloud& cloud_out, double voxel_size)
{
    if (!cloud_in || cloud_in->points.size() <= 100) 
    {
        PCL_ERROR("Input cloud is empty\n");
        return -1.0; 
    }
    if (!cloud_out) cloud_out = std::make_shared<PointCloud>();
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pPointCloud cloud_filtered(new PointCloud);
    vg.setInputCloud (cloud_in);
    vg.setLeafSize (voxel_size, voxel_size, voxel_size);
    vg.filter (*cloud_filtered);
    // //std::cout << "体素滤波前点数： " << cloud_in->points.size() << "  体素滤波后cloud_filtered: " << cloud_filtered->points.size() <<endl;
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold(20);

    seg.setAxis(Eigen::Vector3f(0.0f, 0.0f, 1.0f));     //限制法向量接近z轴
    seg.setEpsAngle(static_cast<float>(50.0 * M_PI / 180.0));
    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers, *coefficients);
   if (inliers->indices.size() <= 100)
    {
        PCL_ERROR("Could not estimate a planar model for the given dataset.");
        return -1.0;
    }
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);    
    extract.setNegative(false);
    extract.filter(*cloud_out);	

    float A = coefficients->values[0];
    float B = coefficients->values[1];
    float C = coefficients->values[2];
    float D = coefficients->values[3];

    float norm = std::sqrt(A * A + B * B + C * C);
    float distance_mm = std::abs(D) / norm;

    //法向量(A,B,C)与XZ平面的夹角a = arcsin(|B|/sqrt{A^2 + B^2 + C^2})
    float angle_rad = std::asin(std::abs(B) / norm);
    float angle_deg = angle_rad * 180.0f / M_PI;
    if (angle_deg > 8.0f)
    {
        if (!cloud_out->points.empty())
        {
            double sum_z = 0.0;
            for (const auto& pt : cloud_out->points)
            {
                sum_z += pt.z;
            }
            distance_mm = static_cast<float>(sum_z / cloud_out->points.size());
            // std::cout << "夹角大于8度: " << distance_mm << std::endl;
        }
        else
        {
            distance_mm = 0.0f; 
        }
    }
    // std::cout << "\n\n\n\n" << A << " " << B << " " << C << " " << D << "\n\n\n\n";
    return distance_mm;
}

float Pclprocess::fitPlaneAndGetCameraDepth(const pPointCloud& cloud)
{
    // 质心
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud, centroid, covariance);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    Eigen::Vector3f normal = solver.eigenvectors().col(0); // 最小特征值

    normal.normalize();

    float d = -normal.dot(centroid.head<3>());

    // std:://std::cout << "质心深度：" << centroid[2] << "  " << "平面深度" << std::abs(d) << endl;

    return std::abs(d);
}

cv::Mat Pclprocess::filterMaskByDepth(const cv::Mat& black_depth,
                          const cv::Mat& valid_mask,
                          float z_min_mm,
                          float z_max_mm)
{
    CV_Assert(!black_depth.empty());
    CV_Assert(!valid_mask.empty());
    CV_Assert(black_depth.size() == valid_mask.size());
    CV_Assert(valid_mask.type() == CV_8U);

    cv::Mat depth_mask = cv::Mat::zeros(valid_mask.size(), CV_8U);

    if (black_depth.type() == CV_16U)
    {
        for (int y = 0; y < black_depth.rows; ++y)
        {
            const uint16_t* d_ptr = black_depth.ptr<uint16_t>(y);
            const uchar* m_ptr = valid_mask.ptr<uchar>(y);
            uchar* o_ptr = depth_mask.ptr<uchar>(y);

            for (int x = 0; x < black_depth.cols; ++x)
            {
                if (m_ptr[x] > 0)
                {
                    uint16_t z = d_ptr[x];  // mm
                    if (z >= z_min_mm && z <= z_max_mm)
                    {
                        o_ptr[x] = 255;
                    }
                }
            }
        }
    }
    else if (black_depth.type() == CV_32F)
    {
        for (int y = 0; y < black_depth.rows; ++y)
        {
            const float* d_ptr = black_depth.ptr<float>(y);
            const uchar* m_ptr = valid_mask.ptr<uchar>(y);
            uchar* o_ptr = depth_mask.ptr<uchar>(y);

            for (int x = 0; x < black_depth.cols; ++x)
            {
                if (m_ptr[x] > 0)
                {
                    float z = d_ptr[x];  // mm
                    if (z >= z_min_mm && z <= z_max_mm)
                    {
                        o_ptr[x] = 255;
                    }
                }
            }
        }
    }
    else
    {
        CV_Error(cv::Error::StsUnsupportedFormat,
                 "black_depth must be CV_16U or CV_32F");
    }

    return depth_mask;
}

void Pclprocess::add_points(const pPointCloud &cloud_out, const std::string& name)
{
    if (!cloud_out || cloud_out->empty())
    {
        std::cerr << "cloud_out is empty, skip save\n";
        return;
    }

    // 生成文件名
    std::string filename = name + ".csv";

    std::ofstream file(filename, std::ios::trunc);

    if (!file.is_open())
    {
        std::cerr << "cannot open file: " << filename << "\n";
        return;
    }

    for (const auto& pt : cloud_out->points)
    {
        file << pt.x << "," << pt.y << "," << pt.z << "\n";
    }

    file.flush();
    file.close();
}


void Pclprocess::clear_pointcloud(pPointCloud& cloud) 
{
    if (!cloud) {
        std::cerr << "智能指针为空!" << std::endl;
        cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        return;
    }
    cloud->clear();
    cloud->width = 0;
    cloud->height = 1;
    cloud->is_dense = true;
}

Pclprocess::~Pclprocess() 
{
}
