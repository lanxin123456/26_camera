/*
 * @Author: 於悦洋 yuyueyang2468@163.com
 * @Date: 2026-01-20 20:42:50
 * @LastEditors: 於悦洋 yuyueyang2468@163.com
 * @LastEditTime: 2026-02-02 15:18:50
 * @FilePath: /ROBOCON2026_base/src/cv/act_d455/include/act_d455/pcl_process.hpp
 * @Description: 
 * 
 * Copyright (c) 2026 by Action, All Rights Reserved. 
 */
#define _CRT_SECURE_NO_WARNINGS

#ifndef PCLPROCESS_HPP
#define PCLPROCESS_HPP

#include <cstdio>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h> 
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>

#include <iostream>
#include <iomanip>
#include "camera/params.hpp"

// #define CLOUD_SAVE //保存点云到文件夹
#define CLOUDPATH "./data/cloud"

using namespace std;
using namespace cv;
using namespace Eigen;

typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> PointCloud;
typedef PointCloud::Ptr pPointCloud;

class Pclprocess
{
public:
    // 白墙灰度图阈值
    int white_gray_min_;
    int white_gray_max_;

    // 白墙 HSV 阈值 (使用 cv::Scalar 直接对应 [H, S, V])
    cv::Scalar white_hsv_min_;
    cv::Scalar white_hsv_max_;

    Pclprocess(const PclParams& pcl_params);
    ~Pclprocess();
    cv::Mat filterMaskByDepth(const cv::Mat& black_depth,
                          const cv::Mat& valid_mask,
                          float z_min_mm,
                          float z_max_mm);
    float fitPlaneAndGetCameraDepth(const pPointCloud& cloud);
    cv::Mat extract_white(cv::Mat& src);
    float PlaneSegmentation(pPointCloud & cloud_in, pPointCloud & cloud_out, double voxel_size);
    pPointCloud processing_filtered(const pPointCloud& points);                                    // 统计离群点滤波
    pPointCloud voxelGridDownsample(const pPointCloud& cloud);
    pPointCloud removeRadiusOutliers(pPointCloud& cloud);
    pPointCloud removeStatisticalOutliers(pPointCloud& cloud);
    void clear_pointcloud(pPointCloud& cloud);
    void add_points(const pPointCloud &cloud_out, const std::string& name);
 
private:

    std::ofstream points;
    pPointCloud roi_cloud;
    pPointCloud voxel_cloud;
    pPointCloud cluster_cloud;
    
    std::vector<Eigen::Vector4f> cluster_centroids;
    std::vector<pcl::PointIndices> cluster_indices;
};

#endif