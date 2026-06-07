/*
 * @Author: 於悦洋 yuyueyang2468@163.com
 * @Date: 2026-01-26 14:44:18
 * @LastEditors: 於悦洋 yuyueyang2468@163.com
 * @LastEditTime: 2026-01-29 22:31:00
 * @FilePath: /ROBOCON2026_base/src/cv/act_d455/include/act_d455/pcl_viewer.hpp
 * @Description: 
 * 
 * Copyright (c) 2026 by Action, All Rights Reserved. 
 */
#ifndef PCL_VIEWER_HPP_
#define PCL_VIEWER_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>


class PclViewer
{
public:
    using PointT = pcl::PointXYZ;
    using CloudT = pcl::PointCloud<PointT>::Ptr;
    ~PclViewer() = default;

    void add_cloud(CloudT cloud, std::string cloud_name)
    {
        if(!cloud || cloud->points.empty())
        {
            std::cout << "进入可视化的" << cloud_name << "，点云为空" << std::endl;
            return;
        }
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        clouds_[cloud_name] = cloud;
        viewer_show_ = true;
    }

    void viewer_run()
    {
        if (viewer_show_) 
        {
            viewer_show_ = false;
            std::vector<std::string> names;
            {
                std::lock_guard<std::mutex> lock(cloud_mutex_);
                for(auto& cloud : clouds_)
                {
                    names.push_back(cloud.first);
                }
            }

            for(const auto& name : names)
            {
                CloudT cloud;
                {
                    std::lock_guard<std::mutex> lock(cloud_mutex_);
                    auto it = clouds_.find(name);
                    if(it == clouds_.end())
                    {
                        continue;
                    }
                    cloud = it->second;
                }

                // 创建或获取窗口
                if(viewers_.find(name) == viewers_.end())
                {
                    auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>(name);
                    viewer->setBackgroundColor(0, 0, 0);
                    viewer->addCoordinateSystem(1.0);
                    viewer->initCameraParameters();
                    viewer->setCameraPosition(0, 3.0, 0, 0, 0, 0, 0, 0, 1, 1);
                    viewers_[name] = viewer;
                }

                auto& viewer = viewers_[name];
                if(!cloud || !viewer)
                {
                    continue;
                }

                if(!viewer->updatePointCloud(cloud, name))
                {
                    viewer->addPointCloud(cloud, name);
                    double r = 1.0, g = 1.0, b = 1.0;
                    if (name == "red")
                    {
                        r = 1.0; g = 0.0; b = 0.0;
                    }
                    else if (name == "blue")
                    {
                        r = 0.0; g = 0.0; b = 1.0;
                    }
                    else r = g = b = 1.0;
                    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, name);
                }
                // std::cout << "PCL Viewer 正在显示点云: " << name << ", 点数: " << cloud->points.size() << std::endl;
                viewer->spinOnce();
            }
        }
        
    }

private:
    mutable std::mutex cloud_mutex_;
    bool viewer_show_ = false;
    std::map<std::string, CloudT> clouds_;
    std::map<std::string, std::shared_ptr<pcl::visualization::PCLVisualizer>> viewers_;
};


#endif
