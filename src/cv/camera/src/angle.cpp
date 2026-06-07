#include <act_d455/angle.hpp>

#include <cmath>

#include <vector>

Angle::~Angle()
{
}

float Angle::depth_mode(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr working_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *working_cloud);

    float sum;
    int n;
    for (const auto& point : working_cloud->points)
    {
        sum += point.z;
        n++;
    }

    return sum / n;
}

float Angle::RANSAC_angle(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    std::cout << "进入RANSAC平面角度测量" << std::endl;
    
    // 检查点云是否为空
    if(!cloud || cloud->empty()) 
    {
        std::cout << "错误：输入点云为空" << std::endl;
        return 99999.f;
    }
    
    std::cout << "点云数量: " << cloud->size() << std::endl;
    
    // 创建一个工作点云，避免修改原始点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr working_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *working_cloud);
    
    // 计算点云范围，用于自适应距离阈值
    pcl::PointXYZ min_pt, max_pt;
    min_pt = working_cloud->points[0];
    max_pt = working_cloud->points[0];
    
    for(const auto& point : working_cloud->points) 
    {
        if(point.x < min_pt.x) min_pt.x = point.x;
        if(point.y < min_pt.y) min_pt.y = point.y;
        if(point.z < min_pt.z) min_pt.z = point.z;
        if(point.x > max_pt.x) max_pt.x = point.x;
        if(point.y > max_pt.y) max_pt.y = point.y;
        if(point.z > max_pt.z) max_pt.z = point.z;
    }
    
    float cloud_size = std::max({max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z});
    float distance_threshold = cloud_size * 0.01f;  // 自适应距离阈值
    distance_threshold = std::max(distance_threshold, 0.001f);  // 最小1mm
    distance_threshold = std::min(distance_threshold, 0.01f);   // 最大1cm
    
    std::cout << "点云尺寸: " << cloud_size << "m, 距离阈值: " << distance_threshold << "m" << std::endl;
    
    float angle_deg = 99999.f;
    
    // 内存分配
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    
    // 设置参数和模式 - 改为拟合平面
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);  // 改为平面模型
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distance_threshold);
    seg.setMaxIterations(10000);
    
    // 执行分割
    seg.setInputCloud(working_cloud);
    seg.segment(*inliers, *coefficients);
    
    Eigen::Vector3f normal;
    // 检查内点数量
    if(inliers->indices.size() > 0)
    {
        float inlier_ratio = static_cast<float>(inliers->indices.size()) / working_cloud->points.size();
        
        std::cout << "平面拟合质量: " << inlier_ratio * 100 << "%" << std::endl;
        
        // 错误识别返回
        if(inlier_ratio < 0.1)
        {
            std::cout << "内点数量太少，结果不可靠" << std::endl;
            return 99999.f;
        }
        
        // 解析平面法向量
        // 平面方程: ax + by + cz + d = 0
        // 法向量: (a, b, c)
        float a = coefficients->values[0];
        float b = coefficients->values[1];
        float c = coefficients->values[2];
        float d = coefficients->values[3];
        
        normal = Eigen::Vector3f(a, b, c);
        
        // 计算法向量的模长
        float normal_length = normal.norm();
        if(normal_length < 0.001f) 
        {
            std::cout << "错误：法向量长度异常" << std::endl;
            return 99999.f;
        }
        
        // 归一化法向量
        normal.normalize();
        // std::cout << "归一化法向量: (" << normal[0] << ", " << normal[1] << ", " << normal[2] << ")" << std::endl;
        
        // 计算法向量在XZ平面上的投影
        Eigen::Vector2f projection(normal[0], normal[2]);  // 忽略Y分量
        
        // 计算投影向量的模长
        float projection_length = projection.norm();
        
        if (projection_length < 0.001f) 
        {
            // 法向量几乎垂直，在XZ平面上投影很小
            std::cout << "法向量接近垂直，在XZ平面上投影太小" << std::endl;
            
            // 检查是否为水平面
            if(std::abs(normal[1]) > 0.9f) 
            {
                // std::cout << "检测到水平面" << std::endl;
                // 水平面的法向量垂直向上/向下，在XZ平面上投影为0
                // 可以根据需要处理，这里返回0
                return 99999.f;
            }
            else 
            {
                std::cout << "警告：法向量异常" << std::endl;
                return 99999.f;
            }
        }
        
        // 归一化投影向量
        projection.normalize();

        //把南向向量修正为向北
        if(projection[1] < 0)
        {
            projection = - projection;
        }

        // 计算角度（弧度）
        float angle_rad = atan2(projection[0],projection[1]);
        
        // 转换为角度
        // 若为正数，车需要往右边转，反之，向左转
        angle_deg = angle_rad * 180.0f / M_PI;


        // if(inlier_ratio * 100 < 99) 
        // {
        //     return 0.0f;
        // }
    }
    else
    {
        std::cout << "未找到平面" << std::endl;
    }

    return angle_deg;
}

void Angle::removeStatisticalOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float stddev_mult = 1.0)
{
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(stddev_mult);
    sor.filter(*cloud);
}

void Angle::calculateWideRodCenter()
{
    if (points.size() < 5) return ;

    // 构建数据矩阵
    cv::Mat data(points.size(), 3, CV_32F);
    for (size_t i = 0; i < points.size(); ++i) 
    {
        data.at<float>(i, 0) = points[i].x;
        data.at<float>(i, 1) = points[i].y;
        data.at<float>(i, 2) = points[i].z;
    }

    // 执行 PCA，保留 2 个主成分
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 2);

    cv::Mat mean = pca.mean;          
    cv::Mat eigenvectors = pca.eigenvectors; 

    cv::Vec3f center_mean(mean.at<float>(0,0), mean.at<float>(0,1), mean.at<float>(0,2));
    cv::Vec3f axis_long(eigenvectors.at<float>(0,0), eigenvectors.at<float>(0,1), eigenvectors.at<float>(0,2)); 
    cv::Vec3f axis_wide(eigenvectors.at<float>(1,0), eigenvectors.at<float>(1,1), eigenvectors.at<float>(1,2)); 

    float min_u = std::numeric_limits<float>::max(), max_u = std::numeric_limits<float>::lowest();
    float min_v = std::numeric_limits<float>::max(), max_v = std::numeric_limits<float>::lowest();

    for (const auto& p : points) 
    {
        cv::Vec3f vec(p.x - center_mean[0], p.y - center_mean[1], p.z - center_mean[2]);
        float u = vec.dot(axis_long);
        float v = vec.dot(axis_wide);

        if (u < min_u) min_u = u;
        if (u > max_u) max_u = u;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }

    float center_u = (min_u + max_u) * 0.5f;
    float center_v = (min_v + max_v) * 0.5f;

    cv::Vec3f final_center = center_mean + axis_long * center_u + axis_wide * center_v;

    center = {final_center[0], final_center[1], final_center[2]};
    points.clear();
}

void Angle::getVector(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    if(!cloud|| cloud->empty()) 
    {
        std::cout << "错误：输入点云为空" << std::endl;
        return;
    }

    std::cout << "点云数量: " << cloud->size() << std::endl;
    
    // 创建一个工作点云，避免修改原始点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr working_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *working_cloud);

    points.reserve(working_cloud->size()); // 预分配内存，提高效率

    for (const auto& p : working_cloud->points) 
    {
        // 可选：过滤无效点 (NaN 或 无穷大)
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) 
        {
            continue;
        }
        points.push_back({p.x, p.y, p.z});
    }
}


