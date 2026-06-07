#include <act_d455/align.hpp>

Align::Align() :
    Node("align")
{
    publisher_  = this->create_publisher<base_interfaces::msg::Align>("d455/align", 10);
}
Align::~Align()
{
}

void Align::getDepth(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr working_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *working_cloud);

    float sum = 0;
    int n = 0;
    for (const auto& point : working_cloud->points)
    {
        sum += point.z;
        n++;
    }
    if(n == 0)
    {
        this->depth = 0;
    }
    else
    {
        this->depth = sum / n;
    }
}



void Align::calculateWideRodCenter()
{
    if (points.size() < 5) return;

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

void Align::getVector(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    if(!cloud|| cloud->empty()) 
    {
        std::cout << "错误：输入点云为空" << std::endl;
        return;
    }

    
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

void Align::publish()
{
    auto align_msg = base_interfaces::msg::Align();

    if(align_start == false)
    {
        align_msg.start = false;
        publisher_->publish(align_msg);
        // std::cout<<"00000000"<<std::endl;
    }
    else
    {
        align_msg.start = true;
        align_msg.should_climb = should_climb;

        align_msg.depth = depth;
        if(use_pca == false)
        {
            align_msg.distance = 450.0f*(align_rect.x + align_rect.width/2 - center_x)/align_rect.width - 11.5;
            // std::cout << "align_msg.distance: " << align_msg.distance << std::endl;
        }
        else
        {
            align_msg.distance = center.x;
            // std::cout << "align_msg.distance: " << align_msg.distance << std::endl;
        }

        if(align_msg.distance > 1200)
        {
            align_msg.distance = 99999.f;
        }
    
        publisher_->publish(align_msg);

    }
}


