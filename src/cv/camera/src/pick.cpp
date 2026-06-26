#include <act_d455/pick.hpp>

Pick::Pick() : 
    Node("pick"),
    use_depth(false), 
    pick_start(false), 
    pick_rect({0, 0, 0, 0}), 
    depth_(0.0f), 
    plane_num_(0),
    close_(0),
    close_depth_(INT_MAX),
    inliers(new pcl::PointIndices),     
    coeff(new pcl::ModelCoefficients)
{
    planes.resize(5); 
    // d455_ = std::make_shared<ActD455>(act_d455_params_);
    publisher_  = this->create_publisher<base_interfaces::msg::Pick>("d455/pick", 10);
    // 配置 seg 
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(20); 
    seg.setProbability(0.99);                  
}

Pick::~Pick()
{
    plane_coeffs.clear();
    plane_indices.clear();
    centroids.clear();
    angles.clear();
}

void Pick::Input(cv::Mat depth, std::vector<cv::Rect> rects, const std::shared_ptr<ActD455>& d455_)
{
    if(rects.empty()) return;
    for(int i = 0, j = 0; i < rects.size(); i++)
    {
        auto rect = rects[i];
        auto cloud = d455_->PointCloudGenerateRect(rect, depth, 2, 2, 0);
        if(!cloud->empty())
        {
            int num = RansacPlane(cloud);
            if(num == 0 || num >= 3)
            {
                continue;
            }
            else
            {
                getDepth(cloud);
                getCentroid(cloud);

                if(centroids[0].y() < 0)
                {
                    continue;
                }
                // cout << j << endl;
                planes[j].depth = depth_;
                if(depth_ <= close_depth_) 
                {
                    close_depth_ = depth_;
                    close_ = j;
                }
                planes[j].plane_num = num;

                if(num == 1)
                {
                    planes[j].center1.x = centroids[0].x();
                    planes[j].center1.y = centroids[0].y();
                    planes[j].center1.z = centroids[0].z();
                    planes[j].angle1 = angles[0];
                }
                else 
                {
                    if(angles[0] < 0 && angles[1] > 0) func1(2, j);
                    else if(angles[1] > 0 && angles[0] < 0) func1(1, j);
                    else if(abs(angles[0]) < abs(angles[1])) func1(1, j);
                    else if(abs(angles[1]) < abs(angles[0])) func1(2, j);
                }
            }

            j++;
        }
    }
    // std::cout << "块的数量" << planes.size() <<  std::endl;
} 

void Pick::func1(int x, int j)
{
    if(x == 1)
    {
        planes[j].center1.x = centroids[0].x();
        planes[j].center1.y = centroids[0].y();
        planes[j].center1.z = centroids[0].z();
        planes[j].angle1 = angles[0];
        planes[j].center2.x = centroids[1].x();
        planes[j].center2.y = centroids[1].y();
        planes[j].center2.z = centroids[1].z();
        planes[j].angle2 = angles[1];
    }
    else if(x == 2)
    {
        planes[j].center1.x = centroids[1].x();
        planes[j].center1.y = centroids[1].y();
        planes[j].center1.z = centroids[1].z();
        planes[j].angle1 = angles[1];
        planes[j].center2.x = centroids[0].x();
        planes[j].center2.y = centroids[0].y();
        planes[j].center2.z = centroids[0].z();
        planes[j].angle2 = angles[0];
    }
}

void Pick::getDepth(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    depth_ = 0;
    float sum = 0.f;
    int n = 0;
    for (const auto& point : cloud->points)
    {
        sum += point.z;
        n++;
    }

    depth_ = sum / n;
}

// void Pick::getDepth(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
// {
//     if (cloud->empty()) {
//         depth = 0.0f;
//         return;
//     }

//     // ✅ 优化：使用中位数代替平均值，防止离群点拉偏深度
//     std::vector<float> z_values;
//     z_values.reserve(cloud->size());
//     for (const auto& point : cloud->points)
//     {
//         if (std::isfinite(point.z) && point.z > 0.1 && point.z < 10.0) 
//             z_values.push_back(point.z);
//     }

//     if (z_values.empty()) {
//         depth = 0.0f;
//         return;
//     }

//     std::sort(z_values.begin(), z_values.end());
//     depth = z_values[z_values.size() / 2]; // 取中位数
// }



int Pick::RansacPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    plane_coeffs.clear();
    plane_indices.clear();

    if(cloud->empty()) 
    {
        std::cout << "点云为空，跳过平面检测" << std::endl;
        return 0;
    }

    // 创建剩余点云的副本用于迭代剔除
    pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZ>(*cloud));
    
    // 初始化原始索引映射：remaining_cloud[i] 对应 original_cloud[original_indices[i]]
    std::vector<int> original_indices;
    original_indices.reserve(cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i) 
    {
        original_indices.push_back(static_cast<int>(i));
    } 

    int plane_count = 0;
    // 使用 size_t 避免有符号/无符号比较警告，且计算更准确
    size_t min_plane_points = static_cast<size_t>(cloud->size() * 0.15);  

    while (remaining_cloud->size() > min_plane_points && plane_count < 6)
    {
        seg.setInputCloud(remaining_cloud);
        seg.segment(*inliers, *coeff);

        // 安全检查：防止 inliers 为空或 coeff 无效
        if (inliers->indices.empty() || inliers->indices.size() <= min_plane_points)
        {
            break; // 未找到足够大的平面，退出循环
        }

        // 1. 映射回原始索引
        pcl::PointIndices original_plane_indices;
        original_plane_indices.indices.reserve(inliers->indices.size());
        
        for (int idx : inliers->indices) 
        {
            // ⚠️ 关键安全检查：防止越界
            if (idx >= 0 && idx < static_cast<int>(original_indices.size())) 
            {
                original_plane_indices.indices.push_back(original_indices[idx]);
            }
        }

        plane_indices.push_back(original_plane_indices);
        plane_coeffs.push_back(*coeff);
        plane_count++;

        // 2. 从 remaining_cloud 中物理移除已提取的点
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*remaining_cloud);

        // 3. 【核心优化】更新索引映射
        // 旧逻辑：分配巨大的 vector<bool> (O(N) 内存 + 初始化开销)，每次循环都做一次
        // 新逻辑：收集被移除的 ID -> 排序 -> 二分查找过滤 (O(M log M + N log M))，内存占用极小
        
        std::vector<int> removed_ids;
        removed_ids.reserve(inliers->indices.size());
        
        // 收集本次被移除的点对应的【原始ID】
        for (int idx : inliers->indices) 
        {
            if (idx >= 0 && idx < static_cast<int>(original_indices.size())) 
            {
                removed_ids.push_back(original_indices[idx]);
            }
        }
        
        // 排序以便进行二分查找
        std::sort(removed_ids.begin(), removed_ids.end()); 

        std::vector<int> new_original_indices;
        new_original_indices.reserve(remaining_cloud->size()); // 预分配，减少重分配
        
        // 遍历当前映射表，只保留不在 removed_ids 中的项
        for (int id : original_indices) 
        {
            // binary_search 时间复杂度 O(log M)，比遍历或大数组标记更快且更安全
            if (!std::binary_search(removed_ids.begin(), removed_ids.end(), id)) 
            {
                new_original_indices.push_back(id);
            }
        }
        
        // 更新映射表
        original_indices = std::move(new_original_indices);
    }
    return plane_count;
    // std::cout << "检测到 " << plane_count << " 个平面" << std::endl;
}

void Pick::getCentroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    angles.clear();
    centroids.clear();
    int j = 0;
    for(int i = 0; i < plane_coeffs.size(); i++)
    {
        Eigen::Vector4f raw_centroid_4f;
        pcl::compute3DCentroid(*cloud, plane_indices[i], raw_centroid_4f);  
        Eigen::Vector3f raw_centroid = raw_centroid_4f.head<3>();
        
        float a = plane_coeffs[i].values[0];
        float b = plane_coeffs[i].values[1];
        float c = plane_coeffs[i].values[2];
        float d = plane_coeffs[i].values[3];
        
        //先计算是否为水平面
        {   
            float norm = sqrt(a*a + b*b + c*c);
            float cos_theta = b / norm;
            cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));
            float theta_rad = acos(cos_theta);
            float theta_deg = theta_rad * 180.0f / M_PI;

            if(fabs(theta_deg) < 10) 
            {
                continue;
            }
            
            j++;
        }

        //计算质心位置
        
        Eigen::Vector3f normal(a, b, c);
        float norm_length = normal.norm();
        
        if (norm_length < 1e-6) 
        {
            std::cerr << "平面法向量长度为0!" << std::endl;
            return;
        }
        
        // 3. 计算点到平面的距离
        float distance = (a * raw_centroid.x() + b * raw_centroid.y() + c * raw_centroid.z() + d) / norm_length;
        
        // 4. 投影到平面上
        Eigen::Vector3f projected_centroid = raw_centroid - distance * normal.normalized();

        // std::cout << "Plane " << j << ": Projected centroid: (" 
        // << projected_centroid.x()  << ", " 
        // << projected_centroid.y()  << ", " 
        // << projected_centroid.z()  << ")" << std::endl;
        
        centroids.push_back(projected_centroid);
        
        //计算与z轴角度
        if(c < 0)
        {
            c = -c;
            a = -a;
        }
        float norm = sqrt(a*a + c*c);
        float cos_theta = c / norm;
        cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));
        float theta_rad = acos(cos_theta);
        float theta_deg = theta_rad * 180.0f / M_PI;

        if(a < 0)
        {
            theta_deg = -theta_deg;
        }

        // std::cout << "Plane " << j << ": Angle: " << theta_deg << " degrees" << std::endl;

        angles.push_back(theta_deg);
    }
    plane_num_ = j;
}

void Pick::getCoordinate(float x, float y, float yaw)
{
    std::lock_guard<std::mutex> lock(mtx);
    x_ = x;
    y_ = y;
    yaw_ = yaw;
}

float Pick::dealX(float x, float y)
{
    float cos_angle = std::cos(yaw_);
    float sin_angle = std::sin(yaw_);

    return x * cos_angle + y * sin_angle + x_;
}

float Pick::dealY(float x, float y)
{
    float cos_angle = std::cos(yaw_);
    float sin_angle = std::sin(yaw_);

    return -x * sin_angle + y * cos_angle + y_;
}
void Pick::publish()
{
    auto pick_msg = base_interfaces::msg::Pick();
    if(!pick_start)
    {
        pick_msg.start = false;
        publisher_->publish(pick_msg);
    }
    else
    {
        pick_msg.start = true;
        if(use_depth || plane_num_ != 0)
        {
            std::lock_guard<std::mutex> lock(mtx);
            pick_msg.depth = true;
            // pick_rect = close_rect_;
            if(planes[close_].plane_num == 1)
            {
                pick_msg.plane_num = 1;
                pick_msg.plane1_centroid_x = dealX(planes[close_].center1.x + 10, (planes[close_].center1.z + 248)); 
                pick_msg.plane1_centroid_y = dealY(planes[close_].center1.x + 10, (planes[close_].center1.z + 248));
                pick_msg.plane1_centroid_z = planes[close_].center1.y;
                pick_msg.plane1_angle = planes[close_].angle1;
                // cout << "distance" << pick_msg.plane1_centroid_x << endl;
                // cout << ((close_rect_.x + close_rect_.width / 2 - center_x) * 350 - 11.5) << endl;
    
                // std::cout << "角度1: " << pick_msg.plane1_angle << std::endl;
                // pick_msg.plane1_pointnum = static_cast<int32_t>(plane_indices[0].indices.size());
                publisher_->publish(pick_msg);
            }
            else if(planes[close_].plane_num == 2)
            {
                pick_msg.plane_num = 2;
                pick_msg.plane1_centroid_x = dealX(planes[close_].center1.x + 10, (planes[close_].center1.z + 248));
                pick_msg.plane1_centroid_y = dealY(planes[close_].center1.x + 10, (planes[close_].center1.z + 248));
                pick_msg.plane1_centroid_z = planes[close_].center1.y;
                pick_msg.plane1_angle = planes[close_].angle1;
                // std::cout << "角度1: " << pick_msg.plane1_angle << std::endl;
                // pick_msg.plane1_pointnum = static_cast<int32_t>(plane_indices[0].indices.size());
                pick_msg.plane2_centroid_x = dealX(planes[close_].center2.x + 10, (planes[close_].center1.z + 248));
                pick_msg.plane2_centroid_y = dealY(planes[close_].center2.x + 10, (planes[close_].center1.z + 248));
                pick_msg.plane2_centroid_z = planes[close_].center2.y;
                pick_msg.plane2_angle = planes[close_].angle2;
                // std::cout << "角度2: " << pick_msg.plane2_angle << std::endl;
                // pick_msg.plane2_pointnum = static_cast<int32_t>(plane_indices[1].indices.size());
                publisher_->publish(pick_msg);
            }
        }
        else
        {
            pick_msg.depth = false;
            pick_msg.distance = (pick_rect.x + pick_rect.width / 2 - center_x) * 350 - 20;
            publisher_->publish(pick_msg);
        }
    }
}

