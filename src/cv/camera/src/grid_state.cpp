/*
 * @Author: 於悦洋 yuyueyang2468@163.com
 * @Date: 2026-01-21 10:45:42
 * @LastEditors: 於悦洋 yuyueyang2468@163.com
 * @LastEditTime: 2026-02-04 19:58:37
 * @FilePath: /ROBOCON2026_base/src/cv/act_d455/src/grid_state.cpp
 * @Description: 
 * 
 * Copyright (c) 2026 by Action, All Rights Reserved. 
 */
#include "act_d455/grid_state.hpp"

Grid_State::Grid_State(const GridParams& grid_params):
    Node("state"),
    width_(640),
    height_(480),
    a_(199+57),
    b_(431+80),
    fx_(384.359),
    fy_(383.817),
    ppx_(321.989),
    ppy_(246.572),
    R_wb_(Eigen::Matrix3d::Identity()),
    cam_pos_w_(160+1500, 10750+60, 960),
    lidar_x_(160+1500),
    lidar_y_(10750+60),
    lidar_z_(960),
    lidar_yaw_(90),
    lidar_pitch_(0),
    lidar_roll_(0),
    grid_state_{{ 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }}
{
    field_ = grid_params.field;
    publisher_state_  = this->create_publisher<base_interfaces::msg::GridState>("d455/gridstate", 1);
    publisher_dist_  = this->create_publisher<base_interfaces::msg::GridDistances>("d455/griddist", 1);
    plane_quad_w_ = {
    {grid_x_, 10500 + loss_, 1860 - loss_},
    {grid_x_, 10500 + loss_, 1360 + loss_},
    {grid_x_, 11000 - loss_, 1360 + loss_},
    {grid_x_, 11000 - loss_, 1860 - loss_}
    };
    if(field_ == 0)
    {
        grid_x_ = 150;
    }
    else if(field_ == 1)
    {
        grid_x_ = -150;
    }
    plane_quads_w_.plane_quad_1_ = {{grid_x_, 9960 + loss_, 2400 - loss_},{grid_x_, 9960 + loss_, 1900 + loss_},{grid_x_, 10460 - loss_, 1900 + loss_},{grid_x_, 10460 - loss_, 2400 - loss_}};
    plane_quads_w_.plane_quad_2_ = {{grid_x_, 10500 + loss_, 2400 - loss_},{grid_x_, 10500 + loss_, 1900 + loss_},{grid_x_, 11000 - loss_, 1900 + loss_},{grid_x_, 11000 - loss_, 2400 - loss_}};
    plane_quads_w_.plane_quad_3_ = {{grid_x_, 11040 + loss_, 2400 - loss_},{grid_x_, 11040 + loss_, 1900 + loss_},{grid_x_, 11540 - loss_, 1900 + loss_},{grid_x_, 11540 - loss_, 2400 - loss_}};
    plane_quads_w_.plane_quad_4_ = {{grid_x_, 9960 + loss_, 1860 - loss_},{grid_x_, 9960 + loss_, 1360 + loss_},{grid_x_, 10460 - loss_, 1360 + loss_},{grid_x_, 10460 - loss_, 1860 - loss_}};
    plane_quads_w_.plane_quad_5_ = {{grid_x_, 10500 + loss_, 1860 - loss_},{grid_x_, 10500 + loss_, 1360 + loss_},{grid_x_, 11000 - loss_, 1360 + loss_},{grid_x_, 11000 - loss_, 1860 - loss_}};
    plane_quads_w_.plane_quad_6_ = {{grid_x_, 11040 + loss_, 1860 - loss_},{grid_x_, 11040 + loss_, 1360 + loss_},{grid_x_, 11540 - loss_, 1360 + loss_},{grid_x_, 11540 - loss_, 1860 - loss_}};
    plane_quads_w_.plane_quad_7_ = {{grid_x_, 9960 + loss_, 1320 - loss_},{grid_x_, 9960 + loss_, 820 + loss_},{grid_x_, 10460 - loss_, 820 + loss_},{grid_x_, 10460 - loss_, 1320 - loss_}};
    plane_quads_w_.plane_quad_8_ = {{grid_x_, 10500 + loss_, 1320 - loss_},{grid_x_, 10500 + loss_, 820 + loss_},{grid_x_, 11000 - loss_, 820 + loss_},{grid_x_, 11000 - loss_, 1320 - loss_}};
    plane_quads_w_.plane_quad_9_ = {{grid_x_, 11040 + loss_, 1320 - loss_},{grid_x_, 11040 + loss_, 820 + loss_},{grid_x_, 11540 - loss_, 820 + loss_},{grid_x_, 11540 - loss_, 1320 - loss_}};

    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);

    gride_state.open("gride_state.csv", std::ios::app);
    if(gride_state.is_open()) gride_state << "\n\n\n\n\n\n"
                                << "时间, " 
                                << std::put_time(localTime, "%Y-%m-%d %H:%M:%S") 
                                << " ================================================" 
                                << endl;
}

void Grid_State::getlidar(const float& lidar_x,
            const float& lidar_y,
            const float& lidar_z,
            const float& lidar_yaw,
            const float& lidar_roll,
            const float& lidar_pitch)
{
    // cout << "--------------------进入里程计回调--------------------" << endl;
    std::lock_guard<std::mutex> lock(mtx);
    {
        lidar_x_ = lidar_x;
        lidar_y_ = lidar_y;
        lidar_z_ = lidar_z;
        lidar_yaw_ = lidar_yaw;
        lidar_roll_ = lidar_roll;
        lidar_pitch_ = lidar_pitch;

        // lidar_x_ = 160+1500;
        // lidar_y_ = 10750+60;
        // lidar_z_ = 940;
        // lidar_yaw_ = 90;
        // lidar_roll_ = 0;
        // lidar_pitch_ = 0;        

        Eigen::Vector3d p_car_w(
        lidar_x_,
        lidar_y_,
        lidar_z_
        );
    // 车体 → 世界
        R_wb_ = eulerZYX(
            lidar_yaw_,
            lidar_pitch_,
            lidar_roll_
        );

        Eigen::Vector3d t_car_cam(-21.0, 199+57-8, 431+80);
        
        cam_pos_w_ = p_car_w + R_wb_ * t_car_cam;
        // cout << "lidar_x_: " << lidar_x_ << " cam_pos_w_.x(): " << cam_pos_w_.x() << endl;
        // if(cam_pos_w_.z() > 800 && cam_pos_w_.z() < 1000) cam_pos_w_.z() = 950.0;
    }
    // computeCameraPosWorld();
    // logState();
}

//按 yaw → pitch → roll 的欧拉角顺序
// yaw  : Z 轴（车在地面转向）
// pitch: X 轴（车头前抬 / 低头）
// roll : Y 轴（左右侧翻）
Eigen::Matrix3d Grid_State::eulerZYX(double yaw_deg, double pitch_deg, double roll_deg)
{
    double yaw   = yaw_deg   * M_PI / 180.0;
    double pitch = pitch_deg * M_PI / 180.0;
    double roll  = roll_deg  * M_PI / 180.0;

    Eigen::AngleAxisd Rz(yaw,   Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd Rx(pitch, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd Ry(roll,  Eigen::Vector3d::UnitY());

    return (Rz * Ry * Rx).toRotationMatrix();
}

// void Grid_State::computeCameraPosWorld() 
// {
//     {
//         std::lock_guard<std::mutex> lock(mtx);
//         Eigen::Vector3d p_car_w(
//             lidar_x_,
//             lidar_y_,
//             lidar_z_
//         );
//     // 车体 → 世界
//         R_wb_ = eulerZYX(
//             lidar_yaw_,
//             lidar_pitch_,
//             lidar_roll_
//         );

//         Eigen::Vector3d t_car_cam(
//             11.0,
//             a_,
//             b_
//         );

//     cam_pos_w_ = p_car_w + R_wb_ * t_car_cam;
//     }
// }

bool Grid_State::pointInQuad(
    const Eigen::Vector3d& P,
    const std::vector<Eigen::Vector3d>& quad,
    const Eigen::Vector3d& normal)
{
    // 利用向量叉积方向一致性
    for (int i = 0; i < 4; ++i)
    {
        const Eigen::Vector3d& A = quad[i];
        const Eigen::Vector3d& B = quad[(i + 1) % 4];

        Eigen::Vector3d edge = B - A;
        Eigen::Vector3d vp   = P - A;

        Eigen::Vector3d c = edge.cross(vp);

        if (c.dot(normal) < 0)
            return false;
    }
    return true;
}

cv::Mat Grid_State::generateVirtualPlaneMasks(const std::vector<trt_yolo::det::Object>& objs) 
{
    cv::Mat state_mask(height_, width_, CV_8UC3, cv::Scalar(0));
    count_ = 0;

    // computeCameraPosWorld();
    // 世界 → 相机旋转 Rcw*转后点 = 转前点
    Eigen::Matrix3d R_bw;
    Eigen::Vector3d cam_pos;
    {
        std::lock_guard<std::mutex> lock(mtx);
        R_wb_ = eulerZYX(
        lidar_yaw_,
        lidar_pitch_,
        lidar_roll_);
        R_bw = R_wb_.transpose();
        cam_pos = cam_pos_w_;
    }
    std::array<std::vector<Eigen::Vector3d>*, 9> quads_w = {
        &plane_quads_w_.plane_quad_1_,
        &plane_quads_w_.plane_quad_2_,
        &plane_quads_w_.plane_quad_3_,
        &plane_quads_w_.plane_quad_4_,
        &plane_quads_w_.plane_quad_5_,
        &plane_quads_w_.plane_quad_6_,
        &plane_quads_w_.plane_quad_7_,
        &plane_quads_w_.plane_quad_8_,
        &plane_quads_w_.plane_quad_9_
    };
    
    // 将平面四点变换到opencv相机坐标系
    std::vector<Eigen::Vector3d> quad_c(4);
    for (int i = 0; i < 4; ++i)
    {
        Eigen::Vector3d Pc_my = R_bw * ((*quads_w[4])[i] - cam_pos);
        quad_c[i] << Pc_my.x(), -Pc_my.z(), Pc_my.y();
    }

    //计算相机坐标系下平面法向
    Eigen::Vector3d n_c =
        (quad_c[1] - quad_c[0]).cross(quad_c[2] - quad_c[0]).normalized();
    // if (n_c.z() > 0) n_c = -n_c;

    // 平面方程：n·X + d = 0
    double d_c = -n_c.dot(quad_c[0]);

    std::array<std::vector<Eigen::Vector3d>, 9> quads_c;
    for (int q = 0; q < 9; ++q)
    {
        quads_c[q].resize(4);
        for (int i = 0; i < 4; ++i)
        {
            Eigen::Vector3d Pc = R_bw * ((*quads_w[q])[i] - cam_pos);
            quads_c[q][i] << Pc.x(), -Pc.z(), Pc.y();
        }
    }
    
    for (int v = 0; v < height_; ++v)
    {
        for (int u = 0; u < width_; ++u)
        {
            // 像素 → 相机射线（相机坐标系）
            Eigen::Vector3d ray;
            ray << (u - ppx_) / fx_,
                   (v - ppy_) / fy_,
                   1.0;

            // 与平面是否平行
            double denom = n_c.dot(ray);
            if (std::abs(denom) < 1e-2)
                continue;

            // 射线参数 t  n_c · (t · ray) + d_c = 0
            double t = -(d_c) / denom;

            // 必须在相机前方
            if (t <= 0)
                continue;

            // 交点
            Eigen::Vector3d P = t * ray;

            // 是否在矩形范围内
            for (int q = 0; q < 9; ++q)
            {
                if (pointInQuad(P, quads_c[q], n_c))
                {
                    state_mask.at<cv::Vec3b>(v, u) = colors[q];
                    break;
                }
            }
        }
    }
    for (const auto& obj : objs)
    {
        int cx = obj.rect.x + obj.rect.width  / 2;
        int cy = obj.rect.y + obj.rect.height / 2;

        // 防止越界（非常重要）
        if (cx < 0 || cx >= width_ || cy < 0 || cy >= height_)
            continue;

        cv::circle(state_mask,cv::Point(cx, cy),3,cv::Scalar(255, 255, 255),-1);
        cv::rectangle(state_mask,obj.rect,cv::Scalar(255, 255, 255),2);
    }
    return state_mask;
}

bool Grid_State::rectCenterInPlane(const cv::Rect_<float>& rect,int plane_id)
{
    if (plane_id < 1 || plane_id > 9)
    {
        cerr << "plane_id: " << plane_id << endl;
        return 0;
    }

    // computeCameraPosWorld();
    Eigen::Matrix3d R_bw;
    Eigen::Vector3d cam_pos;
    {
        std::lock_guard<std::mutex> lock(mtx);
        R_wb_ = eulerZYX(
        lidar_yaw_,
        lidar_pitch_,
        lidar_roll_);
        R_bw = R_wb_.transpose();
        cam_pos = cam_pos_w_;
    }
    // if(gride_state.is_open()) gride_state << "  相机位置：  " << cam_pos.transpose() << endl;

    // === rect 几何中心（像素坐标） ===
    Eigen::Vector2d center_px(
        rect.x + rect.width  * 0.5,
        rect.y + rect.height * 0.5
    );

    // ===  取对应的世界坐标矩形 ===
    std::array<std::vector<Eigen::Vector3d>*, 9> quads_w = {
        &plane_quads_w_.plane_quad_1_,
        &plane_quads_w_.plane_quad_2_,
        &plane_quads_w_.plane_quad_3_,
        &plane_quads_w_.plane_quad_4_,
        &plane_quads_w_.plane_quad_5_,
        &plane_quads_w_.plane_quad_6_,
        &plane_quads_w_.plane_quad_7_,
        &plane_quads_w_.plane_quad_8_,
        &plane_quads_w_.plane_quad_9_
    };

    const auto& quad_w = *quads_w[plane_id - 1];

    std::vector<Eigen::Vector2d> quad_px(4);

    for (int i = 0; i < 4; ++i)
    {
        Eigen::Vector3d Pc =
            R_bw * (quad_w[i] - cam_pos);

        // OpenCV 相机坐标
        double X = Pc.x();
        double Y = -Pc.z();
        double Z = Pc.y();

        if (Z <= 1e-6)
            return 0;  // 在相机后方，直接判否

        // 投影到像素平面
        double u = fx_ * X / Z + ppx_;
        double v = fy_ * Y / Z + ppy_;

        quad_px[i] = Eigen::Vector2d(u, v);
    }

    // === 2D 点是否在四边形内 ===
    for (int i = 0; i < 4; ++i)
    {
        Eigen::Vector2d A = quad_px[i];
        Eigen::Vector2d B = quad_px[(i + 1) % 4];

        Eigen::Vector2d edge = B - A;
        Eigen::Vector2d vp   = center_px - A;

        double cross = edge.x() * vp.y() - edge.y() * vp.x();
        if (cross > 0)
            return false;
    }
    return true;
}

void Grid_State::logState(std::vector<float> distances,const float& nine_square_depth_value, const float& pos_z)
{
    if (!gride_state.is_open())
        return;

    std::ostringstream oss;
    // oss << "  进入里程计回调  ";
    
    // oss << "  pos  = (" << cam_pos_w_.x() << ", "
    //                      << cam_pos_w_.y() << ", "
    //                      << cam_pos_w_.z() << ")";

    // oss << "  rpy  = (" << lidar_roll_ << ", "
    //                      << lidar_pitch_ << ", "
    //                      << lidar_yaw_ << ")";

    oss << " pos_z: " << pos_z << "   Grid State: ";
    for (int i = 0; i < 3; ++i)
    {
        oss << "{";
        for (int j = 0; j < 3; ++j)
        {
            oss << grid_state_[i][j] << " ";
        }
        oss << "}";
        if (i != 2) oss << ", ";
    }

    oss << " distances: [";
    for (int i = 0; i < distances.size(); ++i)
    {
        oss << distances[i] << " ";
        if (i != 2) oss << ", ";
    }
    oss << " ]"  << " 发布的|x|: " << nine_square_depth_value + (199+57-8) + 150 ;

    oss << "\n------------------------------------------\n";

    if (gride_state.is_open())
    {
        gride_state << oss.str();
        gride_state.flush();
    }
}

void Grid_State::publish_state()
{
    auto gridstate_msg = base_interfaces::msg::GridState();

    for(int i = 0; i < 3; i++)
    {
        if(field_ == 0)
        {
            for(int j = 0; j < 3; j++)
            {
                gridstate_msg.grid_state[i*3+j] = grid_state_[i][j];
            }
        }
        else if(field_ == 1)
        {
            for(int j = 2; j > 0; j--)
            {
                gridstate_msg.grid_state[i*3+2-j] = grid_state_[i][j];
            }
        }
    }
    publisher_state_->publish(gridstate_msg);
}

void Grid_State::publish_dist(std::vector<float> distances,const float& nine_square_depth_value, const float& pos_z)
{
    logState(distances, nine_square_depth_value, pos_z);

    auto griddist_msg = base_interfaces::msg::GridDistances();

    if(distances.size() >= 3)
    {
        if(abs(distances[0]) < 2500) griddist_msg.col_first = distances[0];
        if(abs(distances[1]) < 2500) griddist_msg.col_second = distances[1];
        if(abs(distances[2]) < 2500) griddist_msg.col_third = distances[2];
        // if(nine_square_depth_value > 0) griddist_msg.x = nine_square_depth_value + (199+57-8) + 150;
    }
    
    publisher_dist_->publish(griddist_msg);
}

std::vector<Quad2D> Grid_State::getProjectedQuads()
{
    std::vector<Quad2D> quads_2d(9);
    Eigen::Matrix3d R_bw;
    Eigen::Vector3d cam_pos;
    
    // 全局只加这一次锁
    {
        std::lock_guard<std::mutex> lock(mtx);
        R_wb_ = eulerZYX(lidar_yaw_, lidar_pitch_, lidar_roll_);
        R_bw = R_wb_.transpose();
        cam_pos = cam_pos_w_;
    }

    std::array<std::vector<Eigen::Vector3d>*, 9> quads_w = {
        &plane_quads_w_.plane_quad_1_, &plane_quads_w_.plane_quad_2_, &plane_quads_w_.plane_quad_3_,
        &plane_quads_w_.plane_quad_4_, &plane_quads_w_.plane_quad_5_, &plane_quads_w_.plane_quad_6_,
        &plane_quads_w_.plane_quad_7_, &plane_quads_w_.plane_quad_8_, &plane_quads_w_.plane_quad_9_
    };

    for (int i = 0; i < 9; ++i)
    {
        const auto& quad_w = *quads_w[i];
        quads_2d[i].pts.resize(4);
        
        for (int j = 0; j < 4; ++j)
        {
            Eigen::Vector3d Pc = R_bw * (quad_w[j] - cam_pos);
            double X = Pc.x(), Y = -Pc.z(), Z = Pc.y();

            if (Z <= 1e-6) {
                quads_2d[i].valid = false; // 在相机后方，标记为无效
                break; 
            }
            quads_2d[i].pts[j] = Eigen::Vector2d(fx_ * X / Z + ppx_, fy_ * Y / Z + ppy_);
        }
    }
    return quads_2d;
}

bool Grid_State::isPointInQuad2D(const Eigen::Vector2d& pt, const Unet::Quad2D& quad)
{
    if (!quad.valid) return false;
    for (int i = 0; i < 4; ++i)
    {
        // 显式转换 cv::Point2f -> Eigen::Vector2d
        Eigen::Vector2d A(quad.pts[i].x, quad.pts[i].y);
        Eigen::Vector2d B(quad.pts[(i + 1) % 4].x, quad.pts[(i + 1) % 4].y);
        Eigen::Vector2d edge = B - A;
        Eigen::Vector2d vp   = pt - A;
        double cross = edge.x() * vp.y() - edge.y() * vp.x();
        // 假设顶点为逆时针顺序，内部点应满足 cross <= 0
        if (cross > 0) return false;
    }
    return true;
}

void Grid_State::sortObjectsOrder_Red(std::vector<trt_yolo::det::Object>& objs) 
{
    // std::cout << "========== 排序前 ==========\n";
    // for (size_t i = 0; i < objs.size(); ++i) {
    //     std::cout << "Obj[" << i << "] 左上角坐标: (" 
    //               << objs[i].rect.x << ", " << objs[i].rect.y << ")\n";
    // }

    if(field_ == 0)
    {
        std::sort(objs.begin(), objs.end(), [](const trt_yolo::det::Object& a, const trt_yolo::det::Object& b) 
        {
            constexpr float Y_TOLERANCE = 20.0f; 
            
            if (std::abs(a.rect.y - b.rect.y) >= Y_TOLERANCE) {
                return a.rect.y < b.rect.y;
            }
            
            return a.rect.x < b.rect.x;
        });
    }
    else if(field_ == 1)
    {
        std::sort(objs.begin(), objs.end(), [](const trt_yolo::det::Object& a, const trt_yolo::det::Object& b) 
        {
            constexpr float Y_TOLERANCE = 20.0f; 
            
            if (std::abs(a.rect.y - b.rect.y) >= Y_TOLERANCE) {
                return a.rect.y < b.rect.y;
            }
            
            return a.rect.x > b.rect.x;
        });
    }


    // std::cout << "\n========== 排序后 ==========\n";
    // for (size_t i = 0; i < objs.size(); ++i) {
    //     std::cout << "Obj[" << i << "] 左上角坐标: (" 
    //               << objs[i].rect.x << ", " << objs[i].rect.y << ")\n";
    // }
    // cout << "排序完成" << endl;
}