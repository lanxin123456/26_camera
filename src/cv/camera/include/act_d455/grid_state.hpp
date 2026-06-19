/*
 * @Author: 於悦洋 yuyueyang2468@163.com
 * @Date: 2026-01-21 10:47:24
 * @LastEditors: 於悦洋 yuyueyang2468@163.com
 * @LastEditTime: 2026-02-03 10:38:42
 * @FilePath: /ROBOCON2026_base/src/cv/act_d455/include/act_d455/grid_state.hpp
 * @Description: 
 * 
 * Copyright (c) 2026 by Action, All Rights Reserved. 
 */


#include <rclcpp/rclcpp.hpp>

#include <opencv2/opencv.hpp>
#include <trt_yolo/yolo.hpp>
#include <trt_seg/trt_node.hpp>
#include <fstream>
#include <iomanip> 
#include <ctime>

#include <array>
#include "base_interfaces/msg/grid_state.hpp"
#include "base_interfaces/msg/grid_distances.hpp"

#include <mutex>
#include <Eigen/Dense>

#include "camera/params.hpp"

using namespace std;

struct PlaneQuad
{
    std::vector<Eigen::Vector3d> plane_quad_1_;
    std::vector<Eigen::Vector3d> plane_quad_2_;
    std::vector<Eigen::Vector3d> plane_quad_3_;
    std::vector<Eigen::Vector3d> plane_quad_4_;
    std::vector<Eigen::Vector3d> plane_quad_5_;
    std::vector<Eigen::Vector3d> plane_quad_6_;
    std::vector<Eigen::Vector3d> plane_quad_7_;
    std::vector<Eigen::Vector3d> plane_quad_8_;
    std::vector<Eigen::Vector3d> plane_quad_9_;
};

struct PlaneGrid
{
    std::vector<Eigen::Vector3d> plane_grid_1_;
    std::vector<Eigen::Vector3d> plane_grid_2_;
    std::vector<Eigen::Vector3d> plane_grid_3_;
    std::vector<Eigen::Vector3d> plane_grid_4_;
    std::vector<Eigen::Vector3d> plane_grid_5_;
    std::vector<Eigen::Vector3d> plane_grid_6_;
    std::vector<Eigen::Vector3d> plane_grid_7_;
    std::vector<Eigen::Vector3d> plane_grid_8_;
};

static const cv::Vec3b colors[9] = {
    {255, 0, 0},    // 蓝
    {0, 255, 0},    // 绿
    {0, 0, 255},    // 红
    {255, 255, 0},
    {255, 0, 255},
    {0, 255, 255},
    {128, 128, 255},
    {128, 255, 128},
    {255, 128, 128}
};

//存放 2D 像素四边形
struct Quad2D 
{
    bool valid = true; // 是否在相机前方
    std::vector<Eigen::Vector2d> pts;
};

class Grid_State : public rclcpp::Node
{
public:
    std::mutex mtx;
    void getlidar(const float& lidar_x,
                const float& lidar_y,
                const float& lidar_z,
                const float& lidar_yaw,
                const float& lidar_roll,
                const float& lidar_pitch);
    Grid_State(const GridParams& grid_params);

    std::ofstream gride_state;
    int grid_state_[3][3];

    cv::Mat gridMasks();
    cv::Mat generateVirtualPlaneMasks(const std::vector<trt_yolo::det::Object>& objs);
    bool rectCenterInPlane(const cv::Rect_<float>& rect,int plane_id);
    float getx() const {return lidar_x_;};
    std::vector<Quad2D> getProjectedQuads();
    bool isPointInQuad2D(const Eigen::Vector2d& pt, const Unet::Quad2D& quad);
    void sortObjectsOrder_Red(std::vector<trt_yolo::det::Object>& objs);
    float get_x() {return static_cast<float>(cam_pos_w_.x());};
    void publish_state();
    void publish_dist(std::vector<float> distances, const float& nine_square_depth_value, const float& pos_z);

private:

    int field_;
    
    int loss_{0};
    float lidar_x_;
    float lidar_y_;
    float lidar_z_;
    float lidar_yaw_;
    float lidar_roll_;
    float lidar_pitch_;

    float a_;          // 相机在车中心前方 a mm
    float b_;          // 相机在车中心上方 b mm
    float camera_x_;
    float camera_y_;
    float camera_z_;
    Eigen::Vector3d cam_pos_w_; 
    Eigen::Matrix3d R_wb_;
    Eigen::Matrix3d R_bw_;

    int width_;
    int height_;
    double grid_x_{0.0};
    double fx_;
    double fy_;
    double ppx_;
    double ppy_;
    std::vector<Eigen::Vector3d> plane_quad_w_;
    PlaneQuad plane_quads_w_;
    PlaneGrid plane_grid_w_;

    int count_;
    rclcpp::Publisher<base_interfaces::msg::GridState>::SharedPtr publisher_state_;
    rclcpp::Publisher<base_interfaces::msg::GridDistances>::SharedPtr publisher_dist_;
    
private:
    Eigen::Matrix3d eulerZYX(double yaw_deg, double pitch_deg, double roll_deg);
    void computeCameraPosWorld();

    bool pointInQuad(
        const Eigen::Vector3d& P,
        const std::vector<Eigen::Vector3d>& quad,
        const Eigen::Vector3d& normal);
    void thickenQuad(std::vector<Eigen::Vector3d>& quad, double scale);
    void logState(std::vector<float> distances, const float& nine_square_depth_value, const float& pos_z);
};