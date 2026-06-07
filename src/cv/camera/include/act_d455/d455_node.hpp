/*
 * @Author: 於悦洋 yuyueyang2468@163.com
 * @Date: 2026-01-16 19:53:02
 * @LastEditors: 於悦洋 yuyueyang2468@163.com
 * @LastEditTime: 2026-02-03 10:05:35
 * @FilePath: /ROBOCON2026_base/src/cv/act_d455/include/act_d455/d455_node.hpp
 * @Description: 
 * 
 * Copyright (c) 2026 by Action, All Rights Reserved. 
 */
#pragma once

#include <rclcpp/rclcpp.hpp>

#include <act_d455/act_d455.hpp>
#include <act_d455/grid_state.hpp>
#include <act_d455/pick.hpp>
#include "act_d455/save_video.hpp"
#include "camera/params.hpp"

#include <trt_yolo/yolo.hpp>
#include <trt_seg/trt_node.hpp>

#include "pcl_process.hpp"
#include <pcl/common/centroid.h>
#include "pcl_viewer.hpp"
#include "align.hpp"
#include "kun.hpp"
#include <act_d455/Buffer.hpp>
#include "base_interfaces/msg/align.hpp"
#include "base_interfaces/msg/grid_state.hpp"
#include "base_interfaces/msg/grid_start.hpp"


#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <fstream>
#include <iomanip>
#include <ctime>
#include <ostream>
#include <cmath>

#include <nav_msgs/msg/odometry.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#define NINE_GRID_THICKNESS 300


class D455Node : public rclcpp::Node
{
public:
    //当只传路径时，options 会自动使用默认值
    explicit D455Node(const std::string& config_path, const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~D455Node() override; 

    void start();
    void stop();
    PclViewer& getViewer() { return cloud_viewer_; }
    PclViewer& getWallViewer() { return wall_cloud_viewer_; }
    // PclViewer& getGanViewer() { return gan_cloud_viewer_; }

private:

    std::vector<float> distances_;
    float real_width_mm[10] = {530,540,530,530,540,530,530,540,530};
    float calculateRealDistance(float a,float b, float x1, float x2) 
    {
        float integral_x2 = 0.5 * a * x2 * x2 + b * x2;
        float integral_x1 = 0.5 * a * x1 * x1 + b * x1;

        return (integral_x2 - integral_x1);
    }
    std::vector<float> fitPixelScale(const std::vector<Unet::Quad2D>& quads) 
    {
        std::vector<cv::Point2f> samples;
        std::vector<float> mid_distance;

        float mid_sumx1=0.0;
        float mid_sumx2=0.0;
        float mid_sumx3=0.0;

        int count1 = 0, count2 = 0, count3 = 0;

        for (int i = 0; i < 9; ++i) {
            if (!quads[i].valid || quads[i].pts.size() < 4)
                continue;

            const auto& q = quads[i];

            // pts 的顺序：0-左上, 1-左下, 2-右下, 3-右上
            float left_x = 0.5f * (q.pts[0].x + q.pts[1].x);
            float right_x = 0.5f * (q.pts[2].x + q.pts[3].x);
            float pixel_width = std::fabs(right_x - left_x);

            // 计算格子的中心X坐标，作为该比例尺对应的横坐标位置
            float center_x = 0.25f * (q.pts[1].x + q.pts[2].x + q.pts[3].x + q.pts[0].x);

            switch(i)
            {
                case 0: case 3: case 6:
                    // mid_sumx1 += (center_x*(1-5.0/real_width_mm[i]));
                    mid_sumx1 += center_x;
                    count1++;
                    break;
                case 1: case 4: case 7:
                    mid_sumx2 += center_x;
                    count2++;
                    break;
                case 2: case 5: case 8:
                    // mid_sumx3 += (center_x*(1+5.0/real_width_mm[i]));
                    mid_sumx3 += center_x;
                    count3++;
                    break;
            }

            float scale = real_width_mm[i] / pixel_width;

            samples.emplace_back(center_x, scale);
        }

        if (samples.size() < 2) 
        {
            std::cerr << "有效采样点太少 (" << samples.size() << ")，无法拟合像素比例尺" << std::endl;
            return { 0.0, 0.0, 0.0 };
        }

        // 线性最小二乘法拟合 (y = ax + b)
        cv::Mat A(samples.size(), 2, CV_32F);
        cv::Mat B(samples.size(), 1, CV_32F);

        for (size_t i = 0; i < samples.size(); ++i) {
            float x = samples[i].x;
            float y = samples[i].y;

            A.at<float>(i, 0) = x;         
            A.at<float>(i, 1) = 1.0;       
            B.at<float>(i, 0) = y;         
        }

        cv::Mat coeff;
        // DECOMP_SVD (奇异值分解) 能有效处理病态矩阵，求解最稳健
        cv::solve(A, B, coeff, cv::DECOMP_SVD);

        //y=ax+b
        float a = coeff.at<float>(0); // 斜率
        float b = coeff.at<float>(1); // 截距

        float avg_x1 = (count1 > 0) ? (mid_sumx1 / count1) : 317.011;
        float avg_x2 = (count2 > 0) ? (mid_sumx2 / count2) : 317.011;
        float avg_x3 = (count3 > 0) ? (mid_sumx3 / count3) : 317.011;

        // ==================== 【1. 新增：打印所有 quads 详细状态与坐标】 ====================
        // std::cout << "\n==================== Quads 原始点及有效性状态 ====================\n";
        // for (int i = 0; i < 9; ++i) {
        //     // 终端高亮显示：true 为绿，false 为红
        //     std::cout << "Quad[" << i << "] valid=" << (quads[i].valid ? "\033[1;32mtrue\033[0m" : "\033[1;31mfalse\033[0m");
            
        //     if (quads[i].pts.size() < 4) {
        //         std::cout << " | \033[1;33m点数不足 (" << quads[i].pts.size() << ")\033[0m\n";
        //     } else {
        //         std::cout << " | 点坐标:\n"
        //                   << "  [0] 左上: (" << quads[i].pts[0].x << ", " << quads[i].pts[0].y << ")\n"
        //                   << "  [1] 左下: (" << quads[i].pts[1].x << ", " << quads[i].pts[1].y << ")\n"
        //                   << "  [2] 右下: (" << quads[i].pts[2].x << ", " << quads[i].pts[2].y << ")\n"
        //                   << "  [3] 右上: (" << quads[i].pts[3].x << ", " << quads[i].pts[3].y << ")\n";
                
        //         // 重新算一下当前宽度与比例尺供 debug 查验
        //         float left_x = 0.5f * (quads[i].pts[0].x + quads[i].pts[1].x);
        //         float right_x = 0.5f * (quads[i].pts[2].x + quads[i].pts[3].x);
        //         float pixel_width = std::fabs(right_x - left_x);
        //         float scale = real_width_mm[i] / pixel_width;
                
        //         std::cout << "  -> 计算值: 像素宽度=" << pixel_width << ", 比例尺(scale)=" << scale << "\n";
        //     }
        //     std::cout << "------------------------------------------------------------------\n";
        // }
        // ==================================================================================

        // std::cout << "count1: " << count1 << " count2: " << count2 << " count3: " << count3 << std::endl;
        
        mid_distance.push_back(calculateRealDistance(a,b, 317.011, avg_x1) - 5.0);
        mid_distance.push_back(calculateRealDistance(a,b, 317.011, avg_x2));
        mid_distance.push_back(calculateRealDistance(a,b, 317.011, avg_x3) + 5.0);

        // std::cout << "\n" << " avg_x1: " << avg_x1 << 
        //         " avg_x2: " << avg_x2 << 
        //         " avg_x3: " << avg_x3 << std::endl;
        for(const auto& d: mid_distance) std::cout << d << " " << "=======\n" << std::endl;
        return mid_distance;
    }

    rclcpp::Subscription<base_interfaces::msg::GridStart>::SharedPtr if_start_unet_;
    std::mutex start_mutex_;

    Eigen::Matrix3d eulerZYX(float yaw_deg, float pitch_deg, float roll_deg)
    {
        float yaw   = yaw_deg   * M_PI / 180.0;
        float pitch = pitch_deg * M_PI / 180.0;
        float roll  = roll_deg  * M_PI / 180.0;

        Eigen::AngleAxisd Rz(yaw,   Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd Rx(pitch, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd Ry(roll,  Eigen::Vector3d::UnitY());

        return (Rz * Ry * Rx).toRotationMatrix();
    }

    int start_unet_{1};


    //-----------------------------
    std::vector<Unet::Quad2D> quads_;
    std::chrono::steady_clock::time_point quads_timestamp_;
    struct QuadsData{
        std::vector<Unet::Quad2D> quads;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::vector<QuadsData> quads_buffer_;
    std::mutex buffer_mutex_;
    const size_t max_buffer_size_ = 5;
    std::vector<Unet::Quad2D> getNearestQuads(std::chrono::steady_clock::time_point target_time);
    //-----------------------------


    std::shared_ptr<TRTNode> trt_seg_;
    Eigen::Matrix3d R_wb_;
    Eigen::Matrix3d R_bw_;
    Eigen::Vector3d cam_pos_;

    std::mutex mut_pos_;
    
    cv::Mat latest_img_state_;

    void task_callback(const base_interfaces::msg::GridStart::SharedPtr msg);

private:
    int field_;
    bool default_display_ = true;
    bool show_params_ = true;
    bool show_test_ = true;

    PclParams pcl_params_;
    GridParams grid_params_;
    ActDParams act_d455_params_;
    void loadYamlConfig(const std::string& path);

    std::ofstream d455_log_;
    struct Dataframe
    {
        Dataframe() = default;                   
        ~Dataframe() = default;
        Dataframe(const Dataframe& other);                 //Dataframe b = a;   Dataframe b(a); 
        Dataframe& operator=(const Dataframe& other);      //Dataframe b; b = a;
        Dataframe(Dataframe&& other);                      //Dataframe c = std::move(a);
        Dataframe& operator=(Dataframe&& other);           //Dataframe b; b = std::move(a);
        cv::Mat src, depth;
        //int kfs_mode;
        //int gan_mode;
    };
    void captureLoop();
    void process_unet_Loop();
    void process_yolo_Loop();
    void process_wall_Loop();

    void process_state_Loop();
    void displayLoop();
    void pickLoop();
    void alignLoop();
    
    //int kfs_mode = 0;
    //int gan_mode = 0;
    std::shared_ptr<ActD455> d455_;
    std::shared_ptr<Pclprocess> pcl_;
    std::shared_ptr<Grid_State> gstate_;
    std::shared_ptr<Align> align_;
    std::shared_ptr<Pick> pick_;
    
    DealImg dealImg_;

    std::mutex disp_mutex_;
    std::mutex disp_trt_mutex_;
    cv::Mat disp_kfs_;
    cv::Mat disp_wall_;
    cv::Mat disp_src_;

    std::shared_ptr<cv::Mat> shared_src_ptr_;
    std::shared_ptr<cv::Mat> shared_depth_ptr_;

    cv::Mat src_;
    cv::Mat src_unet_;
    cv::Mat depth_;
    cv::Mat disp_;
    cv::Mat latest_depth_state_;
    cv::Mat white_mask_;
    cv::Mat show_wall_;

    Dataframe dataframe_align_;
    buffer::DataChannel<Dataframe> align_buffer_;
    Dataframe dataframe_pick_;
    buffer::DataChannel<Dataframe> pick_buffer_;
    
    PclViewer cloud_viewer_;
    PclViewer wall_cloud_viewer_;
    // PclViewer gan_cloud_viewer_;
    
    std::condition_variable get_wall_or_grid_cloud_;
    std::condition_variable get_frame_;
    std::condition_variable get_frame_unet_;

    std::condition_variable get_wall_;
    std::condition_variable get_kfs_;
    std::condition_variable final_state_;

    std::mutex frame_mutex;
    std::mutex frame_unet_mutex;

    std::mutex compute_wall_or_grid_;
    std::mutex compute_state_;
    std::mutex gan_mut_;
    std::mutex wall_mut_;
    std::mutex kfs_mut_;
    std::mutex lidar_;

    bool is_identify_KFS_{true};
    
    bool start_compute_{false};//计算九宫格深度
    
    bool has_frame_{false};
    bool has_frame_unet_{false};

    bool has_obj_kfs_{false};
    bool has_float_{false};

    std::atomic<bool> done_kfs_{true};
    std::atomic<bool> done_wall_{true};
    std::atomic<bool> retain_{true};//防止在等 has_float_ 时 objs_kfs_ 更新
    std::atomic<bool> detect_obj_wall_{false};
    std::atomic<int>  cloud_state_{0};//0：用雷达的数算 1：得到wallCloud 2：得到gridCloud

    std::atomic<bool> running_{true};

    
    std::thread capture_thread_;
    std::thread process_unet_grid_thread_;

    std::thread process_yolo_kfs_thread_;
    std::thread process_yolo_wall_thread_;
    
    std::thread process_state_thread_;
    std::thread display_thread_;

    std::thread align_thread_;

    std::thread pick_thread_;

    bool display_{true};

    bool save_frames_{false};
    bool save_video_{true};

    std::string frames_dir_{"frames"};
    std::string frame_ext_{"png"};

    trt_yolo::YOLOv8Config config_;
    trt_yolo::YOLOv8 *yolo_detector_;
    
    pPointCloud kfs_show_cloud;
    pPointCloud gan_show_cloud;

    int grid_pre_state_[3][3]={0};
    int grid_state_[3][3]={0};
    int count = 0;

    cv::VideoWriter video_writer;
	cv::Size frame_size{640, 480};

    rclcpp::TimerBase::SharedPtr timer_1;
    rclcpp::TimerBase::SharedPtr timer_2;
    rclcpp::Publisher<base_interfaces::msg::Align>::SharedPtr publisher_1;
    //rclcpp::Subscription<base_interfaces::msg::AlignStart>::SharedPtr d455_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    
    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
    //void d455_callback(const base_interfaces::msg::AlignStart::ConstSharedPtr msg);
    float lidar_x_{1500.f};
    float lidar_y_{1500.f};
    float lidar_z_;
    float lidar_x_pick_;
    float lidar_y_pick_;
    float lidar_yaw_;
    float lidar_roll_;
    float lidar_pitch_;
    float nine_square_depth_value_;
    float dispersion_threshold_;        //kfs点云方差阈值
    
    std::vector<trt_yolo::det::Object> objs_kfs_red_state_;
    std::vector<trt_yolo::det::Object> objs_kfs_blue_state_;
    std::vector<trt_yolo::det::Object> objs_kfs_red_r1_;
    std::vector<trt_yolo::det::Object> objs_kfs_red_r2_;
    std::vector<trt_yolo::det::Object> objs_kfs_blue_r1_;
    std::vector<trt_yolo::det::Object> objs_kfs_blue_r2_;

    std::vector<trt_yolo::det::Object> objs_kfs_red_;
    std::vector<trt_yolo::det::Object> objs_kfs_blue_;
    std::vector<trt_yolo::det::Object> objs_gw_gan1_;//绿白杆
    std::vector<trt_yolo::det::Object> objs_gy_gan2_;//绿黄杆
    std::vector<trt_yolo::det::Object> objs_wall_;

    SaveVideoD video_saver_;

    // 位姿信息
    struct pos_data_t
    {
        float x;            // x坐标
        float y;            // y坐标
        float z;            // z坐标 
        float vx;           // x速度
        float vy;           // y速度
        float vz;           // z速度
        float omega;        // 角速度
        float pitch;        // 俯仰角
        float roll;         // 翻滚角
        float yaw;          // 航向角
        float a1;           // 加速度x
        float a2;           // 加速度y
        float a3;           // 加速度z
    };
    pos_data_t nowPos;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_highfreq_subscriber_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

private:
    cv::Mat keep_roi(const cv::Mat& src, const cv::Rect& roi); 
    float rectangle_depth(const cv::Rect &roiRect , const cv::Mat &depthimg, std::ostringstream& oss, int& row, int& col);
    float computeDepthByMode(int current_mode, float lidar_x, pPointCloud& wall_cloud_out);
};