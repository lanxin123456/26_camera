#ifndef TRT_NODE_HPP_
#define TRT_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <chrono>

#include "trt_seg/cuda_kernels.hpp"
#include <Eigen/Dense>

namespace Unet {
    // 预先声明外部或后续用到的几何结构体
    struct Quad2D {
        std::vector<cv::Point2f> pts;
        bool valid = true;
    };
}

// ==================== 全局配置常量 ====================
// 【修改】移除 BATCH 和 PATCH，替换为全图推理所需的尺寸常量
static constexpr int IMG_W = 960;         // 原始图宽
static constexpr int IMG_H = 720;         // 原始图高
static constexpr int NET_W = 960;         // 网络输入宽
static constexpr int NET_H = 736;         // 网络输入高 (包含16像素白边)

static constexpr float THRESH = 0.5f;

static const std::string FAREM_PATH  = "/home/action/code/ROBOCON2026_CHANLLENGE/src/cv/output_526_1280x960.avi";

// ==================== TensorRT 日志类 ====================
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

class TRTNode : public rclcpp::Node {
public:
    TRTNode(const std::string& engine_path);
    ~TRTNode();
    std::vector<Unet::Quad2D> getgridquads(const cv::Mat& frame,
                                    Eigen::Matrix3d R_wb_, 
                                    Eigen::Vector3d cam_pos_);
    void detect(const cv::Mat& frame);
    float Getdep() const { return grid_depth_; };
    cv::Mat Getcanvas()  {
        std::lock_guard<std::mutex> lock(mat_mutex_); 
        return canvas_show_.clone(); 
    }
    cv::Mat Getmask()  {
        std::lock_guard<std::mutex> lock(mat_mutex_); 
        return mask_show_.clone(); 
    }

    cv::Mat Getmerged_mask()  {
        std::lock_guard<std::mutex> lock(mat_mutex_); 
        return merged_mask_show_.clone(); 
    }
private:
    cv::Mat resized_host_; // 用于存放缩放后的图像
    bool is_registered_ = false; // 标记是否已注册

    float grid_depth_ = -1.0f;
    std::string engine_path_;  

    cv::Mat mask_show_;
    cv::Mat merged_mask_show_;
    cv::Mat canvas_show_;

    cv::Mat mask_;
    cv::Mat canvas_;
    cv::Mat merged_mask_;
    mutable std::mutex mat_mutex_; 

    float sim_avg_step_v_{100};
    float sim_avg_step_h_{100};

    std::vector<Unet::Quad2D> quads_;
    Eigen::Matrix3d R_wb_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d cam_pos_ = Eigen::Vector3d::Zero();
    
    Eigen::Matrix3d eulerZYX(double yaw_deg, double pitch_deg, double roll_deg)
    {
        double yaw   = yaw_deg   * M_PI / 180.0;
        double pitch = pitch_deg * M_PI / 180.0;
        double roll  = roll_deg  * M_PI / 180.0;

        Eigen::AngleAxisd Rz(yaw,   Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd Rx(pitch, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd Ry(roll,  Eigen::Vector3d::UnitY());

        return (Rz * Ry * Rx).toRotationMatrix();
    }

    void computeCameraPosWorld() 
    {
            Eigen::Vector3d p_car_w(
                1800,
                11285,
                800
            );
        // 车体 → 世界
            R_wb_ = eulerZYX(
                90,
                0,
                0
            );

            Eigen::Vector3d t_car_cam(
                11.0,
                198.35,
                468
            );

        cam_pos_ = p_car_w + R_wb_ * t_car_cam;
    }

    // 【修改】由于不再需要分块滑窗，移除 void build_windows();
    void init_trt();
    void release();

    struct LineCandidate {
        std::vector<cv::Point> points;
        float intercept = 0.0f;    // 竖线为中心Y轴切出的X截距，横线为中心X轴切出的Y截距
        float angle = 0.0f;        // 直线弧度角
        cv::Point2f pt1;    
        cv::Point2f pt2;    
    };

    struct TrackedLine {
        int id = -1;           
        float intercept = 0.0f; 
        float angle = 0.0f;     
        bool is_visible = false;
    };

    struct PlaneGrid {
        std::vector<Eigen::Vector3d> line_1_;
        std::vector<Eigen::Vector3d> line_2_;
        std::vector<Eigen::Vector3d> line_3_;
        std::vector<Eigen::Vector3d> line_4_;
        std::vector<Eigen::Vector3d> line_5_;
        std::vector<Eigen::Vector3d> line_6_;
        std::vector<Eigen::Vector3d> line_7_;
        std::vector<Eigen::Vector3d> line_8_;
    };

    struct SimLine {
        LineCandidate line;
        bool valid = false;
    };

    cv::Point2f computeIntersection(const cv::Vec4f& line1, const cv::Vec4f& line2);
    cv::Vec4f slotToVec4f(const TrackedLine& tl, bool is_vertical);
    void computeGridQuads(int src_w, int src_h, int net_w, int net_h);

    std::vector<LineCandidate> extractAndClusterLines(const cv::Mat& morph_mask, 
                                                     cv::Mat& merged_mask, 
                                                     bool is_vertical, 
                                                     float dist_thresh, 
                                                     float angle_thresh);

    void gridMasks(Eigen::Matrix3d R_wb, Eigen::Vector3d cam_pos);

private:
    PlaneGrid plane_grid_w_; 
    std::vector<SimLine> sim_v_slots_{4}; // 对应 3D 里的 1,2,3,4 号线投影
    std::vector<SimLine> sim_h_slots_{4}; // 对应 3D 里的 5,6,7,8 号线投影

    bool test_{false};
    int frame_idx_ = 0;
    
    int W_ = 0;
    int H_ = 0;
    
    // 【修改】移除了分块所需的 xs_, ys_ 容器

    std::thread th_camera_;
    std::thread th_process_;
    std::mutex global_mtx_;
    std::condition_variable cv_frame_;
    bool has_frame_ = false;
    cv::Mat global_frame_;
    std::atomic<bool> running_{false};

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    const char* input_name_ = nullptr;
    const char* output_name_ = nullptr;
    cudaStream_t stream_{};

    void *d_input_ = nullptr;
    void *d_output_ = nullptr;
    uchar3 *d_img_full_ = nullptr; 
    // 【修改】彻底移除了 d_score_, d_xs_, d_ys_ 显存指针
    uint8_t* d_mask_out_ = nullptr;

    std::vector<TrackedLine> tracked_h_{4}; 
    std::vector<TrackedLine> tracked_v_{4}; 

    // 【修改】对应目标分辨率 960x720，内参缩放比由 *2 调整为 *1.5
    double fx_ = 384.359 * 1.5;
    double fy_ = 383.817 * 1.5;
    double ppx_ = (640 - 1 - 321.989) * 1.5;
    double ppy_ = (480 - 1 - 246.572) * 1.5;
};

#endif // TRT_NODE_HPP_