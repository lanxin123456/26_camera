#pragma once

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>          //提供标准异常类.std::runtime_error
#include <thread>
#include <opencv2/opencv.hpp>

#include <glob.h>                    //glob(pattern, flags, &glob_result)
#include <linux/videodev2.h>         //视频设备 v4l2_capability结构体和 VIDIOC_QUERYCAP命令用于查询摄像头属性
#include <sys/ioctl.h>               //ioctl(fd,...)
#include <fcntl.h>                  //open(fd) O_RDONLY
#include <unistd.h>                 //close(fd)

namespace camera_backward
{
    class ActCamera
    {
    public:
        ActCamera();
        ~ActCamera();

        void Init();
        cv::Mat Update();


        // 如果需要，外界可以使用clone方法获取帧
        cv::Mat GetFrame()
        { 
            std::lock_guard<std::mutex> lock(frame_mutex_);
            return srcframe; 
        }

        int exposure_mode_ = 1;
    private:
        cv::Mat temp_frame;
        cv::Mat srcframe;
        cv::Mat dstframe;
        cv::VideoCapture cap;
        
        std::queue<cv::Mat> buffer_queue_;

        std::mutex queue_mutex_;
        std::mutex frame_mutex_;

        int width_;
        int height_;
        int fps_;
        std::string video_device_;

        //标定参数
        cv::Mat camera_matrix_;      // 相机内参矩阵
        cv::Mat distortion_coeffs_;  // 畸变系数
        cv::Mat map1_, map2_;        // 畸变校正映射表
        bool calibration_loaded_;    // 标定参数是否加载成功
        
    private:
        void getBuffer();
        void clearBuffer();
        void release();
        std::string find_lrcp_camera();
        bool loadCalibrationParameters(const std::string& calibration_file = "ost.yaml");
    };

    // 最新帧缓存
    class LatestFrame 
    {
    public:
        void push(cv::Mat f) {
            std::lock_guard<std::mutex> lock(mtx_);
            latest_ = std::move(f);  // 直接覆盖旧帧
        }

        cv::Mat get_latest() {
            std::lock_guard<std::mutex> lock(mtx_);
            return latest_.clone();   // 非阻塞获取当前最新帧
        }

    private:
        cv::Mat latest_;
        std::mutex mtx_;
    };
}