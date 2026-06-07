#ifndef PARAMS_HPP
#define PARAMS_HPP

struct PclParams {
    // 白墙灰度图阈值
    int white_gray_min;
    int white_gray_max;

    // 白墙 HSV 阈值 (使用 cv::Scalar 直接对应 [H, S, V])
    cv::Scalar white_hsv_min;
    cv::Scalar white_hsv_max;
};

struct GridParams {
    int field;       // 0 红场  1 蓝场
};

struct ActDParams {
    std::string bag_path;       
};

struct WeaponParams {
    bool visualize;
    int field;       // 0 红场  1 蓝场
    cv::Rect rect;        //装配时武器前方矩形
    cv::Rect rect_fixed;  //交并比矩形
    int interval_y;       //rect 与 yolo框的间隔
    float iou_thresh;
    std::string image_path;
    std::string video_path;
};

#endif