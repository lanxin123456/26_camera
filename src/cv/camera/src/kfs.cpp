#include "act_d455/kfs.hpp"
/******************************** *****************/
#define CAM_PLACE 2 //1在上，2在下

KFS::KFS():
    Node{"deal"}
{

    kfs_log_.open("kfs.csv", std::ios::trunc);
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);
    if(kfs_log_.is_open()) kfs_log_ << "\n\n\n\n\n"
                                    << "时间, " 
                                    << put_time(localTime, "%Y-%m-%d %H:%M:%S") 
                                    << " ===================================================================================================================================" 
                                    << endl;    
}

KFS::~KFS()
{
}


cv::Mat KFS::Kfs(const Mat &frame, cv::Rect& rect, const Mat &depth)
{
    Mat hsv;
    binary_ = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // cout << hsv.at<Vec3b>(360, 239) << frame.at<Vec3b>(360, 239) << endl;
    cv::Rect img_rect(0, 0, frame.cols, frame.rows);
    cv::Rect validated_rect = rect & img_rect;
    
    for (int i = validated_rect.y; i < validated_rect.y + validated_rect.height; i++) 
    {
        for (int j = validated_rect.x; j < validated_rect.x + validated_rect.width; j++) 
        {
            float pixel_depth = depth.at<float>(i,j);
            if(pixel_depth > 2000) continue;
            Vec3b hsv_pixel = hsv.at<Vec3b>(i, j);
            Vec3b bgr_pixel = frame.at<Vec3b>(i, j);

            // 红色判断条件：
            bool red_bgr = (bgr_pixel[0] < 235 && bgr_pixel[1] < 235 && bgr_pixel[2] > 30);
            bool red_hsv_h = (hsv_pixel[0] > 140 || hsv_pixel[0] < 15);
            bool red_hsv_s = (hsv_pixel[1] > 20);
            
            if (red_bgr && red_hsv_h && red_hsv_s) 
            {
                binary_.at<uchar>(i, j) = 255;
            }

            //蓝色判断条件：
            bool blue_bgr = (bgr_pixel[2] < 235 && bgr_pixel[1] < 235 && bgr_pixel[0] > 30);
            bool blue_hsv_h = (hsv_pixel[0] > 90 && hsv_pixel[0] <= 140);
            bool blue_hsv_s = (hsv_pixel[1] > 20);
            if (blue_bgr && blue_hsv_h && blue_hsv_s) 
            {
                binary_.at<uchar>(i, j) = 255;
            }
        }
    }

    cv::namedWindow("binary_", cv::WINDOW_NORMAL);
    cv::imshow("binary_", binary_);

    Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    morphologyEx(binary_, binary_, MORPH_CLOSE, kernel); // 闭运算 填空隙
    morphologyEx(binary_, binary_, MORPH_OPEN, kernel); 
    // // // 中值滤波 - 有效去除颗粒噪点
    // cv::Mat a = binary_.clone();

    // cv::medianBlur(binary_, binary_, 5);

    return binary_;
}

KFS::LineKB KFS::ransacLineFit( const std::vector<cv::Point>& pts )
{
    LineKB best;
    if (pts.size() < 10) return best;

    std::mt19937 rng(static_cast<unsigned>(time(nullptr)));
    std::uniform_int_distribution<int> dist(0, pts.size() - 1);

    int best_inliers = 0;

    for (int it = 0; it < iterations_; it++)
    {
        const cv::Point& p1 = pts[dist(rng)];
        const cv::Point& p2 = pts[dist(rng)];
        if (std::abs(p1.y - p2.y) < 2) continue;

        double k = (p2.x - p1.x) / double(p2.y - p1.y);
        double b = p1.x - k * p1.y;

        if(abs(k) > 0.2) continue;

        std::vector<cv::Point> current_inliers;
        current_inliers.reserve(pts.size()); 

        for (const auto& p : pts)
        {
            double x_est = k * p.y + b;
            double d = std::abs(p.x - x_est);
            if (d < dist_thresh_)
            {
                current_inliers.push_back(p);  
            }
        }

        int inliers = static_cast<int>(current_inliers.size());

        if (inliers > best_inliers && inliers > min_inliers_)
        {
            best_inliers = inliers;
            best.k = k;
            best.b = b;
            best.inliers.swap(current_inliers); 
            best.valid = true;
        }
    }
    // cout << "best.k: " << best.k << endl;
    return best;
}

void KFS::boudary(const Mat& gray, const cv::Rect& rect)
{
    // cout << rect << endl;
    cv::Mat kernel_y = (Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat gray_y;
    filter2D(gray, gray_y, CV_32F, kernel_y);
    convertScaleAbs(gray_y, gray_y);

    points_left_.clear();
    points_right_.clear();
    cv::imshow("gray_y", gray_y);
    for(int i = 0; i < gray_y.rows; i++)
    {
        int left_x = -1;
        int right_x = -1;
        for(int j = 0; j < gray_y.cols; j++)
        {
            if(gray_y.at<uchar>(i,j) >= 200 && j < (rect.x + rect.width*0.1))
            {
                left_x = j;
                break;
            }
        }
        for(int j = gray_y.cols - 1; j >= 0; j--)
        {
            if(gray_y.at<uchar>(i,j) >= 200 && j > (rect.x + rect.width*0.9))
            {
                right_x = j;
                break;
            }
        }
        if (left_x != -1) points_left_.emplace_back(left_x, i);
        if (right_x != -1) points_right_.emplace_back(right_x, i);
    }
    points_left_ = ransacLineFit(points_left_).inliers;
    points_right_ = ransacLineFit(points_right_).inliers;

    // double mean_left_x = 0.0, mean_right_x = 0.0;
    // if (!points_left_.empty()) {
    //     double sum_left = 0.0;
    //     for (const auto& pt : points_left_) sum_left += pt.x;
    //     mean_left_x = sum_left / points_left_.size();
    // }
    // if (!points_right_.empty()) {
    //     double sum_right = 0.0;
    //     for (const auto& pt : points_right_) sum_right += pt.x;
    //     mean_right_x = sum_right / points_right_.size();
    // }
    // std::cout << "rect: " << rect
    //             << "points_left_ average x: " << mean_left_x
    //           << ", points_right_ average x: " << mean_right_x 
    //          << "points_left_.size(): " << points_left_.size() << " points_right_.size(): " << points_right_.size() << endl;
}

