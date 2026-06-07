#include <iostream>
#include <string>
#include <chrono>
#include <atomic>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

class SaveVideoF
{
public:
    void start(const cv::Mat& frame)
    {
        std::lock_guard<std::mutex> lock(mtx_); // 加上锁，确保线程安全
        if(opened_) return; // writer_ 只打开一次

        opened_ = true;
        stop_.store(false); // 确保每次 start 时 stop 状态被正确重置

        // 在生成新文件前清理旧文件，传入当前目录 "."，保留最新的 10 个（含即将生成的这1个）
        // 这里你之前改成了 3，我保持你的 3
        cleanup_old_files(".", 3);  

        filename_ = make_filename();  // 生成新文件名
        
        writer_.open(
            filename_,
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            // cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            60.0,                                    // 相机 fps
            frame.size(),
            true
        );
        std::cout << "[SaveVideo] record ON -> " << filename_ << "\n";
    }

    void stop()
    {
        stop_.store(true);
        std::lock_guard<std::mutex> lock(mtx_); // 保护 writer_ 和 opened_ 的状态

        if (writer_.isOpened())
        {
            writer_.release();
            opened_ = false; // 释放后重置标记，允许下一次 start
            std::cout << "[SaveVideo] record OFF\n";
        }
    }

    void write(const cv::Mat& frame)
    {
        if (stop_.load()) return;
 
        // 使用 try-lock 或者简单的无锁检查来避免写入时的极高锁竞争开销
        if (!writer_.isOpened())
        {
            // 降低错误输出频率，避免刷屏
            // std::cerr << "[SaveVideo] open failed: " << filename_ << "\n";
            return;
        }

        writer_.write(frame);
    }

private:
    // 清理旧文件逻辑
    void cleanup_old_files(const std::string& directory, size_t max_files)
    {
        std::vector<fs::path> files;
        std::string extension = ".avi";

        try 
        {
            // 遍历目录查找符合前缀和后缀的文件
            for (const auto& entry : fs::directory_iterator(directory)) {
                if (entry.is_regular_file()) {
                    std::string fname = entry.path().filename().string();
                    // 使用私有成员变量 file_prefix_ 替代硬编码
                    if (fname.find(file_prefix_) == 0 && entry.path().extension() == extension) {
                        files.push_back(entry.path());
                    }
                }
            }

            // 因为马上要新建 1 个文件，所以当前目录最多只能保留 max_files - 1 个旧文件
            if (files.size() >= max_files) {
                // 由于文件名包含 YYYY-MM-DD_HH-MM-SS，字典序排序即为时间排序（旧 -> 新）
                std::sort(files.begin(), files.end());

                // 计算需要删除多少个旧文件
                size_t files_to_delete = files.size() - max_files + 1;
                
                for (size_t i = 0; i < files_to_delete; ++i) {
                    fs::remove(files[i]);
                    std::cout << "[SaveVideo] Deleted old video: " << files[i].filename().string() << "\n";
                }
            }
        } 
        catch (const fs::filesystem_error& e) {
            std::cerr << "[SaveVideo] Filesystem error during cleanup: " << e.what() << "\n";
        }
    }

    // 生成时间戳文件名
    std::string make_filename()
    {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t); // Windows 下的安全版本
#else
        localtime_r(&t, &tm); // Linux 下的安全版本
#endif

        std::ostringstream oss;
        // 使用私有成员变量 file_prefix_ 替代硬编码
        oss << file_prefix_
            << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S")
            << ".avi";

        return oss.str();
    }

private:
    cv::VideoWriter writer_;
    std::atomic<bool> stop_{false};
    bool opened_{false};
    std::mutex mtx_;
    std::string filename_;
    const std::string file_prefix_ = "camera_for_"; // 将前缀提取为私有常量成员
};