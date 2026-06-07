#include "trt_seg/cuda_kernels.hpp"

// 基础常量定义
static constexpr int IMG_W = 960;
static constexpr int IMG_H = 720;
static constexpr int NET_W = 960;
static constexpr int NET_H = 736; // 720 + 16 像素白边

__device__ inline float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

// =====================================================
// 1. 全图高效前处理核函数（包含自动补白边 16 像素）
// =====================================================
__global__ void preprocess_and_white_pad_kernel(
    const uchar3* __restrict__ src_img, // 输入 960 x 720 原始图像
    float* __restrict__ dst_tensor)     // 输出 1 x 3 x 736 x 960 网络张量
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int plane_size = NET_W * NET_H; // 960 * 736
    int total_elements = plane_size * 3;

    if (idx >= total_elements) return;

    int c = idx / plane_size;         // 判断当前是 R, G 还是 B 通道
    int residual = idx % plane_size;
    int x = residual % NET_W;
    int y = residual / NET_W;

    if (y < IMG_H)
    {
        // 1. 处于 720 行以内的真实图像区域：正常读取、BGR->RGB 并归一化
        uchar3 pixel = src_img[y * IMG_W + x];
        float val = 0.0f;
        if (c == 0)      val = pixel.z * 0.0039215686f; // R
        else if (c == 1) val = pixel.y * 0.0039215686f; // G
        else if (c == 2) val = pixel.x * 0.0039215686f; // B
        dst_tensor[idx] = val;
    }
    else
    {
        // 2. 处于底部 720~735 行的人工填充区域：直接填充全白边
        // 纯白边像素为 255，归一化后 255 / 255.0 = 1.0f
        dst_tensor[idx] = 1.0f;
    }
}

// =====================================================
// 2. 全图高效后处理核函数（包含自动卡阈值、裁剪回 720）
// =====================================================
__global__ void finalize_and_crop_mask_kernel(
    const float* __restrict__ output_logits, // 输入 736 x 960 的网络原始输出
    uint8_t* __restrict__ mask_out,          // 输出 720 x 960 的最终 Mask 
    float thresh)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int total_pixels = IMG_W * IMG_H; // 960 * 720

    if (idx >= total_pixels) return;

    int x = idx % IMG_W;
    int y = idx / IMG_W;

    // 根据当前的 720 坐标，精确映射回网络输出的 736 维度的线性索引
    int net_idx = y * NET_W + x;

    // 计算 Sigmoid 并根据阈值二值化
    float prob = sigmoidf(output_logits[net_idx]);
    mask_out[idx] = (prob > thresh) ? 255 : 0;
}

// =====================================================
// 包装函数实现
// =====================================================
void launch_preprocess_and_pad(const uchar3* d_img_full, float* d_input, cudaStream_t stream)
{
    int total = 3 * NET_W * NET_H;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    preprocess_and_white_pad_kernel<<<blocks, threads, 0, stream>>>(d_img_full, d_input);
}

void launch_finalize_and_crop(const float* d_output, uint8_t* d_mask, float thresh, cudaStream_t stream)
{
    int total = IMG_W * IMG_H; // 960 * 720
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    finalize_and_crop_mask_kernel<<<blocks, threads, 0, stream>>>(d_output, d_mask, thresh);
}