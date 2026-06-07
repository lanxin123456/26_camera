#ifndef CUDA_KERNELS_HPP_
#define CUDA_KERNELS_HPP_

#include <cuda_runtime.h>
#include <cstdint>

/**
 * @brief 1. 全图高效前处理：将 960x720 原始图像转换为 1x3x736x960 的网络输入张量（自动在最下方补 16 像素白边）
 * @param d_img_full  指向 GPU 端输入的 960x720 原始 BGR 图像 (uchar3 类型)
 * @param d_input     指向 GPU 端网络输入层的 Tensor 显存空间 (1 * 3 * 736 * 960 * sizeof(float))
 * @param stream      CUDA 异步流句柄
 */
void launch_preprocess_and_pad(
    const uchar3* d_img_full, 
    float* d_input, 
    cudaStream_t stream
);

/**
 * @brief 2. 全图高效后处理：对 736x960 网络输出做 Sigmoid、卡阈值，并自动裁剪抛弃下方 16 像素，还原 960x720 的 Mask
 * @param d_output    指向 GPU 端网络原始输出层结果 (1 * 1 * 736 * 960 * sizeof(float))
 * @param d_mask      指向 GPU 端最终用于存放 960x720 掩码的显存空间 (960 * 720 * sizeof(uint8_t))
 * @param thresh      Sigmoid 二值化分类阈值（如 0.5f）
 * @param stream      CUDA 异步流句柄
 */
void launch_finalize_and_crop(
    const float* d_output, 
    uint8_t* d_mask, 
    float thresh, 
    cudaStream_t stream
);

#endif // CUDA_KERNELS_HPP_