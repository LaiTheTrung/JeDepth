#include <cuda_runtime.h>

namespace ct_uav_stereo {

// CUDA kernel for preprocessing (BGR to RGB, normalize, HWC to CHW)
__global__ void preprocessKernel(const uchar3* input, float* output, 
                                 int height, int width,
                                 float mean_r, float mean_g, float mean_b,
                                 float std_r, float std_g, float std_b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int hw = height * width;
    int idx = y * width + x;
    
    // Input: BGR format (uchar3)
    uchar3 pixel = input[idx];
    
    // Convert to RGB and normalize
    float r = (pixel.z / 255.0f - mean_r) / std_r;  // B->R
    float g = (pixel.y / 255.0f - mean_g) / std_g;  // G->G
    float b = (pixel.x / 255.0f - mean_b) / std_b;  // R->B
    
    // Output: CHW format
    output[0 * hw + idx] = r;
    output[1 * hw + idx] = g;
    output[2 * hw + idx] = b;
}

// Wrapper function to launch kernel from C++
void launchPreprocessKernel(const void* input, void* output,
                           int height, int width,
                           float mean_r, float mean_g, float mean_b,
                           float std_r, float std_g, float std_b,
                           void* stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    // Cast void* to proper CUDA types
    const uchar3* d_input = static_cast<const uchar3*>(input);
    float* d_output = static_cast<float*>(output);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    
    preprocessKernel<<<grid, block, 0, cuda_stream>>>(
        d_input, d_output, height, width,
        mean_r, mean_g, mean_b,
        std_r, std_g, std_b
    );
}

} // namespace ct_uav_stereo
