#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ct_uav_stereo {

namespace {

constexpr int kKernelFloat32 = 0;
constexpr int kKernelFloat16 = 1;

__device__ __forceinline__ float toFloat(float v) {
    return v;
}

__device__ __forceinline__ float toFloat(half v) {
    return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float v);

template <>
__device__ __forceinline__ float fromFloat<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ half fromFloat<half>(float v) {
    return __float2half(v);
}

template <typename Tout>
__global__ void preprocessBgrToRgbChwKernel(const uchar3* input_bgr,
                                            Tout* output_chw,
                                            int height,
                                            int width) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int idx = y * width + x;
    const int hw = height * width;

    const uchar3 p = input_bgr[idx];  // OpenCV BGR layout

    // Model expects RGB input in CHW layout, value range [0, 255]
    output_chw[idx] = fromFloat<Tout>(static_cast<float>(p.z));
    output_chw[hw + idx] = fromFloat<Tout>(static_cast<float>(p.y));
    output_chw[2 * hw + idx] = fromFloat<Tout>(static_cast<float>(p.x));
}

template <typename Tin, typename Tout>
__global__ void castKernel(const Tin* input, Tout* output, size_t count) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    output[idx] = fromFloat<Tout>(toFloat(input[idx]));
}

template <typename Tin, typename Tout>
__global__ void gwcVolumeKernel(const Tin* left_feat,
                                const Tin* right_feat,
                                Tout* output,
                                int batch,
                                int channels,
                                int height,
                                int width,
                                int num_groups,
                                int max_disp,
                                bool normalize) {
    const size_t total = static_cast<size_t>(batch) * num_groups *
                         max_disp * height * width;
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }

    size_t t = linear;
    const int x = static_cast<int>(t % width);
    t /= width;
    const int y = static_cast<int>(t % height);
    t /= height;
    const int d = static_cast<int>(t % max_disp);
    t /= max_disp;
    const int g = static_cast<int>(t % num_groups);
    const int b = static_cast<int>(t / num_groups);

    const int d_eff = max_disp < width ? max_disp : width;
    if (d >= d_eff) {
        output[linear] = fromFloat<Tout>(0.0f);
        return;
    }

    const int xr = x - d;
    if (xr < 0) {
        output[linear] = fromFloat<Tout>(0.0f);
        return;
    }

    const int channels_per_group = channels / num_groups;
    const int c_start = g * channels_per_group;

    float dot = 0.0f;
    float norm_l = 0.0f;
    float norm_r = 0.0f;

    for (int k = 0; k < channels_per_group; ++k) {
        const int c = c_start + k;
        const size_t left_idx =
            ((static_cast<size_t>(b) * channels + c) * height + y) * width + x;
        const size_t right_idx =
            ((static_cast<size_t>(b) * channels + c) * height + y) * width + xr;

        const float l = toFloat(left_feat[left_idx]);
        const float r = toFloat(right_feat[right_idx]);
        dot += l * r;

        if (normalize) {
            norm_l += l * l;
            norm_r += r * r;
        }
    }

    if (normalize) {
        const float denom = sqrtf(norm_l * norm_r) + 1e-5f;
        dot /= denom;
    }

    output[linear] = fromFloat<Tout>(dot);
}

template <typename Tout>
void launchPreprocessTyped(const void* input_bgr,
                           void* output_chw,
                           int height,
                           int width,
                           cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    preprocessBgrToRgbChwKernel<Tout><<<grid, block, 0, stream>>>(
        static_cast<const uchar3*>(input_bgr),
        static_cast<Tout*>(output_chw),
        height,
        width);
}

template <typename Tin, typename Tout>
void launchCastTyped(const void* input,
                     void* output,
                     size_t count,
                     cudaStream_t stream) {
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    castKernel<Tin, Tout><<<grid, block, 0, stream>>>(
        static_cast<const Tin*>(input),
        static_cast<Tout*>(output),
        count);
}

template <typename Tin, typename Tout>
void launchGwcTyped(const void* left_feat,
                    const void* right_feat,
                    void* output_volume,
                    int batch,
                    int channels,
                    int height,
                    int width,
                    int num_groups,
                    int max_disp,
                    bool normalize,
                    cudaStream_t stream) {
    const size_t total = static_cast<size_t>(batch) * num_groups *
                         max_disp * height * width;
    const int block = 128;
    const int grid = static_cast<int>((total + block - 1) / block);

    gwcVolumeKernel<Tin, Tout><<<grid, block, 0, stream>>>(
        static_cast<const Tin*>(left_feat),
        static_cast<const Tin*>(right_feat),
        static_cast<Tout*>(output_volume),
        batch,
        channels,
        height,
        width,
        num_groups,
        max_disp,
        normalize);
}

}  // namespace

void launchFastFSPreprocessKernel(const void* input_bgr,
                                  void* output_chw,
                                  int height,
                                  int width,
                                  int output_type,
                                  void* stream) {
    auto cuda_stream = static_cast<cudaStream_t>(stream);

    if (output_type == kKernelFloat16) {
        launchPreprocessTyped<half>(input_bgr, output_chw, height, width, cuda_stream);
    } else {
        launchPreprocessTyped<float>(input_bgr, output_chw, height, width, cuda_stream);
    }
}

void launchFastFSCastKernel(const void* input,
                            void* output,
                            size_t count,
                            int input_type,
                            int output_type,
                            void* stream) {
    auto cuda_stream = static_cast<cudaStream_t>(stream);

    if (input_type == kKernelFloat32 && output_type == kKernelFloat32) {
        launchCastTyped<float, float>(input, output, count, cuda_stream);
    } else if (input_type == kKernelFloat16 && output_type == kKernelFloat16) {
        launchCastTyped<half, half>(input, output, count, cuda_stream);
    } else if (input_type == kKernelFloat16 && output_type == kKernelFloat32) {
        launchCastTyped<half, float>(input, output, count, cuda_stream);
    } else if (input_type == kKernelFloat32 && output_type == kKernelFloat16) {
        launchCastTyped<float, half>(input, output, count, cuda_stream);
    }
}

void launchFastFSGwcVolumeKernel(const void* left_feat,
                                 const void* right_feat,
                                 void* output_volume,
                                 int batch,
                                 int channels,
                                 int height,
                                 int width,
                                 int num_groups,
                                 int max_disp,
                                 bool normalize,
                                 int input_type,
                                 int output_type,
                                 void* stream) {
    auto cuda_stream = static_cast<cudaStream_t>(stream);

    if (input_type == kKernelFloat32 && output_type == kKernelFloat32) {
        launchGwcTyped<float, float>(
            left_feat, right_feat, output_volume,
            batch, channels, height, width,
            num_groups, max_disp, normalize,
            cuda_stream);
    } else if (input_type == kKernelFloat16 && output_type == kKernelFloat16) {
        launchGwcTyped<half, half>(
            left_feat, right_feat, output_volume,
            batch, channels, height, width,
            num_groups, max_disp, normalize,
            cuda_stream);
    } else if (input_type == kKernelFloat16 && output_type == kKernelFloat32) {
        launchGwcTyped<half, float>(
            left_feat, right_feat, output_volume,
            batch, channels, height, width,
            num_groups, max_disp, normalize,
            cuda_stream);
    } else if (input_type == kKernelFloat32 && output_type == kKernelFloat16) {
        launchGwcTyped<float, half>(
            left_feat, right_feat, output_volume,
            batch, channels, height, width,
            num_groups, max_disp, normalize,
            cuda_stream);
    }
}

}  // namespace ct_uav_stereo
