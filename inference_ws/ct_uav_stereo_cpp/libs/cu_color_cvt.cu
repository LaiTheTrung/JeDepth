#include <stdio.h>
#include <ct_uav_stereo_cpp/cu_color_cvt.h>

__device__ inline uint8_t clamp(float val, float mn, float mx) {
  return (uint8_t) ((val >= mn)? ((val <= mx)? val : mx) : mn);
}

__global__ void convert_kernel( CUsurfObject surface1, CUsurfObject surface2,
                    unsigned int width, unsigned int height,
			       uint8_t* out, int conversionType) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int nx = blockDim.x * gridDim.x;
  int ny = blockDim.y * gridDim.y;

  for (int col = x; col < width; col += nx) {
    for (int row = y; row < height; row += ny) {
      uchar1 Ydata, Cbdata, Crdata;
      surf2Dread(&Ydata, surface1, col, row);
      surf2Dread(&Cbdata, surface2, ((int) col / 2) * 2 + 1, (int) (row / 2));
      surf2Dread(&Crdata, surface2, ((int) col / 2) * 2 + 0, (int) (row / 2));

      uint8_t Bval = clamp(Ydata.x + 1.402f * (Crdata.x - 128), 0.0f, 255.0f);
      uint8_t Gval = clamp(Ydata.x - 0.344136f * (Cbdata.x - 128) - 0.714136 * (Crdata.x - 128), 0.0f, 255.0f);
      uint8_t Rval = clamp(Ydata.x + 1.772f * (Cbdata.x - 128), 0.0f, 255.0f);
     
      if (conversionType == cvt_YUV2RGB) {
        out[3 * (row * width + col) + 0] = Rval;
        out[3 * (row * width + col) + 1] = Gval;
        out[3 * (row * width + col) + 2] = Bval;
      }
      else if (conversionType == cvt_YUV2BGR) {
        out[3 * (row * width + col) + 0] = Bval;
        out[3 * (row * width + col) + 1] = Gval;
        out[3 * (row * width + col) + 2] = Rval;
      }
      else {
        out[row * width + col] = Rval/3 + Gval/3 + Bval/3;
      }
    }
  }
}

static inline int get_channels(int conversionType) {
  if (conversionType == cvt_YUV2RGB || conversionType == cvt_YUV2BGR) {
    return 3;
  }
  return 1;
}

float run_smem_atomics(CUsurfObject surface1, CUsurfObject surface2,
		       unsigned int width, unsigned int height,
		       uint8_t* oBuffer, int conversionType) {
  cudaError_t err = cudaSuccess;
  dim3 block(32, 32);
  dim3 grid(4, 4);

  int channels = get_channels(conversionType);
  size_t buffer_bytes = static_cast<size_t>(width) * height * channels * sizeof(uint8_t);

  uint8_t* d_buffer;
  err = cudaMalloc(&d_buffer, buffer_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device buffer (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
    
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&stop);
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);

  convert_kernel<<<grid, block>>>(surface1, surface2, width, height, d_buffer, conversionType);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed_millis;
  cudaEventElapsedTime(&elapsed_millis, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  err = cudaMemcpy(oBuffer, d_buffer, buffer_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy into host buffer (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_buffer);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device buffer (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  return elapsed_millis;
}

float convert(CUsurfObject surface1, CUsurfObject surface2,
	      unsigned int width, unsigned int height,
	      uint8_t* oBuffer, int conversionType) {
  return run_smem_atomics(surface1, surface2, width, height, oBuffer, conversionType);
}

float convert(CUsurfObject surface1, CUsurfObject surface2,
	      unsigned int width, unsigned int height,
	      cv::Mat& outputMat, int conversionType) {
  int channels = get_channels(conversionType);
  int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
  outputMat.create(height, width, type);

  size_t buffer_bytes = static_cast<size_t>(width) * height * channels * sizeof(uint8_t);
  uint8_t* d_buffer = nullptr;
  cudaError_t err = cudaMalloc(&d_buffer, buffer_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device buffer (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  dim3 block(32, 32);
  dim3 grid(4, 4);
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&stop);
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);

  convert_kernel<<<grid, block>>>(surface1, surface2, width, height, d_buffer, conversionType);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed_millis;
  cudaEventElapsedTime(&elapsed_millis, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  err = cudaMemcpy(outputMat.data, d_buffer, buffer_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy into host buffer (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_buffer);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device buffer (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  return elapsed_millis;
}

float convert(CUsurfObject surface1, CUsurfObject surface2,
	      unsigned int width, unsigned int height,
	      cv::cuda::GpuMat& outputMat, int conversionType) {
  int channels = get_channels(conversionType);
  int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
  outputMat.create(height, width, type);

  size_t buffer_bytes = static_cast<size_t>(width) * height * channels * sizeof(uint8_t);
  uint8_t* d_buffer = nullptr;
  cudaError_t err = cudaMalloc(&d_buffer, buffer_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device buffer (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  dim3 block(32, 32);
  dim3 grid(4, 4);
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&stop);
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);

  convert_kernel<<<grid, block>>>(surface1, surface2, width, height, d_buffer, conversionType);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed_millis;
  cudaEventElapsedTime(&elapsed_millis, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  size_t row_bytes = static_cast<size_t>(width) * channels * sizeof(uint8_t);
  err = cudaMemcpy2D(outputMat.data, outputMat.step, d_buffer, row_bytes,
                     row_bytes, height, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy into GpuMat (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_buffer);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device buffer (%s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  return elapsed_millis;
}
