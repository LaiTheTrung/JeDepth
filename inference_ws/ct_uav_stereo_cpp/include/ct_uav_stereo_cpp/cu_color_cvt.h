#ifndef CONVERT_H
#define CONVERT_H

#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

#define cvt_YUV2GRAY 0
#define cvt_YUV2RGB 1
#define cvt_YUV2BGR 2

extern float convert(CUsurfObject surface1, CUsurfObject surface2,
		     unsigned int width, unsigned int height,
		     uint8_t* oBuffer, int conversionType);
extern float convert(CUsurfObject surface1, CUsurfObject surface2,
		     unsigned int width, unsigned int height,
		     cv::Mat& outputMat, int conversionType);
extern float convert(CUsurfObject surface1, CUsurfObject surface2,
		     unsigned int width, unsigned int height,
		     cv::cuda::GpuMat& outputMat, int conversionType);
#endif // CONVERT_H