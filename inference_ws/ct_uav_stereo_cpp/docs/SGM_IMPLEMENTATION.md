# SGM Depth Estimation Implementation Summary

## Code đã được tạo

Đã implement depth estimation hoàn chỉnh dựa trên libSGM (CUDA SGM) với các files sau:

### 1. Header File
- **File**: `include/ct_uav_stereo_cpp/infer_sgm.hpp`
- **Nội dung**: 
  - Class `SGMInference` với API đơn giản
  - Config struct cho các parameters
  - Methods: initialize(), infer(), disparityToDepth(), colorizeDisparity()

### 2. Implementation File  
- **File**: `libs/infer_sgm.cpp`
- **Nội dung**:
  - Sử dụng `sgm::LibSGMWrapper` (wrapper chính thức của libSGM)
  - Tự động quản lý memory
  - Hỗ trợ CUDA acceleration
  - Xử lý preprocessing (grayscale conversion, type conversion)

### 3. Example Program
- **File**: `src/sgm_example.cpp`
- **Nội dung**:
  - Demo hoàn chỉnh cách sử dụng
  - Load stereo images
  - Compute disparity
  - Convert to depth
  - Visualization với colormap

### 4. Python Test Script
- **File**: `scripts/test_sgm.py`
- **Nội dung**:
  - Script Python để test nhanh (dùng OpenCV StereoSGBM)
  - So sánh với C++ CUDA version

### 5. Documentation
- **File**: `docs/SGM_USAGE.md`
- **Nội dung**:
  - Hướng dẫn sử dụng chi tiết (tiếng Việt)
  - Giải thích parameters
  - Performance tips
  - Example code với ROS2

## Key Features

### ✅ Đã Implement
1. **CUDA Acceleration**: Sử dụng libSGM với CUDA
2. **LibSGMWrapper**: Wrapper chính thức - đơn giản và hiệu quả
3. **Auto Memory Management**: Tự động allocate/deallocate buffers
4. **Flexible Input**: Hỗ trợ CV_8U, CV_16U, CV_32S
5. **Auto Preprocessing**: Tự động convert grayscale, type conversion
6. **Subpixel Accuracy**: Hỗ trợ disparity với độ chính xác 1/16 pixel
7. **Depth Conversion**: Chuyển disparity sang depth (meters)
8. **Visualization**: Colorize disparity và depth maps
9. **Error Handling**: Kiểm tra CUDA availability, input validation

### 🎯 Advantages vs Raw libSGM

**Before (Raw StereoSGM)**:
```cpp
// Phải tự manage memory, specify bit depths, execute modes
sgm::StereoSGM::Parameters params(...);
sgm::StereoSGM* sgm = new sgm::StereoSGM(
    width, height, disp_size, 
    input_depth_bits, output_depth_bits,
    sgm::EXECUTE_INOUT_HOST2HOST, params);
sgm->execute(left.data, right.data, disparity.data);
```

**After (LibSGMWrapper)**:
```cpp
// Wrapper tự động quản lý mọi thứ
sgm::LibSGMWrapper sgm(disp_size, P1, P2, uniqueness, 
                       subpixel, path_type, min_disp, 
                       LR_max_diff, census_type);
sgm.execute(left, right, disparity);  // Tự động allocate disparity!
```

## Cách Sử dụng

### Build
```bash
cd /home/ctuav/trung/ws_obstacle_avoidance
colcon build --packages-select ct_uav_stereo_cpp
source install/setup.bash
```

### Run Example
```bash
# Với stereo images
ros2 run ct_uav_stereo_cpp sgm_example left.png right.png

# Với options
ros2 run ct_uav_stereo_cpp sgm_example left.png right.png \
    --disp_size 128 --P1 10 --P2 120 --subpixel
```

### Trong Code
```cpp
#include "ct_uav_stereo_cpp/infer_sgm.hpp"

// Configure
ct_uav_stereo::SGMInference::Config config;
config.width = 640;
config.height = 480;
config.disparity_size = 128;
config.subpixel = true;

// Initialize
ct_uav_stereo::SGMInference sgm(config);
sgm.initialize();

// Infer
cv::Mat disparity;
float time_ms;
sgm.infer(left_img, right_img, disparity, time_ms);

// Convert to depth
cv::Mat depth;
sgm.disparityToDepth(disparity, depth, 0.12, 400.0, 16.0);
```

## CMakeLists.txt Updates

Đã update CMakeLists.txt để:
1. Add libSGM include path
2. Build libSGM as subdirectory
3. Add infer_sgm.cpp to library
4. Link sgm library
5. Build sgm_example executable

## Performance (Expected)

Trên Jetson Orin với CUDA:
- **640×480, disp=128, 8-path**: ~15-25 FPS
- **1280×720, disp=128, 8-path**: ~5-10 FPS
- **640×480, disp=64, 4-path**: ~30-40 FPS

## Next Steps

1. **Test**: Build và test với stereo images
2. **Tune**: Điều chỉnh P1, P2 cho dataset cụ thể
3. **Integrate**: Tích hợp vào ROS2 node
4. **Optimize**: Profile và optimize nếu cần

## Files Modified/Created

```
include/ct_uav_stereo_cpp/
  └── infer_sgm.hpp                 ✨ NEW

libs/
  └── infer_sgm.cpp                 ✨ NEW

src/
  └── sgm_example.cpp               ✨ NEW

scripts/
  └── test_sgm.py                   ✨ NEW

docs/
  └── SGM_USAGE.md                  ✨ NEW

CMakeLists.txt                      ✏️ MODIFIED
```

## References

- libSGM example: `libs/libSGM/sample/stereosgm_image_cv_gpumat.cpp`
- LibSGMWrapper API: `libs/libSGM/include/libsgm_wrapper.h`
- Documentation: `docs/SGM_USAGE.md`
