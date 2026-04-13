# SGM Depth Estimation using libSGM (CUDA)

Triển khai depth estimation dựa trên thuật toán Semi-Global Matching (SGM) với tăng tốc CUDA.

## Tổng quan

Code này cung cấp:
- **SGM Inference Class**: Wrapper C++ cho libSGM với giao diện đơn giản
- **CUDA Acceleration**: Xử lý hoàn toàn trên GPU cho hiệu năng cao
- **Depth Conversion**: Chuyển đổi disparity sang depth map
- **Visualization**: Hỗ trợ colorize disparity và depth maps
- **Example Program**: Demo đầy đủ cách sử dụng

## Cấu trúc Files

```
include/ct_uav_stereo_cpp/
  └── infer_sgm.hpp           # Header file - khai báo API

libs/
  ├── infer_sgm.cpp           # Implementation - thuật toán SGM
  └── libSGM/                 # Thư viện libSGM (CUDA SGM)

src/
  └── sgm_example.cpp         # Ví dụ sử dụng

CMakeLists.txt                # Build configuration
```

## Cài đặt và Build

### 1. Yêu cầu

- CUDA Toolkit (đã có trên Jetson)
- OpenCV 4.x
- CMake >= 3.8
- libSGM (đã có trong project)

### 2. Build

```bash
cd /home/ctuav/trung/ws_obstacle_avoidance
colcon build --packages-select ct_uav_stereo_cpp
source install/setup.bash
```

## Sử dụng

### 1. Trong Code C++

```cpp
#include "ct_uav_stereo_cpp/infer_sgm.hpp"

// Cấu hình SGM
ct_uav_stereo::SGMInference::Config config;
config.width = 640;
config.height = 480;
config.disparity_size = 128;      // 64, 128, hoặc 256
config.P1 = 10;                    // Penalty cho disparity thay đổi ±1
config.P2 = 120;                   // Penalty cho disparity thay đổi >1
config.subpixel = true;            // Bật subpixel precision
config.use_8path = true;           // Dùng 8-path (tốt hơn 4-path)
config.LR_max_diff = 1;            // Left-Right consistency check

// Khởi tạo
ct_uav_stereo::SGMInference sgm(config);
if (!sgm.initialize()) {
    std::cerr << "Failed to initialize SGM!" << std::endl;
    return -1;
}

// Tính disparity
cv::Mat left_img, right_img;  // Load rectified stereo images
cv::Mat disparity;
float time_ms;

if (!sgm.infer(left_img, right_img, disparity, time_ms)) {
    std::cerr << "Inference failed!" << std::endl;
    return -1;
}

std::cout << "Processing time: " << time_ms << " ms" << std::endl;

// Chuyển disparity sang depth
cv::Mat depth;
float baseline = 0.12;        // Baseline in meters (12 cm)
float focal_length = 400.0;   // Focal length in pixels
float scale_factor = 16.0;    // 16.0 cho subpixel, 1.0 cho normal

sgm.disparityToDepth(disparity, depth, baseline, focal_length, scale_factor);

// Visualization
cv::Mat disparity_color;
sgm.colorizeDisparity(disparity, disparity_color);
cv::imshow("Disparity", disparity_color);
```

### 2. Chạy Example Program

```bash
# Với stereo images
ros2 run ct_uav_stereo_cpp sgm_example left.png right.png

# Với tùy chọn
ros2 run ct_uav_stereo_cpp sgm_example left.png right.png \
    --width 640 \
    --height 480 \
    --disp_size 128 \
    --baseline 0.12 \
    --focal 400 \
    --P1 10 \
    --P2 120 \
    --subpixel \
    --4path
```

## Tham số SGM

### Disparity Size
- `64`, `128`, hoặc `256`
- Giá trị lớn hơn → phát hiện vật xa hơn nhưng chậm hơn
- Recommend: `128` cho cân bằng tốc độ/độ chính xác

### P1 và P2 (Smoothness Penalties)
- **P1**: Penalty khi disparity thay đổi ±1 pixel
  - Default: `10`
  - Tăng → disparity mượt hơn
  
- **P2**: Penalty khi disparity thay đổi >1 pixel  
  - Default: `120`
  - Luôn phải P2 > P1
  - Tăng → giảm noise nhưng mất detail

### Subpixel Estimation
- `subpixel = true`: Disparity có độ chính xác 1/16 pixel
- Cần `output_depth_bits = 16`
- Tốt hơn cho depth accuracy

### Path Type
- **8-path**: Scan theo 8 hướng (horizontal, vertical, diagonal)
  - Chính xác hơn, chậm hơn
  
- **4-path**: Scan theo 4 hướng (horizontal, vertical)
  - Nhanh hơn, ít chính xác hơn

### Left-Right Consistency Check
- `LR_max_diff >= 0`: Bật consistency check
- `LR_max_diff = -1`: Tắt check (nhanh hơn)
- Recommend: `1` để loại bỏ outliers

## Tính Depth từ Disparity

Công thức:
```
depth (meters) = (baseline × focal_length) / disparity
```

Với:
- **baseline**: Khoảng cách giữa 2 camera (meters)
- **focal_length**: Tiêu cự camera (pixels)
- **disparity**: Giá trị disparity (pixels)

Nếu dùng subpixel, disparity được scale 16x:
```
depth = (baseline × focal_length) / (disparity / 16.0)
```

## Performance

Trên Jetson Orin:
- **640×480**: ~15-25 FPS (disp=128, 8-path)
- **1280×720**: ~5-10 FPS (disp=128, 8-path)

Tips tăng tốc:
1. Dùng 4-path thay vì 8-path
2. Giảm disparity size (128 → 64)
3. Resize ảnh nhỏ hơn
4. Tắt LR consistency check

## Output

### Disparity Map
- **Type**: `CV_16SC1` (16-bit signed) nếu `output_depth_bits=16`
- **Type**: `CV_8UC1` (8-bit unsigned) nếu `output_depth_bits=8`
- **Range**: `[0, disparity_size * scale_factor]`
- **Invalid**: Giá trị đặc biệt (lấy từ `getInvalidDisparity()`)

### Depth Map
- **Type**: `CV_32FC1` (32-bit float)
- **Unit**: Meters
- **Invalid**: `0.0`

## Ví dụ với ROS2 Stereo Images

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include "ct_uav_stereo_cpp/infer_sgm.hpp"

class StereoDepthNode : public rclcpp::Node {
public:
    StereoDepthNode() : Node("stereo_depth_sgm") {
        // Initialize SGM
        ct_uav_stereo::SGMInference::Config config;
        config.width = 640;
        config.height = 480;
        config.disparity_size = 128;
        config.subpixel = true;
        
        sgm_ = std::make_shared<ct_uav_stereo::SGMInference>(config);
        sgm_->initialize();
        
        // Subscribe to stereo images
        left_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/stereo/left/image_rect", 10,
            std::bind(&StereoDepthNode::leftCallback, this, std::placeholders::_1));
            
        right_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/stereo/right/image_rect", 10,
            std::bind(&StereoDepthNode::rightCallback, this, std::placeholders::_1));
        
        // Publish disparity and depth
        disparity_pub_ = create_publisher<sensor_msgs::msg::Image>("/stereo/disparity", 10);
        depth_pub_ = create_publisher<sensor_msgs::msg::Image>("/stereo/depth", 10);
    }
    
private:
    void processStereoPair() {
        if (left_img_.empty() || right_img_.empty()) return;
        
        cv::Mat disparity;
        float time_ms;
        
        if (sgm_->infer(left_img_, right_img_, disparity, time_ms)) {
            // Publish disparity
            auto disp_msg = cv_bridge::CvImage(std_msgs::msg::Header(), 
                                               "16SC1", disparity).toImageMsg();
            disparity_pub_->publish(*disp_msg);
            
            // Convert to depth
            cv::Mat depth;
            sgm_->disparityToDepth(disparity, depth, 0.12, 400.0, 16.0);
            
            // Publish depth
            auto depth_msg = cv_bridge::CvImage(std_msgs::msg::Header(), 
                                                "32FC1", depth).toImageMsg();
            depth_pub_->publish(*depth_msg);
            
            RCLCPP_INFO(get_logger(), "SGM: %.2f ms (%.1f FPS)", 
                       time_ms, 1000.0/time_ms);
        }
    }
    
    void leftCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        left_img_ = cv_bridge::toCvShare(msg, "mono8")->image;
        processStereoPair();
    }
    
    void rightCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        right_img_ = cv_bridge::toCvShare(msg, "mono8")->image;
    }
    
    std::shared_ptr<ct_uav_stereo::SGMInference> sgm_;
    cv::Mat left_img_, right_img_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_sub_, right_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr disparity_pub_, depth_pub_;
};
```

## Troubleshooting

### 1. Build Errors

```bash
# Kiểm tra CUDA
nvcc --version

# Kiểm tra OpenCV
pkg-config --modversion opencv4

# Clean build
cd ws_obstacle_avoidance
rm -rf build/ install/ log/
colcon build --packages-select ct_uav_stereo_cpp
```

### 2. Runtime Errors

**"disparity_size must be 64, 128, or 256"**
→ Chỉ hỗ trợ 3 giá trị này

**"output_depth_bits must be 16 when subpixel is enabled"**
→ Subpixel cần 16-bit output

**"Image size mismatch"**
→ Left và right images phải cùng kích thước với config

### 3. Kết quả không tốt

- **Nhiều noise**: Tăng P2 (ví dụ: 120 → 200)
- **Mất detail**: Giảm P1, P2
- **Outliers**: Bật LR consistency check
- **Thiếu disparity**: Tăng disparity_size
- **Ảnh mờ**: Kiểm tra rectification, calibration

## Tham khảo

- [libSGM GitHub](https://github.com/fixstars/libSGM)
- [SGM Paper](https://core.ac.uk/download/pdf/11134866.pdf)
- OpenCV Stereo Matching Tutorial

## License

Code sử dụng libSGM (Apache 2.0 License)
