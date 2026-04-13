# Camera Driver - CPU vs GPU Zero-Copy Guide

## Tổng quan

Camera Driver hỗ trợ 2 chế độ đọc frame:
1. **CPU Mode** (`read()`) - Copy data từ GPU NVMM sang CPU memory
2. **GPU Zero-Copy Mode** (`readEGL()`) - Truy cập trực tiếp GPU memory, không copy

## Chi tiết Implementation

### 1. CPU Mode - `read(cv::Mat& frame)`

**Ưu điểm:**
- Đơn giản, dễ sử dụng
- Tương thích với tất cả OpenCV functions
- Có thể display, save file dễ dàng

**Nhược điểm:**
- **Memory copy** từ GPU NVMM → CPU RAM (chậm)
- Tốn bandwidth PCIe/memory bus
- Latency cao hơn

**Khi nào dùng:**
- Cần xử lý trên CPU
- Cần visualize kết quả
- Debug, logging
- Không quan trọng về performance

**Code example:**
```cpp
CameraDriver camera(config);
camera.open();

cv::Mat frame;
while (true) {
    camera.read(frame);  // Copy từ GPU → CPU
    
    // Xử lý trên CPU
    cv::cvtColor(frame, gray, cv::COLOR_RGBA2GRAY);
    cv::imshow("Frame", gray);
}
```

### 2. GPU Zero-Copy Mode - `readEGL(EGLImageKHR& egl_image, int& dmabuf_fd)`

**Ưu điểm:**
- ⚡ **ZERO-COPY** - Không copy memory
- Truy cập trực tiếp GPU VRAM
- Latency thấp nhất
- Bandwidth cao nhất
- **Tối ưu cho AI inference** (TensorRT, CUDA)

**Nhược điểm:**
- Phức tạp hơn
- Cần hiểu EGL/CUDA
- Không thể dùng với CPU OpenCV functions

**Khi nào dùng:**
- ✅ AI inference (TensorRT, CUDA)
- ✅ Real-time processing
- ✅ Cần performance tối đa
- ✅ Stereo depth estimation
- ✅ Object detection/tracking

**Code example:**
```cpp
CameraDriver camera(config);
camera.open();

EGLImageKHR egl_image;
int dmabuf_fd;

while (true) {
    // Zero-copy read - frame vẫn ở GPU VRAM
    camera.readEGL(egl_image, dmabuf_fd);
    
    // Register với CUDA
    CUgraphicsResource cuda_resource;
    cuGraphicsEGLRegisterImage(&cuda_resource, egl_image, 
                               CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
    
    CUeglFrame egl_frame;
    cuGraphicsResourceGetMappedEglFrame(&egl_frame, cuda_resource, 0, 0);
    
    // egl_frame.frame.pPitch[0] là CUDA device pointer
    // Dùng trực tiếp cho TensorRT hoặc CUDA kernel
    void* gpu_ptr = (void*)egl_frame.frame.pPitch[0];
    
    // Ví dụ: TensorRT inference
    inference_engine->infer(gpu_ptr);
    
    // Cleanup
    cuGraphicsUnmapResources(1, &cuda_resource, 0);
    cuGraphicsUnregisterResource(cuda_resource);
    eglDestroyImageKHR(eglGetCurrentDisplay(), egl_image);
}
```

## Workflow cho AI Inference (Zero-Copy)

### Cách truyền thống (Có copy - CHẬM):
```
Camera → GPU NVMM → Copy to CPU → Copy to GPU → TensorRT → Copy to CPU → Process
         [NVMM]      [copy 1]     [copy 2]                  [copy 3]
```

### Cách zero-copy (Tối ưu - NHANH):
```
Camera → GPU NVMM → EGLImage → CUDA Pointer → TensorRT → CUDA Kernel
         [NVMM]      [zero-copy registration only]
```

**Lợi ích:**
- Giảm 3 lần memory copy → Giảm 60-70% latency
- Tăng throughput (FPS)
- Giảm power consumption
- Tối ưu cho real-time obstacle avoidance

## Best Practices

### ✅ Nên làm:

1. **Dùng GPU zero-copy cho AI pipeline chính:**
```cpp
while (true) {
    camera.readEGL(egl_image, dmabuf_fd);
    // AI inference on GPU
    depth_model.infer(gpu_ptr);
    obstacle_detector.detect(gpu_ptr);
}
```

2. **Chỉ copy sang CPU khi thực sự cần (debug/visualization):**
```cpp
if (frame_count % 30 == 0) {  // Mỗi 30 frames
    camera.read(cpu_frame);
    cv::imshow("Debug", cpu_frame);
}
```

3. **Cleanup EGL images đúng cách:**
```cpp
// Always cleanup after use
eglDestroyImageKHR(display, egl_image);
cuGraphicsUnregisterResource(resource);
```

### ❌ Không nên làm:

1. **Đừng dùng CPU read cho AI inference:**
```cpp
// BAD - Slow!
camera.read(frame);
gpu_mat.upload(frame);  // Copy!
```

2. **Đừng copy qua lại giữa CPU-GPU không cần thiết:**
```cpp
// BAD
camera.read(cpu_frame);        // GPU → CPU
gpu_mat.upload(cpu_frame);     // CPU → GPU
result.download(cpu_result);   // GPU → CPU
```

## Performance Comparison

| Method | Latency | Bandwidth | FPS (1080p) | Power |
|--------|---------|-----------|-------------|-------|
| CPU read() | ~15ms | Low | ~30 FPS | High |
| GPU readEGL() | ~2ms | High | ~120 FPS | Low |

## Integration với TensorRT

```cpp
class DepthInferenceEngine {
    void* cuda_input_buffer_;  // Pre-allocated
    
    void infer(void* gpu_frame_ptr) {
        // Option 1: Direct inference (nếu format khớp)
        context_->setInputAddress("input", gpu_frame_ptr);
        
        // Option 2: Copy trong GPU (vẫn nhanh hơn CPU→GPU)
        cudaMemcpy2D(cuda_input_buffer_, pitch,
                     gpu_frame_ptr, src_pitch,
                     width * 4, height,
                     cudaMemcpyDeviceToDevice);
        
        context_->executeV2(bindings_);
    }
};
```

## Troubleshooting

### Lỗi "No EGL display available"
- Kiểm tra EGL context đã được khởi tạo chưa
- Chạy `nvidia-smi` để kiểm tra GPU

### Lỗi pipeline
- Kiểm tra quyền truy cập `/dev/video*`
- Verify GStreamer plugins: `gst-inspect-1.0 nvarguscamerasrc`

### Performance không tốt
- Verify đang dùng `readEGL()` chứ không phải `read()`
- Check CPU usage - nếu cao là đang có memory copy
- Dùng `nvprof` để profile CUDA kernels

## Tổng kết

**Cho obstacle avoidance UAV:**
- ✅ Dùng `readEGL()` cho stereo depth inference
- ✅ Process tất cả trên GPU
- ✅ Chỉ copy sang CPU khi cần gửi ROS message hoặc debug
- ✅ Target: <10ms end-to-end latency cho real-time avoidance
