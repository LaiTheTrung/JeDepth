#ifndef INFER_FASTACV_HPP_
#define INFER_FASTACV_HPP_

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ct_uav_stereo {

// Forward declaration of CUDA kernel wrapper
// Using void* to avoid CUDA type dependencies in header
void launchPreprocessKernel(const void* input, void* output,
                           int height, int width,
                           float mean_r, float mean_g, float mean_b,
                           float std_r, float std_g, float std_b,
                           void* stream);

/**
 * FastACV TensorRT Inference - Direct C++ Implementation
 * Uses TensorRT C++ API directly for maximum performance
 * Features:
 * - Direct TensorRT C++ API (no Python wrapper)
 * - Zero-copy GPU pipeline
 * - Optimized preprocessing/postprocessing on GPU
 * - Minimal latency for real-time stereo processing
 */
class FastACVInferenceOptimized {
public:
    struct Config {
        std::string engine_path;
        int input_width;
        int input_height;
        int max_disparity;
        
        Config() : input_width(480), input_height(288), max_disparity(192) {}
    };
    
    FastACVInferenceOptimized(const Config& config);
    ~FastACVInferenceOptimized();
    
    bool initialize();
    bool infer(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity, float& time_ms);
    
private:
    Config config_;
    bool initialized_;
    
    // GPU buffers (if using CUDA directly)
    void* d_left_input_;
    void* d_right_input_;
    void* d_output_;
    
    // TensorRT context
    void* trt_runtime_;
    void* trt_engine_;
    void* trt_context_;
    void* cuda_stream_;
    
    // CPU buffers for preprocessing
    std::vector<float> h_left_buffer_;
    std::vector<float> h_right_buffer_;
    std::vector<float> h_output_buffer_;
    
    bool preprocessGPU(const cv::Mat& left, const cv::Mat& right);
    bool runTensorRT();
    bool postprocessGPU(cv::Mat& disparity);
};

} // namespace ct_uav_stereo

#endif // INFER_FASTACV_HPP_
