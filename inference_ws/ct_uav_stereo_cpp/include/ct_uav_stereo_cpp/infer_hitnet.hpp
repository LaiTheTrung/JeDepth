#ifndef INFER_HITNET_HPP_
#define INFER_HITNET_HPP_

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ct_uav_stereo {

/**
 * HitNet TensorRT Inference - Direct C++ Implementation
 * Uses TensorRT C++ API directly for maximum performance
 * Features:
 * - Direct TensorRT C++ API (no Python wrapper)
 * - Zero-copy GPU pipeline
 * - Optimized preprocessing/postprocessing on GPU
 * - Minimal latency for real-time stereo processing
 * - Automatic padding to multiple of 32
 */
class HitNetInferenceOptimized {
public:
    struct Config {
        std::string engine_path;
        std::string onnx_path;      // Optional: for building engine if not exists
        int input_width;
        int input_height;
        int max_disparity;
        bool fp16;                   // Use FP16 precision
        
        Config() : input_width(320), input_height(240), 
                   max_disparity(192), fp16(true) {}
    };
    
    HitNetInferenceOptimized(const Config& config);
    ~HitNetInferenceOptimized();
    
    /**
     * Initialize TensorRT engine
     * @return true if initialization successful
     */
    bool initialize();
    
    /**
     * Run inference on stereo pair
     * @param left Left image (grayscale or BGR)
     * @param right Right image (grayscale or BGR)
     * @param disparity Output disparity map (CV_32FC1)
     * @param time_ms Processing time in milliseconds
     * @return true if inference successful
     */
    bool infer(const cv::Mat& left, const cv::Mat& right, 
               cv::Mat& disparity, float& time_ms);
    
private:
    Config config_;
    bool initialized_;
    
    // Actual input dimensions (padded to multiple of 32)
    int padded_width_;
    int padded_height_;
    
    // GPU buffers
    void* d_input_;              // Input tensor (1, 2, H, W)
    void* d_output_;             // Output tensor (1, H, W)
    
    // TensorRT context
    void* trt_runtime_;
    void* trt_engine_;
    void* trt_context_;
    void* cuda_stream_;
    
    // CPU buffers
    std::vector<float> h_input_buffer_;
    std::vector<float> h_output_buffer_;
    
    // Tensor names
    std::string input_tensor_name_;
    std::string output_tensor_name_;
    
    /**
     * Build TensorRT engine from ONNX if not exists
     * @return true if successful
     */
    bool buildEngine();
    
    /**
     * Load TensorRT engine from file
     * @return true if successful
     */
    bool loadEngine();
    
    /**
     * Preprocess stereo pair to input tensor format
     * @param left Left image
     * @param right Right image
     * @return true if successful
     */
    bool preprocess(const cv::Mat& left, const cv::Mat& right);
    
    /**
     * Run TensorRT inference
     * @return true if successful
     */
    bool runTensorRT();
    
    /**
     * Postprocess output tensor to disparity map
     * @param disparity Output disparity map
     * @return true if successful
     */
    bool postprocess(cv::Mat& disparity);
    
    /**
     * Calculate padded dimensions (multiple of 32)
     */
    void calculatePaddedDimensions();
};

} // namespace ct_uav_stereo

#endif // INFER_HITNET_HPP_
