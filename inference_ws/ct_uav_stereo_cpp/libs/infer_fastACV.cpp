#include "ct_uav_stereo_cpp/infer_fastACV.hpp"
#include <chrono>
#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

namespace ct_uav_stereo {

// ============================================================================
// FastACVInferenceOptimized Implementation (TensorRT C++ Direct)
// ============================================================================

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

// Helper function to check CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

// ============================================================================
// FastACVInferenceOptimized Implementation
// ============================================================================

FastACVInferenceOptimized::FastACVInferenceOptimized(const Config& config)
    : config_(config),
      initialized_(false),
      d_left_input_(nullptr),
      d_right_input_(nullptr),
      d_output_(nullptr),
      trt_runtime_(nullptr),
      trt_engine_(nullptr),
      trt_context_(nullptr),
      cuda_stream_(nullptr) {
}

FastACVInferenceOptimized::~FastACVInferenceOptimized() {
    // Free CUDA resources
    if (d_left_input_) cudaFree(d_left_input_);
    if (d_right_input_) cudaFree(d_right_input_);
    if (d_output_) cudaFree(d_output_);
    
    if (cuda_stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(cuda_stream_));
    }
    
    // Delete TensorRT objects in correct order: context -> engine -> runtime
    if (trt_context_) {
        delete static_cast<nvinfer1::IExecutionContext*>(trt_context_);
    }
    if (trt_engine_) {
        delete static_cast<nvinfer1::ICudaEngine*>(trt_engine_);
    }
    if (trt_runtime_) {
        delete static_cast<nvinfer1::IRuntime*>(trt_runtime_);
    }
}

bool FastACVInferenceOptimized::initialize() {
    if (initialized_) {
        return true;
    }
    
    std::cout << "Initializing FastACV TensorRT inference (C++ Direct)..." << std::endl;
    std::cout << "  Engine: " << config_.engine_path << std::endl;
    std::cout << "  Input size: " << config_.input_width << "x" << config_.input_height << std::endl;
    std::cout << "  Max disparity: " << config_.max_disparity << std::endl;
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cuda_stream_ = stream;
    
    // Load TensorRT engine
    std::ifstream engine_file(config_.engine_path, std::ios::binary);
    if (!engine_file) {
        std::cerr << "Failed to open engine file: " << config_.engine_path << std::endl;
        return false;
    }
    
    engine_file.seekg(0, std::ios::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    
    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);
    engine_file.close();
    
    // Deserialize engine
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }
    
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(
        engine_data.data(), engine_size);
    
    if (!engine) {
        std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
        delete runtime;
        return false;
    }
    
    // Store runtime and engine (will be deleted in destructor)
    trt_runtime_ = runtime;
    trt_engine_ = engine;
    
    // Create execution context
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }
    
    trt_context_ = context;
    
    // Allocate GPU buffers
    // Input: 2 images (left, right) of shape [1, 3, H, W] in FP32
    size_t input_size = 1 * 3 * config_.input_height * config_.input_width * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_left_input_, input_size));
    CHECK_CUDA(cudaMalloc(&d_right_input_, input_size));
    
    // Output: disparity map of shape [1, H, W] in FP32
    size_t output_size = 1 * config_.input_height * config_.input_width * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_output_, output_size));
    
    // Allocate CPU buffers
    h_left_buffer_.resize(1 * 3 * config_.input_height * config_.input_width);
    h_right_buffer_.resize(1 * 3 * config_.input_height * config_.input_width);
    h_output_buffer_.resize(1 * config_.input_height * config_.input_width);
    
    initialized_ = true;
    std::cout << "FastACV TensorRT inference initialized successfully" << std::endl;
    
    return true;
}

bool FastACVInferenceOptimized::infer(const cv::Mat& left, const cv::Mat& right, 
                                       cv::Mat& disparity, float& time_ms) {
    if (!initialized_) {
        std::cerr << "Inference not initialized" << std::endl;
        return false;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Store original size
    cv::Size orig_size(left.cols, left.rows);
    
    // Preprocess on GPU
    if (!preprocessGPU(left, right)) {
        std::cerr << "Preprocessing failed" << std::endl;
        return false;
    }
    
    // Run TensorRT inference
    if (!runTensorRT()) {
        std::cerr << "TensorRT inference failed" << std::endl;
        return false;
    }
    
    // Postprocess on GPU
    if (!postprocessGPU(disparity)) {
        std::cerr << "Postprocessing failed" << std::endl;
        return false;
    }
    
    // Resize back to original size if needed
    if (disparity.rows != orig_size.height || disparity.cols != orig_size.width) {
        cv::Mat disparity_resized;
        cv::resize(disparity, disparity_resized, orig_size, 0, 0, cv::INTER_LINEAR);
        disparity = disparity_resized;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    return true;
}

bool FastACVInferenceOptimized::preprocessGPU(const cv::Mat& left, const cv::Mat& right) {
    // ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
    
    auto preprocessImage = [this, &mean, &std](const cv::Mat& img, void* d_output) -> bool {
        // Resize using OpenCV CPU
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(config_.input_width, config_.input_height), 
                  0, 0, cv::INTER_LINEAR);
        
        // Allocate temporary GPU buffer for input image
        void* d_temp;
        size_t temp_size = resized.total() * resized.elemSize();
        CHECK_CUDA(cudaMalloc(&d_temp, temp_size));
        
        // Upload resized image to GPU
        CHECK_CUDA(cudaMemcpyAsync(d_temp, resized.data, temp_size,
                                   cudaMemcpyHostToDevice,
                                   static_cast<cudaStream_t>(cuda_stream_)));
        
        // Launch CUDA kernel for preprocessing (BGR->RGB, normalize, HWC->CHW)
        launchPreprocessKernel(
            d_temp,
            d_output,
            config_.input_height, config_.input_width,
            mean[0], mean[1], mean[2],
            std[0], std[1], std[2],
            cuda_stream_
        );
        
        // Free temporary buffer
        cudaFree(d_temp);
        
        return true;
    };
    
    // Preprocess left and right images directly to GPU buffers
    if (!preprocessImage(left, d_left_input_)) {
        return false;
    }
    
    if (!preprocessImage(right, d_right_input_)) {
        return false;
    }
    
    // Sync only once after both preprocessing
    CHECK_CUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream_)));
    
    return true;
}

bool FastACVInferenceOptimized::runTensorRT() {
    nvinfer1::IExecutionContext* context = static_cast<nvinfer1::IExecutionContext*>(trt_context_);
    nvinfer1::ICudaEngine* engine = static_cast<nvinfer1::ICudaEngine*>(trt_engine_);
    
    // Set up tensor addresses for TensorRT 10+
    // Model uses: left_image, right_image, output
    const char* input_names[] = {"left_image", "right_image"};
    const char* output_name = "output";
    void* input_buffers[] = {d_left_input_, d_right_input_};
    
    // Set input tensor addresses
    for (int i = 0; i < 2; i++) {
        if (!context->setTensorAddress(input_names[i], input_buffers[i])) {
            std::cerr << "Failed to set tensor address for " << input_names[i] << std::endl;
            // Fallback: print available tensor names
            std::cerr << "Available tensors:" << std::endl;
            for (int j = 0; j < engine->getNbIOTensors(); j++) {
                const char* name = engine->getIOTensorName(j);
                std::cerr << "  [" << j << "] " << name << std::endl;
            }
            return false;
        }
    }
    
    // Set output tensor address
    if (!context->setTensorAddress(output_name, d_output_)) {
        std::cerr << "Failed to set output tensor address" << std::endl;
        return false;
    }
    
    // Execute inference
    bool status = context->enqueueV3(static_cast<cudaStream_t>(cuda_stream_));
    
    if (!status) {
        std::cerr << "TensorRT execution failed" << std::endl;
        return false;
    }
    
    CHECK_CUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream_)));
    
    return true;
}

bool FastACVInferenceOptimized::postprocessGPU(cv::Mat& disparity) {
    // Download output from GPU
    size_t output_size = h_output_buffer_.size() * sizeof(float);
    CHECK_CUDA(cudaMemcpyAsync(h_output_buffer_.data(), d_output_,
                               output_size, cudaMemcpyDeviceToHost,
                               static_cast<cudaStream_t>(cuda_stream_)));
    
    CHECK_CUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream_)));
    
    // Convert to cv::Mat
    disparity = cv::Mat(config_.input_height, config_.input_width, CV_32F,
                        h_output_buffer_.data()).clone();
    
    return true;
}

} // namespace ct_uav_stereo
