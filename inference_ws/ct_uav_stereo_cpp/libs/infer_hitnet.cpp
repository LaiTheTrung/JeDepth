#include "ct_uav_stereo_cpp/infer_hitnet.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <chrono>
#include <cstring>

namespace ct_uav_stereo {

// TensorRT Logger (file-local to avoid conflicts)
namespace {
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            // Only print errors and warnings
            if (severity <= Severity::kWARNING) {
                std::cout << "[TensorRT HitNet] " << msg << std::endl;
            }
        }
    } gLogger;
}

HitNetInferenceOptimized::HitNetInferenceOptimized(const Config& config)
    : config_(config),
      initialized_(false),
      padded_width_(0),
      padded_height_(0),
      d_input_(nullptr),
      d_output_(nullptr),
      trt_runtime_(nullptr),
      trt_engine_(nullptr),
      trt_context_(nullptr),
      cuda_stream_(nullptr),
      input_tensor_name_("input"),
      output_tensor_name_("reference_output_disparity") {
}

HitNetInferenceOptimized::~HitNetInferenceOptimized() {
    // Free CUDA buffers
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    
    // Destroy CUDA stream
    if (cuda_stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(cuda_stream_));
    }
    
    // Destroy TensorRT objects (TensorRT 10.x uses delete instead of destroy())
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

void HitNetInferenceOptimized::calculatePaddedDimensions() {
    // Pad to multiple of 32
    padded_width_ = ((config_.input_width + 31) / 32) * 32;
    padded_height_ = ((config_.input_height + 31) / 32) * 32;
    
    std::cout << "[HitNet] Original size: " << config_.input_width 
              << "x" << config_.input_height << std::endl;
    std::cout << "[HitNet] Padded size: " << padded_width_ 
              << "x" << padded_height_ << std::endl;
}

bool HitNetInferenceOptimized::initialize() {
    if (initialized_) {
        return true;
    }
    
    std::cout << "[HitNet] Initializing HitNet TensorRT inference..." << std::endl;
    
    // Calculate padded dimensions
    calculatePaddedDimensions();
    
    // Load or build engine
    if (!loadEngine()) {
        std::cerr << "[HitNet] Failed to load engine, attempting to build..." << std::endl;
        if (!buildEngine()) {
            std::cerr << "[HitNet] Failed to build engine" << std::endl;
            return false;
        }
        if (!loadEngine()) {
            std::cerr << "[HitNet] Failed to load newly built engine" << std::endl;
            return false;
        }
    }
    
    // Create execution context
    auto* engine = static_cast<nvinfer1::ICudaEngine*>(trt_engine_);
    trt_context_ = engine->createExecutionContext();
    if (!trt_context_) {
        std::cerr << "[HitNet] Failed to create execution context" << std::endl;
        return false;
    }
    
    // Create CUDA stream
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "[HitNet] Failed to create CUDA stream" << std::endl;
        return false;
    }
    cuda_stream_ = static_cast<void*>(stream);
    
    // Allocate GPU buffers
    // Input: (1, 2, padded_height, padded_width) float32
    size_t input_size = 1 * 2 * padded_height_ * padded_width_ * sizeof(float);
    if (cudaMalloc(&d_input_, input_size) != cudaSuccess) {
        std::cerr << "[HitNet] Failed to allocate input buffer" << std::endl;
        return false;
    }
    
    // Output: (1, padded_height, padded_width) float32
    size_t output_size = 1 * padded_height_ * padded_width_ * sizeof(float);
    if (cudaMalloc(&d_output_, output_size) != cudaSuccess) {
        std::cerr << "[HitNet] Failed to allocate output buffer" << std::endl;
        return false;
    }
    
    // Allocate CPU buffers
    h_input_buffer_.resize(1 * 2 * padded_height_ * padded_width_);
    h_output_buffer_.resize(1 * padded_height_ * padded_width_);
    
    // Set tensor addresses for TensorRT 10.x API
    auto* context = static_cast<nvinfer1::IExecutionContext*>(trt_context_);
    context->setTensorAddress(input_tensor_name_.c_str(), d_input_);
    context->setTensorAddress(output_tensor_name_.c_str(), d_output_);
    
    initialized_ = true;
    std::cout << "[HitNet] Initialization successful!" << std::endl;
    std::cout << "[HitNet] Input size: (1, 2, " << padded_height_ 
              << ", " << padded_width_ << ")" << std::endl;
    std::cout << "[HitNet] Output size: (1, " << padded_height_ 
              << ", " << padded_width_ << ")" << std::endl;
    
    return true;
}

bool HitNetInferenceOptimized::loadEngine() {
    std::cout << "[HitNet] Loading engine from: " << config_.engine_path << std::endl;
    
    std::ifstream file(config_.engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[HitNet] Engine file not found: " << config_.engine_path << std::endl;
        return false;
    }
    
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();
    
    trt_runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!trt_runtime_) {
        std::cerr << "[HitNet] Failed to create TensorRT runtime" << std::endl;
        return false;
    }
    
    auto* runtime = static_cast<nvinfer1::IRuntime*>(trt_runtime_);
    trt_engine_ = runtime->deserializeCudaEngine(engine_data.data(), size);
    if (!trt_engine_) {
        std::cerr << "[HitNet] Failed to deserialize engine" << std::endl;
        return false;
    }
    
    std::cout << "[HitNet] Engine loaded successfully" << std::endl;
    return true;
}

bool HitNetInferenceOptimized::buildEngine() {
    if (config_.onnx_path.empty()) {
        std::cerr << "[HitNet] ONNX path not provided, cannot build engine" << std::endl;
        return false;
    }
    
    std::cout << "[HitNet] Building engine from ONNX: " << config_.onnx_path << std::endl;
    std::cout << "[HitNet] This may take several minutes..." << std::endl;
    
    // Use system command to build engine (simpler than using Builder API)
    std::string cmd = "trtexec --onnx=" + config_.onnx_path +
                      " --saveEngine=" + config_.engine_path;
    
    if (config_.fp16) {
        cmd += " --fp16";
    }
    
    cmd += " --memPoolSize=workspace:1024 --verbose 2>&1";
    
    std::cout << "[HitNet] Running: " << cmd << std::endl;
    
    int result = system(cmd.c_str());
    if (result != 0) {
        std::cerr << "[HitNet] Failed to build engine (exit code: " << result << ")" << std::endl;
        return false;
    }
    
    std::cout << "[HitNet] Engine built successfully" << std::endl;
    return true;
}

bool HitNetInferenceOptimized::preprocess(const cv::Mat& left, const cv::Mat& right) {
    // Convert to grayscale if needed
    cv::Mat left_gray, right_gray;
    
    if (left.channels() == 3) {
        cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    } else {
        left_gray = left;
    }
    
    if (right.channels() == 3) {
        cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
    } else {
        right_gray = right;
    }
    
    // Resize to padded dimensions
    cv::Mat left_resized, right_resized;
    cv::resize(left_gray, left_resized, cv::Size(padded_width_, padded_height_));
    cv::resize(right_gray, right_resized, cv::Size(padded_width_, padded_height_));
    
    // Convert to float32 and normalize to [0, 1]
    cv::Mat left_float, right_float;
    left_resized.convertTo(left_float, CV_32F, 1.0 / 255.0);
    right_resized.convertTo(right_float, CV_32F, 1.0 / 255.0);
    
    // Pack into input buffer: (1, 2, H, W)
    // Channel 0: left image
    // Channel 1: right image
    size_t pixel_count = padded_height_ * padded_width_;
    
    float* input_ptr = h_input_buffer_.data();
    
    // Copy left image (channel 0)
    for (int y = 0; y < padded_height_; y++) {
        const float* row_ptr = left_float.ptr<float>(y);
        std::memcpy(input_ptr + y * padded_width_, row_ptr, padded_width_ * sizeof(float));
    }
    
    // Copy right image (channel 1)
    input_ptr += pixel_count;
    for (int y = 0; y < padded_height_; y++) {
        const float* row_ptr = right_float.ptr<float>(y);
        std::memcpy(input_ptr + y * padded_width_, row_ptr, padded_width_ * sizeof(float));
    }
    
    return true;
}

bool HitNetInferenceOptimized::runTensorRT() {
    // Copy input to GPU
    size_t input_size = h_input_buffer_.size() * sizeof(float);
    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_);
    
    if (cudaMemcpyAsync(d_input_, h_input_buffer_.data(), input_size,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::cerr << "[HitNet] Failed to copy input to GPU" << std::endl;
        return false;
    }
    
    // Execute inference
    auto* context = static_cast<nvinfer1::IExecutionContext*>(trt_context_);
    if (!context->enqueueV3(stream)) {
        std::cerr << "[HitNet] Failed to execute inference" << std::endl;
        return false;
    }
    
    // Copy output from GPU
    size_t output_size = h_output_buffer_.size() * sizeof(float);
    if (cudaMemcpyAsync(h_output_buffer_.data(), d_output_, output_size,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::cerr << "[HitNet] Failed to copy output from GPU" << std::endl;
        return false;
    }
    
    // Synchronize stream
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::cerr << "[HitNet] Failed to synchronize stream" << std::endl;
        return false;
    }
    
    return true;
}

bool HitNetInferenceOptimized::postprocess(cv::Mat& disparity) {
    // Output shape: (1, padded_height, padded_width)
    // Create cv::Mat from output buffer
    cv::Mat output_mat(padded_height_, padded_width_, CV_32F, h_output_buffer_.data());
    
    // Resize back to original input size
    cv::resize(output_mat, disparity, cv::Size(config_.input_width, config_.input_height));
    
    // Clone to ensure data is owned by output
    disparity = disparity.clone();
    
    return true;
}

bool HitNetInferenceOptimized::infer(const cv::Mat& left, const cv::Mat& right,
                                     cv::Mat& disparity, float& time_ms) {
    if (!initialized_) {
        std::cerr << "[HitNet] Not initialized" << std::endl;
        return false;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Preprocess
    if (!preprocess(left, right)) {
        std::cerr << "[HitNet] Preprocessing failed" << std::endl;
        return false;
    }
    
    // Run inference
    if (!runTensorRT()) {
        std::cerr << "[HitNet] Inference failed" << std::endl;
        return false;
    }
    
    // Postprocess
    if (!postprocess(disparity)) {
        std::cerr << "[HitNet] Postprocessing failed" << std::endl;
        return false;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    return true;
}

} // namespace ct_uav_stereo
