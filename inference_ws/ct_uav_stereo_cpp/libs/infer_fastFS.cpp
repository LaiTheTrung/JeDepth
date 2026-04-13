#include "ct_uav_stereo_cpp/infer_fastFS.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>

namespace ct_uav_stereo {

void launchFastFSPreprocessKernel(const void* input_bgr, void* output_chw,
                                  int height, int width,
                                  int output_type,
                                  void* stream);

void launchFastFSGwcVolumeKernel(const void* left_feat, const void* right_feat,
                                 void* output_volume,
                                 int batch, int channels, int height, int width,
                                 int num_groups, int max_disp, bool normalize,
                                 int input_type, int output_type,
                                 void* stream);

void launchFastFSCastKernel(const void* input, void* output, size_t count,
                            int input_type, int output_type,
                            void* stream);

namespace {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT FastFS] " << msg << std::endl;
        }
    }
};

Logger g_fastfs_logger;

constexpr int kKernelFloat32 = 0;
constexpr int kKernelFloat16 = 1;

#define CHECK_CUDA(call)                                                                   \
    do {                                                                                   \
        cudaError_t status = (call);                                                       \
        if (status != cudaSuccess) {                                                       \
            std::cerr << "[FastFS] CUDA Error: " << cudaGetErrorString(status)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;             \
            return false;                                                                  \
        }                                                                                  \
    } while (0)

size_t dtypeElementSize(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT:
            return sizeof(float);
        case nvinfer1::DataType::kHALF:
            return sizeof(uint16_t);
        default:
            return 0;
    }
}

int trtToKernelType(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT:
            return kKernelFloat32;
        case nvinfer1::DataType::kHALF:
            return kKernelFloat16;
        default:
            return -1;
    }
}

size_t numElementsFromShape(const std::vector<int>& shape) {
    if (shape.empty()) {
        return 0;
    }
    size_t elems = 1;
    for (int d : shape) {
        if (d < 0) {
            return 0;
        }
        elems *= static_cast<size_t>(d);
    }
    return elems;
}

std::vector<int> dimsToVector(const nvinfer1::Dims& dims) {
    std::vector<int> out;
    out.reserve(dims.nbDims);
    for (int i = 0; i < dims.nbDims; ++i) {
        out.push_back(dims.d[i]);
    }
    return out;
}

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string pickTensorNameContains(const std::vector<std::string>& names,
                                   const std::string& needle_lower,
                                   const std::string& fallback,
                                   int fallback_index) {
    for (const auto& n : names) {
        if (toLower(n).find(needle_lower) != std::string::npos) {
            return n;
        }
    }

    if (!fallback.empty()) {
        return fallback;
    }

    if (fallback_index >= 0 && fallback_index < static_cast<int>(names.size())) {
        return names[static_cast<size_t>(fallback_index)];
    }

    return std::string();
}

std::vector<std::string> getIOTensorNames(nvinfer1::ICudaEngine* engine,
                                          nvinfer1::TensorIOMode mode) {
    std::vector<std::string> names;
    const int n = engine->getNbIOTensors();
    names.reserve(static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == mode) {
            names.emplace_back(name);
        }
    }
    return names;
}

bool setInputShapeChecked(nvinfer1::IExecutionContext* ctx,
                          const std::string& name,
                          const std::vector<int>& shape) {
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int>(shape.size());
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = shape[static_cast<size_t>(i)];
    }
    return ctx->setInputShape(name.c_str(), dims);
}

bool isPositiveShape(const std::vector<int>& shape) {
    return !shape.empty() &&
           std::all_of(shape.begin(), shape.end(), [](int d) { return d > 0; });
}

}  // namespace

FastFSInferenceOptimized::FastFSInferenceOptimized(const Config& config)
    : config_(config),
      initialized_(false),
      cuda_stream_(nullptr),
      trt_runtime_(nullptr),
      feature_engine_(nullptr),
      feature_context_(nullptr),
      post_engine_(nullptr),
      post_context_(nullptr),
      d_left_input_(nullptr),
      d_right_input_(nullptr),
      d_left_u8_(nullptr),
      d_right_u8_(nullptr),
      left_input_kernel_dtype_(-1),
      right_input_kernel_dtype_(-1),
      d_post_output_fp32_(nullptr),
      feat_batch_(0),
      feat_channels_(0),
      feat_height_(0),
      feat_width_(0),
      gwc_disp_(0) {
}

FastFSInferenceOptimized::~FastFSInferenceOptimized() {
    releaseCudaBuffers();
    releaseTensorRT();
}

void FastFSInferenceOptimized::releaseCudaBuffers() {
    if (d_left_input_) {
        cudaFree(d_left_input_);
        d_left_input_ = nullptr;
    }
    if (d_right_input_) {
        cudaFree(d_right_input_);
        d_right_input_ = nullptr;
    }
    if (d_left_u8_) {
        cudaFree(d_left_u8_);
        d_left_u8_ = nullptr;
    }
    if (d_right_u8_) {
        cudaFree(d_right_u8_);
        d_right_u8_ = nullptr;
    }

    if (gwc_volume_.device_ptr) {
        cudaFree(gwc_volume_.device_ptr);
        gwc_volume_.device_ptr = nullptr;
    }

    if (post_output_.device_ptr) {
        cudaFree(post_output_.device_ptr);
        post_output_.device_ptr = nullptr;
    }

    if (d_post_output_fp32_) {
        cudaFree(d_post_output_fp32_);
        d_post_output_fp32_ = nullptr;
    }

    for (auto& kv : feature_outputs_) {
        if (kv.second.device_ptr) {
            cudaFree(kv.second.device_ptr);
            kv.second.device_ptr = nullptr;
        }
    }
    feature_outputs_.clear();

    for (auto& kv : post_input_converted_) {
        if (kv.second.device_ptr) {
            cudaFree(kv.second.device_ptr);
            kv.second.device_ptr = nullptr;
        }
    }
    post_input_converted_.clear();

    if (cuda_stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(cuda_stream_));
        cuda_stream_ = nullptr;
    }
}

void FastFSInferenceOptimized::releaseTensorRT() {
    if (feature_context_) {
        delete static_cast<nvinfer1::IExecutionContext*>(feature_context_);
        feature_context_ = nullptr;
    }
    if (post_context_) {
        delete static_cast<nvinfer1::IExecutionContext*>(post_context_);
        post_context_ = nullptr;
    }
    if (feature_engine_) {
        delete static_cast<nvinfer1::ICudaEngine*>(feature_engine_);
        feature_engine_ = nullptr;
    }
    if (post_engine_) {
        delete static_cast<nvinfer1::ICudaEngine*>(post_engine_);
        post_engine_ = nullptr;
    }
    if (trt_runtime_) {
        delete static_cast<nvinfer1::IRuntime*>(trt_runtime_);
        trt_runtime_ = nullptr;
    }
}

bool FastFSInferenceOptimized::loadEngines() {
    auto* runtime = static_cast<nvinfer1::IRuntime*>(trt_runtime_);

    auto load_one = [runtime](const std::string& path, void*& engine_out) -> bool {
        std::ifstream file(path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "[FastFS] Engine file not found: " << path << std::endl;
            return false;
        }

        file.seekg(0, std::ios::end);
        const size_t size = static_cast<size_t>(file.tellg());
        file.seekg(0, std::ios::beg);

        std::vector<char> data(size);
        file.read(data.data(), static_cast<std::streamsize>(size));
        file.close();

        auto* engine = runtime->deserializeCudaEngine(data.data(), size);
        if (!engine) {
            std::cerr << "[FastFS] Failed to deserialize engine: " << path << std::endl;
            return false;
        }

        engine_out = engine;
        return true;
    };

    if (!load_one(config_.feature_engine_path, feature_engine_)) {
        return false;
    }
    if (!load_one(config_.post_engine_path, post_engine_)) {
        return false;
    }

    auto* feature_engine = static_cast<nvinfer1::ICudaEngine*>(feature_engine_);
    auto* post_engine = static_cast<nvinfer1::ICudaEngine*>(post_engine_);

    feature_context_ = feature_engine->createExecutionContext();
    post_context_ = post_engine->createExecutionContext();

    if (!feature_context_ || !post_context_) {
        std::cerr << "[FastFS] Failed to create TensorRT execution context" << std::endl;
        return false;
    }

    return true;
}

bool FastFSInferenceOptimized::configureFeatureBindings() {
    auto* feature_engine = static_cast<nvinfer1::ICudaEngine*>(feature_engine_);
    auto* feature_ctx = static_cast<nvinfer1::IExecutionContext*>(feature_context_);

    auto feature_inputs = getIOTensorNames(feature_engine, nvinfer1::TensorIOMode::kINPUT);
    if (feature_inputs.size() != 2) {
        std::cerr << "[FastFS] Expected 2 inputs for feature engine, got "
                  << feature_inputs.size() << std::endl;
        return false;
    }

    feature_left_input_name_ = pickTensorNameContains(feature_inputs, "left", "", 0);
    feature_right_input_name_ = pickTensorNameContains(feature_inputs, "right", "", 1);

    if (feature_left_input_name_.empty() || feature_right_input_name_.empty()) {
        std::cerr << "[FastFS] Failed to determine feature input tensor names" << std::endl;
        return false;
    }

    const std::vector<int> input_shape = {1, 3, config_.input_height, config_.input_width};
    if (!setInputShapeChecked(feature_ctx, feature_left_input_name_, input_shape) ||
        !setInputShapeChecked(feature_ctx, feature_right_input_name_, input_shape)) {
        std::cerr << "[FastFS] Failed to set feature input shapes" << std::endl;
        return false;
    }

    const auto left_dtype = feature_engine->getTensorDataType(feature_left_input_name_.c_str());
    const auto right_dtype = feature_engine->getTensorDataType(feature_right_input_name_.c_str());
    const size_t input_elems = static_cast<size_t>(config_.input_width) *
                               static_cast<size_t>(config_.input_height) * 3;

    const size_t left_bytes = dtypeElementSize(left_dtype) * input_elems;
    const size_t right_bytes = dtypeElementSize(right_dtype) * input_elems;

    left_input_kernel_dtype_ = trtToKernelType(left_dtype);
    right_input_kernel_dtype_ = trtToKernelType(right_dtype);

    if (left_bytes == 0 || right_bytes == 0 ||
        left_input_kernel_dtype_ < 0 || right_input_kernel_dtype_ < 0) {
        std::cerr << "[FastFS] Unsupported feature input dtype" << std::endl;
        return false;
    }

    CHECK_CUDA(cudaMalloc(&d_left_input_, left_bytes));
    CHECK_CUDA(cudaMalloc(&d_right_input_, right_bytes));

    const size_t image_u8_bytes = static_cast<size_t>(config_.input_width) *
                                  static_cast<size_t>(config_.input_height) * 3;
    CHECK_CUDA(cudaMalloc(&d_left_u8_, image_u8_bytes));
    CHECK_CUDA(cudaMalloc(&d_right_u8_, image_u8_bytes));

    feature_output_names_ = getIOTensorNames(feature_engine, nvinfer1::TensorIOMode::kOUTPUT);
    if (feature_output_names_.empty()) {
        std::cerr << "[FastFS] Feature engine has no outputs" << std::endl;
        return false;
    }

    for (const auto& name : feature_output_names_) {
        const auto dtype = feature_engine->getTensorDataType(name.c_str());
        const int kernel_type = trtToKernelType(dtype);
        const size_t elem_size = dtypeElementSize(dtype);

        if (kernel_type < 0 || elem_size == 0) {
            std::cerr << "[FastFS] Unsupported feature output dtype for tensor: "
                      << name << std::endl;
            return false;
        }

        const auto shape = dimsToVector(feature_ctx->getTensorShape(name.c_str()));
        if (!isPositiveShape(shape)) {
            std::cerr << "[FastFS] Invalid shape for feature output tensor: " << name << std::endl;
            return false;
        }

        TensorBinding binding;
        binding.kernel_dtype = kernel_type;
        binding.shape = shape;
        binding.bytes = elem_size * numElementsFromShape(shape);

        CHECK_CUDA(cudaMalloc(&binding.device_ptr, binding.bytes));

        feature_outputs_[name] = binding;
    }

    feat_left_04_name_ = pickTensorNameContains(feature_output_names_, "features_left_04", "", -1);
    feat_right_04_name_ = pickTensorNameContains(feature_output_names_, "features_right_04", "", -1);

    if (feat_left_04_name_.empty() || feat_right_04_name_.empty()) {
        std::cerr << "[FastFS] Required tensors features_left_04/features_right_04 not found." << std::endl;
        std::cerr << "[FastFS] Available feature outputs:" << std::endl;
        for (const auto& name : feature_output_names_) {
            std::cerr << "  - " << name << std::endl;
        }
        return false;
    }

    const auto& feat_shape = feature_outputs_[feat_left_04_name_].shape;
    if (feat_shape.size() != 4) {
        std::cerr << "[FastFS] features_left_04 must be 4D NCHW" << std::endl;
        return false;
    }

    feat_batch_ = feat_shape[0];
    feat_channels_ = feat_shape[1];
    feat_height_ = feat_shape[2];
    feat_width_ = feat_shape[3];

    if (feat_batch_ <= 0 || feat_channels_ <= 0 || feat_height_ <= 0 || feat_width_ <= 0) {
        std::cerr << "[FastFS] Invalid features_left_04 shape values" << std::endl;
        return false;
    }

    if (feat_channels_ % config_.cv_group != 0) {
        std::cerr << "[FastFS] Channel count " << feat_channels_
                  << " is not divisible by cv_group=" << config_.cv_group << std::endl;
        return false;
    }

    gwc_disp_ = std::max(1, config_.max_disparity / 4);
    return true;
}

bool FastFSInferenceOptimized::configurePostBindings() {
    auto* post_engine = static_cast<nvinfer1::ICudaEngine*>(post_engine_);
    auto* post_ctx = static_cast<nvinfer1::IExecutionContext*>(post_context_);

    post_input_names_ = getIOTensorNames(post_engine, nvinfer1::TensorIOMode::kINPUT);
    if (post_input_names_.empty()) {
        std::cerr << "[FastFS] Post engine has no input tensors" << std::endl;
        return false;
    }

    post_input_source_.clear();

    for (const auto& input_name : post_input_names_) {
        if (toLower(input_name) == "gwc_volume") {
            const std::vector<int> gwc_shape = {
                feat_batch_, config_.cv_group, gwc_disp_, feat_height_, feat_width_};

            if (!setInputShapeChecked(post_ctx, input_name, gwc_shape)) {
                std::cerr << "[FastFS] Failed to set post input shape for gwc_volume" << std::endl;
                return false;
            }

            auto shape_runtime = dimsToVector(post_ctx->getTensorShape(input_name.c_str()));
            if (!isPositiveShape(shape_runtime)) {
                shape_runtime = gwc_shape;
            }

            const auto dtype = post_engine->getTensorDataType(input_name.c_str());
            const size_t elem_size = dtypeElementSize(dtype);
            const int kernel_dtype = trtToKernelType(dtype);

            if (elem_size == 0 || kernel_dtype < 0) {
                std::cerr << "[FastFS] Unsupported dtype for gwc_volume input" << std::endl;
                return false;
            }

            gwc_volume_.shape = shape_runtime;
            gwc_volume_.kernel_dtype = kernel_dtype;
            gwc_volume_.bytes = elem_size * numElementsFromShape(shape_runtime);
            CHECK_CUDA(cudaMalloc(&gwc_volume_.device_ptr, gwc_volume_.bytes));

            // Use engine-resolved D if available
            if (gwc_volume_.shape.size() == 5 && gwc_volume_.shape[2] > 0) {
                gwc_disp_ = gwc_volume_.shape[2];
            }

            continue;
        }

        auto it = feature_outputs_.find(input_name);
        if (it == feature_outputs_.end()) {
            std::cerr << "[FastFS] Post input tensor not found in feature outputs: "
                      << input_name << std::endl;
            std::cerr << "[FastFS] Available feature outputs:" << std::endl;
            for (const auto& kv : feature_outputs_) {
                std::cerr << "  - " << kv.first << std::endl;
            }
            return false;
        }

        const auto& src_binding = it->second;
        if (!setInputShapeChecked(post_ctx, input_name, src_binding.shape)) {
            std::cerr << "[FastFS] Failed to set post input shape for: " << input_name << std::endl;
            return false;
        }

        post_input_source_[input_name] = input_name;

        const auto post_dtype = post_engine->getTensorDataType(input_name.c_str());
        const int post_kernel_dtype = trtToKernelType(post_dtype);
        const size_t post_elem_size = dtypeElementSize(post_dtype);

        if (post_kernel_dtype < 0 || post_elem_size == 0) {
            std::cerr << "[FastFS] Unsupported post input dtype for tensor: "
                      << input_name << std::endl;
            return false;
        }

        if (post_kernel_dtype != src_binding.kernel_dtype) {
            TensorBinding cast_binding;
            cast_binding.kernel_dtype = post_kernel_dtype;
            cast_binding.shape = src_binding.shape;
            cast_binding.bytes = post_elem_size * numElementsFromShape(src_binding.shape);
            CHECK_CUDA(cudaMalloc(&cast_binding.device_ptr, cast_binding.bytes));
            post_input_converted_[input_name] = cast_binding;
        }
    }

    auto post_outputs = getIOTensorNames(post_engine, nvinfer1::TensorIOMode::kOUTPUT);
    if (post_outputs.empty()) {
        std::cerr << "[FastFS] Post engine has no outputs" << std::endl;
        return false;
    }

    post_output_name_ = pickTensorNameContains(post_outputs, "disp", post_outputs[0], 0);

    const auto post_output_dtype = post_engine->getTensorDataType(post_output_name_.c_str());
    const int post_output_kernel_dtype = trtToKernelType(post_output_dtype);
    const size_t post_output_elem_size = dtypeElementSize(post_output_dtype);

    if (post_output_kernel_dtype < 0 || post_output_elem_size == 0) {
        std::cerr << "[FastFS] Unsupported post output dtype" << std::endl;
        return false;
    }

    post_output_.shape = dimsToVector(post_ctx->getTensorShape(post_output_name_.c_str()));
    if (!isPositiveShape(post_output_.shape)) {
        std::cerr << "[FastFS] Invalid post output shape for tensor: " << post_output_name_ << std::endl;
        return false;
    }

    post_output_.kernel_dtype = post_output_kernel_dtype;
    post_output_.bytes = post_output_elem_size * numElementsFromShape(post_output_.shape);
    CHECK_CUDA(cudaMalloc(&post_output_.device_ptr, post_output_.bytes));

    const size_t output_count = numElementsFromShape(post_output_.shape);
    h_post_output_.resize(output_count, 0.0f);

    if (post_output_.kernel_dtype == kKernelFloat16) {
        CHECK_CUDA(cudaMalloc(&d_post_output_fp32_, output_count * sizeof(float)));
    }

    return true;
}

bool FastFSInferenceOptimized::initialize() {
    if (initialized_) {
        return true;
    }

    std::cout << "[FastFS] Initializing Fast Foundation Stereo TensorRT inference..." << std::endl;
    std::cout << "[FastFS] feature engine: " << config_.feature_engine_path << std::endl;
    std::cout << "[FastFS] post engine: " << config_.post_engine_path << std::endl;
    std::cout << "[FastFS] input size: " << config_.input_width << "x" << config_.input_height << std::endl;
    std::cout << "[FastFS] max disparity: " << config_.max_disparity << std::endl;
    std::cout << "[FastFS] Refined Itteration: "<< config_.cv_group << std::endl;


    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cuda_stream_ = stream;

    trt_runtime_ = nvinfer1::createInferRuntime(g_fastfs_logger);
    if (!trt_runtime_) {
        std::cerr << "[FastFS] Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    if (!loadEngines()) {
        return false;
    }

    if (!configureFeatureBindings()) {
        return false;
    }

    if (!configurePostBindings()) {
        return false;
    }

    initialized_ = true;

    std::cout << "[FastFS] Initialization successful" << std::endl;
    std::cout << "[FastFS] features_left_04 shape: ("
              << feat_batch_ << ", " << feat_channels_ << ", "
              << feat_height_ << ", " << feat_width_ << ")" << std::endl;

    return true;
}

bool FastFSInferenceOptimized::preprocessGPU(const cv::Mat& left, const cv::Mat& right) {
    auto preprocess_one = [this](const cv::Mat& src, void* dst_u8, void* dst_input, int output_kernel_dtype) -> bool {
        cv::Mat src_bgr;
        if (src.channels() == 3) {
            src_bgr = src;
        } else if (src.channels() == 1) {
            cv::cvtColor(src, src_bgr, cv::COLOR_GRAY2BGR);
        } else if (src.channels() == 4) {
            cv::cvtColor(src, src_bgr, cv::COLOR_BGRA2BGR);
        } else {
            std::cerr << "[FastFS] Unsupported input channel count: " << src.channels() << std::endl;
            return false;
        }

        cv::Mat resized;
        cv::resize(src_bgr, resized,
                   cv::Size(config_.input_width, config_.input_height),
                   0, 0, cv::INTER_LINEAR);

        if (!resized.isContinuous()) {
            resized = resized.clone();
        }

        const size_t bytes_u8 = static_cast<size_t>(config_.input_width) *
                                static_cast<size_t>(config_.input_height) * 3;

        CHECK_CUDA(cudaMemcpyAsync(dst_u8, resized.data, bytes_u8,
                                   cudaMemcpyHostToDevice,
                                   static_cast<cudaStream_t>(cuda_stream_)));

        launchFastFSPreprocessKernel(
            dst_u8, dst_input,
            config_.input_height, config_.input_width,
            output_kernel_dtype,
            cuda_stream_);

        return true;
    };

    if (!preprocess_one(left, d_left_u8_, d_left_input_, left_input_kernel_dtype_)) {
        return false;
    }
    if (!preprocess_one(right, d_right_u8_, d_right_input_, right_input_kernel_dtype_)) {
        return false;
    }

    return true;
}

bool FastFSInferenceOptimized::runFeatureTensorRT() {
    auto* feature_ctx = static_cast<nvinfer1::IExecutionContext*>(feature_context_);

    if (!feature_ctx->setTensorAddress(feature_left_input_name_.c_str(), d_left_input_)) {
        std::cerr << "[FastFS] Failed to bind feature left input" << std::endl;
        return false;
    }
    if (!feature_ctx->setTensorAddress(feature_right_input_name_.c_str(), d_right_input_)) {
        std::cerr << "[FastFS] Failed to bind feature right input" << std::endl;
        return false;
    }

    for (const auto& name : feature_output_names_) {
        auto it = feature_outputs_.find(name);
        if (it == feature_outputs_.end() || !it->second.device_ptr) {
            std::cerr << "[FastFS] Missing feature output buffer: " << name << std::endl;
            return false;
        }
        if (!feature_ctx->setTensorAddress(name.c_str(), it->second.device_ptr)) {
            std::cerr << "[FastFS] Failed to bind feature output: " << name << std::endl;
            return false;
        }
    }

    if (!feature_ctx->enqueueV3(static_cast<cudaStream_t>(cuda_stream_))) {
        std::cerr << "[FastFS] Feature TensorRT execution failed" << std::endl;
        return false;
    }

    return true;
}

bool FastFSInferenceOptimized::buildGwcVolume() {
    auto it_left = feature_outputs_.find(feat_left_04_name_);
    auto it_right = feature_outputs_.find(feat_right_04_name_);
    if (it_left == feature_outputs_.end() || it_right == feature_outputs_.end()) {
        std::cerr << "[FastFS] Missing left/right feature tensors for gwc_volume" << std::endl;
        return false;
    }

    const auto& left_binding = it_left->second;
    const auto& right_binding = it_right->second;

    if (left_binding.kernel_dtype != right_binding.kernel_dtype) {
        std::cerr << "[FastFS] features_left_04 and features_right_04 have different dtypes" << std::endl;
        return false;
    }

    launchFastFSGwcVolumeKernel(
        left_binding.device_ptr,
        right_binding.device_ptr,
        gwc_volume_.device_ptr,
        feat_batch_, feat_channels_, feat_height_, feat_width_,
        config_.cv_group, gwc_disp_, config_.normalize_gwc,
        left_binding.kernel_dtype, gwc_volume_.kernel_dtype,
        cuda_stream_);

    // Cast other post inputs when their dtype doesn't match feature outputs
    for (const auto& kv : post_input_converted_) {
        const std::string& name = kv.first;
        const TensorBinding& dst_binding = kv.second;

        auto src_it = feature_outputs_.find(name);
        if (src_it == feature_outputs_.end()) {
            std::cerr << "[FastFS] Missing source tensor for cast: " << name << std::endl;
            return false;
        }

        const auto& src_binding = src_it->second;
        const size_t count = numElementsFromShape(src_binding.shape);

        launchFastFSCastKernel(
            src_binding.device_ptr,
            dst_binding.device_ptr,
            count,
            src_binding.kernel_dtype,
            dst_binding.kernel_dtype,
            cuda_stream_);
    }

    return true;
}

bool FastFSInferenceOptimized::runPostTensorRT() {
    auto* post_ctx = static_cast<nvinfer1::IExecutionContext*>(post_context_);

    for (const auto& name : post_input_names_) {
        void* ptr = nullptr;

        if (toLower(name) == "gwc_volume") {
            ptr = gwc_volume_.device_ptr;
        } else {
            auto cast_it = post_input_converted_.find(name);
            if (cast_it != post_input_converted_.end()) {
                ptr = cast_it->second.device_ptr;
            } else {
                auto feat_it = feature_outputs_.find(name);
                if (feat_it == feature_outputs_.end()) {
                    std::cerr << "[FastFS] Missing tensor for post input: " << name << std::endl;
                    return false;
                }
                ptr = feat_it->second.device_ptr;
            }
        }

        if (!post_ctx->setTensorAddress(name.c_str(), ptr)) {
            std::cerr << "[FastFS] Failed to bind post input: " << name << std::endl;
            return false;
        }
    }

    if (!post_ctx->setTensorAddress(post_output_name_.c_str(), post_output_.device_ptr)) {
        std::cerr << "[FastFS] Failed to bind post output: " << post_output_name_ << std::endl;
        return false;
    }

    if (!post_ctx->enqueueV3(static_cast<cudaStream_t>(cuda_stream_))) {
        std::cerr << "[FastFS] Post TensorRT execution failed" << std::endl;
        return false;
    }

    return true;
}

bool FastFSInferenceOptimized::postprocess(cv::Mat& disparity, const cv::Size& orig_size) {
    const size_t output_count = numElementsFromShape(post_output_.shape);
    if (output_count == 0 || h_post_output_.size() != output_count) {
        std::cerr << "[FastFS] Invalid post output size" << std::endl;
        return false;
    }

    if (post_output_.kernel_dtype == kKernelFloat16) {
        if (!d_post_output_fp32_) {
            std::cerr << "[FastFS] Missing fp32 output scratch buffer" << std::endl;
            return false;
        }
        launchFastFSCastKernel(post_output_.device_ptr, d_post_output_fp32_, output_count,
                               kKernelFloat16, kKernelFloat32, cuda_stream_);
        CHECK_CUDA(cudaMemcpyAsync(h_post_output_.data(), d_post_output_fp32_,
                                   output_count * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   static_cast<cudaStream_t>(cuda_stream_)));
    } else if (post_output_.kernel_dtype == kKernelFloat32) {
        CHECK_CUDA(cudaMemcpyAsync(h_post_output_.data(), post_output_.device_ptr,
                                   output_count * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   static_cast<cudaStream_t>(cuda_stream_)));
    } else {
        std::cerr << "[FastFS] Unsupported post output kernel dtype" << std::endl;
        return false;
    }

    CHECK_CUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream_)));

    if (post_output_.shape.size() < 2) {
        std::cerr << "[FastFS] Post output shape has less than 2 dims" << std::endl;
        return false;
    }

    const int out_h = post_output_.shape[post_output_.shape.size() - 2];
    const int out_w = post_output_.shape[post_output_.shape.size() - 1];

    if (out_h <= 0 || out_w <= 0) {
        std::cerr << "[FastFS] Invalid post output spatial shape" << std::endl;
        return false;
    }

    cv::Mat disp_model(out_h, out_w, CV_32F, h_post_output_.data());
    cv::Mat disp_clamped;
    cv::max(disp_model, 0.0f, disp_clamped);

    const float scale_x = static_cast<float>(orig_size.width) /
                          static_cast<float>(std::max(1, config_.input_width));
    disp_clamped *= scale_x;

    if (orig_size.width != out_w || orig_size.height != out_h) {
        cv::resize(disp_clamped, disparity, orig_size, 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        disparity = disp_clamped;
    }

    disparity = disparity.clone();
    return true;
}

bool FastFSInferenceOptimized::infer(const cv::Mat& left, const cv::Mat& right,
                                     cv::Mat& disparity, float& time_ms) {
    if (!initialized_) {
        std::cerr << "[FastFS] Inference not initialized" << std::endl;
        return false;
    }

    if (left.empty() || right.empty()) {
        std::cerr << "[FastFS] Empty input image" << std::endl;
        return false;
    }

    const cv::Size orig_size(left.cols, left.rows);

    auto start = std::chrono::high_resolution_clock::now();

    if (!preprocessGPU(left, right)) {
        std::cerr << "[FastFS] Preprocess failed" << std::endl;
        return false;
    }

    if (!runFeatureTensorRT()) {
        std::cerr << "[FastFS] Feature inference failed" << std::endl;
        return false;
    }

    if (!buildGwcVolume()) {
        std::cerr << "[FastFS] Build gwc_volume failed" << std::endl;
        return false;
    }

    if (!runPostTensorRT()) {
        std::cerr << "[FastFS] Post inference failed" << std::endl;
        return false;
    }

    if (!postprocess(disparity, orig_size)) {
        std::cerr << "[FastFS] Postprocess failed" << std::endl;
        return false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    time_ms = std::chrono::duration<float, std::milli>(end - start).count();

    return true;
}

}  // namespace ct_uav_stereo
