#ifndef INFER_FASTFS_HPP_
#define INFER_FASTFS_HPP_

#include <opencv2/opencv.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace ct_uav_stereo {

class FastFSInferenceOptimized {
public:
    struct Config {
        std::string feature_engine_path;
        std::string post_engine_path;
        int input_width;
        int input_height;
        int max_disparity;
        int cv_group;
        bool normalize_gwc;

        Config()
            : input_width(640),
              input_height(448),
              max_disparity(192),
              cv_group(8),
              normalize_gwc(false) {}
    };

    explicit FastFSInferenceOptimized(const Config& config);
    ~FastFSInferenceOptimized();

    bool initialize();
    bool infer(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity, float& time_ms);

private:
    struct TensorBinding {
        void* device_ptr;
        size_t bytes;
        int kernel_dtype;
        std::vector<int> shape;

        TensorBinding() : device_ptr(nullptr), bytes(0), kernel_dtype(-1) {}
    };

    Config config_;
    bool initialized_;

    void* cuda_stream_;

    void* trt_runtime_;
    void* feature_engine_;
    void* feature_context_;
    void* post_engine_;
    void* post_context_;

    std::string feature_left_input_name_;
    std::string feature_right_input_name_;
    std::vector<std::string> feature_output_names_;
    std::string feat_left_04_name_;
    std::string feat_right_04_name_;

    std::vector<std::string> post_input_names_;
    std::string post_output_name_;

    void* d_left_input_;
    void* d_right_input_;
    void* d_left_u8_;
    void* d_right_u8_;

    int left_input_kernel_dtype_;
    int right_input_kernel_dtype_;

    std::unordered_map<std::string, TensorBinding> feature_outputs_;
    std::unordered_map<std::string, TensorBinding> post_input_converted_;
    std::unordered_map<std::string, std::string> post_input_source_;

    TensorBinding gwc_volume_;
    TensorBinding post_output_;

    void* d_post_output_fp32_;

    int feat_batch_;
    int feat_channels_;
    int feat_height_;
    int feat_width_;
    int gwc_disp_;

    std::vector<float> h_post_output_;

    bool loadEngines();
    bool configureFeatureBindings();
    bool configurePostBindings();
    bool preprocessGPU(const cv::Mat& left, const cv::Mat& right);
    bool runFeatureTensorRT();
    bool buildGwcVolume();
    bool runPostTensorRT();
    bool postprocess(cv::Mat& disparity, const cv::Size& orig_size);

    void releaseCudaBuffers();
    void releaseTensorRT();
};

}  // namespace ct_uav_stereo

#endif  // INFER_FASTFS_HPP_
