#include "ct_uav_stereo_cpp/infer_sgm.hpp"
#include <libsgm_wrapper.h>
#include <iostream>
#include <chrono>

namespace ct_uav_stereo {

SGMInference::SGMInference(const Config& config)
    : config_(config), initialized_(false), sgm_matcher_(nullptr) {
}

SGMInference::~SGMInference() {
    if (sgm_matcher_) {
        delete static_cast<sgm::LibSGMWrapper*>(sgm_matcher_);
        sgm_matcher_ = nullptr;
    }
}

bool SGMInference::initialize() {
    if (initialized_) {
        std::cout << "[SGMInference] Already initialized" << std::endl;
        return true;
    }
    
    try {
        // Validate configuration
        if (config_.disparity_size != 64 && config_.disparity_size != 128 && config_.disparity_size != 256) {
            std::cerr << "[SGMInference] Error: disparity_size must be 64, 128, or 256" << std::endl;
            return false;
        }
        
        // Setup SGM parameters
        sgm::PathType path_type = config_.use_8path ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;
        
        // Create LibSGMWrapper - much simpler than raw StereoSGM!
        sgm_matcher_ = new sgm::LibSGMWrapper(
            config_.disparity_size,    // numDisparity
            config_.P1,                // P1
            config_.P2,                // P2
            config_.uniqueness,        // uniquenessRatio
            config_.subpixel,          // subpixel
            path_type,                 // pathType
            config_.min_disparity,     // minDisparity
            config_.LR_max_diff,       // lrMaxDiff
            sgm::CensusType::SYMMETRIC_CENSUS_9x7  // censusType
        );
        
        initialized_ = true;
        
        std::cout << "[SGMInference] Initialized successfully:" << std::endl;
        std::cout << "  - Image size: " << config_.width << "x" << config_.height << std::endl;
        std::cout << "  - Disparity size: " << config_.disparity_size << std::endl;
        std::cout << "  - Min disparity: " << config_.min_disparity << std::endl;
        std::cout << "  - P1/P2: " << config_.P1 << "/" << config_.P2 << std::endl;
        std::cout << "  - Subpixel: " << (config_.subpixel ? "enabled" : "disabled") << std::endl;
        std::cout << "  - Path type: " << (config_.use_8path ? "8-path" : "4-path") << std::endl;
        std::cout << "  - LR check: " << (config_.LR_max_diff >= 0 ? "enabled" : "disabled") << std::endl;
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[SGMInference] Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool SGMInference::infer(const cv::Mat& left, const cv::Mat& right, 
                         cv::Mat& disparity, float& time_ms) {
    if (!initialized_) {
        std::cerr << "[SGMInference] Not initialized. Call initialize() first." << std::endl;
        return false;
    }
    
    if (!validateInputs(left, right)) {
        return false;
    }
    
    try {
        // Start timing
        auto start = std::chrono::high_resolution_clock::now();
        
        // Preprocess images (convert to grayscale if needed)
        preprocessImages(left, right);
        
        // Execute SGM using LibSGMWrapper - handles all the complexity internally!
        auto* sgm = static_cast<sgm::LibSGMWrapper*>(sgm_matcher_);
        sgm->execute(left_buffer_, right_buffer_, disparity);
        
        // Stop timing
        auto end = std::chrono::high_resolution_clock::now();
        time_ms = std::chrono::duration<float, std::milli>(end - start).count();
        
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "[SGMInference] CUDA inference failed: " << e.what() << std::endl;
        if (e.code == cv::Error::GpuNotSupported) {
            std::cerr << "GPU not supported! Check CUDA installation." << std::endl;
        }
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "[SGMInference] Inference failed: " << e.what() << std::endl;
        return false;
    }
}

void SGMInference::disparityToDepth(const cv::Mat& disparity, cv::Mat& depth,
                                   float baseline, float focal_length,
                                   float scale_factor) {
    // Create output depth map
    depth = cv::Mat::zeros(disparity.size(), CV_32FC1);
    
    // Get invalid disparity value
    int invalid_disp = getInvalidDisparity();
    
    // Convert disparity to depth: depth = (baseline * focal_length) / disparity
    // Note: disparity might be scaled by SUBPIXEL_SCALE if subpixel is enabled
    for (int y = 0; y < disparity.rows; y++) {
        for (int x = 0; x < disparity.cols; x++) {
            float disp_value;
            
            if (disparity.type() == CV_16SC1 || disparity.type() == CV_16UC1) {
                int16_t disp = disparity.at<int16_t>(y, x);
                if (disp == invalid_disp || disp <= 0) {
                    depth.at<float>(y, x) = 0.0f;
                    continue;
                }
                disp_value = static_cast<float>(disp) / scale_factor;
            } else { // CV_8UC1
                uint8_t disp = disparity.at<uint8_t>(y, x);
                if (disp == invalid_disp || disp == 0) {
                    depth.at<float>(y, x) = 0.0f;
                    continue;
                }
                disp_value = static_cast<float>(disp);
            }
            
            // Calculate depth in meters
            depth.at<float>(y, x) = (baseline * focal_length) / disp_value;
        }
    }
}

int SGMInference::getInvalidDisparity() const {
    if (!initialized_ || !sgm_matcher_) {
        return -1;
    }
    
    auto* sgm = static_cast<sgm::LibSGMWrapper*>(sgm_matcher_);
    return sgm->getInvalidDisparity();
}

void SGMInference::colorizeDisparity(const cv::Mat& disparity, cv::Mat& disparity_color) {
    cv::Mat disp_8u;
    
    // Convert to 8-bit for colorization
    if (disparity.type() == CV_16SC1 || disparity.type() == CV_16UC1) {
        double scale = 255.0 / config_.disparity_size;
        if (config_.subpixel) {
            scale /= sgm::StereoSGM::SUBPIXEL_SCALE;
        }
        disparity.convertTo(disp_8u, CV_8U, scale);
    } else {
        disp_8u = disparity.clone();
    }
    
    // Apply color map
    cv::applyColorMap(disp_8u, disparity_color, cv::COLORMAP_TURBO);
    
    // Mask invalid disparities
    int invalid_disp = getInvalidDisparity();
    cv::Mat mask;
    if (disparity.type() == CV_16SC1 || disparity.type() == CV_16UC1) {
        mask = (disparity == invalid_disp);
    } else {
        mask = (disparity == static_cast<uint8_t>(invalid_disp));
    }
    
    disparity_color.setTo(cv::Scalar(0, 0, 0), mask);
}

bool SGMInference::validateInputs(const cv::Mat& left, const cv::Mat& right) {
    if (left.empty() || right.empty()) {
        std::cerr << "[SGMInference] Error: Empty input images" << std::endl;
        return false;
    }
    
    if (left.size() != right.size()) {
        std::cerr << "[SGMInference] Error: Left and right images must have the same size" << std::endl;
        return false;
    }
    
    if (left.cols != config_.width || left.rows != config_.height) {
        std::cerr << "[SGMInference] Error: Image size mismatch. Expected " 
                  << config_.width << "x" << config_.height 
                  << ", got " << left.cols << "x" << left.rows << std::endl;
        return false;
    }
    
    return true;
}

void SGMInference::preprocessImages(const cv::Mat& left, const cv::Mat& right) {
    // Convert to grayscale if needed
    if (left.channels() == 3) {
        cv::cvtColor(left, left_buffer_, cv::COLOR_BGR2GRAY);
    } else {
        left_buffer_ = left.clone();
    }
    
    if (right.channels() == 3) {
        cv::cvtColor(right, right_buffer_, cv::COLOR_BGR2GRAY);
    } else {
        right_buffer_ = right.clone();
    }
    
    // LibSGMWrapper accepts CV_8U, CV_16U, or CV_32S
    // Convert to CV_8U if needed (most common case)
    if (left_buffer_.type() != CV_8U && left_buffer_.type() != CV_16U && left_buffer_.type() != CV_32S) {
        left_buffer_.convertTo(left_buffer_, CV_8U);
    }
    
    if (right_buffer_.type() != CV_8U && right_buffer_.type() != CV_16U && right_buffer_.type() != CV_32S) {
        right_buffer_.convertTo(right_buffer_, CV_8U);
    }
}

} // namespace ct_uav_stereo
