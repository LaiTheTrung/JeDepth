#ifndef INFER_SGM_HPP_
#define INFER_SGM_HPP_

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

namespace ct_uav_stereo {

/**
 * SGM (Semi-Global Matching) Depth Estimation using libSGM (CUDA)
 * Features:
 * - CUDA-accelerated stereo matching
 * - Real-time performance on GPU
 * - Configurable SGM parameters
 * - Support for different disparity sizes
 * - Subpixel disparity computation
 * - Left-Right consistency check
 * - Uses LibSGMWrapper for automatic memory management
 */
class SGMInference {
public:
    struct Config {
        // Image dimensions
        int width;
        int height;
        
        // Disparity parameters
        int disparity_size;        // Must be 64, 128 or 256
        int min_disparity;         // Minimum disparity value
        
        // SGM parameters
        int P1;                    // Penalty for disparity change by ±1
        int P2;                    // Penalty for disparity change by >1
        float uniqueness;          // Uniqueness ratio (0-1)
        bool subpixel;             // Enable subpixel estimation
        bool use_8path;            // true: 8-path, false: 4-path
        int LR_max_diff;           // Max diff for LR consistency check (-1 to disable)
        
        // Default configuration
        Config() 
            : width(640), height(480),
              disparity_size(128), min_disparity(0),
              P1(10), P2(120), uniqueness(0.95f),
              subpixel(true), use_8path(true), LR_max_diff(1) {}
    };
    
    SGMInference(const Config& config);
    ~SGMInference();
    
    /**
     * Initialize the SGM matcher
     * @return true if initialization successful
     */
    bool initialize();
    
    /**
     * Compute disparity map from stereo pair
     * @param left Left rectified image (will convert to grayscale if needed)
     * @param right Right rectified image (will convert to grayscale if needed)
     * @param disparity Output disparity map (CV_16SC1 format, auto-allocated)
     * @param time_ms Processing time in milliseconds
     * @return true if inference successful
     */
    bool infer(const cv::Mat& left, const cv::Mat& right, 
               cv::Mat& disparity, float& time_ms);
    
    /**
     * Convert disparity to depth map
     * @param disparity Input disparity map
     * @param depth Output depth map (in meters)
     * @param baseline Stereo baseline in meters
     * @param focal_length Focal length in pixels
     * @param scale_factor Scale factor (1.0 for pixel disparity, 16.0 for subpixel)
     */
    void disparityToDepth(const cv::Mat& disparity, cv::Mat& depth,
                         float baseline, float focal_length,
                         float scale_factor = 16.0f);
    
    /**
     * Get invalid disparity value
     * @return The value representing invalid disparity
     */
    int getInvalidDisparity() const;
    
    /**
     * Colorize disparity map for visualization
     * @param disparity Input disparity map
     * @param disparity_color Output colorized disparity
     */
    void colorizeDisparity(const cv::Mat& disparity, cv::Mat& disparity_color);
    
private:
    Config config_;
    bool initialized_;
    
    // libSGM wrapper (opaque pointer to avoid header dependency)
    void* sgm_matcher_;
    
    // Internal buffers for preprocessing
    cv::Mat left_buffer_;
    cv::Mat right_buffer_;
    
    // Helper functions
    bool validateInputs(const cv::Mat& left, const cv::Mat& right);
    void preprocessImages(const cv::Mat& left, const cv::Mat& right);
};

} // namespace ct_uav_stereo

#endif // INFER_SGM_HPP_
