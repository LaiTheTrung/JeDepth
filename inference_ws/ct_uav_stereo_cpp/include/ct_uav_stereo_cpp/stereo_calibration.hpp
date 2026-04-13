#ifndef STEREO_CALIBRATION_HPP_
#define STEREO_CALIBRATION_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

namespace ct_uav_stereo {

/**
 * Stereo Camera Calibration and Rectification
 * Based on Fusiello's method for stereo rectification
 * Compatible with Kalibr YAML format
 */
class StereoCalibration {
public:
    struct CameraParams {
        cv::Mat K;           // 3x3 intrinsic matrix
        cv::Mat D;           // Distortion coefficients
        cv::Size image_size; // Image resolution
        std::string model;   // Camera model (pinhole, etc.)
        
        CameraParams() : K(cv::Mat::eye(3, 3, CV_64F)), 
                        D(cv::Mat::zeros(4, 1, CV_64F)),
                        image_size(640, 480),
                        model("pinhole") {}
    };
    
    struct StereoParams {
        CameraParams cam0;  // Left camera
        CameraParams cam1;  // Right camera
        cv::Mat R;          // Rotation from cam0 to cam1
        cv::Mat t;          // Translation from cam0 to cam1
        double baseline;    // Baseline in meters
        
        StereoParams() : R(cv::Mat::eye(3, 3, CV_64F)),
                        t(cv::Mat::zeros(3, 1, CV_64F)),
                        baseline(0.12) {}
    };
    
    StereoCalibration();
    ~StereoCalibration();
    
    /**
     * Load calibration from Kalibr YAML format
     * @param yaml_path Path to YAML calibration file
     * @return true if successful
     */
    bool loadFromYAML(const std::string& yaml_path);
    
    /**
     * Load calibration from OpenCV XML/YAML format
     * @param xml_path Path to OpenCV calibration file
     * @return true if successful
     */
    bool loadFromOpenCV(const std::string& xml_path);
    
    /**
     * Compute stereo rectification using Fusiello's method
     * More stable than OpenCV's stereoRectify for some cases
     * @return true if successful
     */
    bool computeRectification();
    
    /**
     * Get rectification maps for left camera
     */
    void getLeftRectificationMaps(cv::Mat& map1, cv::Mat& map2) const {
        map1 = map1_left_.clone();
        map2 = map2_left_.clone();
    }
    
    /**
     * Get rectification maps for right camera
     */
    void getRightRectificationMaps(cv::Mat& map1, cv::Mat& map2) const {
        map1 = map1_right_.clone();
        map2 = map2_right_.clone();
    }
    
    /**
     * Apply rectification to stereo pair
     */
    void rectify(const cv::Mat& left_raw, const cv::Mat& right_raw,
                 cv::Mat& left_rect, cv::Mat& right_rect) const;
    
    /**
     * Get stereo parameters
     */
    const StereoParams& getStereoParams() const { return stereo_params_; }
    
    /**
     * Get rectified camera matrix (same for both cameras after rectification)
     */
    const cv::Mat& getRectifiedCameraMatrix() const { return K_rect_; }
    
    /**
     * Get baseline in meters
     */
    double getBaseline() const { return stereo_params_.baseline; }
    
    /**
     * Get focal length in pixels (after rectification)
     */
    double getFocalLength() const { 
        return K_rect_.at<double>(0, 0); 
    }
    
    /**
     * Check if calibration is loaded and rectification computed
     */
    bool isReady() const { return is_ready_; }
    
    /**
     * Get left camera valid ROI mask (CPU) for rectified image
     * Pixels outside this mask should have depth set to 0
     * @param erosion_size Size of erosion kernel to shrink valid region (default: 5)
     * @return CPU mask (CV_32FC1) where 1.0 = valid, 0.0 = invalid
     */
    cv::Mat getLeftValidROI(int erosion_size = 5) const;
    
    /**
     * Draw epipolar lines for debugging
     */
    void drawEpipolarLines(cv::Mat& left_img, cv::Mat& right_img, 
                          int num_lines = 20) const;

private:
    StereoParams stereo_params_;
    
    // Rectification matrices
    cv::Mat R_left_;   // Rectification rotation for left camera
    cv::Mat R_right_;  // Rectification rotation for right camera
    cv::Mat K_rect_;   // Rectified camera matrix (common for both)
    
    // Rectification maps
    cv::Mat map1_left_, map2_left_;
    cv::Mat map1_right_, map2_right_;
    
    // Valid ROI for rectified images (CPU cache)
    mutable cv::Rect valid_roi_left_;
    mutable cv::Mat valid_mask_cache_;
    mutable int cached_erosion_size_;
    
    bool is_ready_;
    
    // Helper functions
    bool parseKalibrYAML(const std::string& yaml_path);
    cv::Mat computeFusielloRectification(const cv::Mat& R_rel, 
                                         const cv::Mat& t_rel,
                                         cv::Mat& R_left_out,
                                         cv::Mat& R_right_out);
};

} // namespace ct_uav_stereo

#endif // STEREO_CALIBRATION_HPP_
