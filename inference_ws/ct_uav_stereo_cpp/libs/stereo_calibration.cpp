#include "ct_uav_stereo_cpp/stereo_calibration.hpp"
#include <iostream>
#include <fstream>

namespace ct_uav_stereo {

StereoCalibration::StereoCalibration() : is_ready_(false), cached_erosion_size_(-1) {
}

StereoCalibration::~StereoCalibration() {
}

bool StereoCalibration::loadFromYAML(const std::string& yaml_path) {
    try {
        return parseKalibrYAML(yaml_path);
    } catch (const std::exception& e) {
        std::cerr << "[StereoCalibration] Error loading YAML: " << e.what() << std::endl;
        return false;
    }
}

bool StereoCalibration::parseKalibrYAML(const std::string& yaml_path) {
    YAML::Node config = YAML::LoadFile(yaml_path);
    
    if (!config["cam0"] || !config["cam1"]) {
        std::cerr << "[StereoCalibration] Missing cam0 or cam1 in YAML" << std::endl;
        return false;
    }
    
    // Parse cam0 (left camera)
    YAML::Node cam0 = config["cam0"];
    std::vector<int> resolution0 = cam0["resolution"].as<std::vector<int>>();
    stereo_params_.cam0.image_size = cv::Size(resolution0[0], resolution0[1]);
    stereo_params_.cam0.model = cam0["camera_model"].as<std::string>();
    
    // Intrinsics: [fu, fv, cu, cv]
    std::vector<double> intrinsics0 = cam0["intrinsics"].as<std::vector<double>>();
    stereo_params_.cam0.K = (cv::Mat_<double>(3, 3) << 
        intrinsics0[0], 0, intrinsics0[2],
        0, intrinsics0[1], intrinsics0[3],
        0, 0, 1);
    
    // Distortion coefficients
    std::vector<double> distortion0 = cam0["distortion_coeffs"].as<std::vector<double>>();
    stereo_params_.cam0.D = cv::Mat(distortion0).clone();
    
    // Parse cam1 (right camera)
    YAML::Node cam1 = config["cam1"];
    std::vector<int> resolution1 = cam1["resolution"].as<std::vector<int>>();
    stereo_params_.cam1.image_size = cv::Size(resolution1[0], resolution1[1]);
    stereo_params_.cam1.model = cam1["camera_model"].as<std::string>();
    
    // Intrinsics
    std::vector<double> intrinsics1 = cam1["intrinsics"].as<std::vector<double>>();
    stereo_params_.cam1.K = (cv::Mat_<double>(3, 3) << 
        intrinsics1[0], 0, intrinsics1[2],
        0, intrinsics1[1], intrinsics1[3],
        0, 0, 1);
    
    // Distortion coefficients
    std::vector<double> distortion1 = cam1["distortion_coeffs"].as<std::vector<double>>();
    stereo_params_.cam1.D = cv::Mat(distortion1).clone();
    
    // Parse T_cam1_cam0 (4x4 transformation matrix)
    std::vector<std::vector<double>> T_cam1_cam0 = 
        cam1["T_cn_cnm1"].as<std::vector<std::vector<double>>>();
    
    // Extract R and t
    stereo_params_.R = cv::Mat::zeros(3, 3, CV_64F);
    stereo_params_.t = cv::Mat::zeros(3, 1, CV_64F);
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            stereo_params_.R.at<double>(i, j) = T_cam1_cam0[i][j];
        }
        stereo_params_.t.at<double>(i, 0) = T_cam1_cam0[i][3];
    }
    
    // Compute baseline
    stereo_params_.baseline = cv::norm(stereo_params_.t);
    
    std::cout << "[StereoCalibration] Loaded calibration from: " << yaml_path << std::endl;
    std::cout << "  Left camera: " << stereo_params_.cam0.image_size << std::endl;
    std::cout << "  Right camera: " << stereo_params_.cam1.image_size << std::endl;
    std::cout << "  Baseline: " << stereo_params_.baseline << " m" << std::endl;
    
    return true;
}

bool StereoCalibration::loadFromOpenCV(const std::string& xml_path) {
    cv::FileStorage fs(xml_path, cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        std::cerr << "[StereoCalibration] Failed to open: " << xml_path << std::endl;
        return false;
    }
    
    // Read camera matrices
    fs["K1"] >> stereo_params_.cam0.K;
    fs["D1"] >> stereo_params_.cam0.D;
    fs["K2"] >> stereo_params_.cam1.K;
    fs["D2"] >> stereo_params_.cam1.D;
    
    // Read extrinsics
    fs["R"] >> stereo_params_.R;
    fs["T"] >> stereo_params_.t;
    
    // Read image size
    int width, height;
    fs["image_width"] >> width;
    fs["image_height"] >> height;
    stereo_params_.cam0.image_size = cv::Size(width, height);
    stereo_params_.cam1.image_size = cv::Size(width, height);
    
    fs.release();
    
    stereo_params_.baseline = cv::norm(stereo_params_.t);
    
    std::cout << "[StereoCalibration] Loaded OpenCV calibration" << std::endl;
    std::cout << "  Baseline: " << stereo_params_.baseline << " m" << std::endl;
    
    return true;
}

cv::Mat StereoCalibration::computeFusielloRectification(
    const cv::Mat& R_rel, const cv::Mat& t_rel,
    cv::Mat& R_left_out, cv::Mat& R_right_out) {
    
    // Fusiello's method for stereo rectification
    // Reference: A. Fusiello, E. Trucco, A. Verri
    // "A compact algorithm for rectification of stereo pairs", 1999
    
    // Compute optical centers (in cam0's coordinate system)
    cv::Mat c1 = cv::Mat::zeros(3, 1, CV_64F);  // cam0 is at origin
    cv::Mat c2 = -R_rel.t() * t_rel;             // cam1 position in cam0 coords
    
    // New x-axis: direction of baseline (normalized)
    cv::Mat v1 = c1 - c2;  // Python uses c1 - c2, not c2 - c1
    v1 = v1 / cv::norm(v1);
    
    // Get mean old z-axis
    cv::Mat z0 = (cv::Mat_<double>(3, 1) << 0, 0, 1);
    cv::Mat z1 = R_rel.row(2).t();
    cv::Mat z_mean = (z0 + z1) / 2.0;
    z_mean = z_mean / cv::norm(z_mean);
    
    // New y-axis: orthogonal to new x and mean old z
    cv::Mat v2 = z_mean.cross(v1);
    v2 = v2 / cv::norm(v2);
    
    // New z-axis: orthogonal to new x and new y
    cv::Mat v3 = v1.cross(v2);
    v3 = v3 / cv::norm(v3);
    
    // Create rotation matrix (rows are new axes)
    cv::Mat R_rect = cv::Mat::zeros(3, 3, CV_64F);
    v1.reshape(1, 1).copyTo(R_rect.row(0));
    v2.reshape(1, 1).copyTo(R_rect.row(1));
    v3.reshape(1, 1).copyTo(R_rect.row(2));
    
    // Rectification rotations
    R_left_out = R_rect.clone();              // From cam0 to rectified
    R_right_out = R_rect * R_rel.t();         // From cam1 to rectified
    
    return R_rect;
}

bool StereoCalibration::computeRectification() {
    if (stereo_params_.cam0.K.empty() || stereo_params_.cam1.K.empty()) {
        std::cerr << "[StereoCalibration] Camera matrices not loaded" << std::endl;
        return false;
    }
    
    // Compute rectification using Fusiello's method
    cv::Mat R_rect = computeFusielloRectification(
        stereo_params_.R, stereo_params_.t, R_left_, R_right_);
    
    // New intrinsic matrix (average of both cameras)
    K_rect_ = (stereo_params_.cam0.K + stereo_params_.cam1.K) / 2.0;
    
    // Compute rectification maps
    cv::initUndistortRectifyMap(
        stereo_params_.cam0.K,
        stereo_params_.cam0.D,
        R_left_,
        K_rect_,
        stereo_params_.cam0.image_size,
        CV_32FC1,
        map1_left_,
        map2_left_
    );
    
    cv::initUndistortRectifyMap(
        stereo_params_.cam1.K,
        stereo_params_.cam1.D,
        R_right_,
        K_rect_,
        stereo_params_.cam1.image_size,
        CV_32FC1,
        map1_right_,
        map2_right_
    );
    
    // Compute valid ROI for left camera
    valid_roi_left_ = cv::getValidDisparityROI(
        cv::Rect(0, 0, stereo_params_.cam0.image_size.width, 
                       stereo_params_.cam0.image_size.height),
        cv::Rect(0, 0, stereo_params_.cam1.image_size.width,
                       stereo_params_.cam1.image_size.height),
        0,  // minDisparity
        128, // numDisparities (use conservative value)
        5    // SADWindowSize
    );
    
    is_ready_ = true;
    
    std::cout << "[StereoCalibration] Rectification computed successfully" << std::endl;
    std::cout << "  Rectified focal length: " << getFocalLength() << " pixels" << std::endl;
    std::cout << "  Valid ROI (left): " << valid_roi_left_ << std::endl;
    
    return true;
}

void StereoCalibration::rectify(const cv::Mat& left_raw, const cv::Mat& right_raw,
                                cv::Mat& left_rect, cv::Mat& right_rect) const {
    if (!is_ready_) {
        std::cerr << "[StereoCalibration] Rectification not ready!" << std::endl;
        return;
    }
    
    cv::remap(left_raw, left_rect, map1_left_, map2_left_, cv::INTER_LINEAR);
    cv::remap(right_raw, right_rect, map1_right_, map2_right_, cv::INTER_LINEAR);
}

cv::Mat StereoCalibration::getLeftValidROI(int erosion_size) const {
    if (!is_ready_) {
        std::cerr << "[StereoCalibration] Rectification not ready!" << std::endl;
        return cv::Mat();
    }
    
    // Check cache
    if (!valid_mask_cache_.empty() && cached_erosion_size_ == erosion_size) {
        return valid_mask_cache_;
    }
    
    // Create mask on CPU
    cv::Mat mask = cv::Mat::zeros(stereo_params_.cam0.image_size, CV_8UC1);
    
    // Set valid ROI to 255
    if (valid_roi_left_.width > 0 && valid_roi_left_.height > 0) {
        mask(valid_roi_left_) = 255;
    } else {
        // If no valid ROI computed, use whole image
        mask.setTo(255);
    }
    
    // Apply erosion to shrink valid region for safety
    if (erosion_size > 0) {
        cv::Mat element = cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
            cv::Point(erosion_size, erosion_size)
        );
        cv::erode(mask, mask, element);
    }
    
    // Convert to float32 (0.0 or 1.0) and cache
    mask.convertTo(valid_mask_cache_, CV_32F, 1.0/255.0);
    cached_erosion_size_ = erosion_size;
    
    return valid_mask_cache_;
}

void StereoCalibration::drawEpipolarLines(cv::Mat& left_img, cv::Mat& right_img,
                                          int num_lines) const {
    int height = left_img.rows;
    int step = height / num_lines;
    
    for (int i = 0; i < num_lines; i++) {
        int y = i * step;
        cv::Scalar color = (i % 2 == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);
        
        cv::line(left_img, cv::Point(0, y), cv::Point(left_img.cols, y), color, 1);
        cv::line(right_img, cv::Point(0, y), cv::Point(right_img.cols, y), color, 1);
    }
}

} // namespace ct_uav_stereo
