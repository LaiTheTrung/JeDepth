#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

namespace omni_conversion
{
class CassiniConversion
{
public:
  // Load configuration from file
  bool loadConfig(const std::string& path_to_config) {
    cv::FileStorage fs(path_to_config, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      std::cerr << "Failed to open config file: " << path_to_config << std::endl;
      return false;
    }
    
    fs["mapx"] >> mapx;
    fs["mapy"] >> mapy;
    fs["mask"] >> mask;
    fs.release();
    
    // Validate that all required data was loaded
    if (mapx.empty() || mapy.empty() || mask.empty()) {
      std::cerr << "Failed to load required mapping data from config file" << std::endl;
      return false;
    }
    
    // Convert mask once during initialization
    cv::cvtColor(mask, mask3, cv::COLOR_GRAY2BGR);
    is_initialized = true;
    
    std::cout << "Successfully loaded config from: " << path_to_config << std::endl;
    return true;
  }
  
  // Convert perspective image to Cassini projection
  bool perspectiveToCassini(const cv::Mat& src, cv::Mat& cass) const {
    if (!is_initialized) {
      std::cerr << "Error: Config not loaded. Call loadConfig() first." << std::endl;
      return false;
    }
    
    if (src.empty()) {
      std::cerr << "Error: Input image is empty" << std::endl;
      return false;
    }
    
    // Perform remapping
    cv::remap(src, cass, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    
    // Apply mask
    cv::bitwise_and(cass, mask3, cass);
    
    return true;
  }
  
  // Check if the converter is ready to use
  bool isInitialized() const {
    return is_initialized;
  }

private:
  cv::Mat mapx, mapy, mask, mask3;
  bool is_initialized = false;
};
  
} // namespace omni_conversion