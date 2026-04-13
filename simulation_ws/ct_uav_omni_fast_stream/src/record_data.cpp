// ============================================================================

#include "record_data.hpp"
#include <cv_bridge/cv_bridge.h>
#include <rclcpp_components/register_node_macro.hpp>

namespace ct_uav_omni
{

DataRecorderNode::DataRecorderNode(const rclcpp::NodeOptions & options)
    : Node("data_recorder_node", options),
      coordinator_fps_counter_(30)
{
    // ========== Declare Parameters ==========
    this->declare_parameter<std::string>("host_ip", "localhost");
    this->declare_parameter<int>("host_port", 41451);
    this->declare_parameter<std::string>("vehicle_name", "");
    this->declare_parameter<int>("record_mode", 0); // 0: RGB+Seg, 1: RGB only
    this->declare_parameter<bool>("inter_publish", true);
    this->declare_parameter<int>("jpeg_quality", 70);
    this->declare_parameter<double>("sim_advance_time_ms", 1.0);
    this->declare_parameter<bool>("is_vulkan", false);
    
    // ========== Get Parameters ==========
    this->get_parameter("host_ip", host_ip_);
    int port_int;
    this->get_parameter("host_port", port_int);
    host_port_ = static_cast<uint16_t>(port_int);
    this->get_parameter("vehicle_name", vehicle_name_);
    this->get_parameter("inter_publish", inter_publish_);
    this->get_parameter("jpeg_quality", jpeg_quality_);
    this->get_parameter("sim_advance_time_ms", sim_advance_time_ms_);
    this->get_parameter("is_vulkan", is_vulkan_);
    int record_mode = 0;
    this->get_parameter("record_mode", record_mode);

    if (record_mode == 0) {
        channel_configs_ = seg_channel_configs_;
    } 
    else if(record_mode == 1) {
        channel_configs_ = rgb_channel_configs_;
    }
    else if(record_mode == 2) {
        channel_configs_ = depth_channel_configs_;
    }
    else {
        RCLCPP_ERROR(this->get_logger(), "Invalid record_mode: %d", record_mode);
        throw std::runtime_error("Invalid record_mode parameter");
    }

    RCLCPP_INFO(this->get_logger(), "=== Data Recorder Node Configuration ===");
    RCLCPP_INFO(this->get_logger(), "Host: %s:%d", host_ip_.c_str(), host_port_);
    RCLCPP_INFO(this->get_logger(), "Vehicle: %s", vehicle_name_.empty() ? "default" : vehicle_name_.c_str());
    RCLCPP_INFO(this->get_logger(), "Inter-publish: %s", inter_publish_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "JPEG Quality: %d", jpeg_quality_);
    RCLCPP_INFO(this->get_logger(), "Perfect Sync: %s", PERFECT_SYNC ? "ENABLED" : "DISABLED");
    RCLCPP_INFO(this->get_logger(), "Vulkan Mode: %s", is_vulkan_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "Channels: %zu", NUM_CHANNELS);
    
    // Log channel configurations
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        const auto& cfg = channel_configs_[i];
        std::string type_name;
        switch (cfg.image_type) {
            case msr::airlib::ImageCaptureBase::ImageType::Scene:
                type_name = "RGB";
                break;
            case msr::airlib::ImageCaptureBase::ImageType::Segmentation:
                type_name = "SEG";
                break;
            case msr::airlib::ImageCaptureBase::ImageType::DepthPerspective:
                type_name = "DEPTH";
                break;
            default:
                type_name = "OTHER";
                break;
        }
        RCLCPP_INFO(this->get_logger(), "  Channel %zu: %s [%s] -> %s", 
            i, cfg.camera_name.c_str(), type_name.c_str(), cfg.topic_name.c_str());
    }
    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "Record Mode: %d", record_mode);
    if (record_mode == 0) {
        RCLCPP_INFO(this->get_logger(), "  Mode 0: RGB + Segmentation");
    } else if (record_mode == 1) {
        RCLCPP_INFO(this->get_logger(), "  Mode 1: RGB only (4 cameras)");
    } else if (record_mode == 2) {
        RCLCPP_INFO(this->get_logger(), "  Mode 2: RGB + Depth (cam0, cam1)");
        RCLCPP_INFO(this->get_logger(), "  Depth format: uint16 (millimeters), PNG compression");
    }
    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "Segmentation Info:");
    RCLCPP_INFO(this->get_logger(), "  - Each pixel RGB = object ID encoded in color space");
    RCLCPP_INFO(this->get_logger(), "  - Use simSetSegmentationObjectID() to set IDs in AirSim");
    RCLCPP_INFO(this->get_logger(), "  - Example: Ground=20, Sky=42, Buildings=100+");
    RCLCPP_INFO(this->get_logger(), "");
    
    // ========== Initialize AirSim ==========
    try {
        initializeAirSimClients();
        RCLCPP_INFO(this->get_logger(), "AirSim clients initialized successfully");
        
        // Save camera info to YAML (one-time at startup)
        saveCameraInfo();
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize AirSim: %s", e.what());
        throw;
    }
    
    // ========== Create Publishers ==========
    initializePublishers();
    
    // FPS publisher
    fps_pub_ = this->create_publisher<std_msgs::msg::Float32>("~/fps", 10);
    
    // ========== Start Threads ==========
    startThreads();
    RCLCPP_INFO(this->get_logger(), "All threads started. Multi-channel data recording...");
}

DataRecorderNode::~DataRecorderNode()
{
    stopThreads();
    RCLCPP_INFO(this->get_logger(), "Data Recorder Node stopped");
}

void DataRecorderNode::initializeAirSimClients()
{
    // Control client for pause/unpause
    if (PERFECT_SYNC) {
        sim_control_client_ = std::make_unique<msr::airlib::MultirotorRpcLibClient>(
            host_ip_, host_port_);
        sim_control_client_->confirmConnection();
        RCLCPP_INFO(this->get_logger(), "Sim control client connected");
    }
    
    // One client per channel thread
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        channel_clients_[i] = std::make_unique<msr::airlib::MultirotorRpcLibClient>(
            host_ip_, host_port_);
        channel_clients_[i]->confirmConnection();
        RCLCPP_INFO(this->get_logger(), "Channel %zu client connected", i);
    }
}

void DataRecorderNode::initializePublishers()
{
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        const auto& cfg = channel_configs_[i];
        
        if (inter_publish_) {
            compressed_pubs_[i] = this->create_publisher<sensor_msgs::msg::CompressedImage>(
                "~/" + cfg.topic_name + "/compressed", 10);
            RCLCPP_INFO(this->get_logger(), "Publishing compressed: %s", cfg.topic_name.c_str());
        } else {
            raw_pubs_[i] = this->create_publisher<sensor_msgs::msg::Image>(
                "~/" + cfg.topic_name, 10);
            RCLCPP_INFO(this->get_logger(), "Publishing raw: %s", cfg.topic_name.c_str());
        }
    }
}

void DataRecorderNode::saveCameraInfo()
{
    RCLCPP_INFO(this->get_logger(), "=== Saving Camera Information ===");
    
    // Get package share directory path
    // Assuming config folder is at: <workspace>/src/ct_uav_omni_fast_stream/config/
    std::string config_path = "/mnt/d/trung_Nav_team/Cosys-AirSim/ros2/src/ct_uav_omni_fast_stream/config/";
    
    // Create config directory if it doesn't exist
    std::string mkdir_cmd = "mkdir -p " + config_path;
    int ret = system(mkdir_cmd.c_str());
    (void)ret;  // Suppress unused warning
    
    // Collect unique camera names
    std::set<std::string> unique_cameras;
    for (const auto& cfg : channel_configs_) {
        unique_cameras.insert(cfg.camera_name);
    }
    
    // Query camera info for each unique camera
    for (const auto& cam_name : unique_cameras) {
        try {
            RCLCPP_INFO(this->get_logger(), "Querying camera info: %s", cam_name.c_str());
            
            // Use first available client to query camera info
            auto& client = channel_clients_[0];
            // Get camera info via simGetCameraInfo
            client->simSetCameraFov(cam_name, 140.0f);
            auto cam_info = client->simGetCameraInfo(cam_name, vehicle_name_);
            
            // Get additional CinemAirSim parameters (may not be available on all setups)
            std::string focal_length_str = "N/A";
            std::string fov_str = "N/A";
            std::string filmback_str = "N/A";
            std::string lens_str = "N/A";
            std::string focus_distance_str = "N/A";
            std::string focus_aperture_str = "N/A";
            std::vector<float> distortion_coeffs;
        
            try {
                float focal_length = client->simGetFocalLength(cam_name, vehicle_name_);
                focal_length_str = std::to_string(focal_length);
            } catch (...) {
                RCLCPP_DEBUG(this->get_logger(), "  Focal length not available");
            }
            
            try {
                fov_str = client->simGetCurrentFieldOfView(cam_name, vehicle_name_);
            } catch (...) {
                RCLCPP_DEBUG(this->get_logger(), "  FOV not available");
            }
            
            try {
                filmback_str = client->simGetFilmbackSettings(cam_name, vehicle_name_);
            } catch (...) {
                RCLCPP_DEBUG(this->get_logger(), "  Filmback settings not available");
            }
            
            try {
                lens_str = client->simGetLensSettings(cam_name, vehicle_name_);
            } catch (...) {
                RCLCPP_DEBUG(this->get_logger(), "  Lens settings not available");
            }
            
            try {
                float focus_dist = client->simGetFocusDistance(cam_name, vehicle_name_);
                focus_distance_str = std::to_string(focus_dist);
            } catch (...) {
                RCLCPP_DEBUG(this->get_logger(), "  Focus distance not available");
            }
            
            try {
                float focus_ap = client->simGetFocusAperture(cam_name, vehicle_name_);
                focus_aperture_str = std::to_string(focus_ap);
            } catch (...) {
                RCLCPP_DEBUG(this->get_logger(), "  Focus aperture not available");
            }
            
            try {
                distortion_coeffs = client->simGetDistortionParams(cam_name, vehicle_name_);
            } catch (...) {
                RCLCPP_DEBUG(this->get_logger(), "  Distortion coefficients not available");
            }
            // Create YAML file
            std::string yaml_filename = config_path + "/" + cam_name + "_info.yaml";
            std::ofstream yaml_file(yaml_filename);
            
            if (!yaml_file.is_open()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to create YAML file: %s", yaml_filename.c_str());
                continue;
            }
            
            // Write YAML content
            yaml_file << "# Camera Information for: " << cam_name << "\n";
            yaml_file << "# Generated at startup by DataRecorderNode\n";
            yaml_file << "# Timestamp: " << std::chrono::system_clock::now().time_since_epoch().count() << "\n";
            yaml_file << "\n";
            
            yaml_file << "camera_name: \"" << cam_name << "\"\n";
            yaml_file << "\n";
            
            // Pose information
            yaml_file << "pose:\n";
            yaml_file << "  position:\n";
            yaml_file << "    x: " << cam_info.pose.position.x() << "\n";
            yaml_file << "    y: " << cam_info.pose.position.y() << "\n";
            yaml_file << "    z: " << cam_info.pose.position.z() << "\n";
            yaml_file << "  orientation:\n";
            yaml_file << "    w: " << cam_info.pose.orientation.w() << "\n";
            yaml_file << "    x: " << cam_info.pose.orientation.x() << "\n";
            yaml_file << "    y: " << cam_info.pose.orientation.y() << "\n";
            yaml_file << "    z: " << cam_info.pose.orientation.z() << "\n";
            yaml_file << "\n";
            
            // FOV
            yaml_file << "fov_degrees: " << cam_info.fov << "\n";
            yaml_file << "\n";

            // Distortion coefficients
            yaml_file << "distortion_coefficients:\n";
            yaml_file << "  data: [";
            for (size_t i = 0; i < distortion_coeffs.size(); ++i) {
                yaml_file << distortion_coeffs[i];
                if (i < distortion_coeffs.size() - 1) {
                    yaml_file << ", ";
                }
            }
            yaml_file << "]\n";
            yaml_file << "\n";
            
            // CinemAirSim parameters (if available)
            yaml_file << "cinemairsim:\n";
            yaml_file << "  focal_length_mm: " << focal_length_str << "\n";
            yaml_file << "  field_of_view: \"" << fov_str << "\"\n";
            yaml_file << "  filmback_settings: \"" << filmback_str << "\"\n";
            yaml_file << "  lens_settings: \"" << lens_str << "\"\n";
            yaml_file << "  focus_distance: " << focus_distance_str << "\n";
            yaml_file << "  focus_aperture: " << focus_aperture_str << "\n";
            yaml_file << "\n";
            
            // Additional metadata
            yaml_file << "metadata:\n";
            yaml_file << "  vehicle_name: \"" << vehicle_name_ << "\"\n";
            yaml_file << "  host_ip: \"" << host_ip_ << "\"\n";
            yaml_file << "  host_port: " << host_port_ << "\n";
            
            yaml_file.close();
            
            RCLCPP_INFO(this->get_logger(), "  Saved camera info to: %s", yaml_filename.c_str());
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to query camera info for %s: %s", 
                cam_name.c_str(), e.what());
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "Camera info saved successfully");
    RCLCPP_INFO(this->get_logger(), "");
}

void DataRecorderNode::startThreads()
{
    stop_flag_ = false;
    
    // Start coordinator thread
    coordinator_thread_ = std::thread(&DataRecorderNode::coordinatorThreadFunc, this);
    
    // Start channel threads
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        channel_threads_[i] = std::thread(&DataRecorderNode::channelThreadFunc, this, i);
    }
}

void DataRecorderNode::stopThreads()
{
    stop_flag_ = true;
    
    // Wake up all threads
    capture_cv_.notify_all();
    
    // Join coordinator
    if (coordinator_thread_.joinable()) {
        coordinator_thread_.join();
    }
    
    // Join channel threads
    for (auto& thread : channel_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void DataRecorderNode::coordinatorThreadFunc()
{
    RCLCPP_INFO(this->get_logger(), "[Coordinator] Started");
    
    if (PERFECT_SYNC) {
        RCLCPP_INFO(this->get_logger(), "[Coordinator] PERFECT SYNC: PAUSE → CAPTURE → UNPAUSE");
    }
    
    uint64_t loop_count = 0;
    
    while (!stop_flag_ && rclcpp::ok()) {
        // ===== Track FPS =====
        auto [inst_fps, avg_fps] = coordinator_fps_counter_.tick();
        
        // ===== STEP 1: PAUSE simulation =====
        if (PERFECT_SYNC && sim_control_client_) {
            try {
                sim_control_client_->simPause(true);
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(), "[Coordinator] Pause error: %s", e.what());
            }
        }
        
        // ===== STEP 2: Broadcast capture signal =====
        {
            std::lock_guard<std::mutex> lock(capture_mutex_);
            channels_done_counter_ = 0;
            capture_counter_++;
        }
        capture_cv_.notify_all();
        
        // ===== STEP 3: Wait for all channels to complete =====
        {
            std::unique_lock<std::mutex> lock(capture_mutex_);
            auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            
            while (channels_done_counter_ < NUM_CHANNELS && !stop_flag_) {
                if (capture_cv_.wait_until(lock, timeout) == std::cv_status::timeout) {
                    RCLCPP_WARN(this->get_logger(), 
                        "[Coordinator] Timeout: only %zu/%zu channels completed", 
                        channels_done_counter_.load(), NUM_CHANNELS);
                    break;
                }
            }
        }
        
        // ===== STEP 4: UNPAUSE simulation =====
        if (PERFECT_SYNC && sim_control_client_) {
            try {
                sim_control_client_->simPause(false);
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(), "[Coordinator] Unpause error: %s", e.what());
            }
        }
        
        // ===== STEP 5: Publish all channels =====
        for (size_t i = 0; i < NUM_CHANNELS; ++i) {
            std::lock_guard<std::mutex> lock(frame_mutexes_[i]);
            const auto& frame = latest_frames_[i];
            const auto& cfg = channel_configs_[i];
            
            if (!frame.valid || frame.image.empty()) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "Channel %zu (%s) has invalid/empty frame", i, cfg.topic_name.c_str());
                continue;
            }
            
            if (inter_publish_ && compressed_pubs_[i]) {
                auto msg = createCompressedMsg(frame.image, frame.timestamp_ns, cfg.frame_id);
                compressed_pubs_[i]->publish(*msg);
            } else if (raw_pubs_[i]) {
                auto msg = createRawMsg(frame.image, frame.timestamp_ns, cfg.frame_id, cfg.encoding);
                raw_pubs_[i]->publish(*msg);
            }
        }
        
        // Publish FPS
        auto fps_msg = std_msgs::msg::Float32();
        fps_msg.data = static_cast<float>(avg_fps);
        fps_pub_->publish(fps_msg);
        
        // Log periodically
        if (++loop_count % 30 == 0) {
            RCLCPP_INFO(this->get_logger(), 
                "[Coordinator] Loop FPS: %.1f (avg: %.1f)", inst_fps, avg_fps);
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "[Coordinator] Stopped");
}
void DataRecorderNode::channelThreadFunc(size_t channel_index)
{
    const auto& cfg = channel_configs_[channel_index];
    std::string type_name;
    switch (cfg.image_type) {
        case msr::airlib::ImageCaptureBase::ImageType::Scene:
            type_name = "RGB";
            break;
        case msr::airlib::ImageCaptureBase::ImageType::Segmentation:
            type_name = "SEG";
            break;
        case msr::airlib::ImageCaptureBase::ImageType::DepthPerspective:
            type_name = "DEPTH";
            break;
        default:
            type_name = "OTHER";
            break;
    }
    
    RCLCPP_INFO(this->get_logger(), "[Channel %zu (%s/%s)] Thread started", 
        channel_index, cfg.camera_name.c_str(), type_name.c_str());
    
    auto& client = channel_clients_[channel_index];
    uint64_t last_capture_counter = 0;
    
    // Determine if this is a depth channel
    bool is_depth = (cfg.image_type == msr::airlib::ImageCaptureBase::ImageType::DepthPerspective);
    
    // Create image request
    msr::airlib::ImageCaptureBase::ImageRequest request(
        cfg.camera_name,
        cfg.image_type,
        is_depth,  // pixels_as_float = true for depth
        false      // compress = false
    );
    
    while (!stop_flag_ && rclcpp::ok()) {
        // ===== STEP 1: Wait for capture signal =====
        {
            std::unique_lock<std::mutex> lock(capture_mutex_);
            capture_cv_.wait(lock, [&]() {
                return capture_counter_ > last_capture_counter || stop_flag_.load();
            });
            
            if (stop_flag_) break;
            last_capture_counter = capture_counter_;
        }
        
        // ===== STEP 2: Capture image (simulation is PAUSED) =====
        try {
            uint64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            
            std::vector<msr::airlib::ImageCaptureBase::ImageRequest> requests = {request};
            auto responses = client->simGetImages(requests, vehicle_name_);
            
            if (responses.empty()) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "[Channel %zu] Empty response", channel_index);
                
                // Signal done even on error
                {
                    std::lock_guard<std::mutex> lock(capture_mutex_);
                    channels_done_counter_++;
                }
                capture_cv_.notify_all();
                continue;
            }
            
            auto& response = responses[0];
            
            // Convert to cv::Mat based on type
            cv::Mat img;
            if (is_depth) {
                // Process depth image (float array)
                if (response.image_data_float.empty()) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                        "[Channel %zu] Empty depth data", channel_index);
                    
                    {
                        std::lock_guard<std::mutex> lock(capture_mutex_);
                        channels_done_counter_++;
                    }
                    capture_cv_.notify_all();
                    continue;
                }
                
                // Convert float depth (meters) to uint16 (millimeters)
                img = depthFloatToUint16(response);
                
            } else {
                // Process RGB/Segmentation image
                if (response.image_data_uint8.empty()) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                        "[Channel %zu] Empty image data", channel_index);
                    
                    {
                        std::lock_guard<std::mutex> lock(capture_mutex_);
                        channels_done_counter_++;
                    }
                    capture_cv_.notify_all();
                    continue;
                }
                
                img = responseToMat(response, cfg.encoding);
            }
            
            if (img.empty()) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "[Channel %zu] Failed to convert image", channel_index);
                
                {
                    std::lock_guard<std::mutex> lock(capture_mutex_);
                    channels_done_counter_++;
                }
                capture_cv_.notify_all();
                continue;
            }
            
            // ===== STEP 3: Signal completion =====
            {
                std::lock_guard<std::mutex> lock(capture_mutex_);
                channels_done_counter_++;
            }
            capture_cv_.notify_all();
            
            // ===== STEP 4: Update frame buffer =====
            {
                std::lock_guard<std::mutex> lock(frame_mutexes_[channel_index]);
                latest_frames_[channel_index].image = img;
                latest_frames_[channel_index].timestamp_ns = timestamp_ns;
                latest_frames_[channel_index].valid = true;
            }
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "[Channel %zu] Error: %s", channel_index, e.what());
            
            // Signal done even on error
            {
                std::lock_guard<std::mutex> lock(capture_mutex_);
                channels_done_counter_++;
            }
            capture_cv_.notify_all();
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "[Channel %zu] Thread stopped", channel_index);
}

cv::Mat DataRecorderNode::depthFloatToUint16(
    const msr::airlib::ImageCaptureBase::ImageResponse& response)
{
    /**
     * Convert AirSim depth image (float, meters) to uint16 (millimeters)
     * 
     * AirSim returns depth as float array in meters.
     * We convert to uint16 in millimeters:
     * - Range: 0-65535mm (0-65.535m)
     * - Values > 65.535m are clamped to 65535
     * - This matches standard ROS depth image format
     */
    
    // Create float matrix from response
    cv::Mat depth_meters(response.height, response.width, CV_32FC1);
    std::memcpy(depth_meters.data, response.image_data_float.data(), 
                response.image_data_float.size() * sizeof(float));
    
    // Convert meters to millimeters
    cv::Mat depth_mm;
    depth_meters.convertTo(depth_mm, CV_32FC1, 1000.0);  // meters * 1000 = millimeters
    
    // Clamp to uint16 range [0, 65535]
    cv::Mat depth_clamped;
    cv::threshold(depth_mm, depth_clamped, 65535.0, 65535.0, cv::THRESH_TRUNC);
    cv::threshold(depth_clamped, depth_clamped, 0.0, 0.0, cv::THRESH_TOZERO);
    
    // Convert to uint16
    cv::Mat depth_uint16;
    depth_clamped.convertTo(depth_uint16, CV_16UC1);
    
    return depth_uint16;
}

cv::Mat DataRecorderNode::responseToMat(
    const msr::airlib::ImageCaptureBase::ImageResponse& response, 
    const std::string& encoding)
{
    cv::Mat img;
    
    if (encoding == "rgb8") {
        // RGB image (Scene or Segmentation)
        // Note: Segmentation returns RGB where each pixel color = object ID
        if (response.image_data_uint8.size() == response.width * response.height * 3) {
            cv::Mat color(response.height, response.width, CV_8UC3, 
                const_cast<uint8_t*>(response.image_data_uint8.data()));
            
            if (is_vulkan_) {
                img = color.clone();  // Already RGB
            } else {
                cv::cvtColor(color.clone(), img, cv::COLOR_BGR2RGB);  // Convert BGR to RGB
            }
        } else {
            // Try to decode compressed
            std::vector<uint8_t> data(response.image_data_uint8.begin(), 
                response.image_data_uint8.end());
            img = cv::imdecode(data, cv::IMREAD_COLOR);
            
            if (!img.empty() && !is_vulkan_) {
                cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            }
        }
    } else if (encoding == "mono8") {
        // Grayscale image
        if (response.image_data_uint8.size() == response.width * response.height * 3) {
            // RGB image - convert to grayscale
            cv::Mat color(response.height, response.width, CV_8UC3, 
                const_cast<uint8_t*>(response.image_data_uint8.data()));
            
            if (is_vulkan_) {
                cv::cvtColor(color.clone(), img, cv::COLOR_RGB2GRAY);
            } else {
                cv::cvtColor(color.clone(), img, cv::COLOR_BGR2GRAY);
            }
        } else if (response.image_data_uint8.size() == response.width * response.height) {
            // Already grayscale
            img = cv::Mat(response.height, response.width, CV_8UC1,
                const_cast<uint8_t*>(response.image_data_uint8.data())).clone();
        } else {
            // Try to decode compressed
            std::vector<uint8_t> data(response.image_data_uint8.begin(), 
                response.image_data_uint8.end());
            img = cv::imdecode(data, cv::IMREAD_GRAYSCALE);
        }
    }
    
    return img;
}

sensor_msgs::msg::CompressedImage::SharedPtr DataRecorderNode::createCompressedMsg(
    const cv::Mat& image, uint64_t timestamp_ns, const std::string& frame_id)
{
    auto msg = std::make_shared<sensor_msgs::msg::CompressedImage>();
    
    // Set header
    msg->header.stamp.sec = timestamp_ns / 1000000000ULL;
    msg->header.stamp.nanosec = timestamp_ns % 1000000000ULL;
    msg->header.frame_id = frame_id;
    
    // Check image type to determine compression format
    if (image.type() == CV_16UC1) {
        // Depth image: use PNG (lossless)
        msg->format = "png";
        std::vector<uint8_t> buffer;
        std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 3};  // 0-9, 3 is good balance
        cv::imencode(".png", image, buffer, params);
        msg->data = buffer;
    } else {
        // RGB/Segmentation: use JPEG
        msg->format = "jpeg";
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
        std::vector<uint8_t> buffer;
        cv::imencode(".jpg", image, buffer, params);
        msg->data = buffer;
    }
    
    return msg;
}

sensor_msgs::msg::Image::SharedPtr DataRecorderNode::createRawMsg(
    const cv::Mat& image, uint64_t timestamp_ns, const std::string& frame_id, 
    const std::string& encoding)
{
    auto msg = std::make_shared<sensor_msgs::msg::Image>();
    
    msg->header.stamp.sec = timestamp_ns / 1000000000ULL;
    msg->header.stamp.nanosec = timestamp_ns % 1000000000ULL;
    msg->header.frame_id = frame_id;
    
    msg->height = image.rows;
    msg->width = image.cols;
    msg->encoding = encoding;
    msg->is_bigendian = false;
    
    if (encoding == "rgb8") {
        msg->step = image.cols * 3;
        size_t size = image.rows * image.cols * 3;
        msg->data.resize(size);
        std::memcpy(msg->data.data(), image.data, size);
    } else if (encoding == "mono8") {
        msg->step = image.cols;
        size_t size = image.rows * image.cols;
        msg->data.resize(size);
        std::memcpy(msg->data.data(), image.data, size);
    } else if (encoding == "16UC1") {
        // Depth image: uint16, 1 channel
        msg->step = image.cols * 2;  // 2 bytes per pixel
        size_t size = image.rows * image.cols * 2;
        msg->data.resize(size);
        std::memcpy(msg->data.data(), image.data, size);
    }
    
    return msg;
}

} // namespace ct_uav_omni
using namespace ct_uav_omni;
// ========== Register as composable node ==========
RCLCPP_COMPONENTS_REGISTER_NODE(ct_uav_omni::DataRecorderNode)


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<ct_uav_omni::DataRecorderNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}