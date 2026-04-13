#include "fast_stream_stereo.hpp"
#include <cv_bridge/cv_bridge.h>
#include <rclcpp_components/register_node_macro.hpp>

namespace ct_uav_stereo
{

FastStreamNode::FastStreamNode(const rclcpp::NodeOptions & options)
    : Node("fast_stream_node", options),
      coordinator_fps_counter_(30)
{
    // ========== Declare Parameters ==========
    this->declare_parameter<std::string>("host_ip", "localhost");
    this->declare_parameter<int>("host_port", 41451);
    this->declare_parameter<std::string>("vehicle_name", "");
    this->declare_parameter<bool>("inter_publish", true);
    this->declare_parameter<int>("jpeg_quality", 70);
    this->declare_parameter<double>("sim_advance_time_ms", 1.0);
    
    // ========== Get Parameters ==========
    this->get_parameter("host_ip", host_ip_);
    int port_int;
    this->get_parameter("host_port", port_int);
    host_port_ = static_cast<uint16_t>(port_int);
    this->get_parameter("vehicle_name", vehicle_name_);
    this->get_parameter("inter_publish", inter_publish_);
    this->get_parameter("jpeg_quality", jpeg_quality_);
    this->get_parameter("sim_advance_time_ms", sim_advance_time_ms_);
    
    RCLCPP_INFO(this->get_logger(), "=== Fast Stream Node Configuration ===");
    RCLCPP_INFO(this->get_logger(), "Host: %s:%d", host_ip_.c_str(), host_port_);
    RCLCPP_INFO(this->get_logger(), "Vehicle: %s", vehicle_name_.empty() ? "default" : vehicle_name_.c_str());
    RCLCPP_INFO(this->get_logger(), "Inter-publish: %s", inter_publish_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "JPEG Quality: %d", jpeg_quality_);
    RCLCPP_INFO(this->get_logger(), "Perfect Sync: %s", PERFECT_SYNC ? "ENABLED" : "DISABLED");
    
    // ========== Create Publishers ==========
    
    if (inter_publish_) {
        compressed_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
            "~/raw_image/compressed", 10);
        RCLCPP_INFO(this->get_logger(), "Publishing compressed images");
    } else {
        raw_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "~/raw_image", 10);
        RCLCPP_INFO(this->get_logger(), "Publishing raw images");
    }
    
    fps_pub_ = this->create_publisher<std_msgs::msg::Float32>("~/fps", 10);
    
    // ========== Initialize AirSim ==========
    try {
        initializeAirSimClients();
        RCLCPP_INFO(this->get_logger(), "AirSim clients initialized successfully");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize AirSim: %s", e.what());
        throw;
    }
    
    // ========== Start Threads ==========
    startThreads();
    RCLCPP_INFO(this->get_logger(), "All threads started. Streaming...");
}

FastStreamNode::~FastStreamNode()
{
    stopThreads();
    RCLCPP_INFO(this->get_logger(), "Fast Stream Node stopped");
}

void FastStreamNode::initializeAirSimClients()
{
    // Control client for pause/unpause
    if (PERFECT_SYNC) {
        sim_control_client_ = std::make_unique<msr::airlib::MultirotorRpcLibClient>(
            host_ip_, host_port_);
        sim_control_client_->confirmConnection();
        RCLCPP_INFO(this->get_logger(), "Sim control client connected");
    }
    
    // One client per camera thread
    for (size_t i = 0; i < NUM_CAMERAS; ++i) {
        camera_clients_[i] = std::make_unique<msr::airlib::MultirotorRpcLibClient>(
            host_ip_, host_port_);
        camera_clients_[i]->confirmConnection();
        RCLCPP_INFO(this->get_logger(), "Camera %zu client connected", i);
    }
}

void FastStreamNode::startThreads()
{
    stop_flag_ = false;
    
    // Start coordinator thread
    coordinator_thread_ = std::thread(&FastStreamNode::coordinatorThreadFunc, this);
    
    // Start camera threads
    for (size_t i = 0; i < NUM_CAMERAS; ++i) {
        camera_threads_[i] = std::thread(&FastStreamNode::cameraThreadFunc, this, i);
    }
}

void FastStreamNode::stopThreads()
{
    stop_flag_ = true;
    
    // Wake up all threads
    capture_cv_.notify_all();
    
    // Join coordinator
    if (coordinator_thread_.joinable()) {
        coordinator_thread_.join();
    }
    
    // Join camera threads
    for (auto& thread : camera_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void FastStreamNode::coordinatorThreadFunc()
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
            cameras_done_counter_ = 0;
            capture_counter_++;
        }
        capture_cv_.notify_all();
        
        // ===== STEP 3: Wait for all cameras to complete =====
        {
            std::unique_lock<std::mutex> lock(capture_mutex_);
            auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            
            while (cameras_done_counter_ < NUM_CAMERAS && !stop_flag_) {
                if (capture_cv_.wait_until(lock, timeout) == std::cv_status::timeout) {
                    RCLCPP_WARN(this->get_logger(), 
                        "[Coordinator] Timeout: only %zu/%zu cameras completed", 
                        cameras_done_counter_.load(), NUM_CAMERAS);
                    break;
                }
            }
        }
        
        // ===== STEP 4: UNPAUSE simulation =====
        if (PERFECT_SYNC && sim_control_client_) {
            try {
                sim_control_client_->simPause(false);
                // // Allow sim to advance slightly
                std::this_thread::sleep_for(
                    std::chrono::microseconds(static_cast<long>(sim_advance_time_ms_ * 1000)));
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(), "[Coordinator] Unpause error: %s", e.what());
            }
        }
        
        // ===== STEP 5: Concatenate and publish =====
        std::array<FrameInfo, NUM_CAMERAS> frames_snapshot;
        bool all_valid = true;

        for (size_t i = 0; i < NUM_CAMERAS; ++i) {
            std::lock_guard<std::mutex> lock(frame_mutexes_[i]);
            frames_snapshot[i] = latest_frames_[i];
            if (!frames_snapshot[i].valid || frames_snapshot[i].image.empty()) {
                all_valid = false;
                RCLCPP_WARN(this->get_logger(), "Camera %zu has invalid/empty frame", i);
            }
        }
        
        if (all_valid) {
            cv::Mat concatenated = concatenateImages(frames_snapshot);
            if (concatenated.empty()) {
                RCLCPP_WARN(this->get_logger(), "Failed to concatenate images, skipping publish");
                continue;
            }
            uint64_t timestamp_ns = frames_snapshot[0].timestamp_ns;
            
            if (inter_publish_ && compressed_pub_) {
                auto msg = createCompressedMsg(concatenated, timestamp_ns);
                compressed_pub_->publish(*msg);
            } else if (raw_pub_) {
                auto msg = createRawMsg(concatenated, timestamp_ns);
                raw_pub_->publish(*msg);
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
    }
    
    RCLCPP_INFO(this->get_logger(), "[Coordinator] Stopped");
}

void FastStreamNode::cameraThreadFunc(size_t camera_index)
{
    const std::string& cam_name = camera_names_[camera_index];
    RCLCPP_INFO(this->get_logger(), "[Camera %zu (%s)] Thread started", camera_index, cam_name.c_str());
    
    auto& client = camera_clients_[camera_index];
    uint64_t last_capture_counter = 0;
    
    // Create image request (grayscale, uncompressed)
    msr::airlib::ImageCaptureBase::ImageRequest request(
        cam_name,
        msr::airlib::ImageCaptureBase::ImageType::Scene,
        false,  // pixels_as_float
        false   // compress
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
            
            if (responses.empty() || responses[0].image_data_uint8.empty()) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "[Camera %zu] Empty response", camera_index);
                
                // Signal done even on error
                {
                    std::lock_guard<std::mutex> lock(capture_mutex_);
                    cameras_done_counter_++;
                }
                capture_cv_.notify_all();
                continue;
            }
            
            auto& response = responses[0];
            
            // Convert to RGB cv::Mat
            cv::Mat img;
            if (response.image_data_uint8.size() == response.width * response.height * 3) {
                // RGB image - convert to BGR for OpenCV
                cv::Mat rgb(response.height, response.width, CV_8UC3, 
                    const_cast<uint8_t*>(response.image_data_uint8.data()));
                cv::cvtColor(rgb.clone(), img, cv::COLOR_RGB2BGR);
            } else if (response.image_data_uint8.size() == response.width * response.height) {
                // Grayscale - convert to BGR
                cv::Mat gray(response.height, response.width, CV_8UC1,
                    const_cast<uint8_t*>(response.image_data_uint8.data()));
                cv::cvtColor(gray.clone(), img, cv::COLOR_GRAY2BGR);
            } else {
                // Try to decode
                std::vector<uint8_t> data(response.image_data_uint8.begin(), 
                    response.image_data_uint8.end());
                cv::Mat decoded = cv::imdecode(data, cv::IMREAD_COLOR);
                if (decoded.empty()) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                        "[Camera %zu] Failed to decode image", camera_index);
                    
                    {
                        std::lock_guard<std::mutex> lock(capture_mutex_);
                        cameras_done_counter_++;
                    }
                    capture_cv_.notify_all();
                    continue;
                }
                img = decoded;
            }
            
            // ===== STEP 3: Signal completion =====
            {
                std::lock_guard<std::mutex> lock(capture_mutex_);
                cameras_done_counter_++;
            }
            capture_cv_.notify_all();
            
            // ===== STEP 4: Update frame buffer =====
            {
                std::lock_guard<std::mutex> lock(frame_mutexes_[camera_index]);
                latest_frames_[camera_index].image = img;
                latest_frames_[camera_index].timestamp_ns = timestamp_ns;
                latest_frames_[camera_index].valid = true;
            }
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "[Camera %zu] Error: %s", camera_index, e.what());
            
            // Signal done even on error
            {
                std::lock_guard<std::mutex> lock(capture_mutex_);
                cameras_done_counter_++;
            }
            capture_cv_.notify_all();
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "[Camera %zu] Thread stopped", camera_index);
}

cv::Mat FastStreamNode::concatenateImages(const std::array<FrameInfo, NUM_CAMERAS>& frames)
{
    std::vector<cv::Mat> valid_images;
    int target_height = -1;
    bool first = true;

    for (const auto& frame : frames) {
        if (frame.image.empty()) {
            RCLCPP_WARN(this->get_logger(), "Skipping empty image in concatenation");
            continue;
        }

        if (first) {
            target_height = frame.image.rows;
            valid_images.push_back(frame.image);
            first = false;
        } else {
            // Resize to match first image's height if needed
            cv::Mat resized;
            if (frame.image.rows != target_height) {
                double scale = static_cast<double>(target_height) / frame.image.rows;
                int new_width = static_cast<int>(frame.image.cols * scale);
                cv::resize(frame.image, resized, cv::Size(new_width, target_height), 0, 0, cv::INTER_LINEAR);
                valid_images.push_back(resized);
                RCLCPP_DEBUG(this->get_logger(), "Resized image to %dx%d", new_width, target_height);
            } else {
                valid_images.push_back(frame.image);
            }
        }
    }

    if (valid_images.empty()) {
        RCLCPP_ERROR(this->get_logger(), "No valid images to concatenate!");
        return cv::Mat();
    }

    if (valid_images.size() == 1) {
        return valid_images[0].clone();
    }

    cv::Mat concatenated;
    try {
        cv::hconcat(valid_images, concatenated);
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "hconcat failed: %s", e.what());
        return cv::Mat();
    }

    return concatenated;
}

sensor_msgs::msg::CompressedImage::SharedPtr FastStreamNode::createCompressedMsg(
    const cv::Mat& image, uint64_t timestamp_ns)
{
    auto msg = std::make_shared<sensor_msgs::msg::CompressedImage>();
    
    // Set header
    msg->header.stamp.sec = timestamp_ns / 1000000000ULL;
    msg->header.stamp.nanosec = timestamp_ns % 1000000000ULL;
    msg->header.frame_id = "omni_camera";
    msg->format = "jpeg";
    
    // Compress to JPEG
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
    std::vector<uint8_t> buffer;
    cv::imencode(".jpg", image, buffer, params);
    
    msg->data = buffer;
    
    return msg;
}

sensor_msgs::msg::Image::SharedPtr FastStreamNode::createRawMsg(
    const cv::Mat& image, uint64_t timestamp_ns)
{
    auto msg = std::make_shared<sensor_msgs::msg::Image>();
    
    msg->header.stamp.sec = timestamp_ns / 1000000000ULL;
    msg->header.stamp.nanosec = timestamp_ns % 1000000000ULL;
    msg->header.frame_id = "omni_camera";
    
    msg->height = image.rows;
    msg->width = image.cols;
    msg->encoding = "bgr8";  // RGB (BGR in OpenCV)
    msg->is_bigendian = false;
    msg->step = image.cols * 3;  // 3 bytes per pixel for RGB
    
    size_t size = image.rows * image.cols * 3;
    msg->data.resize(size);
    std::memcpy(msg->data.data(), image.data, size);
    
    return msg;
}

} // namespace ct_uav_stereo

using namespace ct_uav_stereo;
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<FastStreamNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}
// ========== Register as composable node ==========
RCLCPP_COMPONENTS_REGISTER_NODE(ct_uav_stereo::FastStreamNode)