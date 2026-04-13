/**
 * Stereo Depth Node - Comprehensive ROS2 Implementation
 * 
 * This node combines:
 * 1. Camera input processing similar to sgm_example.cpp
 * 2. SGM disparity estimation (640x480) with GPU acceleration
 * 3. Deep learning model inference (HitNet/FastACV) with configurable input size
 * 4. Separate visualization thread showing (left, right, sgm_disp, model_disp)
 * 5. Disparity to depth conversion and ROS2 publishing
 * 
 * Author: CT-UAV Team
 * Date: 2026-01-20
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>

#include "ct_uav_stereo_cpp/stereo_camera_driver.hpp"
#include "ct_uav_stereo_cpp/infer_sgm.hpp"
#include "ct_uav_stereo_cpp/infer_fastACV.hpp"
#include "ct_uav_stereo_cpp/infer_hitnet.hpp"
#include "ct_uav_stereo_cpp/infer_fastFS.hpp"
#include "ct_uav_stereo_cpp/stereo_calibration.hpp"

#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <thread>
#include <mutex>
#include <atomic>
#include <deque>

namespace CtUAVStereoCpp {

class StereoDepthNode : public rclcpp::Node {
public:
    StereoDepthNode() : Node("stereo_depth_node"), 
                        running_(true),
                        frame_count_(0),
                        viz_enabled_(false) {
        
        // ===== Declare Parameters =====
        
        // Debug and visualization
        this->declare_parameter("debug_visualize", true);
        this->declare_parameter("debug_sgm", true);
        this->declare_parameter("use_camera", true);
        
        // Camera parameters
        this->declare_parameter("left_sensor_id", 0);
        this->declare_parameter("right_sensor_id", 1);
        this->declare_parameter("flip_method", 0);
        this->declare_parameter("using_gpu", true);
        this->declare_parameter("conversion_type", cvt_YUV2BGR);
        this->declare_parameter("auto_expose", false);
        this->declare_parameter("assembled_mode", false);
        
        // Calibration file (REQUIRED for rectification)
        this->declare_parameter("calibration_file", "");
        
        // SGM parameters (640x480 processing)
        this->declare_parameter("sgm_width", 640);
        this->declare_parameter("sgm_height", 480);
        this->declare_parameter("sgm_disparity_size", 128);
        this->declare_parameter("sgm_P1", 10);
        this->declare_parameter("sgm_P2", 120);
        this->declare_parameter("sgm_subpixel", true);
        this->declare_parameter("sgm_use_8path", true);
        
        // Model selection and parameters
        this->declare_parameter("model_type", "fastacv");  // "sgm_only", "fastacv", "hitnet", "fastfs"
        this->declare_parameter("onnx_path", "");
        this->declare_parameter("engine_path", "");
        this->declare_parameter("model_input_width", 480);
        this->declare_parameter("model_input_height", 288);
        this->declare_parameter("max_disparity", 192);
        this->declare_parameter("use_fp16", true);
        
        // FastFS (Fast Foundation Stereo) specific parameters
        this->declare_parameter("fastfs_feature_engine_path", "");
        this->declare_parameter("fastfs_post_engine_path", "");
        this->declare_parameter("fastfs_cv_group", 8);
        this->declare_parameter("fastfs_normalize_gwc", false);
        
        // Publishing parameters
        this->declare_parameter("publish_rate", 30.0);
        this->declare_parameter("publish_raw_images", false);
        this->declare_parameter("publish_depth", true);
        this->declare_parameter("publish_depth_viz", true);
        
        // Stereo calibration parameters for depth calculation
        this->declare_parameter("focal_length_default", 100.0);  // Focal length in pixels (will be overridden by calibration)
        this->declare_parameter("baseline", 0.12);       // Baseline in meters (will be overridden by calibration)
        this->declare_parameter("min_depth_m", 0.2);
        this->declare_parameter("max_depth_m", 30.0);
        // ===== Get Parameters =====
        
        viz_enabled_ = this->get_parameter("debug_visualize").as_bool();
        debug_sgm_ = this->get_parameter("debug_sgm").as_bool();
        
        
        left_id = this->get_parameter("left_sensor_id").as_int();
        right_id = this->get_parameter("right_sensor_id").as_int();
        cam_width = STREAM_SIZE.width();
        cam_height = STREAM_SIZE.height();
        framerate = DEFAULT_FRAME_RATE;
        int flip_method = this->get_parameter("flip_method").as_int();
        double rate = this->get_parameter("publish_rate").as_double();

        using_gpu_ = this->get_parameter("using_gpu").as_bool();
        conversion_type_ = this->get_parameter("conversion_type").as_int();
        auto_expose_ = this->get_parameter("auto_expose").as_bool();
        assembled_mode_ = this->get_parameter("assembled_mode").as_bool();
        
        sgm_width_ = this->get_parameter("sgm_width").as_int();
        sgm_height_ = this->get_parameter("sgm_height").as_int();
        
        model_input_width_ = this->get_parameter("model_input_width").as_int();
        model_input_height_ = this->get_parameter("model_input_height").as_int();
        model_type_ = this->get_parameter("model_type").as_string();
        
        publish_raw_ = this->get_parameter("publish_raw_images").as_bool();
        publish_depth_ = this->get_parameter("publish_depth").as_bool();
        publish_viz_ = this->get_parameter("publish_depth_viz").as_bool();
        
        // Get calibration parameters (will be overridden if calibration file exists)
        focal_length_ = this->get_parameter("focal_length_default").as_double();
        baseline_ = this->get_parameter("baseline").as_double();
        min_depth_m_ = this->get_parameter("min_depth_m").as_double();
        max_depth_m_ = this->get_parameter("max_depth_m").as_double();
        
        std::string calib_file = this->get_parameter("calibration_file").as_string();
        
        // ===== Load Stereo Calibration =====
        
        if (!calib_file.empty()) {
            RCLCPP_INFO(this->get_logger(), "Loading stereo calibration from: %s", calib_file.c_str());
            
            stereo_calib_ = std::make_unique<ct_uav_stereo::StereoCalibration>();
            if (!stereo_calib_->loadFromYAML(calib_file)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to load calibration file!");
                rclcpp::shutdown();
                return;
            }
            
            if (!stereo_calib_->computeRectification()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to compute rectification!");
                rclcpp::shutdown();
                return;
            }
            
            // Get calibrated parameters
            baseline_ = stereo_calib_->getBaseline();
            focal_length_ = stereo_calib_->getFocalLength();
            
            // Get rectification maps for GPU processing
            cv::Mat map_left_x, map_left_y, map_right_x, map_right_y;
            stereo_calib_->getLeftRectificationMaps(map_left_x, map_left_y);
            stereo_calib_->getRightRectificationMaps(map_right_x, map_right_y);
            
            // Upload to GPU
            d_map_left_x_.upload(map_left_x);
            d_map_left_y_.upload(map_left_y);
            d_map_right_x_.upload(map_right_x);
            d_map_right_y_.upload(map_right_y);
            
            RCLCPP_INFO(this->get_logger(), "Calibration loaded successfully!");
            RCLCPP_INFO(this->get_logger(), "  Baseline: %.4f m", baseline_);
            RCLCPP_INFO(this->get_logger(), "  Focal length: %.2f pixels", focal_length_);
        } else {
            RCLCPP_WARN(this->get_logger(), "No calibration file provided, using parameter values");
        }
        
        // Scale focal length for SGM processing resolution
        focal_length_sgm_ = focal_length_ * (static_cast<float>(sgm_width_) / cam_width);
        
        // Scale focal length for model input resolution
        focal_length_model_ = focal_length_ * (static_cast<float>(model_input_width_) / cam_width);
        
        RCLCPP_INFO(this->get_logger(), "Focal length (SGM %dx%d): %.2f pixels", 
                    sgm_width_, sgm_height_, focal_length_sgm_);
        RCLCPP_INFO(this->get_logger(), "Focal length (Model %dx%d): %.2f pixels",
                    model_input_width_, model_input_height_, focal_length_model_); 
        
        // ===== Initialize Cameras or Subscribers =====
        use_camera_ = this->get_parameter("use_camera").as_bool();

        if (use_camera_) {
            RCLCPP_INFO(this->get_logger(), "Initializing cameras with Argus...");
            memset(&module_info_, 0, sizeof(ModuleInfo));
            
            module_info_.initialized = false;
            if (!g_display_.initialize()) {
                RCLCPP_ERROR(this->get_logger(), "g_display init failed");
                return;
            }
            camera_provider_ = Argus::UniqueObj<Argus::CameraProvider>(Argus::CameraProvider::create());
            
            if (!initializeStereoDriver()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to initialize stereo hardware sync driver");
                rclcpp::shutdown();
                return;
            }
        }
        else {
            RCLCPP_INFO(this->get_logger(), "Subscribing to compressed image topics...");
            
            // Create subscribers for compressed images
            left_img_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
                "/stereo/left/image_raw/compressed", 10,
                std::bind(&StereoDepthNode::leftImageCallback, this, std::placeholders::_1));
            
            right_img_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
                "/stereo/right/image_raw/compressed", 10,
                std::bind(&StereoDepthNode::rightImageCallback, this, std::placeholders::_1));
            
            RCLCPP_INFO(this->get_logger(), "Subscribed to:");
            RCLCPP_INFO(this->get_logger(), "  - /stereo/left/image_raw/compressed");
            RCLCPP_INFO(this->get_logger(), "  - /stereo/right/image_raw/compressed");
        }
        
        // ===== Initialize SGM =====
        
        ct_uav_stereo::SGMInference::Config sgm_config;
        sgm_config.width = sgm_width_;
        sgm_config.height = sgm_height_;
        sgm_config.disparity_size = this->get_parameter("sgm_disparity_size").as_int();
        sgm_config.P1 = this->get_parameter("sgm_P1").as_int();
        sgm_config.P2 = this->get_parameter("sgm_P2").as_int();
        sgm_config.subpixel = this->get_parameter("sgm_subpixel").as_bool();
        sgm_config.use_8path = this->get_parameter("sgm_use_8path").as_bool();
        
        sgm_inference_ = std::make_unique<ct_uav_stereo::SGMInference>(sgm_config);
        
        if (!sgm_inference_->initialize()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize SGM");
            rclcpp::shutdown();
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "SGM initialized (resolution: %dx%d, disp_size: %d)",
                    sgm_width_, sgm_height_, sgm_config.disparity_size);
        
        // ===== Initialize Model Inference (Optional) =====
        
        if (model_type_ == "fastacv") {
            ct_uav_stereo::FastACVInferenceOptimized::Config model_config;
            model_config.engine_path = this->get_parameter("engine_path").as_string();
            model_config.input_width = model_input_width_;
            model_config.input_height = model_input_height_;
            model_config.max_disparity = this->get_parameter("max_disparity").as_int();
            // Note: FastACV uses fp16 from engine file, parameter not used
            
            fastacv_inference_ = std::make_unique<ct_uav_stereo::FastACVInferenceOptimized>(model_config);
            
            if (!fastacv_inference_->initialize()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to initialize FastACV, falling back to SGM only");
                model_type_ = "sgm_only";
            } else {
                RCLCPP_INFO(this->get_logger(), "FastACV initialized (resolution: %dx%d)",
                            model_input_width_, model_input_height_);
            }
        } else if (model_type_ == "hitnet") {
            ct_uav_stereo::HitNetInferenceOptimized::Config model_config;
            model_config.engine_path = this->get_parameter("engine_path").as_string();
            model_config.onnx_path = this->get_parameter("onnx_path").as_string();
            model_config.input_width = model_input_width_;
            model_config.input_height = model_input_height_;
            model_config.max_disparity = this->get_parameter("max_disparity").as_int();
            model_config.fp16 = this->get_parameter("use_fp16").as_bool();
            
            hitnet_inference_ = std::make_unique<ct_uav_stereo::HitNetInferenceOptimized>(model_config);
            
            if (!hitnet_inference_->initialize()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to initialize HitNet, falling back to SGM only");
                model_type_ = "sgm_only";
            } else {
                RCLCPP_INFO(this->get_logger(), "HitNet initialized (resolution: %dx%d)",
                            model_input_width_, model_input_height_);
            }
        } else if (model_type_ == "fastfs") {
            ct_uav_stereo::FastFSInferenceOptimized::Config model_config;
            model_config.feature_engine_path = this->get_parameter("fastfs_feature_engine_path").as_string();
            model_config.post_engine_path = this->get_parameter("fastfs_post_engine_path").as_string();
            model_config.input_width = model_input_width_;
            model_config.input_height = model_input_height_;
            model_config.max_disparity = this->get_parameter("max_disparity").as_int();
            model_config.cv_group = this->get_parameter("fastfs_cv_group").as_int();
            model_config.normalize_gwc = this->get_parameter("fastfs_normalize_gwc").as_bool();
            
            fastfs_inference_ = std::make_unique<ct_uav_stereo::FastFSInferenceOptimized>(model_config);
            
            if (!fastfs_inference_->initialize()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to initialize FastFS, falling back to SGM only");
                model_type_ = "sgm_only";
            } else {
                RCLCPP_INFO(this->get_logger(), "FastFS (Fast Foundation Stereo) initialized (resolution: %dx%d)",
                            model_input_width_, model_input_height_);
            }
        } else {
            RCLCPP_INFO(this->get_logger(), "Using SGM only (no deep learning model)");
            model_type_ = "sgm_only";
        }
        
        // ===== Create Publishers =====
        
        if (publish_raw_) {
            left_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                "/stereo/left/image_raw", 10);
            right_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                "/stereo/right/image_raw", 10);
        }
        
        if (publish_depth_) {
            depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                "/stereo/depth_map", 10);
        }
        
        if (publish_viz_) {
            depth_viz_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                "/stereo/depth/image_color", 10);
        }
        


        // ===== Start Visualization Thread =====
        
        if (viz_enabled_) {
            viz_thread_ = std::thread(&StereoDepthNode::visualizationLoop, this);
            RCLCPP_INFO(this->get_logger(), "Visualization thread started");
        }
        
        // ===== Create Timer for Capture and Inference =====
        
        auto capture_period = std::chrono::duration<double>(1.0 / rate);
        capture_timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::milliseconds>(capture_period),
            std::bind(&StereoDepthNode::captureAndInferCallback, this));
        
        RCLCPP_INFO(this->get_logger(), "=== Stereo Depth Node Started ===");
        RCLCPP_INFO(this->get_logger(), "  Mode: %s", use_camera_ ? "Camera" : "Subscriber");
        RCLCPP_INFO(this->get_logger(), "  Resolution: %dx%d @ %d fps", cam_width, cam_height, framerate);
        RCLCPP_INFO(this->get_logger(), "  SGM: %dx%d", sgm_width_, sgm_height_);
        RCLCPP_INFO(this->get_logger(), "  Model: %s (%dx%d)", 
                    model_type_.c_str(), model_input_width_, model_input_height_);
        RCLCPP_INFO(this->get_logger(), "  Debug visualize: %s", viz_enabled_ ? "enabled" : "disabled");
    }
    
    ~StereoDepthNode() {
        running_ = false;
        
        if (viz_thread_.joinable()) {
            viz_thread_.join();
        }
        
        if (use_camera_) {
            shutdownStereoDriver();
        }
        
        RCLCPP_INFO(this->get_logger(), "Stereo depth node stopped");
    }

private:
    bool initializeStereoDriver() {
        auto* i_camera_provider = Argus::interface_cast<Argus::ICameraProvider>(camera_provider_);
        if (!i_camera_provider) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create Argus camera provider");
            return false;
        }

        i_camera_provider->setSyncSensorSessionsCount(1, 0);

        std::vector<Argus::CameraDevice*> camera_devices;
        i_camera_provider->getCameraDevices(&camera_devices);
        if (camera_devices.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No camera devices found");
            return false;
        }

        int max_id = std::max(left_id, right_id);
        if (max_id < 0 || static_cast<size_t>(max_id) >= camera_devices.size()) {
            RCLCPP_ERROR(this->get_logger(), "Invalid camera ids (left=%d, right=%d)", left_id, right_id);
            return false;
        }

        std::strncpy(module_info_.moduleName, "stereo", MAX_MODULE_STRING - 1);
        module_info_.initialized = true;
        module_info_.sensorCount = 2;
        module_info_.camDevice[left_device] = left_id;
        module_info_.camDevice[right_device] = right_id;
        module_info_.using_gpu = using_gpu_;
        module_info_.assembled_mode = assembled_mode_;
        module_info_.conversion_type = conversion_type_;
        module_info_.auto_expose = auto_expose_;

        for (int i = 0; i < module_info_.sensorCount; ++i) {
            const int cam_index = module_info_.camDevice[i];
            Argus::CameraDevice* camera_device = camera_devices[cam_index];

            module_info_.iCameraProperties[i] = Argus::interface_cast<Argus::ICameraProperties>(camera_device);
            if (!module_info_.iCameraProperties[i]) {
                RCLCPP_ERROR(this->get_logger(), "Failed to get camera properties for sensor %d", i);
                return false;
            }

            std::vector<Argus::SensorMode*> sensor_modes;
            module_info_.iCameraProperties[i]->getAllSensorModes(&sensor_modes);
            module_info_.sensorMode[i] = ArgusSamples::ArgusHelpers::getSensorMode(camera_device, SENSOR_MODE_INDEX);
            module_info_.iSensorMode[i] = Argus::interface_cast<Argus::ISensorMode>(module_info_.sensorMode[i]);
            if (module_info_.iSensorMode[i]) {
                RCLCPP_INFO(this->get_logger(), "sensor %d using mode %d (%dx%d)", i, SENSOR_MODE_INDEX,
                            module_info_.iSensorMode[i]->getResolution().width(),
                            module_info_.iSensorMode[i]->getResolution().height());
            }

            auto* reprocess_info = Argus::interface_cast<Argus::IReprocessInfo>(camera_device);
            if (reprocess_info) {
                reprocess_info->setReprocessingEnable(false);
            }

            module_info_.captureSession[i] = Argus::UniqueObj<Argus::CaptureSession>(
                i_camera_provider->createCaptureSession(camera_device));
            module_info_.iCaptureSession[i] = Argus::interface_cast<Argus::ICaptureSession>(module_info_.captureSession[i]);
            if (!module_info_.iCaptureSession[i]) {
                RCLCPP_ERROR(this->get_logger(), "Failed to create capture session for sensor %d", i);
                return false;
            }

            module_info_.streamSettings[i] = Argus::UniqueObj<Argus::OutputStreamSettings>(
                module_info_.iCaptureSession[i]->createOutputStreamSettings(Argus::STREAM_TYPE_EGL));
            auto* i_stream_settings = Argus::interface_cast<Argus::IOutputStreamSettings>(module_info_.streamSettings[i]);
            auto* i_egl_stream_settings = Argus::interface_cast<Argus::IEGLOutputStreamSettings>(module_info_.streamSettings[i]);
            if (!i_stream_settings || !i_egl_stream_settings) {
                RCLCPP_ERROR(this->get_logger(), "Failed to create stream settings for sensor %d", i);
                return false;
            }

            i_egl_stream_settings->setPixelFormat(Argus::PIXEL_FMT_YCbCr_420_888);
            i_egl_stream_settings->setResolution(CtUAVStereoCpp::STREAM_SIZE);
            i_egl_stream_settings->setMetadataEnable(true);
            i_egl_stream_settings->setMode(Argus::EGL_STREAM_MODE_MAILBOX);
            i_egl_stream_settings->setEGLDisplay(g_display_.get());

            i_stream_settings->setCameraDevice(camera_device);
            module_info_.stream[i] = Argus::UniqueObj<Argus::OutputStream>(
                module_info_.iCaptureSession[i]->createOutputStream(module_info_.streamSettings[i].get()));
        }

        auto* left_egl_stream_settings = Argus::interface_cast<Argus::IEGLOutputStreamSettings>(
            module_info_.streamSettings[left_device]);
        module_info_.stereoYuvConsumer = new CtUAVStereoCpp::SyncStereoConsumerThread(
            left_egl_stream_settings, &module_info_, module_info_.stream[left_device].get());
        if (!module_info_.stereoYuvConsumer->initialize()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize consumer thread");
            return false;
        }
        if (!module_info_.stereoYuvConsumer->waitRunning()) {
            RCLCPP_ERROR(this->get_logger(), "Consumer thread failed to start");
            return false;
        }

        for (int i = 0; i < module_info_.sensorCount; ++i) {
            module_info_.request[i] = Argus::UniqueObj<Argus::Request>(
                module_info_.iCaptureSession[i]->createRequest(Argus::CAPTURE_INTENT_VIDEO_RECORD));
            auto* i_request = Argus::interface_cast<Argus::IRequest>(module_info_.request[i]);
            if (!i_request) {
                RCLCPP_ERROR(this->get_logger(), "Failed to create capture request %d", i);
                return false;
            }
            if (i_request->enableOutputStream(module_info_.stream[i].get()) != Argus::STATUS_OK) {
                RCLCPP_ERROR(this->get_logger(), "Failed to enable output stream %d", i);
                return false;
            }

            if (auto* i_source_settings = Argus::interface_cast<Argus::ISourceSettings>(module_info_.request[i])) {
                uint64_t frame_duration = 1000000000ULL / static_cast<uint64_t>(framerate);
                i_source_settings->setSensorMode(module_info_.sensorMode[i]);
                i_source_settings->setFrameDurationRange(Argus::Range<uint64_t>(frame_duration));
                i_source_settings->setExposureTimeRange(EXPOSURE_TIME_RANGE);
                i_source_settings->setGainRange(Argus::Range<float>(GAIN_RANGE.min()));
            }

			if (auto* i_auto_control = Argus::interface_cast<Argus::IAutoControlSettings>(i_request->getAutoControlSettings())) {
				if (auto_expose_) {
					i_auto_control->setIspDigitalGainRange(Range<float>(ISP_DIGITAL_GAIN_RANGE.min())); // set digital ISP to 1.0x which is minimum 
					if (i_auto_control->setAeLock(AE_LOCK_DISABLE) != Argus::STATUS_OK) RCLCPP_ERROR(this->get_logger(), "Failed to disable AE lock");
					
				} else {
					i_auto_control->setIspDigitalGainRange(ISP_DIGITAL_GAIN_RANGE); // set digital ISP to 1.0x which is minimum 
					if (i_auto_control->setAeLock(AE_LOCK_DISABLE) != Argus::STATUS_OK) {
						RCLCPP_ERROR(this->get_logger(), "Failed to enable AE lock");
					}
				}
			}
            if (module_info_.iCaptureSession[i]->repeat(module_info_.request[i].get()) != Argus::STATUS_OK) {
                RCLCPP_ERROR(this->get_logger(), "Failed to start repeat capture %d", i);
                return false;
            }
        }

        return true;
    }

    void shutdownStereoDriver() {
        for (int i = 0; i < module_info_.sensorCount; ++i) {
            if (module_info_.iCaptureSession[i]) {
                module_info_.iCaptureSession[i]->stopRepeat();
                module_info_.iCaptureSession[i]->waitForIdle();
            }
            if (module_info_.stream[i]) {
                module_info_.stream[i].reset();
            }
        }

        if (module_info_.stereoYuvConsumer) {
            if (!module_info_.stereoYuvConsumer->shutdown()) {
                RCLCPP_ERROR(this->get_logger(), "Consumer thread failed to shutdown");
            }
            delete module_info_.stereoYuvConsumer;
            module_info_.stereoYuvConsumer = nullptr;
        }

        if (!g_display_.cleanup()) {
            RCLCPP_ERROR(this->get_logger(), "g_display cleanup failed");
        }

        for (int i = 0; i < module_info_.sensorCount; ++i) {
            if (module_info_.request[i]) {
                module_info_.request[i].reset();
            }
            if (module_info_.streamSettings[i]) {
                module_info_.streamSettings[i].reset();
            }
            if (module_info_.captureSession[i]) {
                module_info_.captureSession[i].reset();
            }
            module_info_.iCaptureSession[i] = nullptr;
        }

        camera_provider_.reset();
    }

    
    // ===== Compressed Image Callbacks =====
    void leftImageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(image_mutex_);
        
        // Decode JPEG compressed image
        std::vector<uint8_t> data(msg->data.begin(), msg->data.end());
        cv::Mat decoded = cv::imdecode(data, cv::IMREAD_COLOR);
        
        if (!decoded.empty()) {
            left_frame_buffer_ = decoded;
            left_timestamp_ = msg->header.stamp;
            left_received_ = true;
        }
    }
    
    void rightImageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(image_mutex_);
        
        // Decode JPEG compressed image
        std::vector<uint8_t> data(msg->data.begin(), msg->data.end());
        cv::Mat decoded = cv::imdecode(data, cv::IMREAD_COLOR);
        
        if (!decoded.empty()) {
            right_frame_buffer_ = decoded;
            right_timestamp_ = msg->header.stamp;
            right_received_ = true;
        }
    }
    
    // ===== Main Capture and Inference Callback =====
    void captureAndInferCallback() {
        cv::Mat h_left, h_right;
        rclcpp::Time timestamp;
        
        if (use_camera_) {
            if (!module_info_.stereoYuvConsumer) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                     "Stereo consumer not initialized");
                return;
            }
            double latency =0.0;
            module_info_.stereoYuvConsumer->get_frames(h_left, h_right, latency);
            if (h_left.empty() || h_right.empty()) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                     "Failed to capture frames");
                return;
            }
            uint64_t frameExposureTime = 0;
            if (auto_expose_){
                float frameGain = module_info_.stereoYuvConsumer->get_frame_gain();
                frameExposureTime = module_info_.stereoYuvConsumer->get_frame_exposure_time();
                float  ispGain = module_info_.stereoYuvConsumer->get_isp_gain();
                // RCLCPP_INFO(this->get_logger(), "Auto-exposure update: Gain=%.2f, ExposureTime=%.2f ms, ISP Gain=%.2f", frameGain, frameExposureTime / 1e6, ispGain);
                for (uint32_t i = 0; i < 2; i++){
                    ISourceSettings *iSourceSettings = interface_cast<ISourceSettings>(module_info_.request[i]);
                    
                    if(iSourceSettings->setGainRange(Argus::Range<float>(frameGain)) != Argus::STATUS_OK)
                        RCLCPP_ERROR(this->get_logger(),"Failed to set Analog gain");

                    if(iSourceSettings->setExposureTimeRange(Argus::Range<uint64_t>(frameExposureTime)) != Argus::STATUS_OK) 
                        RCLCPP_ERROR(this->get_logger(),"Failed to set Exposure time");

                    auto* i_request = Argus::interface_cast<Argus::IRequest>(module_info_.request[i]);
                    // if (auto* i_auto_control = Argus::interface_cast<Argus::IAutoControlSettings>(i_request->getAutoControlSettings())) {
                    // i_auto_control->setIspDigitalGainRange(Range<float>(ispGain)); 
                    // }
                    /*
                    * The modified request is re-submitted to terminate the previous repeat() with
                    * the old settings and begin captures with the new settings
                    */
                    module_info_.iCaptureSession[i]->repeat(module_info_.request[i].get());
                }
            }
            timestamp = this->now() - rclcpp::Duration::from_seconds(latency) - rclcpp::Duration::from_nanoseconds(frameExposureTime/2); // Use current time minus latency as timestamp            
        } else {
            // Get frames from subscribers
            {
                std::lock_guard<std::mutex> lock(image_mutex_);
                if (!left_received_ || !right_received_) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                         "Waiting for stereo images...");
                    return;
                }
                h_left = left_frame_buffer_.clone();
                h_right = right_frame_buffer_.clone();
                timestamp = left_timestamp_; // Use left timestamp as reference
            }
        }
        
        if (h_left.empty() || h_right.empty()) {
            return;
        }
        
        // Convert to BGR if needed
        cv::Mat h_left_bgr, h_right_bgr;
        convertToBGR(h_left, h_left_bgr);
        convertToBGR(h_right, h_right_bgr);
        
        // Publish raw images if enabled
        if (publish_raw_) {
            publishRawImages(h_left_bgr, h_right_bgr, timestamp);
        }
        
        // Upload to GPU
        cv::cuda::GpuMat d_left, d_right;
        d_left.upload(h_left_bgr);
        d_right.upload(h_right_bgr);
        
        // Apply rectification if calibration is available
        cv::cuda::GpuMat d_left_rect, d_right_rect;
        if (stereo_calib_) {
            cv::cuda::remap(d_left, d_left_rect, d_map_left_x_, d_map_left_y_, 
                           cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
            cv::cuda::remap(d_right, d_right_rect, d_map_right_x_, d_map_right_y_, 
                           cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        } else {
            d_left_rect = d_left;
            d_right_rect = d_right;
        }
        
        // Convert to grayscale on GPU
        cv::cuda::GpuMat d_left_gray, d_right_gray;
        cv::cuda::cvtColor(d_left_rect, d_left_gray, cv::COLOR_BGR2GRAY);
        cv::cuda::cvtColor(d_right_rect, d_right_gray, cv::COLOR_BGR2GRAY);
        
        // Resize to SGM resolution on GPU
        cv::cuda::GpuMat d_left_sgm, d_right_sgm;
        cv::cuda::resize(d_left_gray, d_left_sgm, 
                        cv::Size(sgm_width_, sgm_height_), 
                        0, 0, cv::INTER_LINEAR);
        cv::cuda::resize(d_right_gray, d_right_sgm, 
                        cv::Size(sgm_width_, sgm_height_), 
                        0, 0, cv::INTER_LINEAR);
        
        // Download for CPU-based SGM
        cv::Mat h_left_sgm, h_right_sgm;
        d_left_sgm.download(h_left_sgm);
        d_right_sgm.download(h_right_sgm);
        
        // Run SGM inference
        cv::Mat sgm_disparity;
        float sgm_time_ms;
        if (!sgm_inference_->infer(h_left_sgm, h_right_sgm, sgm_disparity, sgm_time_ms)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "SGM inference failed");
            return;
        }
        
        // Run model inference if enabled
        cv::Mat model_disparity;
        float model_time_ms = 0.0f;
        bool has_model_result = false;
        
        if (model_type_ == "sgm_only") {
            // Skip model inference for sgm_only mode
            has_model_result = false;
        } else if (model_type_ == "fastacv" && fastacv_inference_) {
            // Resize to model input resolution
            cv::cuda::GpuMat d_left_model, d_right_model;
            cv::cuda::resize(d_left_gray, d_left_model, 
                            cv::Size(model_input_width_, model_input_height_), 
                            0, 0, cv::INTER_LINEAR);
            cv::cuda::resize(d_right_gray, d_right_model, 
                            cv::Size(model_input_width_, model_input_height_), 
                            0, 0, cv::INTER_LINEAR);
            
            cv::Mat h_left_model, h_right_model;
            d_left_model.download(h_left_model);
            d_right_model.download(h_right_model);
            
            // Convert grayscale back to BGR for model (if needed)
            cv::Mat h_left_model_bgr, h_right_model_bgr;
            cv::cvtColor(h_left_model, h_left_model_bgr, cv::COLOR_GRAY2BGR);
            cv::cvtColor(h_right_model, h_right_model_bgr, cv::COLOR_GRAY2BGR);
            
            if (fastacv_inference_->infer(h_left_model_bgr, h_right_model_bgr, 
                                          model_disparity, model_time_ms)) {
                has_model_result = true;
            }
        } else if (model_type_ == "hitnet" && hitnet_inference_) {
            // Resize to model input resolution
            cv::cuda::GpuMat d_left_model, d_right_model;
            cv::cuda::resize(d_left_gray, d_left_model, 
                            cv::Size(model_input_width_, model_input_height_), 
                            0, 0, cv::INTER_LINEAR);
            cv::cuda::resize(d_right_gray, d_right_model, 
                            cv::Size(model_input_width_, model_input_height_), 
                            0, 0, cv::INTER_LINEAR);
            
            cv::Mat h_left_model, h_right_model;
            d_left_model.download(h_left_model);
            d_right_model.download(h_right_model);
            
            // HitNet accepts grayscale images directly
            if (hitnet_inference_->infer(h_left_model, h_right_model, 
                                         model_disparity, model_time_ms)) {
                has_model_result = true;
            }
        } else if (model_type_ == "fastfs" && fastfs_inference_) {
            // Resize rectified BGR images to model input resolution
            cv::cuda::GpuMat d_left_model, d_right_model;
            cv::cuda::resize(d_left_rect, d_left_model, 
                            cv::Size(model_input_width_, model_input_height_), 
                            0, 0, cv::INTER_LINEAR);
            cv::cuda::resize(d_right_rect, d_right_model, 
                            cv::Size(model_input_width_, model_input_height_), 
                            0, 0, cv::INTER_LINEAR);
            
            cv::Mat h_left_model, h_right_model;
            d_left_model.download(h_left_model);
            d_right_model.download(h_right_model);
            
            // FastFS takes BGR input, handles BGR->RGB conversion internally
            if (fastfs_inference_->infer(h_left_model, h_right_model, 
                                         model_disparity, model_time_ms)) {
                has_model_result = true;
            }
        }
        
        // Update statistics
        frame_count_++;
        total_sgm_time_ += sgm_time_ms;
        total_model_time_ += model_time_ms;
        
        size_t current_frame = frame_count_.load();
        if (current_frame % 30 == 0) {
            float avg_sgm = total_sgm_time_ / current_frame;
            float avg_model = total_model_time_ / current_frame;
            RCLCPP_INFO(this->get_logger(), 
                       "Frame %zu | SGM: %.2f ms (avg %.2f) | Model: %.2f ms (avg %.2f)",
                       current_frame, sgm_time_ms, avg_sgm, model_time_ms, avg_model);
        }
        
        // Choose disparity to use for depth based on model_type
        cv::Mat disparity_for_depth;
        float focal_for_depth;
        
        if (model_type_ == "sgm_only") {
            // Always use SGM disparity for sgm_only mode
            disparity_for_depth = sgm_disparity;
            focal_for_depth = focal_length_sgm_;
        } else if (has_model_result) {
            // Use model disparity when available
            disparity_for_depth = model_disparity;
            focal_for_depth = focal_length_model_;
        } else {
            // Fallback to SGM if model fails
            disparity_for_depth = sgm_disparity;
            focal_for_depth = focal_length_sgm_;
        }

        // Apply valid ROI mask to zero out invalid regions (CPU-based)
        if (stereo_calib_) {
            // Prepare mask if not cached or size changed
            if (depth_mask_cached_.empty() || 
                cached_depth_size_ != cv::Size(disparity_for_depth.cols, disparity_for_depth.rows)) {
                
                // Get valid ROI mask (with erosion) - returns CPU Mat
                cv::Mat valid_mask = stereo_calib_->getLeftValidROI(5);
                
                // Resize mask to match depth resolution if needed
                if (valid_mask.cols != disparity_for_depth.cols || valid_mask.rows != disparity_for_depth.rows) {
                    cv::resize(valid_mask, depth_mask_cached_, 
                             cv::Size(disparity_for_depth.cols, disparity_for_depth.rows), 
                             0, 0, cv::INTER_NEAREST);
                } else {
                    depth_mask_cached_ = valid_mask;
                }
                
                cached_depth_size_ = cv::Size(disparity_for_depth.cols, disparity_for_depth.rows);
            }
            
            // Apply mask to disparity: zero out invalid regions
            // Handle different disparity types
            if (disparity_for_depth.type() == CV_16SC1) {
                // SGM subpixel format - convert mask to int16
                cv::Mat mask_int16;
                depth_mask_cached_.convertTo(mask_int16, CV_16SC1);
                cv::multiply(disparity_for_depth, mask_int16, disparity_for_depth, 1.0, CV_16SC1);
            } else if (disparity_for_depth.type() == CV_32FC1) {
                // Model float format - multiply directly
                cv::multiply(disparity_for_depth, depth_mask_cached_, disparity_for_depth, 1.0, CV_32FC1);
            }
        }
        
        
        // Convert disparity to depth
        cv::Mat depth = disparityToDepth(disparity_for_depth, focal_for_depth);
        cv::Mat valid_mask = (depth > min_depth_m_) & (depth < max_depth_m_);
        depth.setTo(max_depth_m_, ~valid_mask); // Set invalid depths to max depth
        // Publish depth
        if (publish_depth_ && depth_pub_) {
            //resize depth to 256 -> 144
            cv::resize(depth, depth, cv::Size(256, 144), 0, 0, cv::INTER_NEAREST);
            publishDepth(depth, timestamp);
        }
        
        // Publish depth visualization
        if (publish_viz_ && depth_viz_pub_) {
            publishDepthVisualization(depth, timestamp);
        }
        
        // Update visualization data
        if (viz_enabled_) {
            std::lock_guard<std::mutex> lock(viz_mutex_);
            viz_left_ = h_left_sgm.clone();
            viz_right_ = h_right_sgm.clone();
            viz_sgm_disp_ = sgm_disparity.clone();
            if (has_model_result) {
                viz_model_disp_ = model_disparity.clone();
            } else {
                viz_model_disp_ = cv::Mat();
            }
        }
    }
    
    // ===== Visualization Thread =====
    
    void visualizationLoop() {
        const std::string window_name = "Stereo Depth Visualization";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        
        while (running_ && rclcpp::ok()) {
            cv::Mat left, right, sgm_disp, model_disp;
            
            {
                std::lock_guard<std::mutex> lock(viz_mutex_);
                if (viz_left_.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                left = viz_left_.clone();
                right = viz_right_.clone();
                sgm_disp = viz_sgm_disp_.clone();
                model_disp = viz_model_disp_.clone();
            }
            
            // Convert grayscale to BGR for display
            cv::Mat left_bgr, right_bgr;
            cv::cvtColor(left, left_bgr, cv::COLOR_GRAY2BGR);
            cv::cvtColor(right, right_bgr, cv::COLOR_GRAY2BGR);
            
            // Colorize SGM disparity
            cv::Mat sgm_disp_color;
            colorizeDisparity(sgm_disp, sgm_disp_color, 16.0f);
            
            // Resize SGM disparity to model size for comparison
            cv::Mat sgm_disp_resized_color;
            cv::resize(sgm_disp_color, sgm_disp_resized_color, 
                      cv::Size(model_input_width_, model_input_height_));
            
            // Colorize model disparity (if available)
            cv::Mat model_disp_color;
            if (!model_disp.empty()) {
                colorizeDisparity(model_disp, model_disp_color, 1.0f);
            } else {
                // Create black placeholder
                model_disp_color = cv::Mat::zeros(model_input_height_, model_input_width_, CV_8UC3);
                cv::putText(model_disp_color, "No Model", 
                           cv::Point(model_input_width_/2 - 60, model_input_height_/2),
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            }
            
            // Create combined visualization: 2x2 grid
            // Top row: left (SGM size) | right (SGM size)
            // Bottom row: sgm_disp (model size) | model_disp (model size)
            
            // Resize left and right to model size for uniform display
            cv::Mat left_resized, right_resized;
            cv::resize(left_bgr, left_resized, cv::Size(model_input_width_, model_input_height_));
            cv::resize(right_bgr, right_resized, cv::Size(model_input_width_, model_input_height_));
            
            cv::Mat top_row, bottom_row, combined;
            cv::hconcat(left_resized, right_resized, top_row);
            cv::hconcat(sgm_disp_resized_color, model_disp_color, bottom_row);
            cv::vconcat(top_row, bottom_row, combined);
            
            // Add labels
            cv::putText(combined, "Left", cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(combined, "Right", cv::Point(model_input_width_ + 10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(combined, "SGM Disparity", cv::Point(10, model_input_height_ + 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(combined, model_type_.c_str(), 
                       cv::Point(model_input_width_ + 10, model_input_height_ + 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            cv::imshow(window_name, combined);
            
            char key = cv::waitKey(1);
            if (key == 'q' || key == 'Q') {
                RCLCPP_INFO(this->get_logger(), "Visualization window closed");
                break;
            }
        }
        
        cv::destroyWindow(window_name);
    }
    
    // ===== Helper Functions =====
    
    void convertToBGR(const cv::Mat& input, cv::Mat& output) {
        if (input.channels() == 4) {
            cv::cvtColor(input, output, cv::COLOR_RGBA2BGR);
        } else if (input.channels() == 3) {
            output = input;
        } else {
            cv::cvtColor(input, output, cv::COLOR_GRAY2BGR);
        }
    }
    
    void publishRawImages(const cv::Mat& left, const cv::Mat& right, 
                          const rclcpp::Time& timestamp) {
        auto left_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", left).toImageMsg();
        left_msg->header.stamp = timestamp;
        left_msg->header.frame_id = "left_camera";
        left_pub_->publish(*left_msg);
        
        auto right_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", right).toImageMsg();
        right_msg->header.stamp = timestamp;
        right_msg->header.frame_id = "right_camera";
        right_pub_->publish(*right_msg);
    }
    
    cv::Mat disparityToDepth(const cv::Mat& disparity, float focal_length)
    {
        CV_Assert(disparity.type() == CV_16SC1 || disparity.type() == CV_32FC1);

        cv::Mat depth(disparity.size(), CV_32FC1);
        const float fb = focal_length * baseline_;

        if (disparity.type() == CV_16SC1)
        {
            constexpr float inv_scale = 1.0f / 16.0f;

            cv::parallel_for_(cv::Range(0, disparity.rows), [&](const cv::Range& range)
            {
                for (int y = range.start; y < range.end; ++y)
                {
                    const int16_t* disp = disparity.ptr<int16_t>(y);
                    float* out = depth.ptr<float>(y);

                    for (int x = 0; x < disparity.cols; ++x)
                    {
                        const int16_t d = disp[x];
                        // branchless-ish: invalid → 0
                        out[x] = (d > 0) ? (fb / (d * inv_scale)) : 0.0f;
                    }
                }
            });
        }
        else // CV_32FC1
        {
            constexpr float eps = 1e-6f;

            cv::parallel_for_(cv::Range(0, disparity.rows), [&](const cv::Range& range)
            {
                for (int y = range.start; y < range.end; ++y)
                {
                    const float* disp = disparity.ptr<float>(y);
                    float* out = depth.ptr<float>(y);

                    for (int x = 0; x < disparity.cols; ++x)
                    {
                        const float d = disp[x];
                        out[x] = (d > eps) ? (fb / d) : 0.0f;
                    }
                }
            });
        }

        return depth;
    }
    
    void publishDepth(const cv::Mat& depth, const rclcpp::Time& timestamp) {
        auto depth_msg = std::make_shared<sensor_msgs::msg::Image>();
        depth_msg->header.stamp = timestamp;
        depth_msg->header.frame_id = "left_optical_frame";
        depth_msg->height = depth.rows;
        depth_msg->width = depth.cols;
        depth_msg->encoding = "32FC1";
        depth_msg->is_bigendian = false;
        depth_msg->step = depth_msg->width * 4;
        
        size_t data_size = depth.total() * depth.elemSize();
        depth_msg->data.resize(data_size);
        std::memcpy(depth_msg->data.data(), depth.data, data_size);
        
        depth_pub_->publish(*depth_msg);
    }
    
    void publishDepthVisualization(const cv::Mat& depth, const rclcpp::Time& timestamp) {
        cv::Mat depth_color;
        cv::Mat depth_norm;
        cv::normalize(depth, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(depth_norm, depth_color, cv::COLORMAP_JET);
        
        auto viz_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", depth_color).toImageMsg();
        viz_msg->header.stamp = timestamp;
        viz_msg->header.frame_id = "left_optical_frame";
        depth_viz_pub_->publish(*viz_msg);
    }
    
    void colorizeDisparity(const cv::Mat& disparity, cv::Mat& disparity_color, float scale) {
        cv::Mat disp_8u;
        if (disparity.type() == CV_16SC1) {
            cv::Mat disp_float;
            disparity.convertTo(disp_float, CV_32F, 1.0 / scale);
            cv::normalize(disp_float, disp_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
        } else {
            cv::normalize(disparity, disp_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
        }
        cv::applyColorMap(disp_8u, disparity_color, cv::COLORMAP_TURBO);
    }
    
    // ===== Member Variables =====
    ArgusSamples::EGLDisplayHolder g_display_{true};
    ModuleInfo module_info_{};
    Argus::UniqueObj<Argus::CameraProvider> camera_provider_;
    
    // Stereo calibration
    std::unique_ptr<ct_uav_stereo::StereoCalibration> stereo_calib_;
    cv::cuda::GpuMat d_map_left_x_, d_map_left_y_;
    cv::cuda::GpuMat d_map_right_x_, d_map_right_y_;
    
    // Cached depth mask (CPU, prepared once, reused every frame)
    cv::Mat depth_mask_cached_;
    cv::Size cached_depth_size_;
    
    // Inference engines
    std::unique_ptr<ct_uav_stereo::SGMInference> sgm_inference_;
    std::unique_ptr<ct_uav_stereo::FastACVInferenceOptimized> fastacv_inference_;
    std::unique_ptr<ct_uav_stereo::HitNetInferenceOptimized> hitnet_inference_;
    std::unique_ptr<ct_uav_stereo::FastFSInferenceOptimized> fastfs_inference_;
    
    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_viz_pub_;

    
    // Subscribers (for non-camera mode)
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr left_img_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr right_img_sub_;
    
    // Image buffers for subscriber mode
    std::mutex image_mutex_;
    sensor_msgs::msg::CompressedImage::SharedPtr left_img_msg_;
    sensor_msgs::msg::CompressedImage::SharedPtr right_img_msg_;
    
    cv::Mat left_frame_buffer_;
    cv::Mat right_frame_buffer_;
    rclcpp::Time left_timestamp_;
    rclcpp::Time right_timestamp_;
    bool left_received_ = false;
    bool right_received_ = false;
    
    // Timers
    rclcpp::TimerBase::SharedPtr capture_timer_;
    
    // Visualization thread
    std::thread viz_thread_;
    std::atomic<bool> running_;
    std::mutex viz_mutex_;
    int left_id, right_id;
    int cam_width, cam_height, framerate;
    cv::Mat viz_left_, viz_right_, viz_sgm_disp_, viz_model_disp_;
    
    // Settings
    bool debug_sgm_;
    bool viz_enabled_;
    bool use_camera_;
    bool publish_raw_;
    bool publish_depth_;
    bool publish_viz_;

    bool using_gpu_ = true;
    int conversion_type_ = cvt_YUV2BGR;
    bool auto_expose_ = false;
    bool assembled_mode_ = false;
    
    int sgm_width_;
    int sgm_height_;
    int model_input_width_;
    int model_input_height_;
    std::string model_type_;
    double min_depth_m_;
    double max_depth_m_;

    // Calibration parameters
    double focal_length_;
    double baseline_;
    float focal_length_sgm_;
    float focal_length_model_;
    
    // Statistics
    std::atomic<size_t> frame_count_;
    float total_sgm_time_ = 0.0f;
    float total_model_time_ = 0.0f;

};

} // namespace CtUAVStereoCpp

using namespace CtUAVStereoCpp;
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StereoDepthNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
