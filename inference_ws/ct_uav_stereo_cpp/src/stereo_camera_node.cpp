#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cstring>
#include <vector>

#include "ct_uav_stereo_cpp/stereo_camera_driver.hpp"

using namespace ArgusSamples;

namespace CtUAVStereoCpp
{

EGLDisplayHolder g_display(true);

class StereoCameraNode : public rclcpp::Node {
public:
	explicit StereoCameraNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
	: Node("stereo_camera_node", options) {
		this->declare_parameter("left_sensor_id", 0);
		this->declare_parameter("right_sensor_id", 1);
		this->declare_parameter("publish_rate", 20.0);
		this->declare_parameter("left_frame_id", "left_camera");
		this->declare_parameter("right_frame_id", "right_camera");
		this->declare_parameter("debug", true);
		this->declare_parameter("record_path", "");
		this->declare_parameter("using_gpu", true);
		this->declare_parameter("conversion_type", cvt_YUV2BGR);
		this->declare_parameter("auto_expose", true);
		this->declare_parameter("assembled_mode", true);
		this->declare_parameter("publish_concat", true);

		left_id_ = this->get_parameter("left_sensor_id").as_int();
		right_id_ = this->get_parameter("right_sensor_id").as_int();
		publish_rate_ = this->get_parameter("publish_rate").as_double();
		left_frame_id_ = this->get_parameter("left_frame_id").as_string();
		right_frame_id_ = this->get_parameter("right_frame_id").as_string();
		record_path_ = this->get_parameter("record_path").as_string();
		is_debug_ = this->get_parameter("debug").as_bool();
		using_gpu_ = this->get_parameter("using_gpu").as_bool();
		conversion_type_ = this->get_parameter("conversion_type").as_int();
		auto_expose_ = this->get_parameter("auto_expose").as_bool();
		assembled_mode_ = this->get_parameter("assembled_mode").as_bool();
		publish_concat_ = this->get_parameter("publish_concat").as_bool();
		width_ = 1640;
		height_ = 1232;
		framerate_ = 30;
		STREAM_SIZE = Argus::Size2D<uint32_t>(width_, height_);
		// Initialize the Argus camera provider.
		memset(&module_info_, 0, sizeof(ModuleInfo));
		module_info_.initialized = false;
		// ArgusSamples::Window &window = ArgusSamples::Window::getInstance();
    	// window.setWindowRect(options.windowRect());
        if (!g_display.initialize()) 
            std::cerr<<"Error g_display init failed"<<std::endl;
		camera_provider_ = Argus::UniqueObj<Argus::CameraProvider>(Argus::CameraProvider::create());
		if (!initializeStereoDriver()) {
			RCLCPP_ERROR(this->get_logger(), "Failed to initialize stereo hardware sync driver");
			rclcpp::shutdown();
			return;
		}

		if (!record_path_.empty()) {
			initializeVideoRecording(width_, height_, framerate_);
		}

		concat_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
			"/stereo/concat/image_raw", 10);

		
		if (conversion_type_ == cvt_YUV2GRAY) {
			left_frame_ = cv::Mat(height_, width_, CV_8UC1);
			right_frame_ = cv::Mat(height_, width_, CV_8UC1);
		}
		else {
			left_frame_ = cv::Mat(height_, width_, CV_8UC3);
			right_frame_ = cv::Mat(height_, width_, CV_8UC3);
		}
		auto period = std::chrono::duration<double>(1.0 / publish_rate_);
		timer_ = this->create_wall_timer(
			std::chrono::duration_cast<std::chrono::milliseconds>(period),
			std::bind(&StereoCameraNode::timerCallback, this));
		
		RCLCPP_INFO(this->get_logger(), "Stereo camera node started (HW sync)");
		RCLCPP_INFO(this->get_logger(), "Resolution: %dx%d @ %d fps", width_, height_, framerate_);
	}

	~StereoCameraNode() override {
		shutdownStereoDriver();
		if (video_writer_.isOpened()) {
			video_writer_.release();
			RCLCPP_INFO(this->get_logger(), "Video recording stopped. Total frames: %ld", frame_count_);
			RCLCPP_INFO(this->get_logger(), "Saved to: %s", record_path_.c_str());
		}
		RCLCPP_INFO(this->get_logger(), "Stereo camera node stopped");
	}

private:
	bool initializeStereoDriver() {
		auto* i_camera_provider = Argus::interface_cast<Argus::ICameraProvider>(camera_provider_);
		if (!i_camera_provider) {
			RCLCPP_ERROR(this->get_logger(), "Failed to create Argus camera provider");
			return false;
		}

		i_camera_provider->setSyncSensorSessionsCount(1,0);

		std::vector<Argus::CameraDevice*> camera_devices;
		i_camera_provider->getCameraDevices(&camera_devices);
		if (camera_devices.empty()) {
			RCLCPP_ERROR(this->get_logger(), "No camera devices found");
			return false;
		}

		int max_id = std::max(left_id_, right_id_);
		if (max_id < 0 || static_cast<size_t>(max_id) >= camera_devices.size()) {
			RCLCPP_ERROR(this->get_logger(), "Invalid camera ids (left=%d, right=%d)", left_id_, right_id_);
			return false;
		}

		std::strncpy(module_info_.moduleName, "stereo", MAX_MODULE_STRING - 1);
		module_info_.initialized = true;
		module_info_.sensorCount = 2;
		module_info_.camDevice[left_device] = left_id_;
		module_info_.camDevice[right_device] = right_id_;
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

			module_info_.captureSession[i] = UniqueObj<CaptureSession>(i_camera_provider->createCaptureSession(camera_device));
			module_info_.iCaptureSession[i] = Argus::interface_cast<Argus::ICaptureSession>(module_info_.captureSession[i]);
			if (!module_info_.iCaptureSession[i]) {
				RCLCPP_ERROR(this->get_logger(), "Failed to create capture session for sensor %d", i);
				return false;
			}

			module_info_.streamSettings[i] = UniqueObj<OutputStreamSettings>(
				module_info_.iCaptureSession[i]->createOutputStreamSettings(Argus::STREAM_TYPE_EGL));
			auto* i_stream_settings = Argus::interface_cast<Argus::IOutputStreamSettings>(module_info_.streamSettings[i]);
			auto* i_egl_stream_settings = Argus::interface_cast<Argus::IEGLOutputStreamSettings>(module_info_.streamSettings[i]);
			if (!i_stream_settings || !i_egl_stream_settings) {
				RCLCPP_ERROR(this->get_logger(), "Failed to create stream settings for sensor %d", i);
				return false;
			}

			i_egl_stream_settings->setPixelFormat(Argus::PIXEL_FMT_YCbCr_420_888);
			i_egl_stream_settings->setResolution(STREAM_SIZE);
			i_egl_stream_settings->setMetadataEnable(true);
			i_egl_stream_settings->setMode(Argus::EGL_STREAM_MODE_MAILBOX);
			// i_egl_stream_settings->setFifoLength(EGL_STREAM_BUFFERS);
			i_egl_stream_settings->setEGLDisplay(g_display.get());

			i_stream_settings->setCameraDevice(camera_device);
			module_info_.stream[i] = UniqueObj<OutputStream>(
				module_info_.iCaptureSession[i]->createOutputStream(module_info_.streamSettings[i].get()));
		}

		auto* left_egl_stream_settings = Argus::interface_cast<Argus::IEGLOutputStreamSettings>(
			module_info_.streamSettings[left_device]);
		module_info_.stereoYuvConsumer = new SyncStereoConsumerThread(
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
			module_info_.request[i] = UniqueObj<Request>(
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

			if (auto* i_source_settings = Argus::interface_cast<Argus::ISourceSettings>(module_info_.request[i])) {
				uint64_t frame_duration = 1000000000ULL / static_cast<uint64_t>(framerate_);
				i_source_settings->setSensorMode(module_info_.sensorMode[i]);
				i_source_settings->setFrameDurationRange(Argus::Range<uint64_t>(frame_duration));
				i_source_settings->setExposureTimeRange(EXPOSURE_TIME_RANGE);
				i_source_settings->setGainRange(Argus::Range<float>(GAIN_RANGE.min()));
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

		if (!g_display.cleanup()) {
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

	void initializeVideoRecording(int width, int height, int framerate) {
		std::filesystem::path target_path(record_path_);
		if (std::filesystem::is_directory(target_path) || target_path.extension().empty()) {
			if (std::filesystem::is_directory(target_path)) {
				// keep directory
			} else {
				target_path = target_path.parent_path().empty() ? std::filesystem::path("./") : target_path;
			}
			std::filesystem::path dir = target_path;
			if (!dir.empty() && !std::filesystem::exists(dir)) {
				std::filesystem::create_directories(dir);
				RCLCPP_INFO(this->get_logger(), "Created directory: %s", dir.c_str());
			}

			auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			std::tm tm_buf{};
			localtime_r(&t, &tm_buf);
			std::ostringstream name;
			name << "stereo_hw_" << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << ".mp4";
			target_path /= name.str();
		} else {
			std::filesystem::path parent_dir = target_path.parent_path();
			if (!parent_dir.empty() && !std::filesystem::exists(parent_dir)) {
				std::filesystem::create_directories(parent_dir);
				RCLCPP_INFO(this->get_logger(), "Created directory: %s", parent_dir.c_str());
			}
		}

		int concat_width = width * 2;
		int concat_height = height;
		int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

		video_writer_.open(target_path.string(), fourcc, framerate,
						   cv::Size(concat_width, concat_height), true);

		if (!video_writer_.isOpened()) {
			RCLCPP_ERROR(this->get_logger(),
						"Failed to open video writer for path: %s", target_path.c_str());
			record_path_.clear();
			return;
		}

		record_path_ = target_path.string();
		RCLCPP_INFO(this->get_logger(), "Video recording initialized");
		RCLCPP_INFO(this->get_logger(), "Output path: %s", record_path_.c_str());
		RCLCPP_INFO(this->get_logger(), "Resolution: %dx%d (stereo concatenated)",
				   concat_width, concat_height);
		RCLCPP_INFO(this->get_logger(), "Framerate: %d fps", framerate);
	}


	void timerCallback() {
		if (!module_info_.stereoYuvConsumer) {
			RCLCPP_WARN(this->get_logger(), "Stereo consumer not initialized");
			return;
		}
		
		double latency = 0.0;
		if (is_debug_) {
		cv::Mat concatMat;
		module_info_.stereoYuvConsumer->get_frames(concatMat, latency);
		if (latency > 1000) {
			RCLCPP_INFO(this->get_logger(), "Too high latency: %.2f ms", latency);
			return;
		}
		
		if (auto_expose_){
			float frameGain = module_info_.stereoYuvConsumer->get_frame_gain();
			uint64_t frameExposureTime = module_info_.stereoYuvConsumer->get_frame_exposure_time();
			float  ispGain = module_info_.stereoYuvConsumer->get_isp_gain();
			// RCLCPP_INFO(this->get_logger(), "Auto-exposure update: Gain=%.2f, ExposureTime=%.2f ms, ISP Gain=%.2f", frameGain, frameExposureTime / 1e6, ispGain);
			for (uint32_t i = 0; i < 2; i++){
				ISourceSettings *iSourceSettings = interface_cast<ISourceSettings>(module_info_.request[i]);
				 
				if(iSourceSettings->setGainRange(Argus::Range<float>(frameGain)) != Argus::STATUS_OK)
					RCLCPP_ERROR(this->get_logger(),"Failed to set Analog gain");

				if(iSourceSettings->setExposureTimeRange(Argus::Range<uint64_t>(frameExposureTime)) != Argus::STATUS_OK) 
					RCLCPP_ERROR(this->get_logger(),"Failed to set Exposure time");

				auto* i_request = Argus::interface_cast<Argus::IRequest>(module_info_.request[i]);
				if (auto* i_auto_control = Argus::interface_cast<Argus::IAutoControlSettings>(i_request->getAutoControlSettings())) {
				i_auto_control->setIspDigitalGainRange(Range<float>(ispGain)); 
				}
				/*
				* The modified request is re-submitted to terminate the previous repeat() with
				* the old settings and begin captures with the new settings
				*/
				module_info_.iCaptureSession[i]->repeat(module_info_.request[i].get());
			}
		}
		cv::Mat concat_vis;
		cv::resize(concatMat, concat_vis, cv::Size(1280, 480));
		if (conversion_type_ == cvt_YUV2RGB) {
			cv::cvtColor(concat_vis, concat_vis, cv::COLOR_RGB2BGR);
		}
		cv::imshow("Stereo Camera (SW Sync)", concat_vis);
		cv::waitKey(1);
		}


		if (publish_concat_){
			auto msg = std::make_unique<sensor_msgs::msg::Image>();
			msg->header.frame_id = "stereo_camera";
			module_info_.stereoYuvConsumer->get_image_msg(*msg, latency);
			if (latency > 1000) {
				RCLCPP_INFO(this->get_logger(), "Too high latency: %.2f ms", latency);
				return;
			}
			auto timestamp = this->now() - rclcpp::Duration::from_seconds(latency / 1000.0);
			msg->header.stamp = timestamp;
			concat_pub_->publish(std::move(msg));
		}


		std::vector<uchar> left_buf;
		std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};

		RCLCPP_INFO(this->get_logger(), "Received new stereo frames ");

		frame_count_++;
		if (frame_count_ % 30 == 0) {
			RCLCPP_INFO(this->get_logger(), "Published %ld stereo frames", frame_count_);
		}
	}

	void normalizeFrame(const cv::Mat& input, cv::Mat& output) {
		if (input.channels() == 4) {
			cv::cvtColor(input, output, cv::COLOR_RGBA2BGR);
		} else if (input.channels() == 3) {
			output = input.clone();
		} else if (input.channels() == 1) {
			cv::cvtColor(input, output, cv::COLOR_GRAY2BGR);
		} else {
			output = input.clone();
		}
	}

	void recordStereoFrame(const cv::Mat& left_frame, const cv::Mat& right_frame) {
		cv::Mat left_bgr, right_bgr;

		if (left_frame.channels() == 3) {
			left_bgr = left_frame;
		} else {
			cv::cvtColor(left_frame, left_bgr, cv::COLOR_GRAY2BGR);
		}

		if (right_frame.channels() == 3) {
			right_bgr = right_frame;
		} else {
			cv::cvtColor(right_frame, right_bgr, cv::COLOR_GRAY2BGR);
		}

		cv::Mat concatenated;
		cv::hconcat(left_bgr, right_bgr, concatenated);
		video_writer_.write(concatenated);

		if (frame_count_ % 300 == 0) {
			RCLCPP_INFO(this->get_logger(), "Recorded %ld frames to video", frame_count_);
		}
	}

	int left_id_ = 0;
	int right_id_ = 1;
	int width_ = 1280;
	int height_ = 720;
	int framerate_ = 60;
	double publish_rate_ = 30.0;
	bool using_gpu_ = true;
	int conversion_type_ = cvt_YUV2BGR;
	bool is_debug_ = false;
	bool auto_expose_ = false;
	bool assembled_mode_ = false;
	bool publish_concat_ = true;

	std::string left_frame_id_;
	std::string right_frame_id_;
	std::string record_path_;

	ModuleInfo module_info_{};
	Argus::UniqueObj<Argus::CameraProvider> camera_provider_;

	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr concat_pub_;
	rclcpp::TimerBase::SharedPtr timer_;

	cv::Mat left_frame_;
	cv::Mat right_frame_;
	cv::Mat concat_frame_;

	cv::VideoWriter video_writer_;
	size_t frame_count_ = 0;
	};
} // namespace CtUAVStereoCpp

RCLCPP_COMPONENTS_REGISTER_NODE(CtUAVStereoCpp::StereoCameraNode)

int main(int argc, char** argv) {
	rclcpp::init(argc, argv);
	rclcpp::NodeOptions options;
	auto node = std::make_shared<CtUAVStereoCpp::StereoCameraNode>(options);
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
