#ifndef DATA_RECORDER_NODE_HPP
#define DATA_RECORDER_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/float32.hpp>

#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <array>
#include <chrono>
#include <deque>

namespace ct_uav_omni
{

struct ChannelConfig
{
    std::string camera_name;
    msr::airlib::ImageCaptureBase::ImageType image_type;
    std::string topic_name;
    std::string frame_id;
    std::string encoding;  // "rgb8" or "mono8" or "16UC1"
};

struct FrameInfo
{
    cv::Mat image;
    uint64_t timestamp_ns;
    bool valid;
    
    FrameInfo() : valid(false), timestamp_ns(0) {}
};

struct FPSCounter
{
    std::deque<double> deltas;
    std::chrono::steady_clock::time_point last_time;
    size_t max_size;
    std::mutex mutex;
    
    FPSCounter(size_t window_size = 30) 
        : max_size(window_size), last_time(std::chrono::steady_clock::now()) {}
    
    std::pair<double, double> tick()
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto now = std::chrono::steady_clock::now();
        
        if (last_time.time_since_epoch().count() == 0) {
            last_time = now;
            return {0.0, 0.0};
        }
        
        double dt = std::chrono::duration<double>(now - last_time).count();
        last_time = now;
        
        double inst_fps = (dt > 0) ? (1.0 / dt) : 0.0;
        deltas.push_back(inst_fps);
        
        if (deltas.size() > max_size) {
            deltas.pop_front();
        }
        
        double avg_fps = 0.0;
        for (double val : deltas) {
            avg_fps += val;
        }
        avg_fps /= deltas.size();
        
        return {inst_fps, avg_fps};
    }
};

class DataRecorderNode : public rclcpp::Node
{
public:
    explicit DataRecorderNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~DataRecorderNode();
    
    // Prevent copy
    DataRecorderNode(const DataRecorderNode&) = delete;
    DataRecorderNode& operator=(const DataRecorderNode&) = delete;

private:
    // ========== Configuration ==========
    static constexpr size_t NUM_CHANNELS = 4;
    static constexpr bool PERFECT_SYNC = true;
    
    // Channel configurations: RGB + Segmentation for cam0 and cam1
    // NOTE: Segmentation returns RGB image where each pixel's RGB value represents object ID
    //       - Use simSetSegmentationObjectID() to assign IDs before capture
    //       - Each object gets unique color: [R, G, B] = object_id encoded as RGB
    //       - Example: Ground=20, Sky=42, etc.
    std::array<ChannelConfig, NUM_CHANNELS> channel_configs_;
    const std::array<ChannelConfig, NUM_CHANNELS> seg_channel_configs_{
        ChannelConfig{"cam0", msr::airlib::ImageCaptureBase::ImageType::Scene, 
                      "rgb_cam0", "cam0_rgb_optical_frame", "rgb8"},
        ChannelConfig{"cam1", msr::airlib::ImageCaptureBase::ImageType::Scene, 
                      "rgb_cam1", "cam1_rgb_optical_frame", "rgb8"},
        ChannelConfig{"cam0", msr::airlib::ImageCaptureBase::ImageType::Segmentation, 
                      "segmentation_cam0", "cam0_seg_optical_frame", "rgb8"},
        ChannelConfig{"cam1", msr::airlib::ImageCaptureBase::ImageType::Segmentation, 
                      "segmentation_cam1", "cam1_seg_optical_frame", "rgb8"}
    };
    const std::array<ChannelConfig, NUM_CHANNELS> rgb_channel_configs_{
        ChannelConfig{"cam0", msr::airlib::ImageCaptureBase::ImageType::Scene, 
                      "rgb_cam0", "cam0_rgb_optical_frame", "rgb8"},
        ChannelConfig{"cam1", msr::airlib::ImageCaptureBase::ImageType::Scene, 
                      "rgb_cam1", "cam1_rgb_optical_frame", "rgb8"},
        ChannelConfig{"cam2", msr::airlib::ImageCaptureBase::ImageType::Scene, 
                      "rgb_cam2", "cam2_rgb_optical_frame", "rgb8"},
        ChannelConfig{"cam3", msr::airlib::ImageCaptureBase::ImageType::Scene, 
                      "rgb_cam3", "cam3_rgb_optical_frame", "rgb8"},
    };

    const std::array<ChannelConfig, NUM_CHANNELS> depth_channel_configs_{
        ChannelConfig{"cam0", msr::airlib::ImageCaptureBase::ImageType::Scene, 
                      "rgb_cam0", "cam0_rgb_optical_frame", "rgb8"},
        ChannelConfig{"cam1", msr::airlib::ImageCaptureBase::ImageType::Scene, 
                      "rgb_cam1", "cam1_rgb_optical_frame", "rgb8"},
        ChannelConfig{"cam0", msr::airlib::ImageCaptureBase::ImageType::DepthPerspective, 
                      "depth_cam0", "cam0_depth_optical_frame", "16UC1"},
        ChannelConfig{"cam1", msr::airlib::ImageCaptureBase::ImageType::DepthPerspective, 
                      "depth_cam1", "cam1_depth_optical_frame", "16UC1"}
    };
    // ========== AirSim Clients ==========
    std::unique_ptr<msr::airlib::MultirotorRpcLibClient> sim_control_client_;
    std::array<std::unique_ptr<msr::airlib::MultirotorRpcLibClient>, NUM_CHANNELS> channel_clients_;
    
    // ========== Thread Management ==========
    std::thread coordinator_thread_;
    std::array<std::thread, NUM_CHANNELS> channel_threads_;
    std::atomic<bool> stop_flag_{false};
    
    // ========== Synchronization ==========
    std::mutex capture_mutex_;
    std::condition_variable capture_cv_;
    std::atomic<uint64_t> capture_counter_{0};
    std::atomic<size_t> channels_done_counter_{0};
    
    // ========== Frame Buffers (per channel) ==========
    std::array<FrameInfo, NUM_CHANNELS> latest_frames_;
    std::array<std::mutex, NUM_CHANNELS> frame_mutexes_;
    
    // ========== Publishers (per channel) ==========
    std::array<rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr, NUM_CHANNELS> compressed_pubs_;
    std::array<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr, NUM_CHANNELS> raw_pubs_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr fps_pub_;
    
    // ========== Parameters ==========
    std::string host_ip_;
    uint16_t host_port_;
    std::string vehicle_name_;
    bool inter_publish_;
    int jpeg_quality_;
    double sim_advance_time_ms_;
    bool is_vulkan_;
    
    // ========== FPS Tracking ==========
    FPSCounter coordinator_fps_counter_;
    
    // ========== Core Functions ==========
    void initializeAirSimClients();
    void initializePublishers();
    void saveCameraInfo();
    void startThreads();
    void stopThreads();
    
    // Thread functions
    void coordinatorThreadFunc();
    void channelThreadFunc(size_t channel_index);
    
    // Image processing
    cv::Mat responseToMat(const msr::airlib::ImageCaptureBase::ImageResponse& response, 
                          const std::string& encoding);
    cv::Mat depthFloatToUint16(const msr::airlib::ImageCaptureBase::ImageResponse& response);
    sensor_msgs::msg::CompressedImage::SharedPtr createCompressedMsg(
        const cv::Mat& image, uint64_t timestamp_ns, const std::string& frame_id);
    sensor_msgs::msg::Image::SharedPtr createRawMsg(
        const cv::Mat& image, uint64_t timestamp_ns, const std::string& frame_id, 
        const std::string& encoding);
};

} // namespace ct_uav_omni

#endif // DATA_RECORDER_NODE_HPP