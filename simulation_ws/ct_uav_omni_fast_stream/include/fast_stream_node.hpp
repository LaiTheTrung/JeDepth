#ifndef FAST_STREAM_NODE_HPP
#define FAST_STREAM_NODE_HPP

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

class FastStreamNode : public rclcpp::Node
{
public:
    explicit FastStreamNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~FastStreamNode();
    
    // Prevent copy
    FastStreamNode(const FastStreamNode&) = delete;
    FastStreamNode& operator=(const FastStreamNode&) = delete;

private:
    // ========== Configuration ==========
    static constexpr size_t NUM_CAMERAS = 4;
    static constexpr bool PERFECT_SYNC = true;
    
    // Camera names
    std::array<std::string, NUM_CAMERAS> camera_names_ = {"cam0", "cam1", "cam2", "cam3"};
    
    // ========== AirSim Clients ==========
    std::unique_ptr<msr::airlib::MultirotorRpcLibClient> sim_control_client_;
    std::array<std::unique_ptr<msr::airlib::MultirotorRpcLibClient>, NUM_CAMERAS> camera_clients_;
    
    // ========== Thread Management ==========
    std::thread coordinator_thread_;
    std::array<std::thread, NUM_CAMERAS> camera_threads_;
    std::atomic<bool> stop_flag_{false};
    
    // ========== Synchronization ==========
    std::mutex capture_mutex_;
    std::condition_variable capture_cv_;
    std::atomic<uint64_t> capture_counter_{0};
    std::atomic<size_t> cameras_ready_counter_{0};
    std::atomic<size_t> cameras_done_counter_{0};
    
    // ========== Frame Buffers (per camera) ==========
    std::array<FrameInfo, NUM_CAMERAS> latest_frames_{};
    std::array<std::mutex, NUM_CAMERAS> frame_mutexes_;
    
    // ========== Publishers ==========
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr fps_pub_;
    
    // ========== Parameters ==========
    std::string host_ip_;
    uint16_t host_port_;
    std::string vehicle_name_;
    bool inter_publish_;
    int jpeg_quality_;
    double sim_advance_time_ms_;
    
    // ========== FPS Tracking ==========
    FPSCounter coordinator_fps_counter_;
    
    // ========== Core Functions ==========
    void initializeAirSimClients();
    void startThreads();
    void stopThreads();
    
    // Thread functions
    void coordinatorThreadFunc();
    void cameraThreadFunc(size_t camera_index);
    
    // Image processing
    cv::Mat concatenateImages(const std::array<FrameInfo, NUM_CAMERAS>& frames);
    sensor_msgs::msg::CompressedImage::SharedPtr createCompressedMsg(const cv::Mat& image, uint64_t timestamp_ns);
    sensor_msgs::msg::Image::SharedPtr createRawMsg(const cv::Mat& image, uint64_t timestamp_ns);
};

} // namespace ct_uav_omni

#endif // FAST_STREAM_NODE_HPP
