/**
 * Obstacle Prevention Node
 * 
 * This node:
 * 1. Subscribes to depth map from stereo_depth_node
 * 2. Subscribes to vehicle position/attitude from PX4
 * 3. Rotates and crops depth map based on attitude (20 deg vertical crop)
 * 4. Converts to obstacle distance sectors and publishes to PX4
 * 
 * Author: CT-UAV Team
 * Date: 2026-01-26
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <px4_msgs/msg/obstacle_distance.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <algorithm>
#include <deque>
#include <mutex>
#include <limits>
#include <cstdint>

static constexpr uint16_t UINT16_MAX_VAL = 65535;

namespace ct_uav_stereo {

class PitchQueue {
public:
    struct Sample {
        uint64_t timestamp_us;
        float pitch_rad;
    };

    explicit PitchQueue(size_t max_size = 200)
        : max_size_(std::max<size_t>(10, max_size)) {}

    void setMaxSize(size_t max_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        max_size_ = std::max<size_t>(10, max_size);
        while (samples_.size() > max_size_) {
            samples_.pop_front();
        }
    }

    void push(uint64_t timestamp_us, float pitch_rad) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (samples_.empty() || timestamp_us >= samples_.back().timestamp_us) {
            samples_.push_back(Sample{timestamp_us, pitch_rad});
        } else {
            auto it = std::lower_bound(
                samples_.begin(), samples_.end(), timestamp_us,
                [](const Sample& sample, uint64_t ts) {
                    return sample.timestamp_us < ts;
                });
            samples_.insert(it, Sample{timestamp_us, pitch_rad});
        }

        while (samples_.size() > max_size_) {
            samples_.pop_front();
        }
    }

    bool findCausalNearest(uint64_t target_ts_us,
                           uint64_t tolerance_us,
                           float& out_pitch_rad,
                           uint64_t& out_abs_dt_us) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (samples_.empty()) {
            return false;
        }

        auto it = std::lower_bound(
            samples_.begin(), samples_.end(), target_ts_us,
            [](const Sample& sample, uint64_t ts) {
                return sample.timestamp_us < ts;
            });

        const Sample* best = nullptr;
        if (it == samples_.begin()) {
            best = &samples_.front();
        } else {
            best = &(*std::prev(it));
        }

        out_abs_dt_us = (best->timestamp_us > target_ts_us)
            ? (best->timestamp_us - target_ts_us)
            : (target_ts_us - best->timestamp_us);

        if (out_abs_dt_us > tolerance_us) {
            return false;
        }

        out_pitch_rad = best->pitch_rad;
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return samples_.size();
    }

private:
    mutable std::mutex mutex_;
    std::deque<Sample> samples_;
    size_t max_size_;
};

class ObstaclePreventionNode : public rclcpp::Node {
public:
    ObstaclePreventionNode() : Node("obstacle_prevention_node") {
        
        // ===== Declare Parameters =====
        this->declare_parameter("debug", false);
        this->declare_parameter("obstacle_prevention_enable", true);
        this->declare_parameter("fov_deg", 90.0);
        this->declare_parameter("min_depth_m", 0.2);
        this->declare_parameter("crop_left_angle", 0);
        this->declare_parameter("crop_right_angle", 0);
        this->declare_parameter("max_depth_m", 30.0);
        this->declare_parameter("vertical_crop_deg", 20.0);
        this->declare_parameter("num_sectors", 72);
        this->declare_parameter("pitch_queue_size", 200);
        this->declare_parameter("sync_tolerance_ms", 80.0);
        this->declare_parameter("obstacle_topic", "/fmu/in/obstacle_distance");
        this->declare_parameter("depth_topic", "/stereo/depth_map");
        // ===== Get Parameters =====
        debug_ = this->get_parameter("debug").as_bool();
        enabled_ = this->get_parameter("obstacle_prevention_enable").as_bool();
        fov_deg_ = this->get_parameter("fov_deg").as_double();
        min_depth_m_ = this->get_parameter("min_depth_m").as_double();
        max_depth_m_ = this->get_parameter("max_depth_m").as_double();
        crop_left_ = this->get_parameter("crop_left_angle").as_int();
        crop_right_ = this->get_parameter("crop_right_angle").as_int();
        vertical_crop_deg_ = this->get_parameter("vertical_crop_deg").as_double();
        int queue_size = this->get_parameter("pitch_queue_size").as_int();
        pitch_queue_size_ = static_cast<size_t>(
            std::max(10, queue_size));
        sync_tolerance_us_ = static_cast<uint64_t>(
            std::max(1.0, this->get_parameter("sync_tolerance_ms").as_double()) * 1000.0);
        num_sectors_ = this->get_parameter("num_sectors").as_int();
        if ((num_sectors_ != 72) && (num_sectors_ != 36)) {
            RCLCPP_WARN(this->get_logger(), "num_sectors must be 36, or 72. Setting to 72.");
            num_sectors_ = 72;
        }
        sector_increment_ = 360.0 / static_cast<double>(num_sectors_);
        std::string obstacle_topic = this->get_parameter("obstacle_topic").as_string();
        std::string depth_topic = this->get_parameter("depth_topic").as_string();
        vertical_crop_rad_ = vertical_crop_deg_ * M_PI / 180.0;
        pitch_queue_.setMaxSize(pitch_queue_size_);
        
        if (!enabled_) {
            RCLCPP_WARN(this->get_logger(), "Obstacle prevention is DISABLED");
        }
        
        // ===== QoS Settings =====
        auto qos_sensor = rclcpp::QoS(rclcpp::KeepLast(10))
            .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
            .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
        
        // ===== Create Subscribers =====
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            depth_topic, 10,
            std::bind(&ObstaclePreventionNode::depthCallback, this, std::placeholders::_1));
        
        lpos_sub_ = this->create_subscription<px4_msgs::msg::VehicleLocalPosition>(
            "/fmu/out/vehicle_local_position", qos_sensor,
            std::bind(&ObstaclePreventionNode::vehicleLocalPositionCallback, this, std::placeholders::_1));
        
        att_sub_ = this->create_subscription<px4_msgs::msg::VehicleAttitude>(
            "/fmu/out/vehicle_attitude", qos_sensor,
            std::bind(&ObstaclePreventionNode::vehicleAttitudeCallback, this, std::placeholders::_1));
        
        // ===== Create Publisher =====
        obstacle_pub_ = this->create_publisher<px4_msgs::msg::ObstacleDistance>(
            obstacle_topic, 10);
        
        RCLCPP_INFO(this->get_logger(), "=== Obstacle Prevention Node Started ===");
        RCLCPP_INFO(this->get_logger(), "  Enabled: %s", enabled_ ? "YES" : "NO");
        RCLCPP_INFO(this->get_logger(), "  FOV: %.1f deg", fov_deg_);
        RCLCPP_INFO(this->get_logger(), "  Vertical crop: %.1f deg", vertical_crop_deg_);
        RCLCPP_INFO(this->get_logger(), "  Sectors: %d (%.1f deg increment)", num_sectors_, sector_increment_);
        RCLCPP_INFO(this->get_logger(), "  Depth range: %.2f - %.2f m", min_depth_m_, max_depth_m_);
        RCLCPP_INFO(this->get_logger(), "  Pitch queue size: %zu", pitch_queue_size_);
        RCLCPP_INFO(this->get_logger(), "  Sync tolerance: %.1f ms", static_cast<double>(sync_tolerance_us_) / 1000.0);
        RCLCPP_INFO(this->get_logger(), "  Publishing to: %s", obstacle_topic.c_str());
    }

private:
    
    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg) {
        pose_received_ = true;
    }
    
    void vehicleAttitudeCallback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg) {
        // Convert quaternion to roll, pitch, yaw
        float w = msg->q[0];
        float x = msg->q[1];
        float y = msg->q[2];
        float z = msg->q[3];
        
        // Pitch (y-axis rotation)
        float t2 = 2.0f * (w * y - z * x);
        t2 = std::max(-1.0f, std::min(1.0f, t2));
        pitch_ = std::asin(t2);

        const uint64_t receive_ts_us = static_cast<uint64_t>(this->now().nanoseconds()) / 1000ULL;
        pitch_queue_.push(receive_ts_us, pitch_);


        att_received_ = true;
    }
    
    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (!enabled_ || !pose_received_ || !att_received_) {
            if (!enabled_) {
                return;
            }
            if (!pose_received_) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Waiting for vehicle position...");
            }
            if (!att_received_) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Waiting for vehicle attitude...");
                roll_ = 0; 
                pitch_ = 0;
                yaw_ = 0;
            }
        }
        
        // Convert ROS Image to OpenCV Mat
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        
        cv::Mat depth = cv_ptr->image;
        if (depth.empty()) {
            return;
        }

        const uint64_t depth_receive_ts_us = static_cast<uint64_t>(this->now().nanoseconds()) / 1000ULL;
        float synced_pitch = pitch_;
        uint64_t abs_dt_us = 0;

        if (pitch_queue_.findCausalNearest(depth_receive_ts_us, sync_tolerance_us_, synced_pitch, abs_dt_us)) {
            if (debug_) {
                RCLCPP_DEBUG(this->get_logger(), "Pitch synced, |dt|=%.2f ms", static_cast<double>(abs_dt_us) / 1000.0);
            }
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Pitch sync miss (queue=%zu, tol=%.1f ms). Fallback to latest pitch",
                pitch_queue_.size(), static_cast<double>(sync_tolerance_us_) / 1000.0);
        }
        
        // Process depth map to obstacle distances
        processDepthMap(depth, msg->header.stamp, synced_pitch);
    }
    
    void processDepthMap(const cv::Mat& depth,
                         const builtin_interfaces::msg::Time& stamp,
                         float synced_pitch_rad) {
        int h = depth.rows;
        int w = depth.cols;
        
        // Calculate camera intrinsics
        float fx =  (w / 2.0) / std::tan(fov_deg_ * M_PI / 180.0 / 2.0);  // Focal length in pixels
        float cx = w / 2.0f;
        float cy = h / 2.0f;
        
        // Initialize obstacle distance array
        std::vector<uint16_t> distances_cm(num_sectors_, UINT16_MAX_VAL);
        
        // Calculate vertical crop pixel range
        float vertical_crop_pixel = std::tan(vertical_crop_rad_) * fx;
        // compensate vertical crop for camera tilt (pitch)
        float offset_verticle = std::tan(synced_pitch_rad) * fx;
        int row_min = std::max(0, static_cast<int>(cy - vertical_crop_pixel + offset_verticle));
        int row_max = std::min(h - 1, static_cast<int>(cy + vertical_crop_pixel + offset_verticle));
        // limit boundary to center row to avoid flipping when pitch is large
        row_min = std::min(row_min, static_cast<int>(h)-20); 
        row_max = std::max(row_max, 20);
        // Crop depth map vertically based on camera tilt and vertical FOV
        cv::Mat cropped_depth = depth.rowRange(row_min, row_max + 1);
        cv::Mat valid_mask = (cropped_depth > min_depth_m_) & (cropped_depth < max_depth_m_);
        cropped_depth.setTo(max_depth_m_, ~valid_mask); // ignore invalid
        if (debug_){
            cv::imshow("Cropped Depth", cropped_depth / max_depth_m_);
            cv::waitKey(1);
        }
        // Calculate number of active sectors based on FOV
        int active_sectors = static_cast<int>(num_sectors_ * fov_deg_ / 360.0);
        if (active_sectors <= 0) active_sectors = 1;
        if (active_sectors > num_sectors_) active_sectors = num_sectors_;
        
        // Calculate sector width in pixels
        float sector_width = static_cast<float>(w) / static_cast<float>(active_sectors);
        
        // Process each sector using sliding window
        // index 0 is forward (0 deg), index increases clockwise (right positive, left negative)
        // crop_left_ and crop_right_ are in degrees, convert to sector count inside active FOV
        int crop_left_sectors = std::max(0, static_cast<int>(std::round(crop_left_ / sector_increment_)));
        int crop_right_sectors = std::max(0, static_cast<int>(std::round(crop_right_ / sector_increment_)));
        int valid_active_sectors = active_sectors - crop_left_sectors - crop_right_sectors;
        if (valid_active_sectors <= 0) {
            publishObstacleDistance(distances_cm, stamp);
            return;
        }

        auto wrapSectorIndex = [this](int idx) {
            int wrapped = idx % num_sectors_;
            if (wrapped < 0) wrapped += num_sectors_;
            return wrapped;
        };

        int active_start_global = -(active_sectors / 2) + crop_left_sectors;
        for (int sector_idx = 0; sector_idx < valid_active_sectors; ++sector_idx) {
            int source_sector_idx = sector_idx + crop_left_sectors;
            // Calculate column range for this sector
            int col_start = static_cast<int>(source_sector_idx * sector_width);
            int col_end = static_cast<int>((source_sector_idx + 1) * sector_width);
            col_end = std::min(col_end, w);
            
            if (col_start >= col_end) continue;
            
            // Extract sector region
            cv::Mat sector_region = cropped_depth.colRange(col_start, col_end);
            
            // Find minimum valid depth in this sector
            double min_val;
            cv::minMaxLoc(sector_region, &min_val, nullptr);
            
            // Convert to cm and store
            if (min_val < max_depth_m_) {
                uint16_t dist_cm = static_cast<uint16_t>(std::round(min_val * 100.0f));
                if (dist_cm > 0) {
                    int global_sector_idx = wrapSectorIndex(active_start_global + sector_idx);
                    distances_cm[global_sector_idx] = dist_cm;
                }
            }
        }
        
        publishObstacleDistance(distances_cm, stamp);
    }
    
    void publishObstacleDistance(const std::vector<uint16_t>& distances_cm,
                                 const builtin_interfaces::msg::Time& stamp) {
        auto msg = px4_msgs::msg::ObstacleDistance();
        
        // Timestamp in microseconds
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        
        // Frame of reference
        msg.frame = px4_msgs::msg::ObstacleDistance::MAV_FRAME_BODY_FRD;
        
        // Sensor type
        msg.sensor_type = px4_msgs::msg::ObstacleDistance::MAV_DISTANCE_SENSOR_LASER;
        
        // Angular parameters
        msg.angle_offset = 0.0f;
        msg.increment = static_cast<float>(sector_increment_);
        
        // Distance range in cm
        msg.min_distance = static_cast<uint16_t>(std::round(min_depth_m_ * 100.0f));
        msg.max_distance = static_cast<uint16_t>(std::round(max_depth_m_ * 100.0f));
        
        // Copy distances (msg.distances is std::array<uint16_t, 72>, fixed size)
        for (size_t i = 0; i < msg.distances.size() && i < distances_cm.size(); ++i) {
            msg.distances[i] = distances_cm[i];
        }
        
        obstacle_pub_->publish(msg);
        
        // Log statistics periodically
        static size_t count = 0;
        if (++count % 30 == 0) {
            int valid_sectors = 0;
            for (auto d : distances_cm) {
                if (d != UINT16_MAX_VAL) valid_sectors++;
            }
            RCLCPP_INFO(this->get_logger(), "Published obstacle distance: %d/%d sectors valid",
                       valid_sectors, num_sectors_);
        }
    }
    
    // ===== Member Variables =====
    
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr lpos_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr att_sub_;
    
    // Publisher
    rclcpp::Publisher<px4_msgs::msg::ObstacleDistance>::SharedPtr obstacle_pub_;
    
    // Parameters
    bool debug_;
    bool enabled_;
    double fov_deg_;
    double min_depth_m_;
    double max_depth_m_;
    double vertical_crop_deg_;
    double vertical_crop_rad_;
    double sector_increment_;
    int num_sectors_;
    double focal_length_;
    int crop_left_;
    int crop_right_;
    
    // State
    bool pose_received_ = false;
    bool att_received_ = false;
    float roll_ = 0.0f;
    float pitch_ = 0.0f;
    float yaw_ = 0.0f;

    // Pitch synchronization
    PitchQueue pitch_queue_;
    size_t pitch_queue_size_ = 200;
    uint64_t sync_tolerance_us_ = 80000;
};
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ct_uav_stereo::ObstaclePreventionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
