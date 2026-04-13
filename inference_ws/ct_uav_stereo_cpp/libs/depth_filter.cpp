#include "ct_uav_stereo_cpp/depth_filter.hpp"
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <type_traits>
#include <utility>
#include <vector>

namespace ct_uav_stereo {

namespace {

template <typename T>
bool isValidSample(const T value) {
    if constexpr (std::is_floating_point<T>::value) {
        return std::isfinite(value) && value > static_cast<T>(std::numeric_limits<T>::epsilon());
    }
    return value > static_cast<T>(0);
}

template <typename T>
T zeroSample() {
    return static_cast<T>(0);
}

template <typename T>
T blendSample(const T current, const T state, const float alpha) {
    const float filtered = static_cast<float>(current) * alpha + static_cast<float>(state) * (1.0f - alpha);
    if constexpr (std::is_floating_point<T>::value) {
        return static_cast<T>(filtered);
    }
    return static_cast<T>(filtered + 0.5f);
}

template <typename T>
void recursiveHorizontalPass(cv::Mat& depth, const float alpha, const float delta, const int hole_filling_radius) {
    const int rows = depth.rows;
    const int cols = depth.cols;

    for (int y = 0; y < rows; ++y) {
        T* row = depth.ptr<T>(y);

        T state = row[0];
        int fill_count = 0;
        for (int x = 1; x < cols; ++x) {
            T& sample = row[x];
            if (isValidSample(state) && isValidSample(sample)) {
                fill_count = 0;
                if (std::fabs(static_cast<float>(sample) - static_cast<float>(state)) <= delta) {
                    sample = blendSample(sample, state, alpha);
                }
                state = sample;
            } else if (isValidSample(state) && !isValidSample(sample) && hole_filling_radius > 0) {
                ++fill_count;
                if (fill_count <= hole_filling_radius) {
                    sample = state;
                }
            } else {
                fill_count = 0;
                state = sample;
            }
        }

        state = row[cols - 1];
        fill_count = 0;
        for (int x = cols - 2; x >= 0; --x) {
            T& sample = row[x];
            if (isValidSample(state) && isValidSample(sample)) {
                fill_count = 0;
                if (std::fabs(static_cast<float>(sample) - static_cast<float>(state)) <= delta) {
                    sample = blendSample(sample, state, alpha);
                }
                state = sample;
            } else if (isValidSample(state) && !isValidSample(sample) && hole_filling_radius > 0) {
                ++fill_count;
                if (fill_count <= hole_filling_radius) {
                    sample = state;
                }
            } else {
                fill_count = 0;
                state = sample;
            }
        }
    }
}

template <typename T>
void recursiveVerticalPass(cv::Mat& depth, const float alpha, const float delta) {
    const int rows = depth.rows;
    const int cols = depth.cols;

    for (int x = 0; x < cols; ++x) {
        T state = depth.at<T>(0, x);
        for (int y = 1; y < rows; ++y) {
            T& sample = depth.at<T>(y, x);
            if (isValidSample(state) && isValidSample(sample) &&
                std::fabs(static_cast<float>(sample) - static_cast<float>(state)) <= delta) {
                sample = blendSample(sample, state, alpha);
            }
            state = sample;
        }

        state = depth.at<T>(rows - 1, x);
        for (int y = rows - 2; y >= 0; --y) {
            T& sample = depth.at<T>(y, x);
            if (isValidSample(state) && isValidSample(sample) &&
                std::fabs(static_cast<float>(sample) - static_cast<float>(state)) <= delta) {
                sample = blendSample(sample, state, alpha);
            }
            state = sample;
        }
    }
}

template <typename T>
void applySpatialRecursive(cv::Mat& depth, const DepthFilter::SpatialConfig& config) {
    for (int i = 0; i < config.iterations; ++i) {
        recursiveHorizontalPass<T>(depth, config.alpha, config.delta, config.hole_filling_radius);
        recursiveVerticalPass<T>(depth, config.alpha, config.delta);
    }
}

}  // namespace

bool DepthFilter::isSupportedDepthType(const int type) {
    return type == CV_32FC1 || type == CV_16UC1 || type == CV_16SC1;
}

float DepthFilter::readDepthAt(const cv::Mat& depth, const int y, const int x) {
    if (depth.type() == CV_32FC1) {
        return depth.at<float>(y, x);
    }
    if (depth.type() == CV_16UC1) {
        return static_cast<float>(depth.at<uint16_t>(y, x));
    }
    return static_cast<float>(depth.at<int16_t>(y, x));
}

void DepthFilter::writeDepthZero(cv::Mat& depth, const int y, const int x) {
    if (depth.type() == CV_32FC1) {
        depth.at<float>(y, x) = 0.0f;
        return;
    }
    if (depth.type() == CV_16UC1) {
        depth.at<uint16_t>(y, x) = static_cast<uint16_t>(0);
        return;
    }
    depth.at<int16_t>(y, x) = static_cast<int16_t>(0);
}

bool DepthFilter::isDepthValid(const float value) {
    return std::isfinite(value) && value > 0.0f;
}

bool DepthFilter::applySpeckleFilter(cv::Mat& depth, const SpeckleConfig& config) {
    if (depth.empty()) {
        return false;
    }
    if (!isSupportedDepthType(depth.type())) {
        return false;
    }
    if (config.max_speckle_size <= 0 || config.max_depth_difference <= 0.0f) {
        return false;
    }
    if (config.connectivity != 4 && config.connectivity != 8) {
        return false;
    }

    const int rows = depth.rows;
    const int cols = depth.cols;

    std::vector<uint8_t> visited(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
    auto indexOf = [cols](const int y, const int x) {
        return static_cast<size_t>(y) * static_cast<size_t>(cols) + static_cast<size_t>(x);
    };

    const int neighbors_4[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    const int neighbors_8[8][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1},
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
    };

    std::queue<std::pair<int, int>> frontier;
    std::vector<std::pair<int, int>> component_pixels;

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const size_t seed_idx = indexOf(y, x);
            if (visited[seed_idx] != 0) {
                continue;
            }

            visited[seed_idx] = 1;
            const float seed_depth = readDepthAt(depth, y, x);
            if (!isDepthValid(seed_depth)) {
                continue;
            }

            component_pixels.clear();
            while (!frontier.empty()) {
                frontier.pop();
            }

            frontier.emplace(y, x);
            component_pixels.emplace_back(y, x);

            while (!frontier.empty()) {
                const auto [cur_y, cur_x] = frontier.front();
                frontier.pop();

                const float cur_depth = readDepthAt(depth, cur_y, cur_x);
                const int (*neighbors)[2] = (config.connectivity == 8) ? neighbors_8 : neighbors_4;
                const int count = (config.connectivity == 8) ? 8 : 4;

                for (int i = 0; i < count; ++i) {
                    const int ny = cur_y + neighbors[i][0];
                    const int nx = cur_x + neighbors[i][1];

                    if (ny < 0 || ny >= rows || nx < 0 || nx >= cols) {
                        continue;
                    }

                    const size_t nidx = indexOf(ny, nx);
                    if (visited[nidx] != 0) {
                        continue;
                    }

                    visited[nidx] = 1;
                    const float neighbor_depth = readDepthAt(depth, ny, nx);
                    if (!isDepthValid(neighbor_depth)) {
                        continue;
                    }

                    if (std::fabs(neighbor_depth - cur_depth) <= config.max_depth_difference) {
                        frontier.emplace(ny, nx);
                        component_pixels.emplace_back(ny, nx);
                    }
                }
            }

            if (static_cast<int>(component_pixels.size()) < config.max_speckle_size) {
                for (const auto& pixel : component_pixels) {
                    writeDepthZero(depth, pixel.first, pixel.second);
                }
            }
        }
    }

    return true;
}

bool DepthFilter::applyBrightnessFilter(cv::Mat& depth,
                                        const cv::Mat& left_image,
                                        const cv::Mat& right_image,
                                        const BrightnessConfig& config) {
    if (depth.empty() || left_image.empty() || right_image.empty()) {
        return false;
    }
    if (!isSupportedDepthType(depth.type())) {
        return false;
    }
    if (depth.size() != left_image.size() || depth.size() != right_image.size()) {
        return false;
    }
    if (config.min_brightness < 0 || config.max_brightness > 255 ||
        config.min_brightness > config.max_brightness) {
        return false;
    }

    cv::Mat left_gray;
    cv::Mat right_gray;

    if (left_image.channels() == 1) {
        left_gray = left_image;
    } else {
        cv::cvtColor(left_image, left_gray, cv::COLOR_BGR2GRAY);
    }

    if (right_image.channels() == 1) {
        right_gray = right_image;
    } else {
        cv::cvtColor(right_image, right_gray, cv::COLOR_BGR2GRAY);
    }

    if (left_gray.type() != CV_8UC1) {
        left_gray.convertTo(left_gray, CV_8UC1);
    }
    if (right_gray.type() != CV_8UC1) {
        right_gray.convertTo(right_gray, CV_8UC1);
    }

    #pragma omp parallel for
    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            const uint8_t left_value = left_gray.at<uint8_t>(y, x);
            const uint8_t right_value = right_gray.at<uint8_t>(y, x);

            const bool left_in_range =
                left_value >= config.min_brightness && left_value <= config.max_brightness;
            const bool right_in_range =
                right_value >= config.min_brightness && right_value <= config.max_brightness;

            if (!(left_in_range && right_in_range)) {
                writeDepthZero(depth, y, x);
            }
        }
    }

    return true;
}

bool DepthFilter::applySpatialFilter(cv::Mat& depth, const SpatialConfig& config) {
    if (depth.empty() || !isSupportedDepthType(depth.type())) {
        return false;
    }
    if (config.alpha < 0.0f || config.alpha > 1.0f || config.delta <= 0.0f || config.iterations <= 0) {
        return false;
    }
    if (config.hole_filling_radius < 0) {
        return false;
    }

    if (depth.type() == CV_32FC1) {
        applySpatialRecursive<float>(depth, config);
        return true;
    }
    if (depth.type() == CV_16UC1) {
        applySpatialRecursive<uint16_t>(depth, config);
        return true;
    }

    applySpatialRecursive<int16_t>(depth, config);
    return true;
}

bool DepthFilter::loadParamsFromYaml(const std::string& yaml_path, FilterParams& params) {
    if (yaml_path.empty()) {
        return false;
    }

    try {
        const YAML::Node root = YAML::LoadFile(yaml_path);
        if (!root || !root["depth_filter"]) {
            return false;
        }

        const YAML::Node filter = root["depth_filter"];

        if (filter["speckle"]) {
            const YAML::Node speckle = filter["speckle"];
            if (speckle["enabled"]) {
                params.enable_speckle = speckle["enabled"].as<bool>();
            }
            if (speckle["max_speckle_size"]) {
                params.speckle.max_speckle_size = speckle["max_speckle_size"].as<int>();
            }
            if (speckle["max_depth_difference"]) {
                params.speckle.max_depth_difference = speckle["max_depth_difference"].as<float>();
            }
            if (speckle["connectivity"]) {
                params.speckle.connectivity = speckle["connectivity"].as<int>();
            }
        }

        if (filter["brightness"]) {
            const YAML::Node brightness = filter["brightness"];
            if (brightness["enabled"]) {
                params.enable_brightness = brightness["enabled"].as<bool>();
            }
            if (brightness["min_brightness"]) {
                params.brightness.min_brightness = brightness["min_brightness"].as<int>();
            }
            if (brightness["max_brightness"]) {
                params.brightness.max_brightness = brightness["max_brightness"].as<int>();
            }
        }

        if (filter["spatial"]) {
            const YAML::Node spatial = filter["spatial"];
            if (spatial["enabled"]) {
                params.enable_spatial = spatial["enabled"].as<bool>();
            }
            if (spatial["alpha"]) {
                params.spatial.alpha = spatial["alpha"].as<float>();
            }
            if (spatial["delta"]) {
                params.spatial.delta = spatial["delta"].as<float>();
            }
            if (spatial["iterations"]) {
                params.spatial.iterations = spatial["iterations"].as<int>();
            }
            if (spatial["hole_filling_radius"]) {
                params.spatial.hole_filling_radius = spatial["hole_filling_radius"].as<int>();
            }
        }

        return true;
    }
    catch (const YAML::Exception&) {
        return false;
    }
}

}  // namespace ct_uav_stereo
