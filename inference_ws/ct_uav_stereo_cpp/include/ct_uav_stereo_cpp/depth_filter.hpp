#ifndef DEPTH_FILTER_HPP_
#define DEPTH_FILTER_HPP_

#include <opencv2/opencv.hpp>
#include <string>

namespace ct_uav_stereo {

class DepthFilter {
public:
	struct SpeckleConfig {
		int max_speckle_size;
		float max_depth_difference;
		int connectivity;

		SpeckleConfig()
			: max_speckle_size(100),
			  max_depth_difference(0.08f),
			  connectivity(8) {}
	};

	struct BrightnessConfig {
		int min_brightness;
		int max_brightness;

		BrightnessConfig()
			: min_brightness(20),
			  max_brightness(235) {}
	};

	struct SpatialConfig {
		float alpha;
		float delta;
		int iterations;
		int hole_filling_radius;

		SpatialConfig()
			: alpha(0.5f),
			  delta(0.08f),
			  iterations(2),
			  hole_filling_radius(0) {}
	};

	struct FilterParams {
		bool enable_speckle;
		bool enable_brightness;
		bool enable_spatial;
		SpeckleConfig speckle;
		BrightnessConfig brightness;
		SpatialConfig spatial;

		FilterParams()
			: enable_speckle(true),
			  enable_brightness(true),
			  enable_spatial(true) {}
	};

	static bool applySpeckleFilter(cv::Mat& depth, const SpeckleConfig& config);

	static bool applyBrightnessFilter(cv::Mat& depth,
									  const cv::Mat& left_image,
									  const cv::Mat& right_image,
									  const BrightnessConfig& config);

	static bool applySpatialFilter(cv::Mat& depth, const SpatialConfig& config);

	static bool loadParamsFromYaml(const std::string& yaml_path, FilterParams& params);

private:
	static bool isSupportedDepthType(int type);
	static float readDepthAt(const cv::Mat& depth, int y, int x);
	static void writeDepthZero(cv::Mat& depth, int y, int x);
	static bool isDepthValid(float value);
};

}  // namespace ct_uav_stereo

#endif  // DEPTH_FILTER_HPP_
