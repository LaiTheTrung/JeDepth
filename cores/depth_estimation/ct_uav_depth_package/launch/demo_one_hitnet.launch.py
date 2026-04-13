#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description() -> LaunchDescription:
	# Launch arguments for configuration
	use_compressed = LaunchConfiguration('use_compressed', default='true')
	onnx_path = LaunchConfiguration('onnx_path', default='')
	engine_path = LaunchConfiguration('engine_path', default='')
	update_rate_hz = LaunchConfiguration('update_rate_hz', default='15.0')
	stream_number = LaunchConfiguration('stream_number', default='4')
	fov_deg = LaunchConfiguration('fov_deg', default='140.0')
	vertical_crop_deg = LaunchConfiguration('vertical_crop_deg', default='20.0')
	publish_pc = LaunchConfiguration('publish_pc', default='false')
	publish_depth = LaunchConfiguration('publish_depth', default='false')

	return LaunchDescription([
		DeclareLaunchArgument(
			'use_compressed',
			default_value='true',
			description='Use /fast_stream_node/raw_image/compressed if true, else raw Image.'
		),
		DeclareLaunchArgument(
			'onnx_path',
			default_value='/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_1x240x320_model_float16_quant_opt.onnx',
			description='Path to HitNet ONNX model (optional, used if engine needs building).'
		),
		DeclareLaunchArgument(
			'engine_path',
			default_value='/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_240x320_fp16_opt_level_d2slam.engine',
			description='Path to HitNet TensorRT engine file.'
		),
        DeclareLaunchArgument(
			'update_rate_hz',
			default_value='15.0',
			description='Number of TensorRT streams for HitNet.'
		),
		DeclareLaunchArgument(
			'stream_number',
			default_value='2',
			description='Number of TensorRT streams for HitNet.'
		),
		DeclareLaunchArgument(
			'fov_deg',
			default_value='140.0',
			description='Per-camera horizontal field of view in degrees.'
		),
		DeclareLaunchArgument(
			'vertical_crop_deg',
			default_value='20.0',
			description='Vertical crop for collision avoidance (±deg).'
		),
		DeclareLaunchArgument(
			'publish_pc',
			default_value='false',
			description='Publish merged depth point cloud if true.'
		),
		DeclareLaunchArgument(
			'publish_depth',
			default_value='false',
			description='Publish depth debug image if true.'
		),

		Node(
			package='ct_uav_depth_package',
			executable='demo_one',
			name='stereo_hitnet_px4_cp',
			output='screen',
			parameters=[{
				'use_compressed': use_compressed,
				'onnx_path': onnx_path,
				'update_rate_hz': update_rate_hz,
				'engine_path': engine_path,
				'stream_number': stream_number,
				'fov_deg': fov_deg,
				'vertical_crop_deg': vertical_crop_deg,
				'publish_pc': publish_pc,
				'publish_depth': publish_depth,
				'debug_downsample_factor': 2,
				'pc_stride_u': 4,
                'pc_stride_v': 4,
			}],
		),
	])

