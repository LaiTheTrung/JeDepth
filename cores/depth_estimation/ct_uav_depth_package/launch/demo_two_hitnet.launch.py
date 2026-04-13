#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

onnx_hitnet_path = '/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_1x240x320_model_float16_quant_opt.onnx'
engine_hitnet_path = '/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_240x320_fp16_opt_level_d2slam.engine'

onnx_fastACV_path = '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acvnet_kitti_2012_opset16_256x320.onnx'
engine_fastACV_path = '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acv_256x320_f32.engine'


def generate_launch_description() -> LaunchDescription:
	# Launch arguments for configuration
	stereo_method = LaunchConfiguration('stereo_method', default='sgm')
	onnx_path = LaunchConfiguration('onnx_path', default='')
	engine_path = LaunchConfiguration('engine_path', default='')
	model_width = LaunchConfiguration('model_width', default='320')
	model_height = LaunchConfiguration('model_height', default='256')
	stream_number = LaunchConfiguration('stream_number', default='4')
	fov_deg = LaunchConfiguration('fov_deg', default='140.0')
	vertical_crop_deg = LaunchConfiguration('vertical_crop_deg', default='20.0')
	publish_pc = LaunchConfiguration('publish_pc', default='false')
	publish_depth = LaunchConfiguration('publish_depth', default='false')
	sgm_num_disparities = LaunchConfiguration('sgm_num_disparities', default='64')
	sgm_block_size = LaunchConfiguration('sgm_block_size', default='5')
	sgm_uniqueness_ratio = LaunchConfiguration('sgm_uniqueness_ratio', default='12')

	return LaunchDescription([
		DeclareLaunchArgument(
			'calibration_file',
			default_value='/workspace/ros_packages/src/ct_uav_depth_package/config/stereo_config.yaml',
			description='Path to stereo camera calibration file (YAML format).'
		),
		DeclareLaunchArgument(
			'stereo_method',
			default_value='sgm',
			description='Stereo matching method: "sgm" or "hitnet" or "fastACV"'
		),
		DeclareLaunchArgument(
			'onnx_path',
			default_value= onnx_fastACV_path,
			description='Path to ONNX model (only used if stereo_method=fastACV).'
		),
		DeclareLaunchArgument(
			'engine_path',
			default_value= engine_fastACV_path,
			description='Path to TensorRT engine file (only used if stereo_method=fastACV).'
		),
		DeclareLaunchArgument(
			'stream_number',
			default_value='1',
			description='Number of TensorRT streams for HitNet (only used if stereo_method=hitnet).'
		),
		DeclareLaunchArgument(
			'fov_deg',
			default_value='83.0',
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
			default_value='true',
			description='Publish depth debug image if true.'
		),
		DeclareLaunchArgument(
			'sgm_num_disparities',
			default_value='64',
			description='Number of disparities for SGM (multiple of 16).'
		),
		DeclareLaunchArgument(
			'sgm_block_size',
			default_value='7',
			description='Block size for SGM (odd number >=3).'
		),
		DeclareLaunchArgument(
			'sgm_uniqueness_ratio',
			default_value='12',
			description='Uniqueness ratio for SGM.'
		),

		Node(
			package='ct_uav_depth_package',
			executable='demo_two',
			name='stereo_hitnet_px4_cp',
			output='screen',
			parameters=[{
				'stereo_method': stereo_method,
				'width': 1640,
				'height': 1232,
				'framerate': 30,
				'onnx_path': onnx_path,
				'engine_path': engine_path,
				'model_width': model_width,
				'model_height': model_height,
				'stream_number': stream_number,
				'fov_deg': fov_deg,
				'vertical_crop_deg': vertical_crop_deg,
				'publish_pc': publish_pc,
				'publish_depth': publish_depth,
				'sgm_num_disparities': sgm_num_disparities,
				'sgm_block_size': sgm_block_size,
				'sgm_uniqueness_ratio': sgm_uniqueness_ratio,
			}],
		),
	])

