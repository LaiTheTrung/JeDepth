#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

DEPTH_TYPE = 1  # 0: SGM, 1: HitNet, 2: FastACV
path = {
	0: {
		'name': 'sgm',
		'onnx': '',
		'engine': '',
		'model_w': 480,
		'model_h': 288,
	},
	1: {
		'name': "hitnet",
		'onnx': '/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_1x240x320_model_float16_quant_opt.onnx',
		'engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_240x320_fp16_opt_level_d2slam.engine',
		'model_w': 320,
		'model_h': 240,
	},
	1.1: {
		'name': "hitnet",
		'onnx': '/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_1x240x320_model_float32.onnx',
		'engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_1x240x320_model_float16.engine',
		'model_w': 320,
		'model_h': 240,
	},
	1.2: {
		'name': "hitnet",
		'onnx': '/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_1x240x320_model_float32.onnx',
		'engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_1x240x320_model_float32.engine',
		'model_w': 320,
		'model_h': 240,
	},
	2: {
		'name': 'fastacv',
		'onnx': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acvnet_plus_generalization_opset16_288x480.onnx',
		'engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acv_256x320_f32.engine',	
		'model_w': 320,
		'model_h': 256,
	},
	2.1: {
		'name': 'fastacv',
		'onnx': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acvnet_plus_generalization_opset16_288x480.onnx',
		'engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acv_NGS_288x480_f16.engine',	
		'model_w': 480,
		'model_h': 288,
	},
	2.2: {
		'name': 'fastacv',
		'onnx': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acvnet_plus_generalization_opset16_288x480.onnx',
		'engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acv_NGS_288x480_f32.engine',	
		'model_w': 480,
		'model_h': 288,
	},
	2.3: {
		'name': 'fastacv',
		'onnx': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acvnet_plus_generalization_opset16_288x480.onnx',
		'engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acv_plus_288x480_f32.engine',	
		'model_w': 480,
		'model_h': 288,
	
	},
}

def generate_launch_description() -> LaunchDescription:
	# Launch arguments for configuration
	stereo_method = LaunchConfiguration('stereo_method', default='sgm')
	onnx_path = LaunchConfiguration('onnx_path', default='')
	engine_path = LaunchConfiguration('engine_path', default='')
	stream_number = LaunchConfiguration('stream_number', default='1')
	fov_deg = LaunchConfiguration('fov_deg', default='140.0')
	vertical_crop_deg = LaunchConfiguration('vertical_crop_deg', default='20.0')
	publish_pc = LaunchConfiguration('publish_pc', default='false')
	publish_depth = LaunchConfiguration('publish_depth', default='false')
	sgm_num_disparities = LaunchConfiguration('sgm_num_disparities', default='64')
	sgm_block_size = LaunchConfiguration('sgm_block_size', default='5')
	sgm_uniqueness_ratio = LaunchConfiguration('sgm_uniqueness_ratio', default='10')
	image_topic = LaunchConfiguration('image_topic', default='/fast_stream_node_stereo/fps')
	topic_fps = LaunchConfiguration('topic_fps', default='30')
	model_w = LaunchConfiguration('model_w', default='320')
	model_h = LaunchConfiguration('model_h', default='240')

	return LaunchDescription([
		DeclareLaunchArgument(
			'stereo_method',
			default_value= path[DEPTH_TYPE]['name'],
			description='Stereo matching method: "sgm" or "hitnet" or "fastacv'
		),
		DeclareLaunchArgument(
			'onnx_path',
			default_value= path[DEPTH_TYPE]['onnx'],
			description='Path to HitNet ONNX model (only used if stereo_method=hitnet).'
		),
		DeclareLaunchArgument(
			'engine_path',
			default_value= path[DEPTH_TYPE]['engine'],
			description='Path to HitNet TensorRT engine file (only used if stereo_method=hitnet).'
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
			default_value='128',
			description='Number of disparities for SGM (multiple of 16).'
		),
		DeclareLaunchArgument(
			'sgm_block_size',
			default_value='5',
			description='Block size for SGM (odd number >=3s).'
		),
		DeclareLaunchArgument(
			'sgm_uniqueness_ratio',
			default_value='15',
			description='Uniqueness ratio for SGM.'
		),
		DeclareLaunchArgument(
			'image_topic',
			default_value='/fast_stream_node_stereo/raw_image',
			description='ROS2 Image topic for stereo camera stream.'
		),
		DeclareLaunchArgument(
			'topic_fps',
			default_value='30',
			description='FPS of the image topic (can be set).'
		),
		DeclareLaunchArgument(
			'model_w',
			default_value=str(path[DEPTH_TYPE]['model_w']),
			description='Model input width (e.g., 320 for HitNet, 480 for FastACV).'
		),
		DeclareLaunchArgument(
			'model_h',
			default_value=str(path[DEPTH_TYPE]['model_h']),
			description='Model input height (e.g., 240 for HitNet, 288 for FastACV).'
		),

		Node(
			package='ct_uav_depth_package',
			executable='demo_three',
			name='stereo_hitnet_px4_cp',
			output='screen',
			parameters=[{
				'stereo_method': stereo_method,
				'onnx_path': onnx_path,
				'engine_path': engine_path,
				'stream_number': stream_number,
				'fov_deg': fov_deg,
				'vertical_crop_deg': vertical_crop_deg,
				'publish_pc': publish_pc,
				'publish_depth': publish_depth,
				'sgm_num_disparities': sgm_num_disparities,
				'sgm_block_size': sgm_block_size,
				'sgm_uniqueness_ratio': sgm_uniqueness_ratio,
				'image_topic': image_topic,
				'topic_fps': topic_fps,
				'model_w': model_w,
				'model_h': model_h,
			}],
		),
	])

