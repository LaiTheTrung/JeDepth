#!/usr/bin/env python3
"""
Stereo Obstacle Avoidance Launch File

This launch file starts:
1. stereo_depth_node - Generates depth maps from stereo cameras
2. obstacle_prevention_node - Converts depth to PX4 obstacle distance messages

Author: CT-UAV Team
Date: 2026-01-26
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

# Select depth estimation method
DEPTH_TYPE = 3.3  # 0: SGM only, 1.x: HitNet dajvariants, 2.x: FastACV variants, 3.x: FastFS (Fast Foundation Stereo)
path = {
    0: {
        'name': 'sgm_only',
        'onnx': '',
        'engine': '',
        'model_w': 640,
        'model_h': 480,
        'fp16': False,
    },
	1: {
		'name': "hitnet",
		'onnx': '/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_1x240x320_model_float16_quant_opt.onnx',
		'engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_240x320_fp16_opt_level_d2slam.engine',
		'model_w': 320,
		'model_h': 240,
        'fp16': True,
	},
	1.1: {
		'name': "hitnet",
		'onnx': '/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_1x240x320_model_float32.onnx',
		'engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_1x240x320_model_float16.engine',
		'model_w': 320,
		'model_h': 240,
        'fp16': True,
	},
	1.2: {
		'name': "hitnet",
		'onnx': '/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_1x240x320_model_float32.onnx',
		'engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/hitnet/models/hitnet_1x240x320_model_float32.engine',
		'model_w': 320,
		'model_h': 240,
        'fp16': False,
	},
	2: {
		'name': 'fastacv',
		'onnx': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acvnet_plus_generalization_opset16_288x480.onnx',
		'engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acv_256x320_f32.engine',	
		'model_w': 320,
		'model_h': 256,
        'fp16': False,
	},
	2.1: {
		'name': 'fastacv',
		'onnx': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acvnet_plus_generalization_opset16_288x480.onnx',
		'engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acv_NGS_288x480_f16.engine',	
		'model_w': 480,
		'model_h': 288,
        'fp16': True,
	},
	2.2: {
		'name': 'fastacv',
		'onnx': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acvnet_plus_generalization_opset16_288x480.onnx',
		'engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acv_NGS_288x480_f32.engine',	
		'model_w': 480,
		'model_h': 288,
        'fp16': False,
	},
	2.3: {
		'name': 'fastacv',
		'onnx': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acvnet_plus_generalization_opset16_288x480.onnx',
		'engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_acv_plus/fast_acv_plus_288x480_f32.engine',	
		'model_w': 480,
		'model_h': 288,
        'fp16': False,
	},
	# === Fast Foundation Stereo (two-stage TRT: feature + post) ===
	3: { #12.5 fps
		'name': 'fastfs',
		'onnx': '',
		'engine': '',  # Not used for fastfs (uses feature/post engine paths)
		'feature_engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_fs/output_320_448_96_8/feature_runner.engine',
		'post_engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_fs/output_320_448_96_8/post_runner.engine',
		'model_w': 448,
		'model_h': 320,
		'max_disp': 96,
		'cv_group': 8,
		'normalize_gwc': False,
		'fp16': True,
	},
	3.1: { #10fps
		'name': 'fastfs',
		'onnx': '',
		'engine': '',
		'feature_engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_fs/output_320_448_192_8/feature_runner.engine',
		'post_engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_fs/output_320_448_192_8/post_runner.engine',
		'model_w': 448,
		'model_h': 320,
		'max_disp': 192,
		'cv_group': 8,
		'normalize_gwc': False,
		'fp16': True,
	},
	3.2: { # 5fps
		'name': 'fastfs', 
		'onnx': '',
		'engine': '',
		'feature_engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_fs/output_448_640_192_8/feature_runner.engine',
		'post_engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_fs/output_448_640_192_8/post_runner.engine',
		'model_w': 640,
		'model_h': 448,
		'max_disp': 192,
		'cv_group': 8,
		'normalize_gwc': False,
		'fp16': True,
	},
    3.3: { #21 fps
		'name': 'fastfs', 
		'onnx': '',
		'engine': '',
		'feature_engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_fs/output_224_320_96_8/feature_runner.engine',
		'post_engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_fs/output_224_320_96_8/post_runner.engine',
		'model_w': 320,
		'model_h': 224,
		'max_disp': 96,
		'cv_group': 8,
		'normalize_gwc': False,
		'fp16': True,
	},
    3.4: { #17.8
		'name': 'fastfs', 
		'onnx': '',
		'engine': '',
		'feature_engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_fs/output_224_320_192_8/feature_runner.engine',
		'post_engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_fs/output_224_320_192_8/post_runner.engine',
		'model_w': 320,
		'model_h': 224,
		'max_disp': 192,
		'cv_group': 8,
		'normalize_gwc': False,
		'fp16': True,
	},
    3.5: { # 5fps
		'name': 'fastfs', 
		'onnx': '',
		'engine': '',
		'feature_engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_fs/output_448_640_96_4/feature_runner.engine',
		'post_engine': '/workspace/ros_packages/src/ct_uav_depth_package/models/fast_fs/output_448_640_96_4/post_runner.engine',
		'model_w': 640,
		'model_h': 448,
		'max_disp': 96,
		'cv_group': 4,
		'normalize_gwc': False,
		'fp16': True,
	},
}

def generate_launch_description():
    pkg_dir = get_package_share_directory('ct_uav_stereo_cpp')
    
    # Launch arguments
    calibration_file_arg = DeclareLaunchArgument(
        'calibration_file',
        default_value='/workspace/ros_packages/src/ct_uav_stereo_cpp/config/stereo_calib_example.yaml',
        description='Path to stereo calibration YAML file'
    )

    conversion_type_arg = DeclareLaunchArgument(
        'conversion_type',
        default_value=str(2),
        description='Conversion type (0:GRAY, 1:RGB, 2:BGR)'
    )

    model_type_arg = DeclareLaunchArgument(
        'model_type',
        default_value=path[DEPTH_TYPE]['name'],
        description='Model type: sgm_only, fastacv, hitnet, or fastfs'
    )
    onnx_path_arg = DeclareLaunchArgument(
        'onnx_path', default_value=path[DEPTH_TYPE]['onnx'],
        description='Path to ONNX model file (for first-time TensorRT engine build)'
    )
    engine_path_arg = DeclareLaunchArgument(
        'engine_path',
        default_value=path[DEPTH_TYPE]['engine'],
        description='Path to TensorRT engine file'
    )
    
    model_input_width_arg = DeclareLaunchArgument(
        'model_input_width',
        default_value=str(path[DEPTH_TYPE]['model_w']),
        description='Input width for the depth estimation model'
    )
    model_input_height_arg = DeclareLaunchArgument(
        'model_input_height',
        default_value=str(path[DEPTH_TYPE]['model_h']),
        description='Input height for the depth estimation model'
    )

    max_disparity_arg = DeclareLaunchArgument(
        'max_disparity',
        default_value=str(path[DEPTH_TYPE].get('max_disp', 192)),
        description='Maximum disparity for stereo matching'
    )

    # FastFS-specific arguments
    fastfs_feature_engine_arg = DeclareLaunchArgument(
        'fastfs_feature_engine_path',
        default_value=path[DEPTH_TYPE].get('feature_engine', ''),
        description='Path to FastFS feature extraction TensorRT engine'
    )
    fastfs_post_engine_arg = DeclareLaunchArgument(
        'fastfs_post_engine_path',
        default_value=path[DEPTH_TYPE].get('post_engine', ''),
        description='Path to FastFS post-processing TensorRT engine'
    )
    fastfs_cv_group_arg = DeclareLaunchArgument(
        'fastfs_cv_group',
        default_value=str(path[DEPTH_TYPE].get('cv_group', 8)),
        description='Number of groups for GWC volume in FastFS'
    )
    
    fastfs_normalize_gwc_arg = DeclareLaunchArgument(
        'fastfs_normalize_gwc',
        default_value=str(path[DEPTH_TYPE].get('normalize_gwc', False)).lower(),
        description='Whether to normalize GWC volume (cosine similarity) in FastFS'
    )


    obstacle_enable_arg = DeclareLaunchArgument(
        'obstacle_prevention_enable',
        default_value='true',
        description='Enable obstacle prevention publishing'
    )
    
    debug_viz_arg = DeclareLaunchArgument(
        'debug_visualize',
        default_value='true',
        description='Enable debug visualization windows'
    )
    
    # Stereo Depth Node
    stereo_depth_node = Node(
        package='ct_uav_stereo_cpp',
        executable='stereo_depth_node',
        name='stereo_depth_node',
        output='screen',
        parameters=[{
            # Camera settings
            'use_camera': True,
            'auto_expose':True,
            'left_sensor_id': 1,
            'right_sensor_id': 0,
            'camera_width': 1640,
            'camera_height': 1232,
            'camera_framerate': 30,
            'flip_method': 0,
            'conversion_type': LaunchConfiguration('conversion_type'),
            
            # Calibration
            'calibration_file': LaunchConfiguration('calibration_file'),
            
            # SGM settings
            'sgm_width': 320,
            'sgm_height': 240,
            'sgm_disparity_size': 64,
            'sgm_P1': 10,
            'sgm_P2': 60,
            'sgm_subpixel': False,
            'sgm_use_8path': False,
            
            # Model settings
            'model_type': LaunchConfiguration('model_type'),
            'onnx_path': LaunchConfiguration('onnx_path'),
            'engine_path': LaunchConfiguration('engine_path'),
            'model_input_width': LaunchConfiguration('model_input_width'),
            'model_input_height': LaunchConfiguration('model_input_height'),
            'max_disparity': LaunchConfiguration('max_disparity'),
            'use_fp16': True,
            
            # FastFS (Fast Foundation Stereo) specific settings
            'fastfs_feature_engine_path': LaunchConfiguration('fastfs_feature_engine_path'),
            'fastfs_post_engine_path': LaunchConfiguration('fastfs_post_engine_path'),
            'fastfs_cv_group': LaunchConfiguration('fastfs_cv_group'),
            'fastfs_normalize_gwc': LaunchConfiguration('fastfs_normalize_gwc'),
            
            # Publishing settings
            'publish_rate': 20.0,
            'publish_raw_images': False,
            'publish_depth': True,
            'publish_depth_viz': False,
            'min_depth_m': 0.1,
            'max_depth_m': 10.0,

            # Stereo parameters
            'focal_length_default': 185.5, # Default focal length in pixels (will be overridden by calibration)
            'baseline': 0.06,
            
            # Debug
            'debug_visualize': LaunchConfiguration('debug_visualize'),
        }],
    )
    
    # Obstacle Prevention Node
    obstacle_prevention_node = Node(
        package='ct_uav_stereo_cpp',
        executable='obstacle_prevention_node',
        name='obstacle_prevention_node',
        output='screen',
        parameters=[{
            # Enable/disable
            'debug': True,
            'obstacle_prevention_enable': LaunchConfiguration('obstacle_prevention_enable'),
            
            # Camera FOV and range
            'fov_deg': 90.0,
            'min_depth_m': 0.05,
            'max_depth_m': 10.0,
            
            # Processing settings
            'vertical_crop_deg': 10.0,
            'sector_increment_deg': 5.0,
            'num_sectors': 72,
            'crop_left_angle': 10,
            'crop_right_angle': 0,
            
            # Topics
            'obstacle_topic': '/fmu/in/obstacle_distance',
            'depth_topic': '/stereo/depth_map',
            
            # Stereo parameters (should match stereo_depth_node)
            'focal_length': 187.5,
        }],
    )
    
    return LaunchDescription([
        # Arguments
        calibration_file_arg,
        conversion_type_arg,
        model_type_arg,
        engine_path_arg,
        obstacle_enable_arg,
        debug_viz_arg,
        model_input_width_arg,
        model_input_height_arg,
        max_disparity_arg,
        onnx_path_arg,
        fastfs_feature_engine_arg,
        fastfs_post_engine_arg,
        fastfs_cv_group_arg,
        fastfs_normalize_gwc_arg,
        
        # Nodes
        stereo_depth_node,
        obstacle_prevention_node,
    ])
