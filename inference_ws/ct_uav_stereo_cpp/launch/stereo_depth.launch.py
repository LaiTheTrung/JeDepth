#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

# Select depth estimation method
DEPTH_TYPE = 0 # 0: SGM only, 1.x: HitNet variants, 2.x: FastACV variants
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
}

def generate_launch_description():
    # Get selected model configuration
    selected_model = path[DEPTH_TYPE]
    
    # Declare launch arguments
    
    # === Debug and Visualization ===
    debug_visualize_arg = DeclareLaunchArgument(
        'debug_visualize', default_value='false',
        description='Enable debug visualization window (2x2 grid: left, right, sgm_disp, model_disp)'
    )
    
    # === Camera Parameters ===
    left_sensor_id_arg = DeclareLaunchArgument(
        'left_sensor_id', default_value='1',
        description='Left camera sensor ID'
    )
    
    right_sensor_id_arg = DeclareLaunchArgument(
        'right_sensor_id', default_value='0',
        description='Right camera sensor ID'
    )
    
    camera_width_arg = DeclareLaunchArgument(
        'camera_width', default_value='1640',
        description='Camera width'
    )
    
    camera_height_arg = DeclareLaunchArgument(
        'camera_height', default_value='1232',
        description='Camera height'
    )
    
    framerate_arg = DeclareLaunchArgument(
        'camera_framerate', default_value='30',
        description='Camera framerate'
    )
    
    flip_method_arg = DeclareLaunchArgument(
        'flip_method', default_value='0',
        description='Flip method: 0=none, 2=rotate-180'
    )
    
    # === Calibration ===
    calibration_file_arg = DeclareLaunchArgument(
        'calibration_file', default_value='/workspace/ros_packages/src/ct_uav_stereo_cpp/config/stereo_calib_example.yaml',
        description='Path to stereo calibration YAML file (optional)'
    )
    
    # === SGM Parameters ===
    sgm_width_arg = DeclareLaunchArgument(
        'sgm_width', default_value='480',
        description='SGM processing width'
    )
    
    sgm_height_arg = DeclareLaunchArgument(
        'sgm_height', default_value='288',
        description='SGM processing height'
    )
    
    sgm_disparity_size_arg = DeclareLaunchArgument(
        'sgm_disparity_size', default_value='64',
        description='SGM disparity size (64, 128, or 256)'
    )
    
    sgm_P1_arg = DeclareLaunchArgument(
        'sgm_P1', default_value='10',
        description='SGM P1 penalty parameter'
    )
    
    sgm_P2_arg = DeclareLaunchArgument(
        'sgm_P2', default_value='120',
        description='SGM P2 penalty parameter'
    )
    
    sgm_subpixel_arg = DeclareLaunchArgument(
        'sgm_subpixel', default_value='true',
        description='Enable SGM subpixel estimation'
    )
    
    sgm_use_8path_arg = DeclareLaunchArgument(
        'sgm_use_8path', default_value='true',
        description='Use 8-path (true) or 4-path (false) for SGM'
    )
    
    # === Model Parameters ===
    model_type_arg = DeclareLaunchArgument(
        'model_type', default_value=selected_model['name'],
        description='Model type: sgm_only, fastacv, or hitnet'
    )
    
    onnx_path_arg = DeclareLaunchArgument(
        'onnx_path', default_value=selected_model['onnx'],
        description='Path to ONNX model file (for first-time TensorRT engine build)'
    )
    
    engine_path_arg = DeclareLaunchArgument(
        'engine_path', default_value=selected_model['engine'],
        description='Path to TensorRT engine file'
    )
    
    model_input_width_arg = DeclareLaunchArgument(
        'model_input_width', default_value=str(selected_model['model_w']),
        description='Model input width'
    )
    
    model_input_height_arg = DeclareLaunchArgument(
        'model_input_height', default_value=str(selected_model['model_h']),
        description='Model input height'
    )
    
    max_disparity_arg = DeclareLaunchArgument(
        'max_disparity', default_value='128',
        description='Maximum disparity for model'
    )
    
    use_fp16_arg = DeclareLaunchArgument(
        'use_fp16', default_value='true' if selected_model.get('fp16', False) else 'false',
        description='Use FP16 precision for model inference (affects engine building)'
    )
    
    # === Calibration Parameters (fallback if no calibration file) ===
    focal_length_arg = DeclareLaunchArgument(
        'focal_length', default_value='100.0',
        description='Focal length in pixels (overridden by calibration file if provided)'
    )
    
    baseline_arg = DeclareLaunchArgument(
        'baseline', default_value='0.06',
        description='Baseline distance in meters (overridden by calibration file if provided)'
    )
    
    # === Publishing Parameters ===
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate', default_value='15.0',
        description='Publishing rate in Hz'
    )
    
    publish_raw_images_arg = DeclareLaunchArgument(
        'publish_raw_images', default_value='false',
        description='Publish raw left/right images'
    )
    
    publish_depth_arg = DeclareLaunchArgument(
        'publish_depth', default_value='true',
        description='Publish depth map (32FC1)'
    )
    
    publish_depth_viz_arg = DeclareLaunchArgument(
        'publish_depth_viz', default_value='false',
        description='Publish depth visualization (colorized)'
    )
    
    # Stereo depth node
    stereo_depth_node = Node(
        package='ct_uav_stereo_cpp',
        executable='stereo_depth_node',
        name='stereo_depth_node',
        output='screen',
        parameters=[{
            # Debug and visualization
            'debug_visualize': LaunchConfiguration('debug_visualize'),
            
            # Camera parameters
            'left_sensor_id': LaunchConfiguration('left_sensor_id'),
            'right_sensor_id': LaunchConfiguration('right_sensor_id'),
            'camera_width': LaunchConfiguration('camera_width'),
            'camera_height': LaunchConfiguration('camera_height'),
            'camera_framerate': LaunchConfiguration('camera_framerate'),
            'flip_method': LaunchConfiguration('flip_method'),
            
            # Calibration file
            'calibration_file': LaunchConfiguration('calibration_file'),
            
            # SGM parameters
            'sgm_width': LaunchConfiguration('sgm_width'),
            'sgm_height': LaunchConfiguration('sgm_height'),
            'sgm_disparity_size': LaunchConfiguration('sgm_disparity_size'),
            'sgm_P1': LaunchConfiguration('sgm_P1'),
            'sgm_P2': LaunchConfiguration('sgm_P2'),
            'sgm_subpixel': LaunchConfiguration('sgm_subpixel'),
            'sgm_use_8path': LaunchConfiguration('sgm_use_8path'),
            
            # Model selection and parameters
            'model_type': LaunchConfiguration('model_type'),
            'onnx_path': LaunchConfiguration('onnx_path'),
            'engine_path': LaunchConfiguration('engine_path'),
            'model_input_width': LaunchConfiguration('model_input_width'),
            'model_input_height': LaunchConfiguration('model_input_height'),
            'max_disparity': LaunchConfiguration('max_disparity'),
            'use_fp16': LaunchConfiguration('use_fp16'),
            
            # Calibration parameters (fallback)
            'focal_length': LaunchConfiguration('focal_length'),
            'baseline': LaunchConfiguration('baseline'),
            
            # Publishing parameters
            'publish_rate': LaunchConfiguration('publish_rate'),
            'publish_raw_images': LaunchConfiguration('publish_raw_images'),
            'publish_depth': LaunchConfiguration('publish_depth'),
            'publish_depth_viz': LaunchConfiguration('publish_depth_viz'),
        }]
    )
    
    return LaunchDescription([
        # Debug
        debug_visualize_arg,
        
        # Camera
        left_sensor_id_arg,
        right_sensor_id_arg,
        camera_width_arg,
        camera_height_arg,
        framerate_arg,
        flip_method_arg,
        
        # Calibration
        calibration_file_arg,
        
        # SGM
        sgm_width_arg,
        sgm_height_arg,
        sgm_disparity_size_arg,
        sgm_P1_arg,
        sgm_P2_arg,
        sgm_subpixel_arg,
        sgm_use_8path_arg,
        
        # Model
        model_type_arg,
        onnx_path_arg,
        engine_path_arg,
        model_input_width_arg,
        model_input_height_arg,
        max_disparity_arg,
        use_fp16_arg,
        
        # Calibration fallback
        focal_length_arg,
        baseline_arg,
        
        # Publishing
        publish_rate_arg,
        publish_raw_images_arg,
        publish_depth_arg,
        publish_depth_viz_arg,
        
        # Node
        stereo_depth_node,
    ])
