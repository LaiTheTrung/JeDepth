from datetime import datetime
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
def generate_launch_description():
    # Launch arguments
        # Generate default video filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_record_path = f"~/stereo_recordings/stereo_{timestamp}.mp4"
    
    record_path_arg = DeclareLaunchArgument(
        'record_path',
        default_value="",  # Default path with timestamp
        description='Path to save stereo video (empty disables recording)'
    )

    conversion_type_arg = DeclareLaunchArgument(
        'conversion_type',
        default_value=str(2),
        description='Conversion type (0:GRAY, 1:RGB, 2:BGR)'
    )
    
    # Stereo camera node with hardware sync
    stereo_camera_node = Node(
        package='ct_uav_stereo_cpp',
        executable='stereo_camera_node',
        name='stereo_camera_node',
        output='screen',
        parameters=[{
            'left_sensor_id': 0,
            'right_sensor_id': 1,
            'publish_rate': 10.0,
            'left_frame_id': 'left_camera',
            'right_frame_id': 'right_camera',
            'record_path': LaunchConfiguration('record_path'),
            'using_gpu': True,
            'conversion_type': LaunchConfiguration('conversion_type'),
            'auto_expose': True,
            'debug': True,
            'publish_concat': True,
        }]
    )
    
    return LaunchDescription([
        record_path_arg,
        conversion_type_arg,
        stereo_camera_node,
    ])
