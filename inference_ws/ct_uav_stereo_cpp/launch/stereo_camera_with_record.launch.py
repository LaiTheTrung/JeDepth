from datetime import datetime
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.substitutions import LaunchConfiguration as LaunchConfig

def generate_launch_description():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_dir = os.path.expanduser('~/stereo_recordings')
    os.makedirs(bag_dir, exist_ok=True)
    bag_prefix = '/workspace/ros_record_log/ros2bag_'
    default_record_path = os.path.join(bag_dir, f'stereo_{timestamp}.mp4')

    record_path_arg = DeclareLaunchArgument(
        'record_path',
        default_value="",
        description='Path to save stereo video (empty disables recording)'
    )
    conversion_type_arg = DeclareLaunchArgument(
        'conversion_type',
        default_value='2',
        description='Conversion type (0:GRAY, 1:RGB, 2:BGR)'
    )
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='20.0',
        description='Publishing rate (Hz)'
    )
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug visualization'
    )

    # Single container holding stereo camera + rosbag recorder as composable nodes
    stereo_container = ComposableNodeContainer(
        name='stereo_camera_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        output='screen',
        composable_node_descriptions=[
            # ── Stereo camera (libargus HW-sync) ──────────────────────────────
            ComposableNode(
                package='ct_uav_stereo_cpp',
                plugin='CtUAVStereoCpp::StereoCameraNode',
                name='stereo_camera_node',
                parameters=[{
                    'left_sensor_id':  0,
                    'right_sensor_id': 1,
                    'publish_rate':    LaunchConfiguration('publish_rate'),
                    'left_frame_id':   'left_camera',
                    'right_frame_id':  'right_camera',
                    'record_path':     LaunchConfiguration('record_path'),
                    'using_gpu':       True,
                    'conversion_type': LaunchConfiguration('conversion_type'),
                    'auto_expose':     True,
                    'assembled_mode':  True,
                    'publish_concat':  True,
                    'debug':           True,
                    'auto_expose':     True,
                }],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            # ── ROS 2 bag recorder ────────────────────────────────────────────
            ComposableNode(
                package='rosbag2_composable_recorder',
                plugin='rosbag2_composable_recorder::ComposableRecorder',
                name='recorder',
                parameters=[{
                    'topics': [
                        '/stereo/concat/image_raw',
                        '/fmu/out/sensor_combined',
                        '/fmu/out/vehicle_attitude',
                        '/fmu/out/vehicle_gps_position',
                        '/fmu/out/vehicle_odometry',
                    ],
                    'storage_id': 'sqlite3',
                    'record_all': False,
                    'disable_discovery': False,
                    'bag_prefix': bag_prefix,
                    'start_recording_immediately': False,
                }],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            # ComposableNode(
            #     package='rosbag2_composable_recorder',
            #     plugin='rosbag2_composable_recorder::ComposableRecorder',
            #     name='recorder_imu',
            #     parameters=[{
            #         'topics': [
            #             '/fmu/out/sensor_combined',
            #             '/fmu/out/vehicle_attitude',
            #             '/fmu/out/vehicle_gps_position',
            #             '/fmu/out/vehicle_odometry',
            #         ],
            #         'storage_id': 'sqlite3',
            #         'record_all': False,
            #         'disable_discovery': False,
            #         'bag_prefix': bag_prefix,
            #         'start_recording_immediately': False,
            #     }],
            #     extra_arguments=[{'use_intra_process_comms': True}],
            # ),
        ],
    )

    return LaunchDescription([
        record_path_arg,
        conversion_type_arg,
        publish_rate_arg,
        debug_arg,
        stereo_container,
    ])
