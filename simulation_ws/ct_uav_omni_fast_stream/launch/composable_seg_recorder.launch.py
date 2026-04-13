from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def get_topic_list(record_mode, compressed):
    if record_mode == 0:  # RGB + Segmentation
        topics = ["/segmentation_record_node/rgb_cam0",
                  "/segmentation_record_node/rgb_cam1",
                  "/segmentation_record_node/segmentation_cam0",
                  "/segmentation_record_node/segmentation_cam1",]
    elif record_mode == 1:  # RGB only
        topics = ["/segmentation_record_node/rgb_cam0",
                  "/segmentation_record_node/rgb_cam1",
                  "/segmentation_record_node/rgb_cam2",
                  "/segmentation_record_node/rgb_cam3",]
    elif record_mode == 2:  # 2RGB - Depth
        topics = ["/segmentation_record_node/rgb_cam0",
                  "/segmentation_record_node/rgb_cam1",
                  "/segmentation_record_node/depth_cam0",
                  "/segmentation_record_node/depth_cam1",]
    else:
        topics = []
    if compressed:
        topics = [topic + "/compressed" for topic in topics]
    return topics
def generate_launch_description():
    # Declare launch arguments
    host_ip_arg = DeclareLaunchArgument(
        'host_ip',
        default_value='10.42.0.244',
        description='AirSim host IP address'
    )
    
    host_port_arg = DeclareLaunchArgument(
        'host_port',
        default_value='41451',
        description='AirSim RPC port'
    )
    record_mode_arg = DeclareLaunchArgument(
        'record_mode',
        default_value='0',
        description='Recording mode: 0 for RGB+Segmentation, 1 for RGB only'
    )
    vehicle_name_arg = DeclareLaunchArgument(
        'vehicle_name',
        default_value='',
        description='Vehicle name (empty for default)'
    )
    
    inter_publish_arg = DeclareLaunchArgument(
        'inter_publish',
        default_value='true',
        description='Publish compressed images (true) or raw (false)'
    )
    
    jpeg_quality_arg = DeclareLaunchArgument(
        'jpeg_quality',
        default_value='30',
        description='JPEG compression quality (1-100)'
    )
    
    sim_advance_time_arg = DeclareLaunchArgument(
        'sim_advance_time_ms',
        default_value='1.0',
        description='Time to let simulation advance after unpause (ms)'
    )
    
    # Create composable node
    container = ComposableNodeContainer(
        name='fast_stream_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='ct_uav_omni_fast_stream',
                plugin='ct_uav_omni::DataRecorderNode',
                name='segmentation_record_node',
                parameters=[{
                    'host_ip': LaunchConfiguration('host_ip'),
                    'host_port': LaunchConfiguration('host_port'),
                    'vehicle_name': LaunchConfiguration('vehicle_name'),
                    'record_mode': LaunchConfiguration('record_mode'),
                    'inter_publish': LaunchConfiguration('inter_publish'),
                    'jpeg_quality': LaunchConfiguration('jpeg_quality'),
                    'sim_advance_time_ms': LaunchConfiguration('sim_advance_time_ms'),
                }],
                extra_arguments=[{ 'use_intra_process_comms': True }],
            ),
            ComposableNode(
                package='rosbag2_composable_recorder',
                plugin='rosbag2_composable_recorder::ComposableRecorder',
                name='composable_recorder_node',
                parameters=[{
                    'topics': get_topic_list(2,0),
                    'bag_prefix': '/mnt/d/trung_Nav_team/Cosys-AirSim/ros2/src/ct_uav_omni_fast_stream/recorded_bags/bag',
                }],
                extra_arguments=[{ 'use_intra_process_comms': True }]
            ),
            ComposableNode(
                package='rosbag2_composable_recorder',
                plugin='rosbag2_composable_recorder::ComposableRecorder',
                name='composable_recorder_node',
                parameters=[{
                    'topics': get_topic_list(2,1),
                    'bag_prefix': '/mnt/d/trung_Nav_team/Cosys-AirSim/ros2/src/ct_uav_omni_fast_stream/recorded_bags/bag',
                }],
                extra_arguments=[{ 'use_intra_process_comms': True }]
            ),
        ],
        output='screen',
    )
   
    
    return LaunchDescription([
        host_ip_arg,
        host_port_arg,
        record_mode_arg,
        inter_publish_arg,
        vehicle_name_arg,
        jpeg_quality_arg,
        sim_advance_time_arg,
        container
    ])
