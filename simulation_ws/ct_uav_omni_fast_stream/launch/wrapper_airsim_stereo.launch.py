from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

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
    
    vehicle_name_arg = DeclareLaunchArgument(
        'vehicle_name',
        default_value='',
        description='Vehicle name (empty for default)'
    )
    
    inter_publish_arg = DeclareLaunchArgument(
        'inter_publish',
        default_value='false',
        description='Publish compressed images (true) or raw (false)'
    )
    
    jpeg_quality_arg = DeclareLaunchArgument(
        'jpeg_quality',
        default_value='30',
        description='JPEG compression quality (1-100)'
    )
    
    sim_advance_time_arg = DeclareLaunchArgument(
        'sim_advance_time_ms',
        default_value='10.0',
        description='Time to let simulation advance after unpause (ms)'
    )
    
    # Create node
    fast_stream_node_stereo = Node(
        package='ct_uav_omni_fast_stream',
        executable='fast_stream_node_stereo',
        name='fast_stream_node_stereo',
        output='screen',
        parameters=[{
            'host_ip': LaunchConfiguration('host_ip'),
            'host_port': LaunchConfiguration('host_port'),
            'vehicle_name': LaunchConfiguration('vehicle_name'),
            'inter_publish': LaunchConfiguration('inter_publish'),
            'jpeg_quality': LaunchConfiguration('jpeg_quality'),
            'sim_advance_time_ms': LaunchConfiguration('sim_advance_time_ms'),
            'cassini_map_file': '/mnt/d/trung_Nav_team/Cosys-AirSim/ros2/src/ct_uav_omni_fast_stream/config/cassini_maps.yaml'
        }]
    ) 
    
    return LaunchDescription([
        host_ip_arg,
        host_port_arg,
        inter_publish_arg,
        vehicle_name_arg,
        jpeg_quality_arg,
        sim_advance_time_arg,
        fast_stream_node_stereo
    ])
