import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get configuration file
    config_file = os.path.join(
        get_package_share_directory('ego_px4_bridge'),
        'config',
        'drone_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='ego_px4_bridge',
            executable='bridge_real',  # Make sure setup.py entry_point matches this
            name='bridge_node',
            output='screen',
            parameters=[config_file],
            
        )
    ])
