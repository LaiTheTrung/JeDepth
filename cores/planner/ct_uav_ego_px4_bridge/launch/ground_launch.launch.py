import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    """
    GROUND LAUNCH — Drone is on the ground.
    Bridge will: WAIT → ARM+OFFBOARD → TAKEOFF → MAP_WARMING → READY
    Ego Planner launches concurrently. Bridge holds goal until READY state.
    """
    bridge_pkg  = get_package_share_directory('ego_px4_bridge')

    # Config files
    # NOTE: safety_params.yaml là file duy nhất cần sửa cho bridge config
    safety_params = os.path.join(bridge_pkg, 'config', 'safety_params.yaml')

    # Bridge node — ground mode
    bridge_node = Node(
        package='ego_px4_bridge',
        executable='bridge_production',
        name='bridge_production',
        output='screen',
        parameters=[
            safety_params,
            {'launch_mode': 'ground'},  # Explicit override
        ],
        emulate_tty=True,
    )

    # Ego Planner
    try:
        planner_pkg    = get_package_share_directory('ego_planner')
        planner_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(planner_pkg, 'launch', 'run_in_real.launch.py')
            )
        )
        planner_action = planner_launch
    except Exception as e:
        planner_action = LogInfo(msg=f"⚠️  Ego Planner not found: {e}")

    return LaunchDescription([
        LogInfo(msg="🚀 GROUND LAUNCH — Waiting for pilot to ARM + OFFBOARD..."),
        bridge_node,
        planner_action,
    ])
