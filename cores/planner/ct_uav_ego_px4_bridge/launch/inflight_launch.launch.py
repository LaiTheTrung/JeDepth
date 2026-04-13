import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    """
    IN-FLIGHT LAUNCH — Drone is already airborne.

    ⚠️  SAFETY REQUIREMENT before launching:
      1. Drone must be hovering stably in MANUAL / POSITION mode.
      2. Launch THIS file BEFORE switching to Offboard on RC.
      3. Bridge will hold EXACT position at the moment of Offboard switch.
      4. Only set a goal AFTER log shows [READY].

    Bridge will: WAIT → ARM+OFFBOARD → INAIR_HOLD → MAP_WARMING → READY
    """
    bridge_pkg  = get_package_share_directory('ego_px4_bridge')

    # Config files
    drone_params  = os.path.join(bridge_pkg, 'config', 'drone_params.yaml')
    safety_params = os.path.join(bridge_pkg, 'config', 'safety_params.yaml')

    # Bridge node — inflight mode
    bridge_node = Node(
        package='ego_px4_bridge',
        executable='bridge_production',
        name='bridge_production',
        output='screen',
        parameters=[
            drone_params,
            safety_params,
            {'launch_mode': 'inflight'},  # Explicit override — SKIP TAKEOFF
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
        LogInfo(msg="✈  IN-FLIGHT LAUNCH — Bridge will HOLD current pos on Offboard switch."),
        LogInfo(msg="⚠️  Make sure drone is HOVERING STABLY before switching OFFBOARD!"),
        bridge_node,
        planner_action,
    ])
