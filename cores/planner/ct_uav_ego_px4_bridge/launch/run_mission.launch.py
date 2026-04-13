import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessExit
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    bridge_pkg = get_package_share_directory('ego_px4_bridge')
    
    # 1. System Check
    # Script installed to share/ego_px4_bridge/script/system_check.py
    system_check_script = os.path.join(bridge_pkg, 'script', 'system_check.py')
    
    system_check = ExecuteProcess(
        cmd=['python3', system_check_script, '--timeout', '10.0'],
        output='screen'
    )

    # 2. Bridge Launch
    bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bridge_pkg, 'launch', 'bridge.launch.py')
        )
    )

    # 3. Ego Planner Launch
    try:
        planner_pkg = get_package_share_directory('ego_planner')
        planner_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(planner_pkg, 'launch', 'run_in_real.launch.py')
            )
        )
        planner_action = planner_launch
    except Exception as e:
        planner_action = LogInfo(msg=f"⚠️ Ego Planner launch not found: {e}")

    return LaunchDescription([
        LogInfo(msg="🚀 STARTING MISSION SEQUENCE..."),
        LogInfo(msg="1️⃣  Running System Check (10s)..."),
        system_check,
        
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=system_check,
                on_exit=[
                    LogInfo(msg="✅ System Check Finished. Proceeding to Launch..."),
                    LogInfo(msg="2️⃣  Starting Bridge Node..."),
                    bridge_launch,
                    LogInfo(msg="3️⃣  Starting Ego Planner..."),
                    planner_action
                ]
            )
        )
    ])
