import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    drone_id_ = "0"
    
    # Map size configuration
    map_size_x_ = LaunchConfiguration('map_size_x_', default='50.0')
    map_size_y_ = LaunchConfiguration('map_size_y_', default='50.0')
    map_size_z_ = LaunchConfiguration('map_size_z_', default='40.0')

    # 1. RViz for visualization
    rviz_para_dir = os.path.join(get_package_share_directory('ego_planner'), 'config', 'ros2_ego_rviz.rviz')
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=['-d', rviz_para_dir]
    )

    # 2. Trajectory Server (Ego Planner Core)
    traj_server_node = Node(
        package='ego_planner',
        executable='traj_server',
        output='screen',
        name="drone_"+drone_id_+"_traj_server",
        remappings=[
            ('position_cmd', "/drone_"+drone_id_+"_planning/pos_cmd"),
            ('planning/bspline', "/drone_"+drone_id_+"_planning/bspline"),
        ],
        parameters=[
            # BUG FIX: Pass odom_topic explicitly so traj_server subscribes to the
            # correct topic for yaw initialization (was using hardcoded default).
            {'odom_topic': f'/drone_{drone_id_}_visual_slam/odom'},
            {'traj_server/time_forward': 1.5},
        ],
    )

    # 3. Advanced Param Launch (Planner Logic)
    advanced_param_launch = os.path.join(get_package_share_directory('ego_planner'), "launch",'advanced_param.launch.py')
    included_launch_advanced_param = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(advanced_param_launch),
        launch_arguments={
            'drone_id': drone_id_,
            'map_size_x_': map_size_x_,
            'map_size_y_': map_size_y_,
            'map_size_z_': map_size_z_,
            'obj_num_set': '0',  # No simulated obstacles, we use real depth
            
            # ✅ FIXED: Camera Parameters phải khớp với Stereo Depth Node
            'cx': '128.0',  # 256/2
            'cy': '72.0',   # 144/2
            'fx': '187.8',  # ✅ Đúng như bạn confirm
            'fy': '187.8',  # ✅ Đúng như bạn confirm
            
            # ✅ CRITICAL: Topics phải khớp CHÍNH XÁC
            'camera_pose_topic': f'/drone_{drone_id_}_pcl_render_node/camera_pose',
            'depth_topic': f'/drone_{drone_id_}_pcl_render_node/depth',  # ✅ Bỏ underscore thừa
            'cloud_topic': '/null_topic',
            'odom_world': f'/drone_{drone_id_}_visual_slam/odom',
            
            # ✅ Grid Map PHẢI dùng depth thay vì cloud
            'grid_map/odom': f'/drone_{drone_id_}_visual_slam/odom',
            'grid_map/cloud': '/null_topic',
            'grid_map/pose': f'/drone_{drone_id_}_pcl_render_node/camera_pose',
            'grid_map/depth': f'/drone_{drone_id_}_pcl_render_node/depth',  # ✅ Bỏ underscore thừa
            
            # Planning Parameters
            'max_vel': '1.0',  # RECOMMENDED SAFE VALUE FOR REAL FLIGHT
            'max_acc': '0.8',  # RECOMMENDED SAFE VALUE FOR REAL FLIGHT
            'planning_horizon': '9.0',
            'use_distinctive_trajs': 'True',
            'flight_type': '1',
            
            # Node Name (Required by advanced_param.launch.py)
            # Name
            'name_of_ego_planner_node': f"drone_{drone_id_}_ego_planner_node",
            
            # Initial Goal (Optional) - SAFE DEFAULT IS NO INIT GOAL
            'point_num': '0',  # Wait for Rviz 2D Nav Goal instead of auto-flying
            'point0_x': '0.0',
            'point0_y': '0.0',
            'point0_z': '0.0',
            'point1_x': '0.0',
            'point1_y': '0.0',
            'point1_z': '1.0',
            'point2_x': '0.0',
            'point2_y': '0.0',
            'point2_z': '1.0',
            'point3_x': '0.0',
            'point3_y': '0.0',
            'point3_z': '1.0',
            'point4_x': '0.0',
            'point4_y': '0.0',
            'point4_z': '1.0',
            
            # Topic Remappings (Required by advanced_param.launch.py)
            'planning/bspline': f'/drone_{drone_id_}_planning/bspline',
            'planning/data_display': f'/drone_{drone_id_}_planning/data_display',
            'grid_map/occupancy_inflate': f'/drone_{drone_id_}_ego_planner_node/grid_map/occupancy_inflate',
            'optimal_list': f'/drone_{drone_id_}_ego_planner_node/optimal_list',
            
        }.items()
    )

    ld = LaunchDescription()
    ld.add_action(rviz_node)
    ld.add_action(traj_server_node)
    ld.add_action(included_launch_advanced_param)

    return ld
