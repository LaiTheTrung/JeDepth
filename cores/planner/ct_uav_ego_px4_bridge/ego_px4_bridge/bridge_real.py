import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from px4_msgs.msg import VehicleOdometry, VehicleAttitude, TrajectorySetpoint, OffboardControlMode, VehicleCommand, VehicleStatus
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from quadrotor_msgs.msg import PositionCommand
import numpy as np
from collections import deque
import math
import time
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
from interactive_markers.menu_handler import MenuHandler
import psutil
import subprocess

class BridgeNode(Node):
    def __init__(self):
        super().__init__('bridge_node')

        # --- PARAMETERS ---
        self.declare_parameter('drone_id', 0)
        self.declare_parameter('takeoff_height', 1.0)
        self.declare_parameter('camera_offset_x', 0.1)
        self.declare_parameter('control_mode', 'POSITION')
        self.declare_parameter('publish_pointcloud', False)
        
        # Camera Intrinsics
        self.declare_parameter('camera.width', 256)
        self.declare_parameter('camera.height', 144)
        self.declare_parameter('camera.fx', 128.0)
        self.declare_parameter('camera.fy', 128.0)
        self.declare_parameter('camera.cx', 128.0)
        self.declare_parameter('camera.cy', 72.0)

        # Load Params
        self.drone_id = self.get_parameter('drone_id').value
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.camera_offset_x = self.get_parameter('camera_offset_x').value
        self.control_mode = self.get_parameter('control_mode').value.upper()
        self.should_pub_pcl = self.get_parameter('publish_pointcloud').value
        
        self.cam_width = self.get_parameter('camera.width').value
        self.cam_height = self.get_parameter('camera.height').value
        self.fx = self.get_parameter('camera.fx').value
        self.fy = self.get_parameter('camera.fy').value
        self.cx = self.get_parameter('camera.cx').value
        self.cy = self.get_parameter('camera.cy').value

        self.get_logger().info(f"🚀 REAL BRIDGE STARTED | Mode: {self.control_mode} | Height: {self.takeoff_height}m")
        self.get_logger().info(f"📷 Camera Config: {self.cam_width}x{self.cam_height} | fx={self.fx}")

        # --- STATE & COORDINATES ---
        self.state = "INIT"
        self.start_time = self.get_clock().now()
        self.initial_pos_ned = None
        self.current_pos_ned_abs = None
        self.current_att_q = None
        self.is_rc_override = False
        self.stabilize_start_time = None
        self.nav_state = None
        self.arming_state = None

        # Safety Turn State
        self.is_turning_around = False
        
        # Odom Buffer for Sync (Store last 100 samples)
        self.odom_buffer = deque(maxlen=100) 

        # --- TF & BROADCASTERS ---
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_tf()
        self.create_timer(1.0, self.publish_static_tf)

        # --- QoS ---
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- SUBSCRIBERS ---
        # Depth from Stereo Camera
        self.depth_sub = self.create_subscription(
            Image, '/stereo/depth_map', self.depth_callback, 10
        )
        
        # PX4 Topics
        self.pos_sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.pos_callback, qos_best_effort
        )
        self.att_sub = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', self.att_callback, qos_best_effort
        )
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, qos_best_effort
        )
        
        # Ego Planner Command (Corrected Type)
        self.cmd_sub = self.create_subscription(
            PositionCommand, f'/drone_{self.drone_id}_planning/pos_cmd', self.cmd_callback, 10
        )

        # --- PUBLISHERS ---
        self.odom_pub = self.create_publisher(Odometry, f'/drone_{self.drone_id}_visual_slam/odom', 10)
        self.cam_pose_pub = self.create_publisher(PoseStamped, f'/drone_{self.drone_id}_pcl_render_node/camera_pose', 10)
        self.depth_pub = self.create_publisher(Image, f'/drone_{self.drone_id}_pcl_render_node/depth', 10)
        self.cam_info_pub = self.create_publisher(CameraInfo, f'/drone_{self.drone_id}_pcl_render_node/camera_info', 10)
        
        # PX4 Control
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_best_effort)
        self.offboard_ctrl_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_best_effort)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        # --- TIMERS ---
        self.create_timer(0.05, self.publish_offboard_heartbeat) # 20Hz
        self.create_timer(0.1, self.mission_loop) # 10Hz Logic
        self.last_cmd_time = self.get_clock().now()
        
        # Geofence
        self.GEO_FENCE_XY = 2000.0 
        self.GEO_FENCE_Z_MAX = 5.0
        self.GEO_FENCE_Z_MIN = 0.2

        # --- VISUALIZATION & INTERACTION (Ported from Sim) ---
        # 1. Goal Publisher (To Ego Planner)
        self.goal_pub = self.create_publisher(PoseStamped, f'/drone_{self.drone_id}_ego_planner_node/goal', 10)
        
        # 2. Local Map Visualization (Bounding Box)
        self.local_map_pub = self.create_publisher(Marker, f'/drone_{self.drone_id}_ego_planner_node/local_map_bound', 10)
        self.local_map_size = [15.0, 15.0, 8.0] # Match advanced_param.launch.py

        # 3. Interactive Markers
        self.server = InteractiveMarkerServer(self, "goal_marker_server")
        self.menu_handler = MenuHandler()
        self.menu_handler.insert("EXECUTE GOAL", callback=self.process_marker_feedback)
        
        self.rviz_goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.rviz_goal_callback, 10
        )
        self.pending_goal_pose = None

        # 4. Clicked Point (3D Goal)
        self.clicked_point_sub = self.create_subscription(
            PoseStamped, '/clicked_point', self.clicked_point_callback, 10
        )

        # 5. Resource Monitor
        self.create_timer(5.0, self.log_system_resources)

        # Shutdown Hook
        rclpy.get_default_context().on_shutdown(self.on_shutdown)

    # =========================================================================
    # COORDINATE & SYNC LOGIC
    # =========================================================================
    
    def publish_static_tf(self):
        # Camera Optical Frame: Z-forward, X-right, Y-down
        t = TransformStamped()
        t.header.stamp = rclpy.time.Time().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_link'
        t.transform.translation.x = self.camera_offset_x
        # RPY = [-90, 0, -90] deg
        t.transform.rotation.x = -0.5
        t.transform.rotation.y = 0.5
        t.transform.rotation.z = -0.5
        t.transform.rotation.w = 0.5
        self.tf_static_broadcaster.sendTransform(t)

    def pos_callback(self, msg):
        self.current_pos_ned_abs = np.array([msg.position[0], msg.position[1], msg.position[2]])
        
        # Chỉ set Origin một lần duy nhất
        if self.initial_pos_ned is None:
            # Check for valid position before setting origin
            if not np.isnan(self.current_pos_ned_abs).any():
                self.initial_pos_ned = self.current_pos_ned_abs
                self.get_logger().info(f"📍 ORIGIN SET: {self.initial_pos_ned}")
            return

        if self.current_att_q is None: return

        # Tính toán Odom ENU
        pos_ned_rel = self.current_pos_ned_abs - self.initial_pos_ned
        pos_enu = self.ned_to_enu_pos(pos_ned_rel)
        vel_enu = self.ned_to_enu_pos(np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]]))
        
        q_ned = self.current_att_q
        q_enu = self.ned_to_enu_orientation(q_ned) 

        now = self.get_clock().now()
        
        # --- LƯU BUFFER CHO CAMERA SYNC ---
        odom_data = {
            'stamp': now, # Lưu timestamp ROS
            'pos': pos_enu,
            'vel': vel_enu,
            'quat': q_enu
        }
        self.odom_buffer.append(odom_data)
        
        # --- PUBLISH ODOMETRY ---
        # Quan trọng: Dùng timestamp hiện tại cho TF để tránh lỗi "Message Filter dropping" trên RViz
        self.publish_odom_and_tf(now, pos_enu, vel_enu, q_enu)

    def att_callback(self, msg):
        self.current_att_q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]

    def status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state
        
        # Check Disarm (arming_state: 1=DISARMED, 2=ARMED)
        if msg.arming_state != 2 and self.state != "INIT":
            self.get_logger().info("⚠️ VEHICLE DISARMED. Resetting State to INIT.")
            self.state = "INIT"
            # Return to manual control logic if needed
            self.initial_pos_ned = None 

        # NAVIGATION_STATE_OFFBOARD = 14
        if msg.nav_state != 14:
            if not self.is_rc_override and self.state != "INIT":
                self.get_logger().warn("⚠️ RC OVERRIDE (Not Offboard). Pausing Bridge Control.")
            self.is_rc_override = True
        else:
            if self.is_rc_override:
                self.get_logger().info("✅ OFFBOARD DETECTED. Bridge Active.")
            self.is_rc_override = False

    def depth_callback(self, msg):
        """
        Receive Depth -> Find Past Odom -> Publish Sync
        """
        if not self.odom_buffer: 
            self.get_logger().warn("Depth callback: No odom yet", throttle_duration_sec=2.0)
            return

        # Get depth timestamp
        depth_time = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds
        
        # Find closest Odom
        best_odom = None
        min_diff = float('inf')
        
        for odom in reversed(self.odom_buffer):
            odom_time = odom['stamp'].nanoseconds
            diff = abs(depth_time - odom_time)
            if diff < min_diff:
                min_diff = diff
                best_odom = odom
            else:
                break
        
        if best_odom is None or min_diff > 0.1 * 1e9: # > 100ms diff
            return

        # Publish Synced Camera Pose
        cam_pose = PoseStamped()
        cam_pose.header.stamp = msg.header.stamp # Use DEPTH timestamp
        cam_pose.header.frame_id = "world"
        cam_pose.pose.position.x = float(best_odom['pos'][0])
        cam_pose.pose.position.y = float(best_odom['pos'][1])
        cam_pose.pose.position.z = float(best_odom['pos'][2])
        cam_pose.pose.orientation.x = float(best_odom['quat'][0])
        cam_pose.pose.orientation.y = float(best_odom['quat'][1])
        cam_pose.pose.orientation.z = float(best_odom['quat'][2])
        cam_pose.pose.orientation.w = float(best_odom['quat'][3])
        self.cam_pose_pub.publish(cam_pose)

        # Publish Camera Info
        cam_info = CameraInfo()
        cam_info.header = msg.header
        cam_info.header.frame_id = "camera_link"
        cam_info.width = self.cam_width
        cam_info.height = self.cam_height
        
        # Intrinsics
        cam_info.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        cam_info.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        self.cam_info_pub.publish(cam_info)
        
        # Republish Depth
        self.depth_pub.publish(msg)

    # =========================================================================
    # HELPERS
    # =========================================================================
    def ned_to_enu_pos(self, ned_rel):
        return np.array([ned_rel[1], ned_rel[0], -ned_rel[2]])

    def ned_to_enu_orientation(self, q_ned):
        # q_ned: [w, x, y, z] (PX4)
        r, p, y = self.euler_from_quaternion(q_ned)
        r_enu = r
        p_enu = -p
        y_enu = -y + (math.pi / 2.0)
        
        if y_enu > math.pi: y_enu -= 2*math.pi
        if y_enu < -math.pi: y_enu += 2*math.pi
        
        return self.quaternion_from_euler(r_enu, p_enu, y_enu)

    def euler_from_quaternion(self, q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return roll, pitch, yaw

    def quaternion_from_euler(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [x, y, z, w]

    def publish_odom_and_tf(self, timestamp, pos, vel, q):
        odom = Odometry()
        odom.header.stamp = timestamp.to_msg()
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        odom.pose.pose.orientation.x = float(q[0])
        odom.pose.pose.orientation.y = float(q[1])
        odom.pose.pose.orientation.z = float(q[2])
        odom.pose.pose.orientation.w = float(q[3])
        
        odom.twist.twist.linear.x = float(vel[0])
        odom.twist.twist.linear.y = float(vel[1])
        odom.twist.twist.linear.z = float(vel[2])
        
        self.odom_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = timestamp.to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        t.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    # =========================================================================
    # MISSION LOGIC
    # =========================================================================
    def mission_loop(self):
        if self.initial_pos_ned is None: return
        if self.is_rc_override: return # Do nothing if pilot is in control

        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9

        # Visualize Local Map
        if self.odom_buffer:
             # Use latest odom for visualization center
             latest_odom = self.odom_buffer[-1]
             self.publish_local_map_marker(now, latest_odom['pos'])

        # --- SAFETY LOGIC FOR REAL WORLD ---
        # 1. Wait for Position Lock (Origin Set)
        if self.initial_pos_ned is None:
            if elapsed % 2.0 < 0.1: # Print every 2s
                self.get_logger().info("⏳ WAITING FOR PX4 POSITION...", throttle_duration_sec=2.0)
            return

        # 2. State Machine
        if self.state == "INIT":
            # LOGIC: Wait for Pilot to switch to Offboard Mode explicitly
            if self.nav_state == 14 and self.arming_state == 2:
                # Pilot has Armed AND switched to Offboard -> We take control
                altitude = -self.current_pos_ned_abs[2] + self.initial_pos_ned[2] # Relative alt
                
                # Check height to decide if Takeoff or Just Hold
                if altitude < 0.3:
                    self.get_logger().info(f"🚀 ON GROUND (Alt={altitude:.2f}m). STARTING TAKEOFF SEQUENCE.")
                    self.state = "TAKEOFF"
                    # Capture exact origin again for better accuracy?
                    # self.initial_pos_ned = self.current_pos_ned_abs 
                else:
                    self.get_logger().info(f"🚁 IN AIR (Alt={altitude:.2f}m). HOLDING POSITION & STARTING MISSION.")
                    self.state = "STABILIZE" # Go to stabilize briefly
                    self.stabilize_start_time = now
            else:
                 if elapsed % 5.0 < 0.1:
                    self.get_logger().info("💤 R: Waiting for OFFBOARD switch + ARM...", throttle_duration_sec=5.0)
                
        # REMOVED AUTO ARMING STATE
        
        elif self.state == "TAKEOFF":
            target_z_abs = self.initial_pos_ned[2] - self.takeoff_height
            self.publish_trajectory_setpoint(
                self.initial_pos_ned[0], self.initial_pos_ned[1], target_z_abs, 0.0
            )
            if elapsed > 15.0:
                self.state = "MISSION"
                self.get_logger().info(f"MISSION START")

        elif self.state == "STABILIZE":
             target_z_abs = self.initial_pos_ned[2] - self.takeoff_height
             self.publish_trajectory_setpoint(
                self.initial_pos_ned[0], self.initial_pos_ned[1], target_z_abs, 0.0
             )
             if (now - self.stabilize_start_time).nanoseconds / 1e9 > 4.0:
                 self.state = "MISSION"
                 self.get_logger().info(f"MISSION START (STABILIZED)")

        elif self.state == "MISSION":
            # Watchdog
            time_since_last_cmd = (now - self.last_cmd_time).nanoseconds / 1e9
            if time_since_last_cmd > 0.5:
                if self.current_pos_ned_abs is not None:
                     # SAFETY: If Planner dies, HOLD POSITION
                     self.publish_trajectory_setpoint(
                        self.current_pos_ned_abs[0], 
                        self.current_pos_ned_abs[1], 
                        self.current_pos_ned_abs[2], 
                        float('nan')
                    )

    def cmd_callback(self, msg):
        if self.initial_pos_ned is None: return
        if self.is_rc_override: return 

        self.last_cmd_time = self.get_clock().now()
        
        # 1. Giới hạn không gian an toàn (GeoFence Check)
        target_x = max(-self.GEO_FENCE_XY, min(self.GEO_FENCE_XY, msg.position.x))
        target_y = max(-self.GEO_FENCE_XY, min(self.GEO_FENCE_XY, msg.position.y))
        target_z = max(self.GEO_FENCE_Z_MIN, min(self.GEO_FENCE_Z_MAX, msg.position.z))
        
        # 2. Chuyển đổi tọa độ: ENU (ROS) -> NED (PX4)
        # ROS X (East)  -> PX4 Y (East)
        # ROS Y (North) -> PX4 X (North)
        # ROS Z (Up)    -> PX4 -Z (Up)
        req_ned_rel = np.array([target_y, target_x, -target_z])
        req_ned_abs = req_ned_rel + self.initial_pos_ned
        
        # Yaw: ENU -> NED
        px4_yaw = -msg.yaw + (math.pi / 2.0)

        # 3. Vận tốc dẫn đường (Feedforward Velocity)
        v_x_ned = msg.velocity.y  # ROS Vy -> PX4 Vx
        v_y_ned = msg.velocity.x  # ROS Vx -> PX4 Vy
        v_z_ned = -msg.velocity.z # ROS Vz -> PX4 -Vz

        # --- SAFETY TURN LOGIC (ĐÃ SỬA) ---
        if self.current_att_q is not None:
            speed_sq = msg.velocity.x**2 + msg.velocity.y**2
            if speed_sq > 0.05: # Tăng ngưỡng lên chút để tránh nhiễu
                vel_heading_enu = math.atan2(msg.velocity.y, msg.velocity.x)
                vel_heading_ned = -vel_heading_enu + (math.pi / 2.0)
                
                _, _, current_yaw_ned = self.euler_from_quaternion(self.current_att_q)
                
                heading_diff = vel_heading_ned - current_yaw_ned
                while heading_diff > math.pi: heading_diff -= 2*math.pi
                while heading_diff < -math.pi: heading_diff += 2*math.pi
                
                # Logic Hysteresis
                if not self.is_turning_around and abs(heading_diff) > 2.0: # ~115 độ
                    self.is_turning_around = True
                    self.get_logger().warn(f"🔄 START TURN AROUND")
                elif self.is_turning_around and abs(heading_diff) < 0.5: # ~30 độ
                    self.is_turning_around = False
                    self.get_logger().info(f"✅ TURN COMPLETE")

        # --- XỬ LÝ KHI QUAY ĐẦU ---
        if self.is_turning_around:
            # Khi quay đầu: Dừng di chuyển, chỉ xoay Yaw
            # QUAN TRỌNG: Không được gửi NaN cho Position nếu OffboardMode đang đòi Position
            # Giải pháp: Giữ nguyên vị trí hiện tại (Hover tại chỗ)
            req_ned_abs[0] = self.current_pos_ned_abs[0]
            req_ned_abs[1] = self.current_pos_ned_abs[1]
            req_ned_abs[2] = self.current_pos_ned_abs[2]
            
            # Vận tốc = 0
            v_x_ned = 0.0
            v_y_ned = 0.0
            v_z_ned = 0.0
            
            # Yaw theo hướng vận tốc mong muốn
            # (Phải tính lại vel_heading_ned vì biến kia nằm trong if scope)
            vel_heading_enu = math.atan2(msg.velocity.y, msg.velocity.x)
            px4_yaw = -vel_heading_enu + (math.pi / 2.0)

        # 4. Gửi lệnh xuống PX4
        # Gửi cả Position và Velocity (PX4 sẽ kết hợp chúng)
        self.publish_trajectory_setpoint(
            req_ned_abs[0], req_ned_abs[1], req_ned_abs[2], px4_yaw,
            vx=v_x_ned, vy=v_y_ned, vz=v_z_ned
        )

    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        # SỬA: Luôn kích hoạt cả Position và Velocity nếu ở chế độ MISSION
        # Để tận dụng Trajectory Setpoint đầy đủ (Feed-forward velocity)
        if self.state == "MISSION":
            msg.position = False # Trajectory Setpoint covers position
            msg.velocity = False # Trajectory Setpoint covers velocity
            msg.acceleration = False
        else:
            # Modes like Takeoff/Stabilize need Position control
            msg.position = True
            msg.velocity = False
            msg.acceleration = False
            
        self.offboard_ctrl_pub.publish(msg)

    def publish_trajectory_setpoint(self, x, y, z, yaw, vx=float('nan'), vy=float('nan'), vz=float('nan')):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [float(vx), float(vy), float(vz)]
        msg.yaw = float(yaw)
        self.traj_pub.publish(msg)

    def publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1 = params.get('param1', 0.0)
        msg.param2 = params.get('param2', 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.from_external = True
        self.vehicle_command_pub.publish(msg)

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    def on_shutdown(self):
        self.get_logger().warn("🛑 BRIDGE SHUTTING DOWN! SWITCHING TO HOLD MODE.")
        # Try to switch to Position Mode (Hold)
        # VEHICLE_CMD_DO_SET_MODE = 176
        # Mode: 1 (Custom), Submode: 3 (POSCTL)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=3.0)

    # =========================================================================
    # PORTED FEATURES (VISUALIZATION & INTERACTION)
    # =========================================================================
    def publish_local_map_marker(self, timestamp, pos_enu):
        marker = Marker()
        marker.header.stamp = timestamp.to_msg()
        marker.header.frame_id = "world"
        marker.id = 999
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(pos_enu[0])
        marker.pose.position.y = float(pos_enu[1])
        marker.pose.position.z = float(pos_enu[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.local_map_size[0]
        marker.scale.y = self.local_map_size[1]
        marker.scale.z = self.local_map_size[2]
        marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 0.15 
        self.local_map_pub.publish(marker)

    def clicked_point_callback(self, msg):
        if not self.odom_buffer: return
        if msg.header.frame_id == 'world':
            goal = PoseStamped()
            goal.header = msg.header
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position = msg.pose.position
            goal.pose.orientation.w = 1.0
            self.goal_pub.publish(goal)
            self.get_logger().info(f"🎯 3D GOAL: {msg.pose.position.x:.1f}, {msg.pose.position.y:.1f}, {msg.pose.position.z:.1f}")

    def rviz_goal_callback(self, msg):
        if not self.odom_buffer: return
        current_z = self.odom_buffer[-1]['pos'][2]
        self.server.clear()
        self.pending_goal_pose = msg
        self.pending_goal_pose.pose.position.z = float(current_z)
        self.make_interactive_marker(self.pending_goal_pose.pose)
        self.server.applyChanges()
        self.get_logger().info(f"⚓ 2D Goal Received. Marker spawned at Z={current_z:.2f}m.")

    def make_interactive_marker(self, pose):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "world"
        int_marker.pose = pose
        int_marker.scale = 1.0
        int_marker.name = "goal_marker"
        int_marker.description = "Right-Click to Fly"
        
        control = InteractiveMarkerControl()
        control.orientation.w = 1.0; control.orientation.y = 1.0 
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)
        
        control = InteractiveMarkerControl()
        control.interaction_mode = InteractiveMarkerControl.MENU
        control.always_visible = True
        marker = InteractiveMarkerControl(); marker.always_visible = True
        int_marker.controls.append(control)
        
        self.server.insert(int_marker, feedback_callback=self.process_marker_feedback)
        self.menu_handler.apply(self.server, int_marker.name)

    def process_marker_feedback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            self.pending_goal_pose.pose = feedback.pose
        elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            if self.pending_goal_pose:
                self.pending_goal_pose.header.stamp = self.get_clock().now().to_msg()
                self.goal_pub.publish(self.pending_goal_pose)
                self.get_logger().info(f"🚀 EXECUTING GOAL")
                self.server.clear(); self.server.applyChanges()

    def log_system_resources(self):
        try:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            self.get_logger().info(f"CPU: {cpu}% | RAM: {mem}%", throttle_duration_sec=5.0)
        except: pass

def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
