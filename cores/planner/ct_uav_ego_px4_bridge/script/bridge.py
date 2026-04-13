import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude, TrajectorySetpoint, OffboardControlMode, VehicleCommand
from cv_bridge import CvBridge
import numpy as np
from collections import deque
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs_py import point_cloud2
from quadrotor_msgs.msg import PositionCommand  
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.menu_handler import MenuHandler
import psutil
import subprocess

class BridgeNode(Node):
    def __init__(self):
        super().__init__('bridge_node')
        
        # --- CONFIGURATION ---
        self.drone_id = 0
        self.takeoff_height = 1.2  
        self.camera_offset_x = 0.1 # Vị trí camera so với tâm drone
        
        # CONTROL MODE: 'POSITION' or 'VELOCITY'
        self.declare_parameter('control_mode', 'POSITION')
        self.control_mode = self.get_parameter('control_mode').value.upper()
        self.get_logger().info(f"🎮 BRIDGE CONTROL MODE: {self.control_mode}")

        # Geofence: Giới hạn an toàn so với điểm khởi tạo (mét)
        # Ego Planner không được điều khiển drone bay quá vùng này
        self.GEO_FENCE_XY = 2000.0 
        self.GEO_FENCE_Z_MAX = 5.0
        self.GEO_FENCE_Z_MIN = 0.2

        # State Machine
        self.state = "INIT" # INIT -> ARMING -> TAKEOFF -> MISSION
        self.start_time = self.get_clock().now()
        
        # --- COORDINATE SYSTEM ---
        # Gốc tọa độ cục bộ (Lưu vị trí NED của PX4 tại thời điểm khởi tạo)
        self.initial_pos_ned = None 
        self.current_pos_ned_abs = None # Vị trí tuyệt đối hiện tại
        self.current_att_q = None       # Quaternion hiện tại [w,x,y,z]
        
        # Lưu trạng thái Odom mới nhất (đã convert sang ENU) để sync với Depth
        self.latest_odom_enu = None
        self.latest_odom_time = 0.0

        self.bridge = CvBridge()

        # --- TF Broadcasters ---
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_tf() # Setup Camera Frame đúng chuẩn Optical

        # --- QoS ---
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- SUBSCRIBERS ---
        # 1. Depth Image
        self.depth_sub = self.create_subscription(
            Image, '/cam0/depth_map', self.depth_callback, 10
        )
        
        # 2. PX4 Position
        self.pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.pos_callback, qos_best_effort
        )
        
        # 3. PX4 Attitude
        self.att_sub = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', self.att_callback, qos_best_effort
        )
        
        # 4. Ego Planner Command
        self.cmd_sub = self.create_subscription(
            PositionCommand, f'/drone_{self.drone_id}_planning/pos_cmd', self.cmd_callback, 10
        )
        
        # 5. Goal from RViz (3D Point Cloud Click)
        self.clicked_point_sub = self.create_subscription(
            PoseStamped, '/clicked_point', self.clicked_point_callback, 10
        )

        # --- PUBLISHERS ---
        # To Ego Planner (ROS Standard)
        self.odom_pub = self.create_publisher(Odometry, f'/drone_{self.drone_id}_visual_slam/odom', 10)
        self.cam_pose_pub = self.create_publisher(PoseStamped, f'/drone_{self.drone_id}_pcl_render_node/camera_pose', 10)
        self.depth_pub = self.create_publisher(Image, f'/drone_{self.drone_id}_pcl_render_node/depth', 10)
        self.cam_info_pub = self.create_publisher(CameraInfo, f'/drone_{self.drone_id}_pcl_render_node/camera_info', 10)
        self.cloud_pub = self.create_publisher(PointCloud2, f'/drone_{self.drone_id}_pcl_render_node/cloud', 10)
        # Publish to specific topic so Ego Planner only flies when WE say so (not direct from RViz)
        self.goal_pub = self.create_publisher(PoseStamped, '/drone_0_ego_planner_node/goal', 10)

        # To PX4
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_best_effort)
        self.offboard_ctrl_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_best_effort)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        # --- TIMERS ---
        self.create_timer(0.05, self.publish_offboard_heartbeat) # 20Hz Heartbeat
        self.create_timer(0.1, self.mission_loop) # 10Hz Logic Loop
        
        # Safety Watchdog: Nếu không nhận được lệnh từ Planner quá 0.5s -> Hold
        self.last_cmd_time = self.get_clock().now()
        self.stabilize_start_time = None # For mid-air initialization

        # Resource Monitor (5s interval)
        self.create_timer(5.0, self.log_system_resources)

        # --- INTERACTIVE MARKER SETUP ---
        self.server = InteractiveMarkerServer(self, "goal_marker_server")
        self.menu_handler = MenuHandler()
        self.menu_handler.insert("EXECUTE GOAL", callback=self.process_marker_feedback)
        
        # Subscribe to RViz 2D Nav Goal (Standard)
        self.rviz_goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.rviz_goal_callback, 10
        )
        self.pending_goal_pose = None

        self.get_logger().info("Bridge Node Started. Waiting for PX4 Position...")

    # =========================================================================
    # CORE LOGIC: COORDINATE FRAMES & TF
    # =========================================================================
    
    def publish_static_tf(self):
        """
        Transform from base_link to camera_optical_frame.
        base_link: X-forward (FLU in ENU), matches drone body
        camera_optical: Z-forward, X-right, Y-down (standard ROS camera optical)
        
        Extrinsics: [x=0.1, y=0, z=0, roll=-90°, pitch=0°, yaw=-90°]
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_link' # Optical frame
        
        # Translation
        t.transform.translation.x = self.camera_offset_x
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        
        # Rotation: RPY = [-90°, 0°, -90°] (optical frame)
        roll, pitch, yaw = -1.5708, 0.0, -1.5708  # radians
        qx, qy, qz, qw = self.rpy_to_quat(roll, pitch, yaw)
        
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        
        self.tf_static_broadcaster.sendTransform(t)

    def rpy_to_quat(self, roll, pitch, yaw):
        """Convert Roll-Pitch-Yaw to Quaternion [x,y,z,w]"""
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        return qx, qy, qz, qw


    def ned_to_enu_pos(self, ned_rel):
        # NED (North, East, Down) -> ENU (East, North, Up)
        # x_enu = y_ned
        # y_enu = x_ned
        # z_enu = -z_ned
        return np.array([ned_rel[1], ned_rel[0], -ned_rel[2]])

    def enu_to_ned_pos(self, enu_pos):
        # ENU -> NED
        # x_ned = y_enu
        # y_ned = x_enu
        # z_ned = -z_enu
        return np.array([enu_pos[1], enu_pos[0], -enu_pos[2]])

    def ned_to_enu_orientation(self, q_ned):
        # q_ned: [w, x, y, z] (PX4)
        # Convert to Euler
        r, p, y = self.euler_from_quaternion(q_ned)
        
        # Convert Euler NED -> ENU
        r_enu = r
        p_enu = -p
        y_enu = -y + (math.pi / 2.0)
        
        # Normalize
        if y_enu > math.pi: y_enu -= 2*math.pi
        if y_enu < -math.pi: y_enu += 2*math.pi
        
        # Back to Quaternion [x, y, z, w]
        return self.quaternion_from_euler(r_enu, p_enu, y_enu)

    # =========================================================================
    # CALLBACKS: PERCEPTION & STATE ESTIMATION
    # =========================================================================

    def att_callback(self, msg):
        # Lưu Attitude mới nhất
        self.current_att_q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]

    def pos_callback(self, msg):
        # 1. TIME SYNC FIX: Gán timestamp hiện tại của ROS
        now = self.get_clock().now()
        
        # Cập nhật vị trí tuyệt đối (cho logic bay Mission)
        self.current_pos_ned_abs = np.array([msg.x, msg.y, msg.z])

        # 2. KHỞI TẠO ĐIỂM GỐC (Chỉ chạy 1 lần)
        if self.initial_pos_ned is None:
            self.initial_pos_ned = self.current_pos_ned_abs
            self.get_logger().info(f"📍 MAP ORIGIN SET at AirSim NED: {self.initial_pos_ned}")
            return

        # Nếu chưa có Attitude, chưa thể tính Odom đầy đủ
        if self.current_att_q is None:
            return

        # 3. TÍNH TOÁN RELATIVE POS (ENU)
        # Vị trí tương đối = Tuyệt đối - Gốc
        pos_ned_rel = self.current_pos_ned_abs - self.initial_pos_ned
        
        # Chuyển sang ENU
        pos_enu = self.ned_to_enu_pos(pos_ned_rel)
        vel_enu = self.ned_to_enu_pos(np.array([msg.vx, msg.vy, msg.vz]))
        q_enu = self.ned_to_enu_orientation(self.current_att_q)

        # Lưu lại để dùng cho Depth Callback (Sync đơn giản qua biến global)
        self.latest_odom_enu = {
            'pos': pos_enu,
            'vel': vel_enu,
            'quat': q_enu,
            'stamp': now
        }
        self.latest_odom_time = now.nanoseconds / 1e9

        # 4. PUBLISH ODOMETRY & TF (World -> Base_link)
        self.publish_odom_and_tf(now, pos_enu, vel_enu, q_enu)

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

        # Publish Dynamic TF
        t = TransformStamped()
        t.header.stamp = timestamp.to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        t.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def depth_callback(self, msg):
        """
        POSE_STAMPED Mode: Publish synchronized camera_pose + depth.
        CRITICAL: Both messages MUST have IDENTICAL timestamps for message_filters to sync.
        """
        # 1. Kiểm tra prerequisites
        if self.latest_odom_enu is None:
            self.get_logger().warn("Depth callback: No odom yet", throttle_duration_sec=2.0)
            return
            
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if (now_sec - self.latest_odom_time) > 0.15:
            self.get_logger().warn(f"Depth callback: Odom too old ({now_sec - self.latest_odom_time:.3f}s)", 
                                   throttle_duration_sec=2.0)
            return 

        try:
            odom_data = self.latest_odom_enu
            
            # 2. CREATE CANONICAL TIMESTAMP (CRITICAL FOR SYNC)
            # Use odom timestamp as the authoritative time
            sync_stamp = odom_data['stamp']
            
            # 3. PUBLISH CAMERA POSE (with sync_stamp)
            cam_pose = PoseStamped()
            cam_pose.header.stamp = sync_stamp.to_msg()
            cam_pose.header.frame_id = "world"
            cam_pose.pose.position.x = float(odom_data['pos'][0])
            cam_pose.pose.position.y = float(odom_data['pos'][1])
            cam_pose.pose.position.z = float(odom_data['pos'][2])
            cam_pose.pose.orientation.x = float(odom_data['quat'][0])
            cam_pose.pose.orientation.y = float(odom_data['quat'][1])
            cam_pose.pose.orientation.z = float(odom_data['quat'][2])
            cam_pose.pose.orientation.w = float(odom_data['quat'][3])
            
            self.cam_pose_pub.publish(cam_pose)
            
            # 4. PUBLISH CAMERA INFO (with sync_stamp)
            cam_info = CameraInfo()
            cam_info.header.stamp = sync_stamp.to_msg()
            cam_info.header.frame_id = "camera_link"
            cam_info.width = msg.width
            cam_info.height = msg.height
            
            # Intrinsics
            fx = 128.0; fy = 128.0; cx = 128.0; cy = 72.0
            cam_info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            cam_info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            
            self.cam_info_pub.publish(cam_info)
            
            # 5. REPUBLISH DEPTH with IDENTICAL timestamp
            depth_out = Image()
            depth_out.header.stamp = sync_stamp.to_msg()  # SAME timestamp as camera_pose!
            depth_out.header.frame_id = "camera_link"
            depth_out.height = msg.height
            depth_out.width = msg.width
            depth_out.encoding = msg.encoding  # Should be 32FC1
            depth_out.is_bigendian = msg.is_bigendian
            depth_out.step = msg.step
            depth_out.data = msg.data
            
            self.depth_pub.publish(depth_out)
            
            # 6. DETAILED LOGGING (để verify sync)
            self.get_logger().info(
                f"\n=== DEPTH+POSE SYNC ===\n"
                f"Timestamp (sec.nsec): {sync_stamp.to_msg().sec}.{sync_stamp.to_msg().nanosec}\n"
                f"Camera Pose: ({odom_data['pos'][0]:.2f}, {odom_data['pos'][1]:.2f}, {odom_data['pos'][2]:.2f})\n"
                f"Depth Size: {msg.width}x{msg.height}, Encoding: {msg.encoding}\n"
                f"Topics Published: camera_pose, camera_info, depth\n"
                f"========================",
                throttle_duration_sec=2.0
            )
            
            # 7. OPTIONAL: Publish PointCloud for RViz visualization (non-critical)
            # This is ONLY for human viewing, Ego Planner does NOT use it in POSE_STAMPED mode
            self.publish_pointcloud_optimized(msg, fx, fy, cx, cy, sync_stamp)

        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def publish_pointcloud_optimized(self, depth_msg, fx, fy, cx, cy, timestamp):
        # Convert CV2
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        
        # Resize nhỏ lại nếu ảnh quá lớn để giảm tải CPU (Optional)
        # depth_img = depth_img[::2, ::2] 
        # fx/=2; fy/=2; cx/=2; cy/=2
        
        height, width = depth_img.shape
        
        # LỌC NHIỄU (Safety Filter)
        # Loại bỏ các điểm quá gần (<= 0.3m) -> Tránh Self-collision do nhiễu
        # Loại bỏ quá xa (>= 20m) -> Map không cần thiết quá xa
        mask = (depth_img > 0.3) & (depth_img < 10.0)
        
        if not np.any(mask):
            return

        # Tạo grid tọa độ
        v, u = np.indices((height, width))
        
        # Chỉ lấy các điểm valid
        z_valid = depth_img[mask]
        u_valid = u[mask]
        v_valid = v[mask]
        
        # Back-projection (Optical Frame: Z=Forward, X=Right, Y=Down)
        x_opt = (u_valid - cx) * z_valid / fx
        y_opt = (v_valid - cy) * z_valid / fy
        z_opt = z_valid
        
        # Stack thành (N, 3)
        points = np.stack([x_opt, y_opt, z_opt], axis=1)
        
        # Create PointCloud2 msg
        header = Header()
        header.stamp = timestamp.to_msg()
        header.frame_id = "camera_link"  # CRITICAL: Must match TF camera frame
        pc2_msg = point_cloud2.create_cloud_xyz32(header, points)
        self.cloud_pub.publish(pc2_msg)

    def clicked_point_callback(self, msg):
        """
        Handle 3D goal from RViz "Publish Point" tool.
        User clicks directly on point cloud to select goal with X, Y, Z.
        """
        if self.latest_odom_enu is None:
            self.get_logger().warn("Cannot set goal: No odometry yet", throttle_duration_sec=2.0)
            return
        
        # The clicked point should be in 'world' frame
        if msg.header.frame_id == 'world':
            goal = PoseStamped()
            goal.header = msg.header
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position = msg.pose.position
            goal.pose.orientation.w = 1.0  # Default orientation
            
            self.goal_pub.publish(goal)
            self.get_logger().info(
                f"🎯 3D GOAL SET: X={msg.pose.position.x:.2f}, "
                f"Y={msg.pose.position.y:.2f}, Z={msg.pose.position.z:.2f}"
            )
        else:
            self.get_logger().warn(
                f"Clicked point frame '{msg.header.frame_id}' not supported. "
                f"Please set Fixed Frame to 'world' in RViz Global Options."
            )

    # --- INTERACTIVE MARKER LOGIC ---
    
    def rviz_goal_callback(self, msg):
        """
        Received 2D Goal from RViz. 
        Spawn an Interactive Marker at this location (with current Z) to allow adjustment.
        """
        if self.latest_odom_enu is None:
            self.get_logger().warn("Cannot set goal: No odometry yet")
            return

        # Use current drone height as default Z for the marker
        current_z = self.latest_odom_enu['pos'][2]
        
        # Create marker at (msg.x, msg.y, current_z)
        self.server.clear()
        self.pending_goal_pose = msg
        self.pending_goal_pose.pose.position.z = float(current_z)
        
        self.make_interactive_marker(self.pending_goal_pose.pose)
        self.server.applyChanges()
        
        self.get_logger().info(
            f"⚓ 2D Goal Received. Marker spawned at Z={current_z:.2f}m.\n"
            f"   -> Drag the ARROW to adjust height.\n"
            f"   -> Right-click 'EXECUTE GOAL' to fly."
        )

    def make_interactive_marker(self, pose):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "world"
        int_marker.pose = pose
        int_marker.scale = 1.0
        int_marker.name = "goal_marker"
        int_marker.description = "Right-Click to Fly"

        # 1. Z-Axis Control (Arrow)
        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 1.0 # Rotate to point Z up
        control.orientation.z = 0.0
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)
        
        # 2. Menu Control (Box)
        control = InteractiveMarkerControl()
        control.interaction_mode = InteractiveMarkerControl.MENU
        control.always_visible = True
        
        # Visual Box
        marker = InteractiveMarkerControl()
        marker.always_visible = True
        # (Optional: Add a box marker here if needed, but the arrow is usually enough)
        
        int_marker.controls.append(control)

        self.server.insert(int_marker, feedback_callback=self.process_marker_feedback)
        self.menu_handler.apply(self.server, int_marker.name)

    def process_marker_feedback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            # User is dragging
            self.pending_goal_pose.pose = feedback.pose
            # self.get_logger().info(f"Adjusting Z: {feedback.pose.position.z:.2f}", throttle_duration_sec=0.5)
            
        elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            # User clicked "EXECUTE GOAL"
            if self.pending_goal_pose:
                self.pending_goal_pose.header.stamp = self.get_clock().now().to_msg()
                self.goal_pub.publish(self.pending_goal_pose)
                self.get_logger().info(
                    f"🚀 EXECUTING GOAL: X={self.pending_goal_pose.pose.position.x:.2f}, "
                    f"Y={self.pending_goal_pose.pose.position.y:.2f}, "
                    f"Z={self.pending_goal_pose.pose.position.z:.2f}"
                )
                self.server.clear()
                self.server.applyChanges()


    # =========================================================================
    # LOGIC: CONTROL & COMMAND
    # =========================================================================

    def cmd_callback(self, msg):
        """Nhận lệnh từ Ego Planner (ENU Relative) -> Gửi PX4 (NED Absolute)"""
        if self.initial_pos_ned is None:
            return
            
        self.last_cmd_time = self.get_clock().now()
        
        # 1. Geofence Check (Safety)
        # Nếu Planner yêu cầu bay quá xa gốc tọa độ -> Clamp lại
        target_x = max(-self.GEO_FENCE_XY, min(self.GEO_FENCE_XY, msg.position.x))
        target_y = max(-self.GEO_FENCE_XY, min(self.GEO_FENCE_XY, msg.position.y))
        target_z = max(self.GEO_FENCE_Z_MIN, min(self.GEO_FENCE_Z_MAX, msg.position.z))
        
        # 2. Convert ENU (Relative) -> NED (Relative)
        # x_ned = y_enu, y_ned = x_enu, z_ned = -z_enu
        req_ned_rel = np.array([target_y, target_x, -target_z])
        
        # 3. Cộng với Gốc tọa độ -> NED (Absolute for PX4)
        req_ned_abs = req_ned_rel + self.initial_pos_ned
        
        # 5. Yaw Handling
        # msg.yaw là ENU (Angle from East, CCW)
        # PX4 cần NED (Angle from North, CW)
        px4_yaw = -msg.yaw + (math.pi / 2.0)

        # --- SAFETY TURN CHECK (IMPROVED) ---
        # Vấn đề: msg.yaw thay đổi từ từ nên logic cũ không bắt được.
        # Giải pháp: So sánh hướng mũi drone (Current Yaw) với Hướng di chuyển (Velocity Vector).
        # Nếu đang bay lùi (góc lệch > 90 độ), ép drone đứng yên và xoay theo hướng di chuyển.
        
        if self.current_att_q is not None:
            # 1. Tính hướng di chuyển mong muốn (Velocity Heading in ENU)
            vel_x_enu = msg.velocity.x
            vel_y_enu = msg.velocity.y
            speed_sq = vel_x_enu**2 + vel_y_enu**2
            
            # Chỉ check nếu vận tốc đủ lớn (> 0.1 m/s)
            if speed_sq > 0.01:
                # Heading của vận tốc (ENU)
                vel_heading_enu = math.atan2(vel_y_enu, vel_x_enu)
                
                # Convert sang NED để so sánh với current_yaw_ned
                # ENU -> NED: yaw_ned = -yaw_enu + pi/2
                vel_heading_ned = -vel_heading_enu + (math.pi / 2.0)
                
                # Lấy góc hiện tại
                _, _, current_yaw_ned = self.euler_from_quaternion(self.current_att_q)
                
                # Tính độ lệch
                heading_diff = vel_heading_ned - current_yaw_ned
                
                # Normalize [-pi, pi]
                while heading_diff > math.pi: heading_diff -= 2*math.pi
                while heading_diff < -math.pi: heading_diff += 2*math.pi
                
                # Nếu lệch quá 90 độ (~1.57 rad) -> Đang bay lùi/ngang
                if abs(heading_diff) > 1.0:
                    # self.get_logger().warn(f"🛑 SAFETY TURN: Moving Backwards (Diff {math.degrees(heading_diff):.0f}°). Rotating...", throttle_duration_sec=1.0)
                    
                    # 1. Override Position: Gửi NaN cho X, Y để PX4 kích hoạt chế độ "Velocity Control" (Phanh)
                    # Giữ nguyên Z (độ cao) hiện tại
                    req_ned_abs = np.array([float('nan'), float('nan'), self.current_pos_ned_abs[2]])
                    
                    # 2. Override Velocity: = 0 (Phanh gấp)
                    v_x_ned = 0.0
                    v_y_ned = 0.0
                    v_z_ned = 0.0
                    
                    # 3. Override Yaw: Ép xoay theo hướng vận tốc
                    px4_yaw = vel_heading_ned

        # -------------------------

        # 6. Send to PX4 based on Control Mode
        # 5. Send to PX4 based on Control Mode
        # Calculate Velocity NED (Feedforward) for smoother tracking
        v_x_ned = msg.velocity.y
        v_y_ned = msg.velocity.x
        v_z_ned = -msg.velocity.z

        if self.control_mode == 'VELOCITY':
            self.publish_trajectory_setpoint(
                x=float('nan'), y=float('nan'), z=float('nan'), 
                yaw=px4_yaw,
                vx=v_x_ned, vy=v_y_ned, vz=v_z_ned
            )
        else:
            # POSITION MODE (Default)
            # FIX: Send Position + Velocity Feedforward
            self.publish_trajectory_setpoint(
                req_ned_abs[0], req_ned_abs[1], req_ned_abs[2], px4_yaw,
                vx=v_x_ned, vy=v_y_ned, vz=v_z_ned
            )

    def mission_loop(self):
        # Vòng lặp quản lý trạng thái bay (Init -> Takeoff -> Handover to Planner)
        if self.initial_pos_ned is None:
            return

        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9

        if self.state == "INIT":
            if elapsed > 1.0: # Wait a bit for data to stabilize
                # Check altitude (NED Z is negative up)
                altitude = -self.current_pos_ned_abs[2]
                
                # Logic: Always aim for 1.5m relative to start
                # Note: self.takeoff_height is used as the target height
                self.takeoff_height = 1.5 

                if altitude < 0.5: # On Ground -> Standard Takeoff
                    self.get_logger().info(f"GROUND DETECTED (Alt={altitude:.2f}m). Taking off to {self.takeoff_height}m.")
                    self.state = "ARMING"
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
                    self.arm()
                else: # In Air -> Adjust Height to 1.5m
                    self.get_logger().info(f"AIR DETECTED (Alt={altitude:.2f}m). Adjusting to {self.takeoff_height}m...")
                    self.state = "STABILIZE"
                    self.stabilize_start_time = now
                    # Ensure we are in Offboard mode
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
                
        elif self.state == "ARMING":
            if elapsed > 5.0:
                self.state = "TAKEOFF"
                self.get_logger().info(f"Taking off...")
        
        elif self.state == "TAKEOFF":
            # Bay lên độ cao tương đối so với mặt đất (Target = Initial Z - 1.5m)
            target_z_abs = self.initial_pos_ned[2] - self.takeoff_height
            self.publish_trajectory_setpoint(
                self.initial_pos_ned[0], self.initial_pos_ned[1], target_z_abs, 0.0
            )
            
            # Sau 10s, chuyển quyền cho Ego Planner
            if elapsed > 15.0:
                self.state = "MISSION"
                self.get_logger().info(f">>> MISSION START: HANDOVER TO EGO PLANNER (Mode: {self.control_mode}) <<<")

        elif self.state == "STABILIZE":
             # Adjust to 1.5m height (Initial Z - 1.5m)
             target_z_abs = self.initial_pos_ned[2] - self.takeoff_height
             
             self.publish_trajectory_setpoint(
                self.initial_pos_ned[0], self.initial_pos_ned[1], target_z_abs, 0.0
             )
             
             if (now - self.stabilize_start_time).nanoseconds / 1e9 > 4.0: # Wait 4s to stabilize
                 self.state = "MISSION"
                 self.get_logger().info(f">>> STABILIZED AT 1.5m. HANDOVER TO EGO PLANNER (Mode: {self.control_mode}) <<<")

        elif self.state == "MISSION":
            # Watchdog: Nếu Planner im lặng quá 0.5s -> Giữ vị trí (Hold)
            time_since_last_cmd = (now - self.last_cmd_time).nanoseconds / 1e9
            if time_since_last_cmd > 0.75:
                # Gửi lệnh Hold tại vị trí hiện tại (Absolute)
                # (Đơn giản hóa: Gửi velocity = 0 hoặc giữ pos hiện tại)
                # Ở đây ta giữ Pos hiện tại
                if self.current_pos_ned_abs is not None:
                     self.publish_trajectory_setpoint(
                        self.current_pos_ned_abs[0], 
                        self.current_pos_ned_abs[1], 
                        self.current_pos_ned_abs[2], 
                        float('nan')
                    )

    def log_system_resources(self):
        try:
            # CPU & RAM
            cpu_pct = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            
            # GPU (NVIDIA)
            gpu_str = ""
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                if result.returncode == 0:
                    gpu_util, gpu_mem = result.stdout.strip().split(',')
                    gpu_str = f" | GPU: {gpu_util}% (Mem: {gpu_mem}MB)"
            except:
                pass # No nvidia-smi or error
                
            self.get_logger().info(
                f" CPU: {cpu_pct}% | RAM: {mem.percent}%{gpu_str}", 
                throttle_duration_sec=1.0
            )
        except Exception:
            pass

    # =========================================================================
    # HELPERS & UTILS
    # =========================================================================
    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        # Logic: Only use Velocity Mode when in MISSION phase and configured as VELOCITY
        if self.state == "MISSION" and self.control_mode == 'VELOCITY':
            msg.position = False
            msg.velocity = True
        else:
            # Other phases (Takeoff, Stabilize) or POSITION config -> Use Position Control
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

def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()