#!/usr/bin/env python3
"""
bridge_production_v3.py — Production-grade bridge between EgoPlanner and PX4.

[V3 FIXES vs V2]:
  1. _ned_att_to_enu: Dùng scipy Rotation chuẩn cho NED→ENU, loại bỏ công thức
     Hamilton thủ công dễ gây sign-flip khi drone quay nhanh.
  2. _sanitize_quaternion_ned: Nới lỏng identity-reject, chỉ reject trước khi
     có bất kỳ quaternion hợp lệ nào từ arm. Tránh drop dữ liệu hợp lệ.
  3. _depth_callback: Guard thêm threshold timestamp để tránh publish depth
     với odom cũ > 200ms (trước là 500ms — quá lỏng).

[V2 FEATURES giữ nguyên]:
  1. Fixed 20Hz Control Loop: Đảm bảo PX4 không bao giờ rớt Offboard mode.
  2. Hysteresis Watchdog: Chống ping-pong giật cục khi Ego bị trễ do tải nặng.
  3. VSLAM Timeout: Tự động phanh khẩn cấp nếu mất dấu Odometry quá 0.3s.
  4. Attitude Filter: Chặn quaternion rác từ EKF2 khi khởi tạo.
"""

import rclpy
import math
import numpy as np
import cv2
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from collections import deque
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from std_msgs.msg import String, Bool

# PX4 Messages
from px4_msgs.msg import (
    VehicleOdometry, VehicleAttitude, VehicleStatus,
    TrajectorySetpoint, OffboardControlMode, VehicleCommand
)

# Ego Planner
from quadrotor_msgs.msg import PositionCommand

# TF
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

# Interactive Markers
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler

import psutil


# =============================================================================
# STATE CONSTANTS
# =============================================================================
class State:
    WAITING_FOR_PX4      = "WAITING_FOR_PX4"
    WAITING_FOR_OFFBOARD = "WAITING_FOR_OFFBOARD"
    TAKEOFF              = "TAKEOFF"
    INAIR_HOLD           = "INAIR_HOLD"
    MAP_WARMING          = "MAP_WARMING"
    READY                = "READY"
    EXECUTING_GOAL       = "EXECUTING_GOAL"
    WATCHDOG_HOLD        = "WATCHDOG_HOLD"


# =============================================================================
# MAIN NODE
# =============================================================================
class BridgeProduction(Node):

    def __init__(self):
        super().__init__('bridge_production')

        # ------------------------------------------------------------------ #
        # PARAMETERS
        # ------------------------------------------------------------------ #
        self._declare_all_params()
        self._load_all_params()

        self.get_logger().info(
            f"🚀 BRIDGE PRODUCTION V2 | launch_mode={self.launch_mode} "
            f"| takeoff_height={self.takeoff_height}m"
        )

        # ------------------------------------------------------------------ #
        # STATE MACHINE
        # ------------------------------------------------------------------ #
        self.state: str = State.WAITING_FOR_PX4
        self.offboard_active: bool = False
        self.is_armed: bool = False
        self.last_offboard_cmd_time = None

        # ------------------------------------------------------------------ #
        # COORDINATE TRACKING
        # ------------------------------------------------------------------ #
        self.initial_pos_ned: np.ndarray | None = None
        self.current_pos_ned_abs: np.ndarray | None = None
        self.current_vel_ned: np.ndarray | None = None
        self.current_att_q: list | None = None
        self.last_valid_q_ned: list | None = None  # [FIX V2] Fallback thái độ

        self.hold_pos_ned_abs: np.ndarray | None = None
        self.takeoff_target_ned: np.ndarray | None = None
        self.takeoff_yaw_ned: float = 0.0    # [FIX] Giữ yaw tại thời điểm ARM, không xoay khi takeoff

        self.stabilize_start_time = None
        self.inair_hold_start_time = None

        # ------------------------------------------------------------------ #
        # TARGET TRACKING (For 20Hz Loop) [FIX V2]
        # ------------------------------------------------------------------ #
        self.current_target_ned: np.ndarray | None = None
        self.current_target_yaw: float = 0.0
        self.current_target_vel: list = [float('nan'), float('nan'), float('nan')]

        # ------------------------------------------------------------------ #
        # MAP WARMUP & WATCHDOG
        # ------------------------------------------------------------------ #
        self.depth_frame_count: int = 0
        self.last_ego_cmd_time = self.get_clock().now()
        self.watchdog_recovery_count: int = 0  # [FIX V2] Hysteresis logic

        # ------------------------------------------------------------------ #
        # ODOM BUFFER & SYNC
        # ------------------------------------------------------------------ #
        self.odom_buffer = deque(maxlen=100)
        self.cv_bridge = CvBridge()
        self.pending_goal_pose: PoseStamped | None = None
        
        self.latest_odom_data: dict | None = None
        self.latest_odom_time_sec: float = 0.0

        # ------------------------------------------------------------------ #
        # PUBLISHERS & SUBSCRIBERS
        # ------------------------------------------------------------------ #
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_tf()

        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_px4_status = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscribers
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self._pos_callback, qos_px4)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self._att_callback, qos_px4)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self._status_callback, qos_px4_status)
        self.create_subscription(Image, '/stereo/depth_map', self._depth_callback, 10)
        self.create_subscription(PositionCommand, f'/drone_{self.drone_id}_planning/pos_cmd', self._ego_cmd_callback, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self._rviz_goal_callback, 10)
        self.create_subscription(PoseStamped, '/clicked_point', self._clicked_point_callback, 10)
        self.create_subscription(Bool, '/bridge/force_ready', self._force_ready_callback, 10)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, f'/drone_{self.drone_id}_visual_slam/odom', 10)
        self.cam_pose_pub = self.create_publisher(PoseStamped, f'/drone_{self.drone_id}_pcl_render_node/camera_pose', 10)
        self.depth_pub = self.create_publisher(Image, f'/drone_{self.drone_id}_pcl_render_node/depth', 10)
        self.cam_info_pub = self.create_publisher(CameraInfo, f'/drone_{self.drone_id}_pcl_render_node/camera_info', 10)
        self.goal_pub = self.create_publisher(PoseStamped, f'/drone_{self.drone_id}_ego_planner_node/goal', 10)
        self.local_map_pub = self.create_publisher(Marker, f'/drone_{self.drone_id}_ego_planner_node/local_map_bound', 10)

        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_px4)
        self.offboard_ctrl_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_px4)
        self.vehicle_cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.state_pub = self.create_publisher(String, '/bridge/state', 10)

        # Interactive Markers
        self.marker_server = InteractiveMarkerServer(self, "bridge_goal_server")
        self.menu_handler = MenuHandler()
        self.menu_handler.insert("✈ Execute Goal", callback=self._marker_feedback)

        # ------------------------------------------------------------------ #
        # TIMERS
        # ------------------------------------------------------------------ #
        # [FIX V2] Control Loop chính xác 20Hz cho cả Heartbeat và Trajectory
        self.create_timer(0.05, self._control_loop_20hz)
        # Mission logic loop 10Hz
        self.create_timer(0.1, self._mission_loop)
        self.create_timer(10.0, self._log_resources)

        rclpy.get_default_context().on_shutdown(self._on_shutdown)

    # ======================================================================= #
    # PARAMETERS (Giữ nguyên)
    # ======================================================================= #
    def _declare_all_params(self):
        self.declare_parameter('drone_id', 0)
        self.declare_parameter('takeoff_height', 0.5)
        self.declare_parameter('camera_offset_x', 0.1)
        self.declare_parameter('launch_mode', 'ground')
        self.declare_parameter('camera.width', 256)
        self.declare_parameter('camera.height', 144)
        self.declare_parameter('camera.fx', 187.8)
        self.declare_parameter('camera.fy', 187.8)
        self.declare_parameter('camera.cx', 128.0)
        self.declare_parameter('camera.cy', 72.0)
        self.declare_parameter('ground_altitude_threshold', 0.5)
        self.declare_parameter('map_warmup_frames', 40)
        self.declare_parameter('watchdog_timeout_sec', 0.5)
        self.declare_parameter('stabilize_hold_sec', 3.0)
        self.declare_parameter('inair_hold_sec', 2.0)
        self.declare_parameter('geofence_xy', 30.0)
        self.declare_parameter('geofence_z_max', 5.0)
        self.declare_parameter('geofence_z_min', 0.3)
        self.declare_parameter('depth_filter_min', 0.2)
        self.declare_parameter('depth_filter_max', 6.0)

    def _load_all_params(self):
        self.drone_id           = self.get_parameter('drone_id').value
        self.takeoff_height     = self.get_parameter('takeoff_height').value
        self.camera_offset_x    = self.get_parameter('camera_offset_x').value
        self.launch_mode        = self.get_parameter('launch_mode').value.lower()
        self.cam_width          = self.get_parameter('camera.width').value
        self.cam_height         = self.get_parameter('camera.height').value
        self.cam_fx             = self.get_parameter('camera.fx').value
        self.cam_fy             = self.get_parameter('camera.fy').value
        self.cam_cx             = self.get_parameter('camera.cx').value
        self.cam_cy             = self.get_parameter('camera.cy').value
        self.ground_alt_thresh  = self.get_parameter('ground_altitude_threshold').value
        self.map_warmup_frames  = self.get_parameter('map_warmup_frames').value
        self.watchdog_timeout   = self.get_parameter('watchdog_timeout_sec').value
        self.stabilize_hold_sec = self.get_parameter('stabilize_hold_sec').value
        self.inair_hold_sec     = self.get_parameter('inair_hold_sec').value
        self.geofence_xy        = self.get_parameter('geofence_xy').value
        self.geofence_z_max     = self.get_parameter('geofence_z_max').value
        self.geofence_z_min     = self.get_parameter('geofence_z_min').value
        self.depth_min          = self.get_parameter('depth_filter_min').value
        self.depth_max          = self.get_parameter('depth_filter_max').value

    # ======================================================================= #
    # PX4 CALLBACKS
    # ======================================================================= #
    def _sanitize_quaternion_ned(self, q_raw) -> list | None:
        """Validate and normalize quaternion; reject obvious EKF bootstrap garbage.
        
        [FIX V3] Nới lỏng điều kiện reject identity:
        - Chỉ reject [1,0,0,0] TRƯỚC khi arm VÀ chưa có quaternion hợp lệ nào.
        - Sau khi arm hoặc đã có last_valid_q_ned thì chấp nhận gần-identity
          vì drone có thể đang hover rất thẳng.
        """
        q = np.array(q_raw, dtype=np.float64)
        if q.shape != (4,):
            return None
        if np.any(np.isnan(q)) or np.any(np.isinf(q)):
            return None

        norm = float(np.linalg.norm(q))
        if norm < 1e-6:
            return None

        q = q / norm

        # Chỉ reject identity tuyệt đối (rác EKF) trước khi có bất kỳ dữ liệu tốt nào
        # VÀ chưa arm. Sau khi đã arm hoặc đã có stream tốt → chấp nhận mọi q bình thường.
        if self.last_valid_q_ned is None and not self.is_armed:
            # Exact identity [1,0,0,0] — đây chắc chắn là EKF chưa hội tụ
            if abs(q[0] - 1.0) < 1e-6 and abs(q[1]) < 1e-6 and abs(q[2]) < 1e-6 and abs(q[3]) < 1e-6:
                return None

        return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]

    def _reset_motion_targets(self):
        self.current_target_ned = None
        self.current_target_yaw = 0.0
        self.current_target_vel = [float('nan'), float('nan'), float('nan')]
        self.pending_goal_pose = None

    def _enter_watchdog_hold(self, reason: str):
        self.watchdog_recovery_count = 0
        if self.current_pos_ned_abs is not None:
            self.hold_pos_ned_abs = self.current_pos_ned_abs.copy()
        if self.state != State.WATCHDOG_HOLD:
            self.get_logger().warn(f"🛑 {reason} → WATCHDOG_HOLD.")
            self._set_state(State.WATCHDOG_HOLD)

    def _pos_callback(self, msg: VehicleOdometry):
        pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
        if np.any(np.isnan(pos)):
            return

        self.current_pos_ned_abs = pos
        self.current_vel_ned = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]])

        # Ưu tiên VehicleAttitude nhưng luôn validate trước khi dùng.
        q_ned = None
        if self.current_att_q is not None:
            q_ned = self._sanitize_quaternion_ned(self.current_att_q)
        if q_ned is None:
            q_ned = self._sanitize_quaternion_ned([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])
        if q_ned is None:
            if self.last_valid_q_ned is None:
                return
            q_ned = self.last_valid_q_ned
        else:
            self.last_valid_q_ned = q_ned

        # Ghi nhận Origin duy nhất 1 lần
        if self.initial_pos_ned is None:
            self.initial_pos_ned = pos.copy()
            self.get_logger().info(f"📍 ORIGIN LATCHED (NED): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        pos_rel_ned = pos - self.initial_pos_ned
        pos_enu = self._ned_to_enu(pos_rel_ned)
        vel_enu = self._ned_to_enu(self.current_vel_ned)
        q_enu   = self._ned_att_to_enu(q_ned)

        now = self.get_clock().now()
        odom_data = {'stamp': now, 'pos': pos_enu, 'vel': vel_enu, 'quat': q_enu}
        self.odom_buffer.append(odom_data)
        
        # Cập nhật time cho VSLAM Watchdog [FIX V2]
        self.latest_odom_data = odom_data
        self.latest_odom_time_sec = now.nanoseconds / 1e9

        self._publish_odom_and_tf(now, pos_enu, vel_enu, q_enu)
        self._publish_local_map_marker(now, pos_enu)

        if self.state == State.WAITING_FOR_PX4:
            self._set_state(State.WAITING_FOR_OFFBOARD)

    def _att_callback(self, msg: VehicleAttitude):
        q = self._sanitize_quaternion_ned([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])
        if q is not None:
            self.current_att_q = q
        else:
            self.get_logger().warn("⚠️ Dropped invalid VehicleAttitude quaternion.", throttle_duration_sec=1.0)

    def _status_callback(self, msg: VehicleStatus):
        was_offboard = self.offboard_active
        self.offboard_active = (msg.nav_state == 14)

        prev_armed = self.is_armed
        self.is_armed = (msg.arming_state == 2)

        if not prev_armed and self.is_armed:
            self.get_logger().info("🔒 ARMED detected — will auto-switch to Offboard.")
            # [FIX V2] Đã xoá đoạn reset origin gây teleport TF tại đây.
        elif prev_armed and not self.is_armed:
            self.get_logger().warn("🔓 DISARMED.")

        if was_offboard and not self.offboard_active:
            self.get_logger().warn("⚠️ RC OVERRIDE — Bridge commands FROZEN.")
            if self.state not in (State.WAITING_FOR_PX4, State.WAITING_FOR_OFFBOARD):
                self._set_state(State.WAITING_FOR_OFFBOARD)

        # Only reset state to WAITING_FOR_OFFBOARD if actually disarmed from a previously armed state
        if prev_armed and not self.is_armed and self.state not in (State.WAITING_FOR_PX4, State.WAITING_FOR_OFFBOARD):
            self.get_logger().warn("🔴 DISARMED mid-flight. Resetting state.")
            self._set_state(State.WAITING_FOR_OFFBOARD)

    def _force_ready_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().warn("🚀 FORCE READY ACTIVATED! Bypassing takeoff checks.")
            self._set_state(State.READY)

    # ======================================================================= #
    # DEPTH CALLBACK (Giữ nguyên)
    # ======================================================================= #
    def _depth_callback(self, msg: Image):
        if self.latest_odom_data is None:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        odom_age = now_sec - self.latest_odom_time_sec
        # [FIX V3] Giảm threshold từ 500ms → 200ms để tránh publish depth
        # với pose quá cũ gây ra map artifact (điểm vân vào vị trí sai)
        if odom_age > 0.2:
            if odom_age > 0.5:  # chỉ warn nếu quá lâu để tránh spam
                self.get_logger().warn(
                    f"[depth_callback] Odom too old ({odom_age*1000:.0f}ms), skipping depth frame.",
                    throttle_duration_sec=1.0
                )
            return

        odom_data = self.latest_odom_data
        sync_stamp = odom_data['stamp']

        cam_pose = PoseStamped()
        cam_pose.header.stamp = sync_stamp.to_msg()
        cam_pose.header.frame_id = "world"
        cam_pose.pose.position.x  = float(odom_data['pos'][0])
        cam_pose.pose.position.y  = float(odom_data['pos'][1])
        cam_pose.pose.position.z  = float(odom_data['pos'][2])
        cam_pose.pose.orientation.x = float(odom_data['quat'][0])
        cam_pose.pose.orientation.y = float(odom_data['quat'][1])
        cam_pose.pose.orientation.z = float(odom_data['quat'][2])
        cam_pose.pose.orientation.w = float(odom_data['quat'][3])
        self.cam_pose_pub.publish(cam_pose)

        cam_info = CameraInfo()
        cam_info.header.stamp    = sync_stamp.to_msg()
        cam_info.header.frame_id = "camera_link"
        cam_info.width  = self.cam_width
        cam_info.height = self.cam_height
        cam_info.k = [self.cam_fx, 0.0, self.cam_cx, 0.0, self.cam_fy, self.cam_cy, 0.0, 0.0, 1.0]
        cam_info.p = [self.cam_fx, 0.0, self.cam_cx, 0.0, 0.0, self.cam_fy, self.cam_cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.cam_info_pub.publish(cam_info)

        try:
            cv_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_arr = np.array(cv_depth, dtype=np.float32)
            depth_arr[np.isnan(depth_arr)] = 0.0
            depth_arr[np.isinf(depth_arr)] = 0.0
            depth_arr[depth_arr < self.depth_min] = 0.0
            depth_arr[depth_arr > self.depth_max] = 0.0
            filtered_msg = self.cv_bridge.cv2_to_imgmsg(depth_arr, encoding=msg.encoding)
            filtered_msg.header.stamp    = sync_stamp.to_msg()
            filtered_msg.header.frame_id = "camera_link"
        except Exception as e:
            filtered_msg = msg
            filtered_msg.header.stamp = sync_stamp.to_msg()

        self.depth_pub.publish(filtered_msg)

        if self.state == State.MAP_WARMING:
            self.depth_frame_count += 1

    # ======================================================================= #
    # EGO PLANNER COMMAND CALLBACK
    # ======================================================================= #
    def _ego_cmd_callback(self, msg: PositionCommand):
        """ [FIX V2] Chỉ cập nhật Target biến nội bộ, không Publish ở đây """
        if self.state not in (State.EXECUTING_GOAL, State.WATCHDOG_HOLD):
            return
        if self.initial_pos_ned is None:
            return

        self.last_ego_cmd_time = self.get_clock().now()

        tx = float(np.clip(msg.position.x, -self.geofence_xy, self.geofence_xy))
        ty = float(np.clip(msg.position.y, -self.geofence_xy, self.geofence_xy))
        tz = float(np.clip(msg.position.z, self.geofence_z_min, self.geofence_z_max))

        req_ned_rel = np.array([ty, tx, -tz])
        self.current_target_ned = req_ned_rel + self.initial_pos_ned
        self.current_target_yaw = -msg.yaw + (math.pi / 2.0)
        self.current_target_vel = [msg.velocity.y, msg.velocity.x, -msg.velocity.z]

    # ======================================================================= #
    # GOAL CALLBACKS (Giữ nguyên)
    # ======================================================================= #
    def _rviz_goal_callback(self, msg: PoseStamped):
        if self.state not in (State.READY, State.EXECUTING_GOAL):
            self.get_logger().warn(
                f"⚠️  Goal DROPPED — bridge not ready yet. State=[{self.state}]. "
                f"Wait until state=READY before sending goals.",
                throttle_duration_sec=2.0
            )
            return
        if not self.odom_buffer: return
        
        current_z = self.odom_buffer[-1]['pos'][2]
        self.pending_goal_pose = msg
        self.pending_goal_pose.pose.position.z = float(current_z)

        self.marker_server.clear()
        self._make_interactive_marker(self.pending_goal_pose.pose)
        self.marker_server.applyChanges()
        self.get_logger().info(f"⚓ 2D Goal received. Z={current_z:.2f}m. Right-click → 'Execute Goal'")

    def _clicked_point_callback(self, msg: PoseStamped):
        if self.state not in (State.READY, State.EXECUTING_GOAL):
            self.get_logger().warn(
                f"⚠️  ClickedPoint DROPPED — bridge not ready yet. State=[{self.state}].",
                throttle_duration_sec=2.0
            )
            return
        
        goal = PoseStamped()
        goal.header.stamp    = self.get_clock().now().to_msg()
        goal.header.frame_id = 'world'
        goal.pose.position   = msg.pose.position
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        self._set_state(State.EXECUTING_GOAL)

    def _make_interactive_marker(self, pose):
        im = InteractiveMarker()
        im.header.frame_id = "world"
        im.pose  = pose
        im.scale = 1.0
        im.name  = "goal_marker"
        
        ctrl = InteractiveMarkerControl()
        ctrl.orientation.w, ctrl.orientation.y = 1.0, 1.0
        ctrl.name = "move_z"
        ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        im.controls.append(ctrl)

        menu_ctrl = InteractiveMarkerControl()
        menu_ctrl.interaction_mode = InteractiveMarkerControl.MENU
        menu_ctrl.always_visible   = True
        im.controls.append(menu_ctrl)

        self.marker_server.insert(im, feedback_callback=self._marker_feedback)
        self.menu_handler.apply(self.marker_server, im.name)

    def _marker_feedback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            if self.pending_goal_pose: self.pending_goal_pose.pose = feedback.pose
        elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            if self.pending_goal_pose and self.state in (State.READY, State.EXECUTING_GOAL):
                self.pending_goal_pose.header.stamp = self.get_clock().now().to_msg()
                self.goal_pub.publish(self.pending_goal_pose)
                self._set_state(State.EXECUTING_GOAL)
                self.marker_server.clear()
                self.marker_server.applyChanges()

    # ======================================================================= #
    # [FIX V2] CONTROL LOOP 20HZ (HEARTBEAT + TRAJECTORY)
    # ======================================================================= #
    def _control_loop_20hz(self):
        """ Vòng lặp siêu ưu tiên, bơm máu 20Hz liên tục cho PX4 để giữ Offboard """
        # 1. Heartbeat
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position  = True
        self.offboard_ctrl_pub.publish(msg)

        # Trạng thái pub
        state_msg = String()
        state_msg.data = self.state
        self.state_pub.publish(state_msg)

        if not self.offboard_active:
            return

        now = self.get_clock().now()

        # 2. Ego Watchdog Hysteresis Check
        if self.state == State.EXECUTING_GOAL:
            time_since_cmd = (now - self.last_ego_cmd_time).nanoseconds / 1e9
            if time_since_cmd > self.watchdog_timeout:
                self._enter_watchdog_hold(f"🐕 EGO TIMEOUT ({time_since_cmd:.2f}s)")

        elif self.state == State.WATCHDOG_HOLD:
            time_since_cmd = (now - self.last_ego_cmd_time).nanoseconds / 1e9
            odom_age = ((now.nanoseconds / 1e9) - self.latest_odom_time_sec) if self.latest_odom_time_sec > 0 else float('inf')
            if odom_age < 0.3 and time_since_cmd < 0.1 and self.current_target_ned is not None:
                self.watchdog_recovery_count += 1
                if self.watchdog_recovery_count > 5:  # Phải ổn định 5 frames (0.25s) mới cho chạy
                    self.get_logger().info("✅ Ego Planner stable again. Resuming flight.")
                    self._set_state(State.EXECUTING_GOAL)
            else:
                self.watchdog_recovery_count = 0

        # 3. Publish Trajectory dựa trên State
        if self.state == State.TAKEOFF:
            if self.takeoff_target_ned is not None:
                self._publish_trajectory_setpoint(
                    self.takeoff_target_ned[0], self.takeoff_target_ned[1], self.takeoff_target_ned[2],
                    self.takeoff_yaw_ned  # [FIX] Giữ yaw ARM, không xoay drone khi takeoff
                )
        elif self.state in (State.EXECUTING_GOAL, State.READY):
            if self.current_target_ned is not None:
                self._publish_trajectory_setpoint(
                    self.current_target_ned[0], self.current_target_ned[1], self.current_target_ned[2],
                    self.current_target_yaw, *self.current_target_vel
                )
            else:
                self._hold_current_position()
        elif self.state in (State.INAIR_HOLD, State.MAP_WARMING, State.WATCHDOG_HOLD):
            self._hold_current_position()

    # ======================================================================= #
    # MAIN MISSION LOOP — 10Hz
    # ======================================================================= #
    def _mission_loop(self):
        now = self.get_clock().now()

        # [FIX V2] VSLAM TIMEOUT SUPERVISOR
        if self.latest_odom_time_sec > 0 and self.state not in (State.WAITING_FOR_PX4, State.WAITING_FOR_OFFBOARD):
            odom_age = (now.nanoseconds / 1e9) - self.latest_odom_time_sec
            if odom_age > 0.3:  # 300ms mất Odometry
                self.get_logger().error(f"🚨 VSLAM TIMEOUT ({odom_age:.2f}s)! Forcing HOLD to save drone.", throttle_duration_sec=1.0)
                if self.offboard_active:
                    self._enter_watchdog_hold("VSLAM stream lost")
                return  # Skip logic bên dưới nếu mù

        if self.state == State.WAITING_FOR_OFFBOARD:
            self._handle_waiting_for_offboard()
        elif self.offboard_active:
            if self.state == State.TAKEOFF:
                self._handle_takeoff(now)
            elif self.state == State.INAIR_HOLD:
                self._handle_inair_hold(now)
            elif self.state == State.MAP_WARMING:
                self._handle_map_warming(now)

    def _handle_waiting_for_offboard(self):
        if not self.is_armed: return
        if not self.offboard_active:
            now = self.get_clock().now()
            since_last = ((now - self.last_offboard_cmd_time).nanoseconds / 1e9 if self.last_offboard_cmd_time else 999.0)
            if since_last >= 0.5:
                self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
                self.last_offboard_cmd_time = now
            return

        if self.current_pos_ned_abs is None: return

        alt = self._get_current_altitude_m()
        is_ground = False if self.launch_mode == 'inflight' else (alt < self.ground_alt_thresh)

        if is_ground:
            tx, ty = float(self.current_pos_ned_abs[0]), float(self.current_pos_ned_abs[1])
            tz = float(self.current_pos_ned_abs[2]) - self.takeoff_height
            self.takeoff_target_ned = np.array([tx, ty, tz])
            # [FIX] Capture yaw ngay lúc TAKEOFF để không bị xoay
            self.takeoff_yaw_ned = self._get_current_yaw_ned()
            self.get_logger().info(f"🛫 Takeoff yaw locked = {math.degrees(self.takeoff_yaw_ned):.1f}°")
            self._set_state(State.TAKEOFF)
        else:
            self._set_state(State.INAIR_HOLD)

    def _handle_takeoff(self, now):
        current_alt = self._get_current_altitude_m()
        if abs(current_alt - self.takeoff_height) < 0.15:
            if self.stabilize_start_time is None:
                self.stabilize_start_time = now
        else:
            self.stabilize_start_time = None
        
        if self.stabilize_start_time is not None:
            elapsed = (now - self.stabilize_start_time).nanoseconds / 1e9
            if elapsed >= self.stabilize_hold_sec:
                self.depth_frame_count = 0
                self._set_state(State.MAP_WARMING)

    def _handle_inair_hold(self, now):
        elapsed = (now - self.inair_hold_start_time).nanoseconds / 1e9
        if elapsed >= self.inair_hold_sec:
            self.depth_frame_count = 0
            self._set_state(State.MAP_WARMING)

    def _handle_map_warming(self, now):
        if self.depth_frame_count >= self.map_warmup_frames:
            self._set_state(State.READY)
            self.get_logger().info("✅ MAP READY! Bridge is READY to accept goals!")

    def _get_current_yaw_ned(self) -> float:
        """Lấy góc yaw hiện tại của drone trong NED frame từ quaternion."""
        q = self.last_valid_q_ned
        if q is None:
            return 0.0
        # q = [w, x, y, z] trong PX4 NED convention
        w, x, y, z = q[0], q[1], q[2], q[3]
        # yaw trong NED: atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    # ======================================================================= #
    # PX4 COMMAND HELPERS
    # ======================================================================= #
    def _publish_trajectory_setpoint(self, x, y, z, yaw, vx=float('nan'), vy=float('nan'), vz=float('nan')):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position  = [float(x), float(y), float(z)]
        msg.velocity  = [float(vx), float(vy), float(vz)]
        msg.yaw       = float(yaw)
        self.traj_pub.publish(msg)

    def _publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.timestamp       = int(self.get_clock().now().nanoseconds / 1000)
        msg.command         = command
        msg.param1          = float(params.get('param1', 0.0))
        msg.param2          = float(params.get('param2', 0.0))
        msg.target_system   = 1
        msg.target_component = 1
        msg.from_external   = True
        self.vehicle_cmd_pub.publish(msg)

    def _hold_current_position(self):
        if self.hold_pos_ned_abs is None and self.current_pos_ned_abs is not None:
            self.hold_pos_ned_abs = self.current_pos_ned_abs.copy()
        if self.hold_pos_ned_abs is not None:
            self._publish_trajectory_setpoint(
                self.hold_pos_ned_abs[0], self.hold_pos_ned_abs[1], self.hold_pos_ned_abs[2], float('nan')
            )

    # ======================================================================= #
    # TF & ODOMETRY
    # ======================================================================= #
    def _publish_static_tf(self):
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id  = 'camera_link'
        t.transform.translation.x = float(self.camera_offset_x)
        t.transform.rotation.x, t.transform.rotation.y = -0.5, 0.5
        t.transform.rotation.z, t.transform.rotation.w = -0.5, 0.5
        self.tf_static_broadcaster.sendTransform(t)

    def _publish_odom_and_tf(self, timestamp, pos, vel, q):
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = timestamp.to_msg(), "world", "base_link"
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])

        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = px, py, pz
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y = qx, qy
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = qz, qw
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = vx, vy, vz
        self.odom_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = timestamp.to_msg(), 'world', 'base_link'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = px, py, pz
        t.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def _publish_local_map_marker(self, timestamp, pos_enu):
        m = Marker()
        m.header.stamp, m.header.frame_id = timestamp.to_msg(), "world"
        m.id, m.type, m.action = 998, Marker.CUBE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(pos_enu[0]), float(pos_enu[1]), float(pos_enu[2])
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = 15.0, 15.0, 8.0
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.5, 0.12
        self.local_map_pub.publish(m)

    # ======================================================================= #
    # COORDINATE TRANSFORMS
    # ======================================================================= #
    def _ned_to_enu(self, ned: np.ndarray) -> np.ndarray:
        return np.array([ned[1], ned[0], -ned[2]])

    def _ned_att_to_enu(self, q_ned: list) -> list:
        """[FIX V3] Chuyển đổi quaternion NED → ENU dùng scipy Rotation chuẩn.
        
        Công thức cũ (Hamilton thủ công) có thể gây sign-flip bất thường.
        scipy dùng SLERP-safe convention và đảm bảo continuity tốt hơn.
        
        NED → ENU: rotate by pi around the axis (1/sqrt(2), 1/sqrt(2), 0) in world frame,
        tương đương với nhân bên phải bởi q_correction = [0, sq2, sq2, 0] (xyzw).
        """
        w, x, y, z = q_ned
        # q_ned ở dạng [w, x, y, z] — chuyển sang scipy [x, y, z, w]
        r_ned = Rotation.from_quat([x, y, z, w])

        # Ma trận chuyển NED → ENU:
        #   x_ENU =  y_NED
        #   y_ENU =  x_NED
        #   z_ENU = -z_NED
        # Tương đương với rotate 180° quanh trục (1/sqrt(2), 1/sqrt(2), 0)
        sq2 = 0.70710678118
        r_ned2enu = Rotation.from_quat([sq2, sq2, 0.0, 0.0])  # [x, y, z, w]
        r_enu = r_ned2enu * r_ned

        q_enu_xyzw = r_enu.as_quat()  # returns [x, y, z, w]

        # Đảm bảo continuity (tránh sign-flip giữa các frame liên tiếp)
        q_enu = [float(q_enu_xyzw[0]), float(q_enu_xyzw[1]),
                 float(q_enu_xyzw[2]), float(q_enu_xyzw[3])]
        if hasattr(self, 'last_q_enu'):
            dot = sum(a * b for a, b in zip(q_enu, self.last_q_enu))
            if dot < 0.0:
                q_enu = [-v for v in q_enu]
        self.last_q_enu = q_enu

        return q_enu  # [x, y, z, w] — khớp với geometry_msgs orientation

    def _get_current_altitude_m(self) -> float:
        if self.current_pos_ned_abs is None or self.initial_pos_ned is None: return 0.0
        return -(self.current_pos_ned_abs[2] - self.initial_pos_ned[2])

    def _set_state(self, new_state: str):
        if self.state != new_state:
            self.get_logger().info(f"🔄 State: [{self.state}] → [{new_state}]")
            self.state = new_state

            if new_state in (State.WAITING_FOR_PX4, State.WAITING_FOR_OFFBOARD):
                self.hold_pos_ned_abs = None
                self.takeoff_target_ned = None
                self.stabilize_start_time = None
                self.inair_hold_start_time = None
                self.depth_frame_count = 0
                self.watchdog_recovery_count = 0
                self._reset_motion_targets()
            elif new_state == State.TAKEOFF:
                self.stabilize_start_time = None
                self.hold_pos_ned_abs = None
                self.watchdog_recovery_count = 0
            elif new_state == State.INAIR_HOLD:
                self.inair_hold_start_time = self.get_clock().now()
                self.watchdog_recovery_count = 0
                if self.current_pos_ned_abs is not None:
                    self.hold_pos_ned_abs = self.current_pos_ned_abs.copy()
            elif new_state == State.MAP_WARMING:
                self.depth_frame_count = 0
                if self.hold_pos_ned_abs is None and self.current_pos_ned_abs is not None:
                    self.hold_pos_ned_abs = self.current_pos_ned_abs.copy()
            elif new_state == State.READY:
                self.watchdog_recovery_count = 0
                self._reset_motion_targets()
                self.hold_pos_ned_abs = None
            elif new_state == State.EXECUTING_GOAL:
                self.watchdog_recovery_count = 0
                self.hold_pos_ned_abs = None
            elif new_state == State.WATCHDOG_HOLD:
                self.watchdog_recovery_count = 0
                if self.current_pos_ned_abs is not None:
                    self.hold_pos_ned_abs = self.current_pos_ned_abs.copy()

    # ======================================================================= #
    # MISCELLANEOUS
    # ======================================================================= #
    def _log_resources(self):
        try:
            self.get_logger().info(f"💻 CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}% | State: [{self.state}]")
        except: pass

    def _on_shutdown(self):
        try:
            self.get_logger().warn("🛑 Bridge shutting down → forcing HOLD mode.")
            if rclpy.ok():
                self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=3.0)
        except: pass

def main(args=None):
    rclpy.init(args=args)
    node = BridgeProduction()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok(): rclpy.shutdown()
        except: pass

if __name__ == '__main__':
    main()
