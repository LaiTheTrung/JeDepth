import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude, TrajectorySetpoint, OffboardControlMode, VehicleCommand, VehicleStatus
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from quadrotor_msgs.msg import PositionCommand
import numpy as np
from collections import deque
import math
import time
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
import cv2
from cv_bridge import CvBridge

class BridgeNodeVer2(Node):
    def __init__(self):
        super().__init__('bridge_node_ver2')

        # --- PARAMETERS ---
        self.declare_parameter('drone_id', 0)
        self.declare_parameter('takeoff_height', 1.0)
        self.declare_parameter('camera_offset_x', 0.1)

        # Camera Intrinsics - ✅ SỬ DỤNG GIÁ TRỊ CHÍNH XÁC
        self.declare_parameter('camera.width', 256)
        self.declare_parameter('camera.height', 144)
        self.declare_parameter('camera.fx', 187.8)  # ✅ Fixed
        self.declare_parameter('camera.fy', 187.8)  # ✅ Fixed
        self.declare_parameter('camera.cx', 128.0)
        self.declare_parameter('camera.cy', 72.0)

        # Load Params
        self.drone_id = self.get_parameter('drone_id').value
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.camera_offset_x = self.get_parameter('camera_offset_x').value
        self.cam_width = self.get_parameter('camera.width').value
        self.cam_height = self.get_parameter('camera.height').value
        self.fx = self.get_parameter('camera.fx').value
        self.fy = self.get_parameter('camera.fy').value
        self.cx = self.get_parameter('camera.cx').value
        self.cy = self.get_parameter('camera.cy').value

        self.get_logger().warn(f"🚀 BRIDGE REAL VER 2 STARTED | SAFETY BYPASSED!")

        # --- DEPTH FILTER ---
        self.min_dist = 0.5
        self.max_dist = 7.0  # ✅ Khớp với ego max_ray_length
        self.cv_bridge = CvBridge()

        # --- STATE & COORDINATES ---
        self.start_time = self.get_clock().now()
        
        # Bypassing the requirement for PX4 Local Position to be valid initially
        # We will create a fake origin immediately so Ego Planner gets Odom
        self.has_initial_pos = False
        self.initial_pos_ned = np.array([0.0, 0.0, 0.0])
        self.current_pos_ned_abs = np.array([0.0, 0.0, 0.0])
        self.current_att_q = [1.0, 0.0, 0.0, 0.0]  # W X Y Z Default (Facing North)

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
        self.depth_sub = self.create_subscription(Image, '/stereo/depth_map', self.depth_callback, 10)
        
        # We will still subscribe to PX4 topics to update position if it ever arrives
        self.pos_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.pos_callback, qos_best_effort)
        self.att_sub = self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.att_callback, qos_best_effort)
        self.status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, qos_best_effort)
        
        # Ego Planner Command
        self.cmd_sub = self.create_subscription(PositionCommand, f'/drone_{self.drone_id}_planning/pos_cmd', self.cmd_callback, 10)

        # --- PUBLISHERS ---
        # ✅ CRITICAL: Đảm bảo topic name CHÍNH XÁC (bỏ underscore thừa)
        self.odom_pub = self.create_publisher(Odometry, f'/drone_{self.drone_id}_visual_slam/odom', 10)
        self.cam_pose_pub = self.create_publisher(PoseStamped, f'/drone_{self.drone_id}_pcl_render_node/camera_pose', 10)
        self.depth_pub = self.create_publisher(Image, f'/drone_{self.drone_id}_pcl_render_node/depth', 10)
        self.cam_info_pub = self.create_publisher(CameraInfo, f'/drone_{self.drone_id}_pcl_render_node/camera_info', 10)
        
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_best_effort)
        self.offboard_ctrl_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_best_effort)

        # --- TIMERS ---
        self.create_timer(0.05, self.publish_offboard_heartbeat) # 20Hz
        
        # THAY VÌ CHỜ PX4 PUBLISH ODOM, CHÚNG TA SẼ TỰ ĐỘNG PUBLISH ODOM LIÊN TỤC 30HZ CHO EGO PLANNER
        self.create_timer(0.033, self.fake_odom_loop)

        # RViz 2D Goal
        self.goal_pub = self.create_publisher(PoseStamped, f'/drone_{self.drone_id}_ego_planner_node/goal', 10)
        self.rviz_goal_sub = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.rviz_goal_callback, 10)
        self.local_map_pub = self.create_publisher(Marker, f'/drone_{self.drone_id}_ego_planner_node/local_map_bound', 10)

    # =========================================================================
    # COORDINATE & SYNC LOGIC
    # =========================================================================
    def publish_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
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
        if not np.isnan(msg.x):
            self.current_pos_ned_abs = np.array([msg.x, msg.y, msg.z])
            if not self.has_initial_pos:
                self.initial_pos_ned = np.array([msg.x, msg.y, msg.z])
                self.has_initial_pos = True
                self.get_logger().info(f"📍 ORIGIN SET TO: {self.initial_pos_ned}")

    def att_callback(self, msg):
        self.current_att_q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]

    def status_callback(self, msg):
        pass # Ignore all safety checks

    def fake_odom_loop(self):
        # Even if PX4 is not sending coordinates, we keep publishing [0,0,0] to keep Ego Planner alive
        now = self.get_clock().now()
        
        pos_ned_rel = self.current_pos_ned_abs - self.initial_pos_ned
        pos_enu = self.ned_to_enu_pos(pos_ned_rel)
        vel_enu = np.array([0.0, 0.0, 0.0]) # Ignore velocity for visualization
        
        q_ned = self.current_att_q
        q_enu = self.ned_to_enu_orientation(q_ned) 

        odom_data = {'stamp': now, 'pos': pos_enu, 'vel': vel_enu, 'quat': q_enu}
        self.odom_buffer.append(odom_data)
        
        self.publish_odom_and_tf(now, pos_enu, vel_enu, q_enu)
        
        # Publish Grid Box
        self.publish_local_map_marker(now, pos_enu)

    def depth_callback(self, msg):
        if not self.odom_buffer: return

        # Find closest Odom
        depth_time = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds
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
        
        if best_odom is None: return

        # ✅ CRITICAL FIX: Publish Camera Pose VỚI OFFSET CHÍNH XÁC
        cam_pose = PoseStamped()
        cam_pose.header.stamp = msg.header.stamp
        cam_pose.header.frame_id = "world"
        
        # ✅ Camera offset 0.1m về phía trước drone (theo trục X của base_link)
        # Cần transform từ base_link sang world frame
        q = best_odom['quat']
        # Tính rotation matrix từ quaternion
        x, y, z, w = q[0], q[1], q[2], q[3]
        R11 = 1 - 2*(y*y + z*z); R12 = 2*(x*y - z*w); R13 = 2*(x*z + y*w)
        
        # Camera offset trong base_link frame: [0.1, 0, 0]
        cam_offset_world = np.array([
            R11 * self.camera_offset_x,
            R12 * self.camera_offset_x,
            R13 * self.camera_offset_x
        ])
        
        cam_pose.pose.position.x = float(best_odom['pos'][0] + cam_offset_world[0])
        cam_pose.pose.position.y = float(best_odom['pos'][1] + cam_offset_world[1])
        cam_pose.pose.position.z = float(best_odom['pos'][2] + cam_offset_world[2])
        
        # ✅ Orientation giữ nguyên (camera có fixed rotation trong TF tree)
        cam_pose.pose.orientation.x = float(q[0])
        cam_pose.pose.orientation.y = float(q[1])
        cam_pose.pose.orientation.z = float(q[2])
        cam_pose.pose.orientation.w = float(q[3])
        self.cam_pose_pub.publish(cam_pose)

        # ✅ CRITICAL FIX: Camera Info phải có distortion model
        cam_info = CameraInfo()
        cam_info.header = msg.header
        cam_info.header.frame_id = "camera_link"  # ✅ Phải là camera_link, không phải world
        cam_info.width = self.cam_width
        cam_info.height = self.cam_height
        cam_info.distortion_model = "plumb_bob"  # ✅ Thêm model
        cam_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # ✅ No distortion
        cam_info.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        cam_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # ✅ Identity rotation
        cam_info.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.cam_info_pub.publish(cam_info)
        
        # ✅ CRITICAL FIX: Depth filtering & republish với TIMESTAMP GỐC
        try:
            cv_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_array = np.array(cv_depth, dtype=np.float32)
            
            # ✅ Cleanup NaN/Inf - DÙNG 0.0 thay vì 0.1 để Ego clear space
            depth_array[np.isnan(depth_array)] = 0.0
            depth_array[np.isinf(depth_array)] = 0.0
            
            # ✅ Clamp to valid range
            depth_array = np.clip(depth_array, 0.0, self.max_dist)
            
            # ✅ Zero out noise near drone body (< min_dist)
            depth_array[depth_array < self.min_dist] = 0.0
            
            # ✅ Convert back với encoding 32FC1 (CRITICAL cho Ego Planner)
            filtered_msg = self.cv_bridge.cv2_to_imgmsg(depth_array, encoding="32FC1")
            filtered_msg.header = msg.header  # ✅ GIỮ NGUYÊN timestamp gốc
            filtered_msg.header.frame_id = "camera_link"  # ✅ Frame phải là camera_link
            
        except Exception as e:
            self.get_logger().error(f"❌ Depth filter error: {e}")
            filtered_msg = msg
        
        self.depth_pub.publish(filtered_msg)

    # =========================================================================
    # HELPERS
    # =========================================================================
    def ned_to_enu_pos(self, ned_rel):
        return np.array([ned_rel[1], ned_rel[0], -ned_rel[2]])

    def ned_to_enu_orientation(self, q_ned):
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
    # IGNORING SAFETY (BYPASS COMMANDS)
    # =========================================================================
    def cmd_callback(self, msg):
        target_x = msg.position.x
        target_y = msg.position.y
        target_z = msg.position.z
        
        req_ned_rel = np.array([target_y, target_x, -target_z])
        req_ned_abs = req_ned_rel + self.initial_pos_ned
        px4_yaw = -msg.yaw + (math.pi / 2.0)

        v_x_ned = msg.velocity.y
        v_y_ned = msg.velocity.x
        v_z_ned = -msg.velocity.z

        self.publish_trajectory_setpoint(
            req_ned_abs[0], req_ned_abs[1], req_ned_abs[2], px4_yaw,
            vx=v_x_ned, vy=v_y_ned, vz=v_z_ned
        )

    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
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

    def rviz_goal_callback(self, msg):
        self.goal_pub.publish(msg)
        self.get_logger().info(f"🚀 RVIZ 2D GOAL PASSED TO EGO: {msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}")

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
        marker.scale.x = 15.0
        marker.scale.y = 15.0
        marker.scale.z = 8.0
        marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 0.15 
        self.local_map_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = BridgeNodeVer2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
