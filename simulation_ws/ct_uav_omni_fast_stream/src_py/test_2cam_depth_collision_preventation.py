#!/usr/bin/env python3
"""
ROS2 node: Fuse 2 AirSim depth cameras → PX4 ObstacleDistance (72 sectors)
Now reads vehicle attitude/pose from PX4 topics instead of AirSim:
  - /fmu/out/vehicle_attitude (px4_msgs/VehicleAttitude)
  - /fmu/out/vehicle_local_position (px4_msgs/VehicleLocalPosition)

- Compensate image azimuth using PX4 roll
- Crop vertical FOV to [-20°, +20°]
- Map to 72 sectors (5° each) in MAV_FRAME_BODY_FRD
- Publish ObstacleDistance + cropped depth + optional PointCloud2 + TF

Deps: rclpy, px4_msgs, cosysairsim (for depth), transforms3d
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import QuaternionStamped
from px4_msgs.msg import ObstacleDistance, VehicleAttitude, VehicleLocalPosition
import sensor_msgs_py.point_cloud2 as pc2

import cosysairsim as airsim
from transforms3d.euler import quat2euler

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

UINT16_MAX = 65535

class TwoCamCollisionPreventionPX4(Node):
    def __init__(self):
        super().__init__('two_cam_collision_prevention_px4')
        # ===== Params =====
        self.declare_parameter('host_ip', '127.0.0.1')
        self.declare_parameter('host_port', 41451)
        self.declare_parameter('vehicle_name', '')
        self.declare_parameter('cams', ['cam0'])
        self.declare_parameter('cam_yaws_deg', [0.0])
        self.declare_parameter('update_rate_hz', 10.0)
        self.declare_parameter('fov_deg', 140.0)
        self.declare_parameter('min_depth_m', 0.2)
        self.declare_parameter('max_depth_m', 60.0)
        self.declare_parameter('vertical_crop_deg', 20.0)
        self.declare_parameter('publish_pc', True)
        self.declare_parameter('obstacle_topic', 'fmu/in/obstacle_distance')

        # ---- Read params ----
        self.host_ip = self.get_parameter('host_ip').value
        self.host_port = int(self.get_parameter('host_port').value)
        self.vehicle_name = self.get_parameter('vehicle_name').value
        self.cams = list(self.get_parameter('cams').value)
        self.cam_yaws_deg = list(self.get_parameter('cam_yaws_deg').value)
        self.update_rate_hz = float(self.get_parameter('update_rate_hz').value)
        self.fov_deg = float(self.get_parameter('fov_deg').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.vertical_crop_rad = math.radians(float(self.get_parameter('vertical_crop_deg').value))
        self.publish_pc = bool(self.get_parameter('publish_pc').value)
        self.obstacle_topic = self.get_parameter('obstacle_topic').value
        self.cam_extrinsics = [[0.10, +0.05, 0.00, -1.5708, 0.0, -1.5708]]
        if len(self.cams) !=  len(self.cam_yaws_deg) or len(self.cam_extrinsics) != len(self.cams):
            raise ValueError('Expect number of cams, yaws, extrinsics to be equal') 

        # ===== PX4 attitude / position subscribers =====
        self.last_roll = 0.0  # radians
        self.last_pitch = 0.0
        self.last_yaw = 0.0
        self.last_quat = [0.0, 0.0, 0.0, 1.0]  # [x, y, z, w]
        self.pose_received = False
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        sensor_qos = rclpy.qos.QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.VOLATILE
            )
        self.sub_att = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self._on_vehicle_attitude,
            sensor_qos,
        )
        self.sub_lpos = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self._on_vehicle_local_position,
            sensor_qos,
        )

        # ===== AirSim client (only for depth streams) =====
        self.get_logger().info(f'Connecting to AirSim at {self.host_ip}:{self.host_port}...')
        self.client = airsim.MultirotorClient(ip=self.host_ip, port=self.host_port)
        self.client.confirmConnection()
        self.get_logger().info(f'Connected to AirSim at {self.host_ip}:{self.host_port}')

        # Probe cameras & intrinsics
        self.cam_info = {}
        for cam in self.cams:
            self.client.simSetCameraFov(cam, self.fov_deg)
            resp = self.client.simGetImages([
                airsim.ImageRequest(cam, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
            ], self.vehicle_name)
            if not resp or len(resp) != 1 or len(resp[0].image_data_float) == 0:
                raise RuntimeError(f'Failed to get initial depth from {cam}')
            w, h = resp[0].width, resp[0].height
            fx = w / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
            fy = fx
            cx = w / 2.0
            cy = h / 2.0
            u = np.arange(w)
            v = np.arange(h)
            az_cols = np.arctan2((u - cx) / fx, np.ones_like(u))  # (W,)
            el_rows = np.arctan2((v - cy) / fy, np.ones_like(v))  # (H,)
            self.cam_info[cam] = {'w': w, 'h': h, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                                  'az_cols': az_cols, 'el_rows': el_rows}
            self.get_logger().info(f"{cam}: {w}x{h}, fx={fx:.1f}")

        # Publishers
        sensor_qos = rclpy.qos.QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.VOLATILE
        )
        self.pub_obstacles = self.create_publisher(ObstacleDistance, self.obstacle_topic, 10)
        self.pub_depth_vis = {cam: self.create_publisher(Image, f'{cam}/depth_map', 10) for cam in self.cams}
        self.pub_pc = {cam: self.create_publisher(PointCloud2, f'{cam}/depth/points', 10) for cam in self.cams} if self.publish_pc else {}
        self.pub_airsim_quat = self.create_publisher(QuaternionStamped, 'airsim/orientation', 10)

        # TF
        self.tf_static = StaticTransformBroadcaster(self)
        self.tf_dyn = TransformBroadcaster(self)
        self._broadcast_static_cam_frames()

        # Timer
        self.timer = self.create_timer(1.0 / self.update_rate_hz, self._on_timer)
        self.get_logger().info('Node initialized (PX4 attitude input)')

        # 

    # ===== PX4 callbacks =====
    def _on_vehicle_attitude(self, msg: VehicleAttitude):
        # PX4 quaternion order is [w, x, y, z] in NED frame
        # Need to convert NED to ENU for ROS2: [qw, qx, qy, qz]_NED -> transform
        try:
            q_ned = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]  # [w, x, y, z]
            # NED to ENU: x_ENU = y_NED, y_ENU = x_NED, z_ENU = -z_NED
            # Quaternion NED->ENU rotation: q_ENU = q_rot * q_NED
            # For this transformation: [qw, qy, qx, -qz] approximately
            qw, qx, qy, qz = q_ned
            # ENU quaternion (swap x<->y, negate z)
            self.last_quat = [-qx, -qy, -qz, -qw]  # [x, y, z, w] in ENU
            
            # Also get Euler for roll compensation in obstacle detection
            roll, pitch, yaw = quat2euler(q_ned, axes='sxyz')
            self.last_roll, self.last_pitch, self.last_yaw = roll, pitch, yaw
            self.pose_received = True
        except Exception as e:
            self.get_logger().warn(f'Attitude parse error: {e}')

    def _on_vehicle_local_position(self, msg: VehicleLocalPosition):
        # Store for future use (not required for CP bins, but handy for TF/world mapping)
        self.position["x"], self.position["y"], self.position["z"] = msg.x, msg.y, msg.z

    # ===== TF helpers =====
    def _broadcast_static_cam_frames(self):
        frames = []
        for cam, extr in zip(self.cams, self.cam_extrinsics):
            x, y, z, r, p, yw = extr
            qx, qy, qz, qw = self._rpy_to_quat(r, p, yw)
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'base_link'
            t.child_frame_id = f'{cam}_optical_frame'
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            frames.append(t)
        self.tf_static.sendTransform(frames)

    @staticmethod
    def _rpy_to_quat(roll, pitch, yaw):
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        return qx, qy, qz, qw

    # ===== Main loop =====
    def _on_timer(self):
        if not self.pose_received:
            # Wait for first PX4 attitude to avoid wrong binning
            return

        roll = self.last_roll  # radians
        pitch = self.last_pitch
        yaw = self.last_yaw
        # Use quaternion directly (already converted to ENU in callback)
        qx, qy, qz, qw = [float(v) for v in self.last_quat]

        # Fetch both depth images in one call
        reqs = []
        for i in range(len(self.cams)):
            reqs.append(airsim.ImageRequest(self.cams[i], airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False))

        # Get AirSim state for quaternion
        ros_now = self.get_clock().now()
        
        airsim_state = self.client.simGetVehiclePose(self.vehicle_name)
        airsim_orient = airsim_state.orientation
        # Publish AirSim quaternion
        quat_msg = QuaternionStamped()
        quat_msg.header.stamp = ros_now.to_msg()
        quat_msg.header.frame_id = 'world'
        # AirSim quaternion is w, x, y, z
        quat_msg.quaternion.w = airsim_orient.w_val
        quat_msg.quaternion.x = airsim_orient.x_val
        quat_msg.quaternion.y = airsim_orient.y_val
        quat_msg.quaternion.z = airsim_orient.z_val
        self.pub_airsim_quat.publish(quat_msg)

        # Get ROS clock immediately to calculate latency
        corrected_timestamp_ns = ros_now.nanoseconds
        resps = self.client.simGetImages(reqs, self.vehicle_name)
        

        
        # Estimate latency from AirSim timestamp (ns)
        lantency_ns = self.get_clock().now().nanoseconds - corrected_timestamp_ns
        if lantency_ns > 50_000_000:
            self.get_logger().warn(f'AirSim depth too high latency: {lantency_ns / 1e6:.2f} ms')
        
        # Convert to ROS Time message
        stamp = self.get_clock().now().to_msg()
        stamp.sec = int(corrected_timestamp_ns // 1_000_000_000)
        stamp.nanosec = int(corrected_timestamp_ns % 1_000_000_000)

        # ----- Broadcast dynamic tf: world -> base_link -----
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id = 'base_link'
        # PX4 local_position is in NED, convert to ENU: x<->y, z=-z
        tf_msg.transform.translation.x = float(self.position["x"])
        tf_msg.transform.translation.y = float(-self.position["y"])
        tf_msg.transform.translation.z = float(-self.position["z"])

        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = -qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        self.tf_dyn.sendTransform(tf_msg)

        increment = 5.0
        bins = 72
        distances_cm = np.full((bins,), UINT16_MAX, dtype=np.uint16)

        for cam_idx, cam in enumerate(self.cams):
            resp = resps[cam_idx]
            if len(resp.image_data_float) == 0:
                self.get_logger().warn(f'Empty depth image from {cam}')
                continue
            info = self.cam_info[cam]
            w, h = info['w'], info['h']
            depth = np.array(resp.image_data_float, dtype=np.float32).reshape(h, w)
            depth = np.clip(depth, self.min_depth_m, self.max_depth_m+1.0)
            az_cols = info['az_cols']
            el_rows = info['el_rows']

            # Roll compensation from PX4 attitude
            az_cols_world = az_cols - roll
            # Add camera yaw w.r.t. body forward
            body_az_cols = az_cols_world + math.radians(self.cam_yaws_deg[cam_idx])

            row_mask = np.where(np.abs(el_rows) <= self.vertical_crop_rad)[0]
            if row_mask.size == 0:
                continue

            az_deg = np.degrees((body_az_cols + 2*math.pi) % (2*math.pi))
            col_bins = (az_deg // increment).astype(np.int32)

            for col in range(w):
                b = int(col_bins[col])
                cd = depth[row_mask, col]
                cd = cd[np.isfinite(cd)]
                if cd.size == 0:
                    continue
                d_cm = int(round(float(cd.min()) * 100.0))
                if d_cm <= 0:
                    d_cm = 1
                if distances_cm[b] == UINT16_MAX or d_cm < distances_cm[b]:
                    distances_cm[b] = np.uint16(d_cm)

            self._publish_depth_vis(cam, depth, row_mask, stamp)
            if cam in self.pub_pc:
                self._publish_pointcloud(cam, depth, row_mask, stamp, info)

        msg = ObstacleDistance()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000  # usec
        msg.frame = ObstacleDistance.MAV_FRAME_BODY_FRD
        msg.angle_offset = 0.0
        msg.increment = float(increment)
        msg.min_distance = int(round(self.min_depth_m * 100.0))
        msg.max_distance = int(round(self.max_depth_m * 100.0))
        msg.distances = distances_cm.tolist()
        self.pub_obstacles.publish(msg)

    # ===== Publishers =====
    def _publish_depth_vis(self, cam, depth, row_mask, stamp):
        # vis = np.zeros_like(depth, dtype=np.float32)
        # vis[row_mask, :] = depth[row_mask, :]
        vis =depth.astype(np.float32)
        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = f'{cam}_optical_frame'
        img.height, img.width = vis.shape
        img.encoding = '32FC1'
        img.is_bigendian = False
        img.step = img.width * 4  # 4 bytes per float32
        img.data = vis.tobytes()
        self.pub_depth_vis[cam].publish(img)

    def _publish_pointcloud(self, cam, depth, row_mask, stamp, info):
        h, w = depth.shape
        fx, fy, cx, cy = info['fx'], info['fy'], info['cx'], info['cy']
        u = np.arange(w)
        v = row_mask
        uu, vv = np.meshgrid(u, v)
        z = depth[vv, uu]
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        # remove invalid points
        valid_mask = (z.flatten() < self.max_depth_m) & np.isfinite(pts).all(axis=1)
        pts = pts[valid_mask]
        header = Header()
        header.stamp = stamp
        header.frame_id = f'{cam}_optical_frame'
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc = pc2.create_cloud(header, fields, pts.astype(np.float32))
        self.pub_pc[cam].publish(pc)


def main(argv=None):
    rclpy.init(args=argv)
    node = TwoCamCollisionPreventionPX4()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
