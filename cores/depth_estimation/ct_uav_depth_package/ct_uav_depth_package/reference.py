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
from px4_msgs.msg import ObstacleDistance, VehicleAttitude, VehicleLocalPosition
import sensor_msgs_py.point_cloud2 as pc2

import cosysairsim as airsim
from transforms3d.euler import quat2euler

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

UINT16_MAX = 65535

class TwoCamCollisionPreventionPX4(Node):
    def __init__(self):
        super().__init__('two_cam_collision_prevention_px4')
        # ===== Params =====
        self.declare_parameter('host_ip', '127.0.0.1')
        self.declare_parameter('host_port', 41451)
        self.declare_parameter('vehicle_name', '')
        self.declare_parameter('cams', ['cam0', 'cam1'])
        self.declare_parameter('cam_yaws_deg', [-45.0, +45.0])
        self.declare_parameter('update_rate_hz', 10.0)
        self.declare_parameter('fov_deg', 140.0)
        self.declare_parameter('min_depth_m', 0.2)
        self.declare_parameter('max_depth_m', 30.0)
        self.declare_parameter('vertical_crop_deg', 20.0)
        self.declare_parameter('publish_pc', True)
        self.declare_parameter('obstacle_topic', 'fmu/in/obstacle_distance')
        self.declare_parameter('cam_extrinsics', [
            [0.10, +0.05, 0.00, 0.0, 0.0, -45.0],
            [0.10, -0.05, 0.00, 0.0, 0.0, +45.0],
        ])
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
        self.cam_extrinsics = list(self.get_parameter('cam_extrinsics').value)
        if len(self.cams) != 2 or len(self.cam_yaws_deg) != 2 or len(self.cam_extrinsics) != 2:
            raise ValueError('Expect exactly two cameras: cams, cam_yaws_deg, cam_extrinsics must have 2 entries.')

        # ===== PX4 attitude / position subscribers =====
        self.last_roll = 0.0  # radians
        self.last_pitch = 0.0
        self.last_yaw = 0.0
        self.pose_received = False
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}

        self.sub_att = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self._on_vehicle_attitude,
            10,
        )
        self.sub_lpos = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self._on_vehicle_local_position,
            10,
        )

        # ===== AirSim client (only for depth streams) =====
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
        self.pub_obstacles = self.create_publisher(ObstacleDistance, self.obstacle_topic, 10)
        self.pub_depth_vis = {cam: self.create_publisher(Image, f'{cam}/depth_cropped', 10) for cam in self.cams}
        self.pub_pc = {cam: self.create_publisher(PointCloud2, f'{cam}/depth/points', 10) for cam in self.cams} if self.publish_pc else {}

        # TF
        self.tf_static = StaticTransformBroadcaster(self)
        self.tf_dyn = TransformBroadcaster(self)
        self._broadcast_static_cam_frames()

        # Timer
        self.timer = self.create_timer(1.0 / self.update_rate_hz, self._on_timer)
        self.get_logger().info('Node initialized (PX4 attitude input)')

    # ===== PX4 callbacks =====
    def _on_vehicle_attitude(self, msg: VehicleAttitude):
        # PX4 quaternion order is [w, x, y, z]
        try:
            q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]
            roll, pitch, yaw = quat2euler(q, axes='sxyz')
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
            r, p, yw = map(math.radians, (r, p, yw))
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

        # Fetch both depth images in one call
        reqs = [
            airsim.ImageRequest(self.cams[0], airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False),
            airsim.ImageRequest(self.cams[1], airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False),
        ]
        resps = self.client.simGetImages(reqs, self.vehicle_name)
        if not resps or len(resps) != 2:
            self.get_logger().warn('No depth responses')
            return

        stamp = self.get_clock().now().to_msg()

        increment = 5.0
        bins = 72
        distances_cm = np.full((bins,), UINT16_MAX, dtype=np.uint16)

        for cam_idx, cam in enumerate(self.cams):
            resp = resps[cam_idx]
            if len(resp.image_data_float) == 0:
                continue
            info = self.cam_info[cam]
            w, h = info['w'], info['h']
            depth = np.array(resp.image_data_float, dtype=np.float32).reshape(h, w)
            depth = np.clip(depth, self.min_depth_m, self.max_depth_m)
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
        vis = np.zeros_like(depth, dtype=np.float32)
        vis[row_mask, :] = depth[row_mask, :]
        d8 = np.clip((vis - self.min_depth_m) / (self.max_depth_m - self.min_depth_m), 0.0, 1.0)
        d8 = (d8 * 255.0).astype(np.uint8)
        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = f'{cam}_optical_frame'
        img.height, img.width = d8.shape
        img.encoding = 'mono8'
        img.is_bigendian = False
        img.step = img.width
        img.data = d8.tobytes()
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
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
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
