#!/usr/bin/env python3

import math
import threading
from collections import deque
from typing import List, Tuple

import cv2
import numpy as np
import rclpy
from px4_msgs.msg import ObstacleDistance, VehicleLocalPosition, VehicleAttitude
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import time
from models.hitnet.hitnet import HitnetTRT

UINT16_MAX = 65535


class StereoHitnetPX4Node(Node):
    def __init__(self):
        super().__init__('stereo_hitnet_px4_cp')

        # ===== Parameters =====
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('update_rate_hz', 10.0)
        self.declare_parameter('publish_pc', True)
        self.declare_parameter('publish_depth', False)
        self.declare_parameter('debug_downsample_factor', 2)
        self.declare_parameter('pc_stride_u', 4)
        self.declare_parameter('pc_stride_v', 4)
        self.declare_parameter('fov_deg', 140.0)  # per camera
        self.declare_parameter('min_depth_m', 0.2)
        self.declare_parameter('max_depth_m', 30.0)
        self.declare_parameter('vertical_crop_deg', 20.0)
        self.declare_parameter('obstacle_topic', 'fmu/in/obstacle_distance')
        # HitNet TensorRT wrapper (TensorRT engine)
        self.hitnet = None  # type: HitnetTRT
        self.declare_parameter('onnx_path', '')
        self.declare_parameter('engine_path', '')
        self.declare_parameter('stream_number', 4)

        onnx_path = self.get_parameter('onnx_path').value
        engine_path = self.get_parameter('engine_path').value
        stream_number = int(self.get_parameter('stream_number').value)
        self.use_compressed = bool(self.get_parameter('use_compressed').value)
        self.update_rate_hz = float(self.get_parameter('update_rate_hz').value)
        self.publish_pc = bool(self.get_parameter('publish_pc').value)
        self.publish_depth = bool(self.get_parameter('publish_depth').value)
        self.debug_downsample = int(self.get_parameter('debug_downsample_factor').value)
        if self.debug_downsample < 1:
            self.debug_downsample = 1
        self.pc_stride_u = max(1, int(self.get_parameter('pc_stride_u').value))
        self.pc_stride_v = max(1, int(self.get_parameter('pc_stride_v').value))
        self.fov_deg = float(self.get_parameter('fov_deg').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.vertical_crop_rad = math.radians(float(self.get_parameter('vertical_crop_deg').value))
        self.obstacle_topic = self.get_parameter('obstacle_topic').value

        # Stereo layout (fixed AirSim-style rig)
        self.stereo_pairs = [
            (7, 0),  # stereo0   yaw ~   0 deg
            (1, 2),  # stereo90  yaw ~  90 deg
            (3, 4),  # stereo180 yaw ~ 180 deg
            (5, 6),  # stereo270 yaw ~ 270 deg
        ]
        self.cam_yaws_deg = {
            0: 0.0,
            1: 90.0,
            2: 90.0,
            3: 180.0,
            4: 180.0,
            5: 270.0,
            6: 270.0,
            7: 0.0,
        }

        # ===== PX4 velocity & attitude =====
        qos_sensor = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
        self.last_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.pose_received = False
        self.att_received = False
        self.sub_lpos = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self._on_vehicle_local_position,
            qos_profile=qos_sensor
        )
        self.sub_att = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self._on_vehicle_attitude,
            qos_profile=qos_sensor
        )

        # roll, pitch, yaw (rad) in body frame
        self.last_rpy = np.zeros(3, dtype=np.float32)



        if engine_path:
            self.hitnet = HitnetTRT(show_info=True)
            ret = self.hitnet.init(onnx_path, engine_path, stream_number)
            if ret != 0:
                self.get_logger().error(f'Failed to init HitnetTRT (code={ret}), disabling inference')
                self.hitnet = None
            else:
                self.get_logger().info(f'HitnetTRT initialized with engine: {engine_path}')

        # ===== Image subscriber (stacked 8 cams) =====
        if self.use_compressed:
            self.sub_img = self.create_subscription(
                CompressedImage,
                '/fast_stream_node/raw_image/compressed',
                self._on_image_compressed,
                10,
            )
            self.get_logger().info('Subscribing to /fast_stream_node/raw_image/compressed')
        else:
            self.sub_img = self.create_subscription(
                Image,
                '/fast_stream_node/raw_image',
                self._on_image,
                10,
            )
            self.get_logger().info('Subscribing to /fast_stream_node/raw_image')

                # ===== Publishers =====
        self.pub_obstacles = self.create_publisher(ObstacleDistance, self.obstacle_topic, 10)
        self.pub_pc = self.create_publisher(PointCloud2, 'hitnet/points', 10) if self.publish_pc else None
        self.pub_depth_concat = self.create_publisher(Image, 'hitnet/left_right_depth_debug', 10) if self.publish_depth else None

        # Video writer for debug concat: video_demo1_<date_time>.avi
        self.video_writer = None
        self.video_fps = self.update_rate_hz

        # ===== Queues & threads =====
        self.disparity_queue = deque(maxlen=2)
        self.queue_lock = threading.Lock()

        # Processing thread: disparity -> depth -> bins/pc/publish
        self.proc_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.proc_thread.start()

        # ===== Timer (HitNet trigger) =====
        self.timer = self.create_timer(1.0 / self.update_rate_hz, self._on_timer)

        # Last stacked frame
        self.last_stacked = None  # type: np.ndarray
        self.last_header = None   # type: Header

        self.get_logger().info('StereoHitnetPX4Node initialized (velocity-based stereo selection)')

    # ===== PX4 callbacks =====
    def _on_vehicle_local_position(self, msg: VehicleLocalPosition):
        self.last_vel[...] = [msg.vx, msg.vy, msg.vz]
        self.pose_received = True

    def _on_vehicle_attitude(self, msg: VehicleAttitude):
        # PX4 quaternion order: [w, x, y, z]
        try:
            w, x, y, z = msg.q
            # convert to roll, pitch, yaw (sxyz)
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            roll = math.atan2(t0, t1)

            t2 = +2.0 * (w * y - z * x)
            t2 = max(-1.0, min(1.0, t2))
            pitch = math.asin(t2)

            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            yaw = math.atan2(t3, t4)

            self.last_rpy[...] = [roll, pitch, yaw]
            self.att_received = True
        except Exception as e:
            self.get_logger().warn(f'Attitude parse error: {e}')

    # ===== Image callbacks =====
    def _on_image_compressed(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().warn('Failed to decode compressed image')
            return
        self.last_stacked = img
        self.last_header = msg.header

    def _on_image(self, msg: Image):
        try:
            dtype = np.uint8
            if msg.encoding in ['mono16', '16UC1']:
                dtype = np.uint16
            img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, -1)
        except Exception as e:
            self.get_logger().error(f'Failed to convert Image to numpy: {e}')
            return
        self.last_stacked = img
        self.last_header = msg.header

    # ===== Timer loop =====
    def _on_timer(self):
        t0 = time.time()
        if self.last_stacked is None or not self.pose_received or not self.att_received:
            if self.last_stacked is None:
                self.get_logger().warn('Waiting for stacked image...')
            if not self.pose_received:
                self.get_logger().warn('Waiting for vehicle position...')
            return

        stacked = self.last_stacked
        header = self.last_header

        h, w = stacked.shape[:2]
        if w % 8 != 0:
            self.get_logger().warn(f'Unexpected stacked image width {w}, not divisible by 8')
            return

        cam_w = w // 8
        if cam_w != h:
            self.get_logger().warn(f'Expected each cam {h}x{h}, got {h}x{cam_w}')

        # Split into 8 cam images
        cams = []
        for i in range(8):
            x0 = i * cam_w
            x1 = (i + 1) * cam_w
            cams.append(stacked[:, x0:x1])

        # Select 2 stereo pairs based on velocity
        sel_pairs_idx = self._select_stereo_pairs_from_velocity()
        if not sel_pairs_idx:
            self.get_logger().info('Velocity too low, skipping HitNet processing')
            return

        # Build stereo images for the selected pairs
        stereo_list = []  # List[Tuple[np.ndarray, np.ndarray, float]]: (left, right, pair_yaw_deg)
        for pair_name, (li, ri), yaw_deg in sel_pairs_idx:
            left = cams[li]
            right = cams[ri]
            stereo_list.append((left, right, yaw_deg))

        # TODO: rotate images to compensate roll/pitch if you also subscribe attitude
        # For now we assume roll=pitch=0 in AirSim config.

        # Run HitNet on selected stereos (placeholder, you must implement)
        disparities = self._run_hitnet_on_steros(stereo_list)
        if not disparities:
            self.get_logger().info('No disparities returned from HitNet, skipping processing')
            return

        # Push work to processing thread: disparities + stereo config + header + attitude
        with self.queue_lock:
            self.disparity_queue.append({
                'stereo_list': stereo_list,
                'disparities': disparities,
                'header': header,
                'rpy': self.last_rpy.copy(),
                'timestamp': self.get_clock().now().to_msg(),
            })

        t1 = time.time()
        self.get_logger().info(f'timer loop (HitNet only) time: {(t1 - t0)*1000.0:.2f} ms')


    def _processing_loop(self):
        while rclpy.ok():
            item = None
            with self.queue_lock:
                if self.disparity_queue:
                    item = self.disparity_queue.popleft()
            if item is None:
                time.sleep(0.001)
                continue

            t0 = time.time()
            stereo_list = item['stereo_list']
            disparities = item['disparities']
            header = item['header']
            stamp = item['timestamp']
            rpy = item['rpy']

            increment = 5.0
            bins = 72
            distances_cm = np.full((bins,), UINT16_MAX, dtype=np.uint16)
            depths_for_pc = []

            roll = float(rpy[0])

            for (left, right, yaw_deg), disp in zip(stereo_list, disparities):
                depth = self._disparity_to_depth(disp)
                depths_for_pc.append(depth)
                self._accumulate_obstacles_from_depth_fast(depth, yaw_deg, roll, distances_cm)

            if self.publish_depth and self.pub_depth_concat is not None and depths_for_pc:
                depth0 = depths_for_pc[0]
                depth1 = depths_for_pc[1]
                left0, right0, _ = stereo_list[0]
                left1, right1, _ = stereo_list[1]
                _depth_msg, concat_msg, depth8, concat0 = self._depth_to_image(left0, right0, depth0, header)
                _depth_msg, concat_msg, depth8, concat1 = self._depth_to_image(left1, right1, depth1, header)
                concat = np.vstack((concat0, concat1))
                # Initialize video writer lazily when first frame is ready
                if self.video_writer is None and concat is not None:
                    h_v, w_v = concat.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    import datetime
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    video_name = f'video_demo1_{ts}.avi'
                    self.video_writer = cv2.VideoWriter(video_name, fourcc, self.video_fps, (w_v, h_v))
                    self.get_logger().info(f'Video writer initialized: {video_name} ({w_v}x{h_v}@{self.video_fps}fps)')

                # Write frame if writer is available
                if self.video_writer is not None and concat is not None:
                    self.video_writer.write(concat)

            msg = ObstacleDistance()
            msg.timestamp = self.get_clock().now().nanoseconds // 1000
            msg.frame = ObstacleDistance.MAV_FRAME_BODY_FRD
            msg.angle_offset = 0.0
            msg.increment = float(increment)
            msg.min_distance = int(round(self.min_depth_m * 100.0))
            msg.max_distance = int(round(self.max_depth_m * 100.0))
            msg.distances = distances_cm.tolist()
            self.pub_obstacles.publish(msg)

            if self.publish_pc and self.pub_pc is not None:
                pc = self._depths_to_pointcloud(depths_for_pc, header)
                self.pub_pc.publish(pc)

            t1 = time.time()
            self.get_logger().info(f'processing thread time: {(t1 - t0)*1000.0:.2f} ms')


    # ===== Velocity → stereo selection =====
    def _select_stereo_pairs_from_velocity(self) -> List[Tuple[str, Tuple[int, int], float]]:
        vx, vy, _ = self.last_vel
        vel_norm = math.hypot(vx, vy)
        if vel_norm < 0.1:
            return [("stereo0", (7, 0), 0.0),("stereo180", (3, 4), 180.0),]

        # Angle of velocity in body XY plane, degrees in [0, 360)
        ang = math.degrees(math.atan2(vy, vx))  # atan2(y, x)
        if ang < 0:
            ang += 360.0

        # Map velocity angle to nearest base yaw (0, 90, 180, 270)
        bases = [0.0, 90.0, 180.0, 270.0]
        diffs = [abs(((ang - b + 180) % 360) - 180) for b in bases]
        idx0 = int(np.argmin(diffs))
        yaw0 = bases[idx0]
        yaw1 = (yaw0 + 90.0) % 360.0

        # Stereo pair yaw centers (simple mapping)
        #NED
        pair_defs = [
            ("stereo0", (7, 0), 0.0),
            ("stereo90", (1, 2), 270.0),
            ("stereo180", (3, 4), 180.0),
            ("stereo270", (5, 6), 90.0),
        ]
        #ENU
        # pair_defs = [
        #     ("stereo0", (7, 0), 270.0),
        #     ("stereo90", (1, 2), 0.0),
        #     ("stereo180", (3, 4), 90.0),
        #     ("stereo270", (5, 6), 180.0),
        # ]
        def find_pair(yaw_target):
            best = None
            best_diff = 1e9
            for name, (li, ri), yaw_c in pair_defs:
                d = abs(((yaw_c - yaw_target + 180) % 360) - 180)
                if d < best_diff:
                    best_diff = d
                    best = (name, (li, ri), yaw_c)
            return best

        p0 = find_pair(yaw0)
        # p1 = find_pair(yaw1)
        res = []
        if p0 is not None:
            res.append(p0)
        # if p1 is not None and (p1[0] != p0[0]):
        #     res.append(p1)
        return [("stereo0", (7, 0), 0.0),p0,]

    # ===== HitNet wrapper =====
    def _run_hitnet_on_steros(self, stereo_list: List[Tuple[np.ndarray, np.ndarray, float]]) -> List[np.ndarray]:
        if self.hitnet is None:
            self.get_logger().warn('HitNetTRT is not initialized; skip inference')
            return []
        t0 = time.time()
        # Preprocess each stereo pair to fixed (1, 2, 240, 320) float32 [0,1]
        inputs = []
        for left, right, _yaw in stereo_list:
            if left is None or right is None:
                inputs.append(None)
                continue

            if left.shape != right.shape:
                self.get_logger().warn('Left/right shapes differ, skipping this pair')
                inputs.append(None)
                continue

            # Convert to grayscale
            if left.ndim == 3 and left.shape[2] == 3:
                left_g = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            else:
                left_g = left
            if right.ndim == 3 and right.shape[2] == 3:
                right_g = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            else:
                right_g = right

            # Resize to HitNet input size (fixed 240x320)
            target_W, target_H = 320, 240
            if left_g.shape[:2] != (target_H, target_W):
                left_g = cv2.resize(left_g, (target_W, target_H))
            if right_g.shape[:2] != (target_H, target_W):
                right_g = cv2.resize(right_g, (target_W, target_H))

            stacked = np.array([left_g, right_g], dtype=np.float32)  # (2, 240, 320)
            tensor = stacked[np.newaxis, :, :, :] / 255.0            # (1, 2, 240, 320)
            inputs.append(tensor)

        # Pad or trim inputs to match stream_number
        valid_inputs = [inp for inp in inputs if inp is not None]
        if not valid_inputs:
            return []

        # Build list length = self.hitnet.stream_number
        batch = []
        for i in range(self.hitnet.stream_number):
            if i < len(valid_inputs):
                batch.append(valid_inputs[i])
            else:
                batch.append(valid_inputs[-1])

        # Warm-up once on first call (optional)
        ret = self.hitnet.do_inference(batch)
        if ret != 0:
            self.get_logger().error(f'HitNet inference failed with code {ret}')
            return []


        outputs = self.hitnet.get_output()
        disparities: List[np.ndarray] = []
        # Take as many disparities as stereo_list length
        for i in range(min(len(stereo_list), len(outputs))):
            disp = outputs[i]
            # Ensure 2D disparity map
            disp_2d = np.squeeze(disp)
            disparities.append(disp_2d)
        t1 = time.time()
        self.get_logger().info(f'HitNet inference time: {(t1 - t0)*1000.0:.2f} ms')

        return disparities

    def _disparity_to_depth(self, disp: np.ndarray) -> np.ndarray:
        # Simple pinhole: depth = (baseline * f) / disp
        # Baseline/focal should match your engine training setup.
        baseline = 0.8  # meters
        h, w = disp.shape[:2]
        fx = (w / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)
        disp_safe = np.where(disp > 0.1, disp, 0.1)
        depth = (baseline * fx) / disp_safe
        depth[disp <= 0.1] = 0.0
        depth = np.clip(depth, 0.0, self.max_depth_m)
        return depth

    # ===== Depth → ObstacleDistance bins (optimized, with roll compensation) =====
    def _accumulate_obstacles_from_depth_fast(self, depth: np.ndarray, yaw_center_deg: float, roll_rad: float, distances_cm: np.ndarray):
        h, w = depth.shape
        fx = (w / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)
        cx = w / 2.0
        cy = h / 2.0

        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        az_cols = np.arctan2((u - cx) / fx, np.ones_like(u, dtype=np.float32))
        el_rows = np.arctan2((v - cy) / fx, np.ones_like(v, dtype=np.float32))

        # vertical crop
        row_mask = np.where(np.abs(el_rows) <= self.vertical_crop_rad)[0]
        if row_mask.size == 0:
            return

        depth_crop = depth[row_mask, :]
        valid = (
            np.isfinite(depth_crop)
            & (depth_crop > self.min_depth_m)
            & (depth_crop < self.max_depth_m)
        )
        if not np.any(valid):
            return

        large_val = self.max_depth_m * 2.0
        depth_valid = np.where(valid, depth_crop, large_val)
        min_col_depth = depth_valid.min(axis=0)  # (w,)

        has_valid = min_col_depth < large_val
        if not np.any(has_valid):
            return

        # compensate roll by shifting azimuth
        az_body = az_cols - roll_rad + math.radians(yaw_center_deg)
        increment = 5.0
        bins = 72
        az_deg = np.degrees((az_body + 2 * math.pi) % (2 * math.pi))
        col_bins = (az_deg // increment).astype(np.int32)

        valid_bins = col_bins[has_valid]
        valid_depths_cm = (min_col_depth[has_valid] * 100.0).astype(np.float32)

        mask_in_range = (valid_bins >= 0) & (valid_bins < bins)
        if not np.any(mask_in_range):
            return
        valid_bins = valid_bins[mask_in_range]
        valid_depths_cm = valid_depths_cm[mask_in_range]

        for b, d_cm_f in zip(valid_bins, valid_depths_cm):
            d_cm = int(round(float(d_cm_f)))
            if d_cm <= 0:
                d_cm = 1
            if distances_cm[b] == UINT16_MAX or d_cm < distances_cm[b]:
                distances_cm[b] = np.uint16(d_cm)

    # ===== Depth → merged PointCloud2 =====
    def _depths_to_pointcloud(self, depths: List[np.ndarray], header: Header) -> PointCloud2:
        pts = []
        for depth in depths:
            if depth is None:
                continue
            h, w = depth.shape
            fx = (w / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)
            fy = fx
            cx = w / 2.0
            cy = h / 2.0
            # apply stride to reduce point count
            u = np.arange(0, w, self.pc_stride_u)
            v = np.arange(0, h, self.pc_stride_v)
            uu, vv = np.meshgrid(u, v)
            z = depth
            mask = (z > self.min_depth_m) & (z < self.max_depth_m) & np.isfinite(z)
            if not np.any(mask):
                continue
            zz = z[mask]
            xx = (uu[mask] - cx) * zz / fx
            yy = (vv[mask] - cy) * zz / fy
            p = np.stack([xx, yy, zz], axis=-1)
            pts.append(p)

        if not pts:
            # empty cloud
            header_out = Header()
            header_out.stamp = header.stamp if header is not None else self.get_clock().now().to_msg()
            header_out.frame_id = 'base_link'
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            return pc2.create_cloud(header_out, fields, [])

        pts = np.concatenate(pts, axis=0).astype(np.float32)
        header_out = Header()
        header_out.stamp = header.stamp if header is not None else self.get_clock().now().to_msg()
        header_out.frame_id = 'base_link'
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        return pc2.create_cloud(header_out, fields, pts)

    def _depth_to_image(self, left: np.ndarray, right: np.ndarray, depth: np.ndarray, header: Header):
        """Create depth mono8 image and concatenated left-right-depth BGR image with optional downsampling."""
        if depth is None or left is None or right is None:
            return None, None

        # Ensure left/right are 3-channel BGR
        if left.ndim == 2:
            left_vis = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        elif left.ndim == 3 and left.shape[2] == 1:
            left_vis = cv2.cvtColor(left.squeeze(-1), cv2.COLOR_GRAY2BGR)
        else:
            left_vis = left

        if right.ndim == 2:
            right_vis = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
        elif right.ndim == 3 and right.shape[2] == 1:
            right_vis = cv2.cvtColor(right.squeeze(-1), cv2.COLOR_GRAY2BGR)
        else:
            right_vis = right

        # Resize left/right to match depth size
        dh, dw = depth.shape
        left_vis = cv2.resize(left_vis, (dw, dh))
        right_vis = cv2.resize(right_vis, (dw, dh))
        left_vis = cv2.cvtColor(left_vis, cv2.COLOR_RGB2BGR)
        right_vis = cv2.cvtColor(right_vis, cv2.COLOR_RGB2BGR)
        # Depth mono8 (0-10 m)
        max_vis_depth = 10.0
        d = np.clip(depth, 0.0, max_vis_depth)
        d8 = (d / max_vis_depth * 255.0).astype(np.uint8)
        depth_color = cv2.cvtColor(d8, cv2.COLOR_GRAY2BGR)

        # Optional downsampling for debug visualization
        if self.debug_downsample > 1:
            fx = 1.0 / float(self.debug_downsample)
            fy = 1.0 / float(self.debug_downsample)
            left_vis = cv2.resize(left_vis, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
            right_vis = cv2.resize(right_vis, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
            depth_color = cv2.resize(depth_color, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)

        # Concatenate horizontally: left | right | depth
        concat = np.concatenate([left_vis, right_vis, depth_color], axis=1)

        # Depth mono8 Image msg
        depth_msg = Image()
        # depth_msg.header = header
        # depth_msg.height, depth_msg.width = d8.shape
        # depth_msg.encoding = 'mono8'
        # depth_msg.is_bigendian = False
        # depth_msg.step = depth_msg.width
        # depth_msg.data = d8.tobytes()

        # Concatenated BGR Image msg
        concat_msg = Image()
        # concat_msg.header = header
        # concat_msg.height, concat_msg.width, _ = concat.shape
        # concat_msg.encoding = 'bgr8'
        # concat_msg.is_bigendian = False
        # concat_msg.step = concat_msg.width * 3
        # concat_msg.data = concat.tobytes()

        return depth_msg, concat_msg, d8, concat


def main(argv=None):
    rclpy.init(args=argv)
    node = StereoHitnetPX4Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()