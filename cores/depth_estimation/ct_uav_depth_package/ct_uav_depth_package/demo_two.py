#!/usr/bin/env python3

import math
import threading
from collections import deque
from typing import List, Tuple
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import time
from models.hitnet.hitnet import HitnetTRT
from models.fast_acv_plus.fastACV import FastACVTRT
import matplotlib.pyplot as plt
import yaml


class StereoDepthNode(Node):
    def __init__(self):
        super().__init__('stereo_depth_node')

        # ===== Parameters =====
        self.declare_parameter('update_rate_hz', 10.0)
        self.declare_parameter('publish_depth', True)
        self.declare_parameter('fov_deg', 140.0)  # per camera
        self.declare_parameter('min_depth_m', 0.2)
        self.declare_parameter('max_depth_m', 30.0)
        
        # Camera calibration file
        self.declare_parameter('calibration_file', '')
        
        # Camera parameters
        self.declare_parameter('cam_width', 1640)
        self.declare_parameter('cam_height', 1232)
        self.declare_parameter('cam_fps', 60)
        self.declare_parameter('cam_flip_method', 0)

        self.declare_parameter('stereo_method', 'sgm')  # 'sgm', 'hitnet', or 'fastacv'
        # Stereo SGM parameters
        self.declare_parameter('sgm_num_disparities', 128)
        self.declare_parameter('sgm_block_size', 5)
        self.declare_parameter('sgm_uniqueness_ratio', 10)
        self.declare_parameter('sgm_speckle_window_size', 50)
        self.declare_parameter('sgm_speckle_range', 1)
        self.declare_parameter('sgm_disp12_max_diff', 1)
        self.declare_parameter('sgm_pre_filter_cap', 31)
        self.declare_parameter('sgm_min_disparity', 0)

        # HitNet TensorRT parameters
        self.declare_parameter('onnx_path', '')
        self.declare_parameter('engine_path', '')
        self.declare_parameter('stream_number', 1)
        
        self.update_rate_hz = float(self.get_parameter('update_rate_hz').value)
        self.publish_depth = bool(self.get_parameter('publish_depth').value)
        self.fov_deg = float(self.get_parameter('fov_deg').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        
        # Camera parameters
        self.cam_width = int(self.get_parameter('cam_width').value)
        self.cam_height = int(self.get_parameter('cam_height').value)
        self.cam_fps = int(self.get_parameter('cam_fps').value)
        self.cam_flip_method = int(self.get_parameter('cam_flip_method').value)
        
        # Load camera calibration
        calibration_file = self.get_parameter('calibration_file').value
        self.calib_data = self._load_calibration(calibration_file)
        
        # Prepare rectification maps
        self._prepare_rectification_maps()
        
        # Initialize GStreamer cameras
        self.cap_left = None
        self.cap_right = None
        self._init_cameras()

        # Stereo method selection
        self.stereo_method = self.get_parameter('stereo_method').value.lower()

        # Initialize stereo matcher based on method
        self.hitnet = None
        self.fastacv = None
        self.sgbm = None

        if self.stereo_method == 'hitnet':
            onnx_path = self.get_parameter('onnx_path').value
            engine_path = self.get_parameter('engine_path').value
            stream_number = int(self.get_parameter('stream_number').value)

            if engine_path:
                self.hitnet = HitnetTRT(show_info=True)
                ret = self.hitnet.init(onnx_path, engine_path, stream_number)
                if ret != 0:
                    self.get_logger().error(f'Failed to init HitnetTRT (code={ret}), falling back to SGM')
                    self.hitnet = None
                    self.stereo_method = 'sgm'
                else:
                    self.get_logger().info(f'HitnetTRT initialized with engine: {engine_path}')
            else:
                self.get_logger().warn('No engine_path provided for HitNet, falling back to SGM')
                self.stereo_method = 'sgm'

        elif self.stereo_method == 'fastacv':
            onnx_path = self.get_parameter('onnx_path').value
            engine_path = self.get_parameter('engine_path').value
            stream_number = int(self.get_parameter('stream_number').value)

            if engine_path:
                self.fastacv = FastACVTRT(show_info=True)
                ret = self.fastacv.init(onnx_path, engine_path, stream_number)
                if ret != 0:
                    self.get_logger().error(f'Failed to init FastACVTRT (code={ret}), falling back to SGM')
                    self.fastacv = None
                    self.stereo_method = 'sgm'
                else:
                    self.get_logger().info(f'FastACVTRT initialized with engine: {engine_path}')
            else:
                self.get_logger().warn('No engine_path provided for FastACV, falling back to SGM')
                self.stereo_method = 'sgm'

        if self.stereo_method == 'sgm' or (self.hitnet is None and self.fastacv is None):
            # Create OpenCV SGM matcher
            self.sgbm = self._create_sgbm()
            self.stereo_method = 'sgm'

        # ===== Direct camera capture (no ROS2 subscription) =====
        self.get_logger().info('Using direct GStreamer camera capture')

        # ===== Publishers =====
        self.pub_depth = self.create_publisher(Image, '/stereo/depth_map', 10) if self.publish_depth else None

        # ===== Queues & threads =====
        self.disparity_queue = deque(maxlen=2)  # Queue for disparity data
        self.queue_lock = threading.Lock()

        # Processing thread: process disparity -> depth -> publish
        self.proc_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.proc_thread.start()

        # ===== Timer (Camera read + Inference) =====
        self.timer = self.create_timer(1.0 / self.update_rate_hz, self._on_timer)

        self.get_logger().info(f'Stereo Depth node initialized with {self.stereo_method.upper()} and direct camera capture')

    # ===== Camera calibration loading =====
    def _load_calibration(self, calibration_file: str) -> dict:
        """Load camera calibration from YAML file (Kalibr format)"""
        if not calibration_file or not os.path.exists(calibration_file):
            self.get_logger().warn(f'Calibration file not found: {calibration_file}, using default values')
            # Return default calibration
            return {
                'cam0': {
                    'intrinsics': [1203.26, 1163.82, 869.92, 595.30],
                    'distortion_coeffs': [-0.0328, 0.0512, -0.0033, -0.0006],
                    'resolution': [1640, 1232]
                },
                'cam1': {
                    'intrinsics': [1198.25, 1157.78, 827.98, 623.76],
                    'distortion_coeffs': [-0.0315, 0.0447, 0.0014, 0.0003],
                    'resolution': [1640, 1232],
                    'T_cn_cnm1': np.eye(4)
                }
            }
        
        try:
            with open(calibration_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract camera parameters (cam0 = left, cam1 = right)
            cam0 = config['cam0']
            cam1 = config['cam1']
            
            # Parse intrinsics [fu, fv, cu, cv]
            intrinsics_0 = cam0['intrinsics']
            intrinsics_1 = cam1['intrinsics']
            
            # Extract distortion coefficients
            D_cam0 = np.array(cam0['distortion_coeffs'])
            D_cam1 = np.array(cam1['distortion_coeffs'])
            
            # Extract extrinsics T_cam1_cam0
            T_cam1_cam0 = np.array(cam1['T_cn_cnm1'])
            
            # Image size
            image_size = tuple(cam0['resolution'])
            
            calib_data = {
                'cam0': {
                    'intrinsics': intrinsics_0,
                    'distortion_coeffs': D_cam0.tolist(),
                    'resolution': list(image_size)
                },
                'cam1': {
                    'intrinsics': intrinsics_1,
                    'distortion_coeffs': D_cam1.tolist(),
                    'resolution': list(image_size),
                    'T_cn_cnm1': T_cam1_cam0.tolist()
                }
            }
            
            # Calculate baseline from transformation matrix
            baseline = np.linalg.norm(T_cam1_cam0[0:3, 3])
            
            self.get_logger().info(f'Loaded calibration from: {calibration_file}')
            self.get_logger().info(f'Baseline: {baseline:.6f} m')
            self.get_logger().info(f'Cam0 intrinsics: fx={intrinsics_0[0]:.2f}, fy={intrinsics_0[1]:.2f}, cx={intrinsics_0[2]:.2f}, cy={intrinsics_0[3]:.2f}')
            self.get_logger().info(f'Cam1 intrinsics: fx={intrinsics_1[0]:.2f}, fy={intrinsics_1[1]:.2f}, cx={intrinsics_1[2]:.2f}, cy={intrinsics_1[3]:.2f}')
            
            return calib_data
            
        except Exception as e:
            self.get_logger().error(f'Failed to load calibration file: {e}')
            self.get_logger().warn('Using default calibration values')
            return {
                'cam0': {
                    'intrinsics': [1203.26, 1163.82, 869.92, 595.30],
                    'distortion_coeffs': [-0.0328, 0.0512, -0.0033, -0.0006],
                    'resolution': [1640, 1232]
                },
                'cam1': {
                    'intrinsics': [1198.25, 1157.78, 827.98, 623.76],
                    'distortion_coeffs': [-0.0315, 0.0447, 0.0014, 0.0003],
                    'resolution': [1640, 1232],
                    'T_cn_cnm1': np.eye(4).tolist()
                }
            }

    # ===== Stereo Rectification =====
    def _prepare_rectification_maps(self):
        """Prepare stereo rectification maps using Fusiello's method
        Reference: A.Fusiello, E. Trucco, A. Verri: A compact algorithm for rectification of stereo pairs, 1999
        """
        # Parse intrinsics [fu, fv, cu, cv]
        intrinsics_0 = self.calib_data['cam0']['intrinsics']
        intrinsics_1 = self.calib_data['cam1']['intrinsics']
        
        K_cam0 = np.array([
            [intrinsics_0[0], 0, intrinsics_0[2]],
            [0, intrinsics_0[1], intrinsics_0[3]],
            [0, 0, 1]
        ])
        D_cam0 = np.array(self.calib_data['cam0']['distortion_coeffs'])
        
        K_cam1 = np.array([
            [intrinsics_1[0], 0, intrinsics_1[2]],
            [0, intrinsics_1[1], intrinsics_1[3]],
            [0, 0, 1]
        ])
        D_cam1 = np.array(self.calib_data['cam1']['distortion_coeffs'])
        
        # Extract extrinsics T_cam1_cam0
        T_cam1_cam0 = np.array(self.calib_data['cam1']['T_cn_cnm1'])
        R_cam1_cam0 = T_cam1_cam0[0:3, 0:3]
        t_cam1_cam0 = T_cam1_cam0[0:3, 3].reshape(3, 1)
        
        # Image size
        image_size = tuple(self.calib_data['cam0']['resolution'])
        self.image_size = image_size
        
        # Stereo rectification using Fusiello's method
        # Projection matrices (cam0 as world frame)
        Poa = np.matrix(K_cam0) @ np.hstack((np.matrix(np.eye(3)), np.matrix(np.zeros((3,1)))))
        Pob = np.matrix(K_cam1) @ np.hstack((np.matrix(R_cam1_cam0), np.matrix(t_cam1_cam0)))
        
        # Optical centers (in cam0's coord sys)
        c1 = -np.linalg.inv(Poa[:, 0:3]) @ Poa[:, 3]
        c2 = -np.linalg.inv(Pob[:, 0:3]) @ Pob[:, 3]
        
        # Get "mean" rotation between cams
        old_z_mean = (R_cam1_cam0[2, :].flatten() + np.eye(3)[2, 0:3]) / 2.0
        v1 = c1 - c2  # new x-axis = direction of baseline
        v2 = np.cross(np.matrix(old_z_mean).flatten(), v1.flatten()).T  # new y axis orthogonal to new x and mean old z
        v3 = np.cross(v1.flatten(), v2.flatten()).T  # orthogonal to baseline and new y
        
        # Normalize
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3 / np.linalg.norm(v3)
        
        # Create rotation matrix
        R = np.hstack((np.hstack((v1, v2)), v3)).T
        
        # New intrinsic parameters
        A = (K_cam0 + K_cam1) / 2.0
        
        # Rectifying transforms
        Ra = R  # cam0=world, then to rectified coords
        Rb = R @ np.linalg.inv(R_cam1_cam0)  # to world then to rectified coords
        
        # Compute rectification maps
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            K_cam0, D_cam0, Ra, A, image_size, cv2.CV_16SC2
        )
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            K_cam1, D_cam1, Rb, A, image_size, cv2.CV_16SC2
        )
        
        # Calculate baseline
        baseline = np.linalg.norm(t_cam1_cam0)
        
        # Store rectified projection matrices for depth calculation
        # P1 = [A | 0], P2 = [A | -baseline*fx*[1,0,0]^T]
        self.P1 = np.hstack((A, np.zeros((3, 1))))
        self.P2 = np.hstack((A, np.array([[-baseline * A[0, 0]], [0], [0]])))
        
        # Create Q matrix for disparity-to-depth conversion
        # depth = (baseline * fx) / disparity
        fx = A[0, 0]
        fy = A[1, 1]
        cx = A[0, 2]
        cy = A[1, 2]
        
        self.Q = np.array([
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, fx],
            [0, 0, -1/baseline, 0]
        ])
        
        self.baseline = baseline
        
        self.get_logger().info('Stereo rectification maps prepared (Fusiello method)')
        self.get_logger().info(f'Image size: {self.image_size}')
        self.get_logger().info(f'Baseline: {baseline:.6f} m')
        self.get_logger().info(f'Rectified fx: {fx:.2f}, fy: {fy:.2f}')

    def _rectify_images(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply rectification to stereo image pair"""
        if left is None or right is None:
            return left, right
        
        # Apply rectification maps
        left_rect = cv2.remap(left, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        
        return left_rect, right_rect

    # ===== Camera initialization =====
    def _init_cameras(self):
        """Initialize GStreamer camera captures for camera ID 0 and 1"""
        def build_gstreamer_pipeline(sensor_id, width, height, framerate, flip_method):
            return (
                f"nvarguscamerasrc sensor-id={sensor_id} "
                f"aeantibanding=2 "
                f"exposuretimerange=\"8000000 8000000\" "
                f"aelock=true awblock=true !"
                f"video/x-raw(memory:NVMM), "
                f"width={width}, height={height}, "
                f"format=NV12, framerate={framerate}/1 ! "
                f"nvvidconv flip-method={flip_method} ! "
                f"video/x-raw, format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink sync=false max-buffers=1 drop=true"
            )

        
        # Initialize left camera (sensor_id=1)
        gst_left = build_gstreamer_pipeline(1, self.cam_width, self.cam_height, self.cam_fps, self.cam_flip_method)
        self.cap_left = cv2.VideoCapture(gst_left, cv2.CAP_GSTREAMER)
        
        if not self.cap_left.isOpened():
            self.get_logger().error(f'Failed to open left camera (sensor_id=1)')
            self.get_logger().error(f'GStreamer pipeline: {gst_left}')
        else:
            self.get_logger().info(f'Left camera (sensor_id=1) opened: {self.cam_width}x{self.cam_height}@{self.cam_fps}fps')
        
        # Initialize right camera (sensor_id=0)
        gst_right = build_gstreamer_pipeline(0, self.cam_width, self.cam_height, self.cam_fps, self.cam_flip_method)
        self.cap_right = cv2.VideoCapture(gst_right, cv2.CAP_GSTREAMER)
        
        if not self.cap_right.isOpened():
            self.get_logger().error(f'Failed to open right camera (sensor_id=0)')
            self.get_logger().error(f'GStreamer pipeline: {gst_right}')
        else:
            self.get_logger().info(f'Right camera (sensor_id=0) opened: {self.cam_width}x{self.cam_height}@{self.cam_fps}fps')

    def _create_sgbm(self):
        num_disp = int(self.get_parameter('sgm_num_disparities').value)
        num_disp = max(16, (num_disp // 16) * 16)

        block_size = int(self.get_parameter('sgm_block_size').value)
        block_size = max(3, block_size | 1)

        sgbm_left = cv2.StereoSGBM_create(
            minDisparity=int(self.get_parameter('sgm_min_disparity').value),
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * block_size * block_size,
            P2=32 * block_size * block_size,
            uniquenessRatio=int(self.get_parameter('sgm_uniqueness_ratio').value),
            speckleWindowSize=int(self.get_parameter('sgm_speckle_window_size').value),
            speckleRange=int(self.get_parameter('sgm_speckle_range').value),
            disp12MaxDiff=int(self.get_parameter('sgm_disp12_max_diff').value),
            preFilterCap=int(self.get_parameter('sgm_pre_filter_cap').value),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        sgbm_right = cv2.ximgproc.createRightMatcher(sgbm_left)

        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm_left)
        wls.setLambda(8000)     # smoothness
        wls.setSigmaColor(1.5)  # edge preservation

        self.sgbm_left = sgbm_left
        self.sgbm_right = sgbm_right
        self.wls = wls

        return sgbm_left


    def _read_stereo_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read frames from both cameras"""
        left_frame = None
        right_frame = None
        
        if self.cap_left is not None and self.cap_left.isOpened():
            ret, left_frame = self.cap_left.read()
            if not ret:
                self.get_logger().warn('Failed to read from left camera')
                left_frame = None
        
        if self.cap_right is not None and self.cap_right.isOpened():
            ret, right_frame = self.cap_right.read()
            if not ret:
                self.get_logger().warn('Failed to read from right camera')
                right_frame = None
        
        return left_frame, right_frame

    def __del__(self):
        """Cleanup camera captures"""
        if hasattr(self, 'cap_left') and self.cap_left is not None:
            self.cap_left.release()
        if hasattr(self, 'cap_right') and self.cap_right is not None:
            self.cap_right.release()

    # ===== Timer loop (Camera read + Inference only) =====
    def _on_timer(self):
        t0 = time.time()
        
        # Read frames directly from cameras
        left_frame, right_frame = self._read_stereo_frames()
        
        if left_frame is None or right_frame is None:
            self.get_logger().warn('Failed to capture frames from cameras')
            return

        # Rectify images using calibration
        left_rect, right_rect = self._rectify_images(left_frame, right_frame)

        # Build stereo list with single pair (left=cam0, right=cam1, yaw=0 deg)
        
        stereo_list = [(left_rect, right_rect, 0.0)]  # Forward-facing stereo pair

        # Run stereo matching based on selected method
        if self.stereo_method == 'hitnet':
            disparities = self._run_hitnet_on_steros(stereo_list)
        elif self.stereo_method == 'fastacv':
            disparities = self._run_fastacv_on_steros(stereo_list)
        else:
            disparities = self._run_sgm_on_steros(stereo_list)
        
        if not disparities:
            self.get_logger().warn(f'No disparities returned from {self.stereo_method.upper()}, skipping processing')
            return

        # Push to processing queue
        with self.queue_lock:
            self.disparity_queue.append({
                'stereo_list': stereo_list,
                'disparities': disparities,
                'timestamp': self.get_clock().now().to_msg(),
            })

        t1 = time.time()
        self.get_logger().info(f'Timer ({self.stereo_method.upper()} inference): {(t1 - t0)*1000.0:.2f} ms')

    # ===== Processing loop (runs continuously in separate thread) =====
    def _processing_loop(self):
        """Continuous loop to process disparity from queue"""
        while rclpy.ok():
            item = None
            with self.queue_lock:
                if self.disparity_queue:
                    item = self.disparity_queue.popleft()
            
            if item is None:
                time.sleep(0.001)  # Sleep briefly if no data
                continue

            t0 = time.time()
            stereo_list = item['stereo_list']
            disparities = item['disparities']
            timestamp = item['timestamp']

            # Process disparities
            for (left, right, yaw_deg), disp in zip(stereo_list, disparities):
                if disp is None:
                    self.get_logger().warn('Missing disparity for pair, skipping')
                    continue

                depth = self._disparity_to_depth(disp)
                
                # Show debug windows

                
                if self.publish_depth and self.pub_depth is not None:
                    # Visualize depth
                    d = np.clip(disp / 128.0, 0.0, 1.0)
                    d8 = (np.uint8(d * 255.0))
                    heat_map = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
                    cv2.imshow('Left Image', left)
                    cv2.imshow('Right Image', right)
                    cv2.imshow('Depth Debug', d8)
                    
                    # Publish depth as float32
                    depth_msg = Image()
                    depth_msg.header.stamp = timestamp
                    depth_msg.header.frame_id = 'stereo_depth'
                    depth_msg.height, depth_msg.width = depth.shape
                    depth_msg.encoding = '32FC1'
                    depth_msg.is_bigendian = False
                    depth_msg.step = depth_msg.width * 4
                    depth_msg.data = depth.astype(np.float32).tobytes()
                    self.pub_depth.publish(depth_msg)
                cv2.waitKey(1)
            t1 = time.time()
            self.get_logger().info(f'Processing thread (depth + publish): {(t1 - t0)*1000.0:.2f} ms')


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

        # Run inference
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

    # ===== FastACV wrapper =====
    def _run_fastacv_on_steros(self, stereo_list: List[Tuple[np.ndarray, np.ndarray, float]]) -> List[np.ndarray]:
        if self.fastacv is None:
            self.get_logger().warn('FastACVTRT is not initialized; skip inference')
            return []
        
        t0 = time.time()
        # Preprocess each stereo pair
        left_tensors = []
        right_tensors = []
        orig_sizes = []
        padded_sizes = []
        
        for left, right, _yaw in stereo_list:
            if left is None or right is None:
                left_tensors.append(None)
                right_tensors.append(None)
                orig_sizes.append(None)
                padded_sizes.append(None)
                continue

            if left.shape != right.shape:
                self.get_logger().warn('Left/right shapes differ, skipping this pair')
                left_tensors.append(None)
                right_tensors.append(None)
                orig_sizes.append(None)
                padded_sizes.append(None)
                continue

            # Preprocess using FastACV's method
            target_W, target_H = 320, 256
            if left.shape[:2] != (target_H, target_W):
                left = cv2.resize(left, (target_W, target_H))
            if right.shape[:2] != (target_H, target_W):
                right = cv2.resize(right, (target_W, target_H))
            left_tensor, right_tensor, orig_size, padded_size = self.fastacv.preprocess(left, right)
            left_tensors.append(left_tensor)
            right_tensors.append(right_tensor)
            orig_sizes.append(orig_size)
            padded_sizes.append(padded_size)

        # Filter out None tensors
        valid_indices = [i for i, (lt, rt) in enumerate(zip(left_tensors, right_tensors)) if lt is not None and rt is not None]
        if not valid_indices:
            return []

        valid_left = [left_tensors[i] for i in valid_indices]
        valid_right = [right_tensors[i] for i in valid_indices]
        valid_orig_sizes = [orig_sizes[i] for i in valid_indices]
        valid_padded_sizes = [padded_sizes[i] for i in valid_indices]

        # Pad or trim inputs to match stream_number
        batch_left = []
        batch_right = []
        for i in range(self.fastacv.stream_number):
            if i < len(valid_left):
                batch_left.append(valid_left[i])
                batch_right.append(valid_right[i])
            else:
                batch_left.append(valid_left[-1])
                batch_right.append(valid_right[-1])

        # Run inference
        ret = self.fastacv.do_inference(batch_left, batch_right)
        if ret != 0:
            self.get_logger().error(f'FastACV inference failed with code {ret}')
            return []

        outputs = self.fastacv.get_output()
        disparities: List[np.ndarray] = []
        
        print(len(outputs))
        # Postprocess disparities
        for i in range(min(len(valid_indices), len(outputs))):
            disp = outputs[i]
            disp_processed = np.squeeze(disp)
            disparities.append(disp_processed)
        
        t1 = time.time()
        self.get_logger().info(f'FastACV inference time: {(t1 - t0)*1000.0:.2f} ms')
        return disparities

    # ===== SGM wrapper =====
    def _run_sgm_on_steros(self, stereo_list: List[Tuple[np.ndarray, np.ndarray, float]]) -> List[np.ndarray]:
        if self.sgbm is None:
            self.get_logger().error('SGM matcher is not initialized')
            return []

        t0 = time.time()
        disparities: List[np.ndarray] = []

        for left, right, _yaw in stereo_list:
            left = cv2.resize(left, (640, 480))
            right = cv2.resize(right, (640, 480))
            if left is None or right is None:
                disparities.append(None)
                continue

            if left.shape != right.shape:
                self.get_logger().warn('Left/right shapes differ, skipping this pair')
                disparities.append(None)
                continue

            # Convert to grayscale for SGM
            if left.ndim == 3:
                left_g = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            else:
                left_g = left
            if right.ndim == 3:
                right_g = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            else:
                right_g = right

            disp_raw = self.sgbm.compute(left_g, right_g).astype(np.float32) / 16.0
            disp_raw[disp_raw < 0.1] = 0.0
            disparities.append(disp_raw)

        t1 = time.time()
        self.get_logger().info(f'SGM inference time: {(t1 - t0)*1000.0:.2f} ms')
        return disparities

    def _disparity_to_depth(self, disp: np.ndarray) -> np.ndarray:
        """Convert disparity to depth using calibrated parameters"""
        # Use calibrated baseline and focal length
        baseline = self.calib_data['baseline']
        fx = self.calib_data['left']['fx']  # Use left camera focal length
        
        # depth = (baseline * fx) / disparity
        disp_safe = np.where(disp > 0.1, disp, 0.1)
        depth = (baseline * fx) / disp_safe
        depth[disp <= 0.1] = 0.0
        depth = np.clip(depth, 0.0, self.max_depth_m)
        
        return depth


def main(argv=None):
    rclpy.init(args=argv)
    node = StereoDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()