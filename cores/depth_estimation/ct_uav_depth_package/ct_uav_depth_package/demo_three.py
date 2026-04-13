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
from cv_bridge import CvBridge
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
        self.declare_parameter('max_depth_m', 10.0)
        
        # Camera calibration file
        self.declare_parameter('calibration_file', '')
        
        # Camera parameters
        self.declare_parameter('cam_width', 640)
        self.declare_parameter('cam_height', 480)
        self.declare_parameter('cam_fps', 20)
        self.declare_parameter('cam_flip_method', 0)
        
        # ROS2 Image topic parameters
        self.declare_parameter('image_topic', '/fast_stream_node_stereo/fps')
        self.declare_parameter('topic_fps', 30)  # Can be set

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
        self.declare_parameter('model_w', 320)  # model input width
        self.declare_parameter('model_h', 240)  # model input height
        
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
        
        # Image topic parameters
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.topic_fps = int(self.get_parameter('topic_fps').value)
        
        # Model size parameters
        self.model_height = int(self.get_parameter('model_h').value)
        self.model_width = int(self.get_parameter('model_w').value)
        
        # CvBridge for converting ROS Image messages
        self.bridge = CvBridge()
        
        # Latest stereo frame from topic
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Load camera calibration
        calibration_file = self.get_parameter('calibration_file').value
        self.calib_data = self._load_calibration(calibration_file)
        
        # Prepare rectification maps
        self._prepare_rectification_maps()
        
        # Subscribe to stereo image topic
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            10
        )
        
        self.get_logger().info(f'Subscribed to image topic: {self.image_topic}')

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

        # ===== Publishers =====
        self.pub_depth = self.create_publisher(Image, '/stereo/depth_map', 10) if self.publish_depth else None

        # ===== Queues & threads =====
        self.disparity_queue = deque(maxlen=2)  # Queue for disparity data
        self.queue_lock = threading.Lock()

        # Processing thread: process disparity -> depth -> publish
        self.proc_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.proc_thread.start()

        # ===== Timer (Process + Inference) =====
        self.timer = self.create_timer(1.0 / self.update_rate_hz, self._on_timer)

        self.get_logger().info(f'Stereo Depth node initialized with {self.stereo_method.upper()} and ROS2 image topic subscription')

    # ===== Camera calibration loading =====
    def _load_calibration(self, calibration_file: str) -> dict:
        """Load camera calibration from YAML file"""
        if not calibration_file or not os.path.exists(calibration_file):
            self.get_logger().warn(f'Calibration file not found: {calibration_file}, using default values')
            # Return default calibration
            return {
                'left': {
                    'fx': 367.9,
                    'fy': 367.9,
                    'cx': 320.0,
                    'cy': 240.0,
                    'k1': 0.0,
                    'k2': 0.0,
                    'p1': 0.0,
                    'p2': 0.0
                },
                'right': {
                    'fx': 367.9,
                    'fy': 367.9,
                    'cx': 320.0,
                    'cy': 240.0,
                    'k1': 0.0,
                    'k2': 0.0,
                    'p1': 0.0,
                    'p2': 0.0
                },
                'baseline': 0.16   # giữ nguyên nếu stereo thật
            }

        
        try:
            with open(calibration_file, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            calib = yaml_data['stereo_camera_calibration']
            
            # Extract left camera parameters
            left_intr = calib['left']['intrinsics']
            left_dist = calib['left']['distortion_coeffs']
            
            # Extract right camera parameters
            right_intr = calib['right']['intrinsics']
            right_dist = calib['right']['distortion_coeffs']
            
            # Extract baseline
            baseline = calib['extrinsics']['baseline_m']
            
            calib_data = {
                'left': {
                    'fx': left_intr['fx'],
                    'fy': left_intr['fy'],
                    'cx': left_intr['cx'],
                    'cy': left_intr['cy'],
                    'k1': left_dist['k1'],
                    'k2': left_dist['k2'],
                    'p1': left_dist['p1'],
                    'p2': left_dist['p2']
                },
                'right': {
                    'fx': right_intr['fx'],
                    'fy': right_intr['fy'],
                    'cx': right_intr['cx'],
                    'cy': right_intr['cy'],
                    'k1': right_dist['k1'],
                    'k2': right_dist['k2'],
                    'p1': right_dist['p1'],
                    'p2': right_dist['p2']
                },
                'baseline': 0.16
            }
            
            self.get_logger().info(f'Loaded calibration from: {calibration_file}')
            self.get_logger().info(f'Baseline: {baseline:.6f} m')
            self.get_logger().info(f'Left fx: {left_intr["fx"]:.2f}, fy: {left_intr["fy"]:.2f}')
            self.get_logger().info(f'Right fx: {right_intr["fx"]:.2f}, fy: {right_intr["fy"]:.2f}')
            
            return calib_data
            
        except Exception as e:
            self.get_logger().error(f'Failed to load calibration file: {e}')
            self.get_logger().warn('Using default calibration values')
            return {
                'left': {
                    'fx': 367.9,
                    'fy': 367.9,
                    'cx': 320.0,
                    'cy': 240.0,
                    'k1': 0.0,
                    'k2': 0.0,
                    'p1': 0.0,
                    'p2': 0.0
                },
                'right': {
                    'fx': 367.9,
                    'fy': 367.9,
                    'cx': 320.0,
                    'cy': 240.0,
                    'k1': 0.0,
                    'k2': 0.0,
                    'p1': 0.0,
                    'p2': 0.0
                },
                'baseline': 0.16   # giữ nguyên nếu stereo thật
            }

    # ===== Stereo Rectification =====
    def _prepare_rectification_maps(self):
        """Prepare stereo rectification maps for undistortion and rectification"""
        # Build camera matrices
        K_left = np.array([
            [self.calib_data['left']['fx'], 0, self.calib_data['left']['cx']],
            [0, self.calib_data['left']['fy'], self.calib_data['left']['cy']],
            [0, 0, 1]
        ], dtype=np.float64)
        
        K_right = np.array([
            [self.calib_data['right']['fx'], 0, self.calib_data['right']['cx']],
            [0, self.calib_data['right']['fy'], self.calib_data['right']['cy']],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Build distortion coefficients (k1, k2, p1, p2, k3)
        D_left = np.array([
            self.calib_data['left']['k1'],
            self.calib_data['left']['k2'],
            self.calib_data['left']['p1'],
            self.calib_data['left']['p2'],
            0.0  # k3
        ], dtype=np.float64)
        
        D_right = np.array([
            self.calib_data['right']['k1'],
            self.calib_data['right']['k2'],
            self.calib_data['right']['p1'],
            self.calib_data['right']['p2'],
            0.0  # k3
        ], dtype=np.float64)
        
        # Extract rotation and translation from calibration
        # Assuming identity rotation and translation along X-axis (baseline)
        R = np.eye(3, dtype=np.float64)  # Identity rotation
        T = np.array([self.calib_data['baseline'], 0, 0], dtype=np.float64)  # Translation
        
        # Image size (will be updated when first frame is captured)
        self.image_size = (self.cam_width, self.cam_height)
        
        # Compute stereo rectification
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            K_left, D_left,
            K_right, D_right,
            self.image_size,
            R, T,
            alpha=0,  # 0 = crop to valid pixels only, 1 = keep all pixels
            newImageSize=self.image_size
        )
        
        # Compute rectification maps
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            K_left, D_left, R1, P1, self.image_size, cv2.CV_32FC1
        )
        
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            K_right, D_right, R2, P2, self.image_size, cv2.CV_32FC1
        )
        
        # Store matrices for depth calculation
        self.Q = Q
        self.P1 = P1
        self.P2 = P2
        
        self.get_logger().info('Stereo rectification maps prepared')
        self.get_logger().info(f'Image size: {self.image_size}')

    def _rectify_images(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply rectification to stereo image pair"""
        if left is None or right is None:
            return left, right
        
        # Apply rectification maps
        left_rect = cv2.remap(left, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        
        return left_rect, right_rect

    # ===== Image topic callback =====
    def _image_callback(self, msg: Image):
        """Callback for receiving stereo image from ROS2 topic"""
        try:
            # Convert ROS Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Store the latest frame
            with self.frame_lock:
                self.latest_frame = frame
                
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

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
        """Split the received frame into left and right images"""
        with self.frame_lock:
            frame = self.latest_frame
        
        if frame is None:
            return None, None
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Split frame into left and right
        # left = frame[:, :width//2, :]
        # right = frame[:, width//2:, :]
        left = frame[:, :width//2, :]
        right = frame[:, width//2:, :]
        
        return left, right

    # ===== Timer loop (Process + Inference) =====
    def _on_timer(self):
        t0 = time.time()
        
        # Split latest frame into left and right
        left_frame, right_frame = self._read_stereo_frames()
        
        if left_frame is None or right_frame is None:
            self.get_logger().warn('No stereo frames available from topic')
            return

        # Build stereo list with single pair (left=cam0, right=cam1, yaw=0 deg)
        
        stereo_list = [(left_frame, right_frame, 0.0)]  # Forward-facing stereo pair

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
                    # disp_color = self.visualize_disparity(disp, 128)
                    # cv2.imshow('Left Image', left)
                    # cv2.imshow('Right Image', right)
                    # cv2.imshow('Depth Debug', disp_color)
                    
                    # Resize depth to 256x144
                    depth_resized = cv2.resize(depth, (256, 144), interpolation=cv2.INTER_NEAREST)
                    
                    # Publish depth as float32
                    depth_msg = Image()
                    depth_msg.header.stamp = timestamp
                    depth_msg.header.frame_id = 'stereo_depth'
                    depth_msg.height, depth_msg.width = depth_resized.shape
                    depth_msg.encoding = '32FC1'
                    depth_msg.is_bigendian = False
                    depth_msg.step = depth_msg.width * 4
                    depth_msg.data = depth_resized.astype(np.float32).tobytes()
                    self.pub_depth.publish(depth_msg)
                # cv2.waitKey(1)
            t1 = time.time()
            self.get_logger().info(f'Processing thread (depth + publish): {(t1 - t0)*1000.0:.2f} ms')

    # ===== Input resize and edit calib data =====
    def _resize_and_update_calib(self, img, target_size, is_left=True):
        """Resize input image and update calibration data accordingly"""
        orig_h, orig_w = img.shape[:2]
        target_w, target_h = target_size

        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        # Resize image
        img_resized = cv2.resize(img, (target_w, target_h))

        # Update calibration data
        cam_key = 'left' if is_left else 'right'
        self.calib_data[cam_key]['fx'] *= scale_x
        self.calib_data[cam_key]['fy'] *= scale_y
        self.calib_data[cam_key]['cx'] *= scale_x
        self.calib_data[cam_key]['cy'] *= scale_y

        return img_resized
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

            # Resize to model input size
            if left_g.shape[:2] != (self.model_height, self.model_width):
                left_g = cv2.resize(left_g, (self.model_width, self.model_height))
            if right_g.shape[:2] != (self.model_height, self.model_width):
                right_g = cv2.resize(right_g, (self.model_width, self.model_height))

            stacked = np.array([left_g, right_g], dtype=np.float32)  # (2, H, W)
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
            if left.shape[:2] != (self.model_height, self.model_width):
                left = cv2.resize(left, (self.model_width, self.model_height))
            if right.shape[:2] != (self.model_height, self.model_width):
                right = cv2.resize(right, (self.model_width, self.model_height))
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

            # Compute left and right disparity maps for WLS filter
            disp_left = self.sgbm_left.compute(left_g, right_g)
            disp_right = self.sgbm_right.compute(right_g, left_g)
            
            # Apply WLS filter for better edge-aware smoothing
            disp_filtered = self.wls.filter(disp_left, left_g, None, disp_right)
            
            # Convert to float and scale
            disp_raw = disp_filtered.astype(np.float32) / 16.0
            disp_raw[disp_raw < 0.1] = 0.0
            disparities.append(disp_raw)

        t1 = time.time()
        self.get_logger().info(f'SGM inference time: {(t1 - t0)*1000.0:.2f} ms')
        return disparities

    def _disparity_to_depth(self, disp: np.ndarray) -> np.ndarray:
        """Convert disparity to depth using calibrated parameters"""
        # Use calibrated baseline and focal length
        baseline = np.float32(self.calib_data['baseline'])
        fx = np.float32(self.calib_data['left']['fx'])  # Use left camera focal length
        # depth = (baseline * fx) / disparity
        min_disp = (baseline * fx) / self.max_depth_m
        disp_safe = np.where(disp > min_disp, disp, min_disp).astype(np.float32)
        depth = (baseline * fx) / disp_safe
        depth[disp <= min_disp] = self.max_depth_m
        depth = np.clip(depth, 0.0, self.max_depth_m)
        
        return depth

    def visualize_disparity(self, disp: np.ndarray, maxdisp: int = 128) -> np.ndarray:
        """
        Visualize disparity map with colormap
        Based on test_onnx.py visualization logic
        """
        # Normalize to 0-255
        if len(disp.shape) >= 2:
            if len(disp.shape) == 3 and disp.shape[0] == 1:
                disp = disp[0]
            elif len(disp.shape) == 4:
                disp = disp[0,0]
        print(f"Visualizing disparity with shape: {disp.shape}")
        disp_vis = (disp / maxdisp * 255.0).astype(np.uint8)
        
        # Apply colormap
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        
        return disp_color


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