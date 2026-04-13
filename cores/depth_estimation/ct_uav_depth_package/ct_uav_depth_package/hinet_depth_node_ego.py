#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import cv2
import numpy as np
import threading
import time
from collections import deque
from models.hitnet.hitnet import HitnetTRT

class HitnetDepthNode(Node):
    def __init__(self):
        super().__init__('hitnet_depth_node')

        # --- PARAMETERS ---
        self.declare_parameter('update_rate_hz', 15.0)
        self.declare_parameter('camera_width', 1280)
        self.declare_parameter('camera_height', 720)
        self.declare_parameter('camera_fps', 30)
        self.declare_parameter('onnx_path', '')
        self.declare_parameter('engine_path', '')
        
        # Stereo Params (CRITICAL)
        self.declare_parameter('baseline', 0.12) # (mét) của camera stereo 
        self.declare_parameter('fov_deg', 100.0) # Dùng để tính fx nếu không có calib
        
        # Load params
        self.fps = self.get_parameter('update_rate_hz').value
        self.cam_w = self.get_parameter('camera_width').value
        self.cam_h = self.get_parameter('camera_height').value
        self.cam_fps = self.get_parameter('camera_fps').value
        self.baseline = self.get_parameter('baseline').value
        
        # HitNet Config
        self.hitnet_w = 320
        self.hitnet_h = 240
        
        # Init HitNet
        engine_path = self.get_parameter('engine_path').value
        onnx_path = self.get_parameter('onnx_path').value
        self.hitnet = HitnetTRT(show_info=True)
        if self.hitnet.init(onnx_path, engine_path, 1) != 0:
            self.get_logger().error("Failed to init HitNet!")
            rclpy.shutdown()

        # Init Cameras (GStreamer)
        self.cap_left = self._init_camera(0)
        self.cap_right = self._init_camera(1)

        # Publisher
        self.pub_depth = self.create_publisher(Image, '/cam0/depth_map', 10)

        # Threading
        self.queue = deque(maxlen=2)
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._processing_loop)
        self.thread.start()

        # Timer for Capture
        self.create_timer(1.0 / self.fps, self._capture_callback)
        self.get_logger().info(" HitNet Depth Node Started!")

    def _init_camera(self, sensor_id):
        pipeline = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={self.cam_w}, height={self.cam_h}, "
            f"format=NV12, framerate={self.cam_fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, format=BGRx ! videoconvert ! "
            f"video/x-raw, format=BGR ! appsink drop=1"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    def _capture_callback(self):
        if not self.cap_left.isOpened() or not self.cap_right.isOpened(): return

        ret1, frame_left = self.cap_left.read()
        ret2, frame_right = self.cap_right.read()

        if not ret1 or not ret2: return

        # Timestamp ngay khi chụp
        now = self.get_clock().now()

        with self.lock:
            self.queue.append((frame_left, frame_right, now))

    def _processing_loop(self):
        while self.running and rclpy.ok():
            data = None
            with self.lock:
                if self.queue:
                    data = self.queue.popleft()
            
            if not data:
                time.sleep(0.005)
                continue

            left, right, timestamp = data
            
            # 1. Preprocess
            left_g = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_g = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            left_in = cv2.resize(left_g, (self.hitnet_w, self.hitnet_h))
            right_in = cv2.resize(right_g, (self.hitnet_w, self.hitnet_h))
            
            # Normalize & Batch
            img_L = left_in.astype(np.float32) / 255.0
            img_R = right_in.astype(np.float32) / 255.0
            input_tensor = np.stack([img_L, img_R], axis=0)[np.newaxis, ...] # (1, 2, H, W)

            # 2. Inference
            self.hitnet.do_inference([input_tensor])
            disp = self.hitnet.get_output()[0].squeeze() # (H, W)

            # 3. Disparity -> Depth (Meters)
            # fx cần tính theo resolution của HitNet (320x240)
            # Giả sử FOV ngang ~100 độ (cần calib để chính xác!)
            fov_rad = np.radians(self.get_parameter('fov_deg').value)
            fx = (self.hitnet_w / 2.0) / np.tan(fov_rad / 2.0)
            
            # depth = (fx * baseline) / disparity
            with np.errstate(divide='ignore'):
                depth = (fx * self.baseline) / disp
            
            depth[disp <= 0.1] = 0.0
            depth[depth > 20.0] = 0.0 # Clip max distance
            depth = depth.astype(np.float32)

            # 4. Publish
            msg = Image()
            msg.header = Header()
            msg.header.stamp = timestamp.to_msg()
            msg.header.frame_id = "camera_link" # QUAN TRỌNG
            msg.height = self.hitnet_h
            msg.width = self.hitnet_w
            msg.encoding = "32FC1"
            msg.is_bigendian = False
            msg.step = self.hitnet_w * 4
            msg.data = depth.tobytes()
            
            self.pub_depth.publish(msg)

    def destroy_node(self):
        self.running = False
        self.thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HitnetDepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()