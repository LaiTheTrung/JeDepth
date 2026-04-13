"""
This code aim to use recorder to record the data from stereo camera and IMU sensors.
It will firstly read data from ros2 subcribers and save them into folders:
Folders structure:
data/
    left/
        image_<timestamp>.png
    right/
        image_<timestamp>.png
    imu_pos/
        imu_<timestamp>.yaml
    imu_att/
        imu_<timestamp>.yaml
topic list: 
- /stereo/left/image_raw
- /stereo/right/image_raw
- /fmu/out/vehicle_local_position
- /fmu/out/vehicle_attitude
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from px4_msgs.msg import VehicleLocalPosition
import cv2
import yaml
import numpy as np
from pathlib import Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import queue
import threading


class StereoIMURecorder(Node):
    def __init__(self, data_dir='data'):
        super().__init__('stereo_imu_recorder')
        
        # Setup data directories
        self.data_dir = Path(data_dir)
        self.left_dir = self.data_dir / 'left'
        self.right_dir = self.data_dir / 'right'
        self.imu_dir = self.data_dir / 'imu'
        
        # Create directories
        self.left_dir.mkdir(parents=True, exist_ok=True)
        self.right_dir.mkdir(parents=True, exist_ok=True)
        self.imu_dir.mkdir(parents=True, exist_ok=True)
        
        # Use best-effort QoS for high-rate sensor streams
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Background writer for images to reduce callback latency
        self.image_queue = queue.Queue(maxsize=200)
        self.image_worker = threading.Thread(target=self._image_writer_loop, daemon=True)
        self.image_worker.start()

        # Create subscribers
        self.left_sub = self.create_subscription(
            CompressedImage,
            '/stereo/left/image_raw/compressed',
            self.left_image_callback,
            qos_sensor
        )
        
        self.right_sub = self.create_subscription(
            CompressedImage,
            '/stereo/right/image_raw/compressed',
            self.right_image_callback,
            qos_sensor
        )
        self.imu_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.imu_callback,
            qos_sensor
        )
        
        self.get_logger().info(f'Recorder initialized. Saving data to: {self.data_dir.absolute()}')
        self.get_logger().info('Subscribed to:')
        self.get_logger().info('  - /stereo/left/image_raw/compressed')
        self.get_logger().info('  - /stereo/right/image_raw/compressed')
        self.get_logger().info('  - /fmu/out/vehicle_local_position')
        
        # Statistics
        self.left_count = 0
        self.right_count = 0
        self.imu_count = 0
        self.dropped_images = 0

    def _image_writer_loop(self):
        while True:
            item = self.image_queue.get()
            if item is None:
                break
            side, timestamp, cv_image = item
            try:
                if side == 'left':
                    filename = self.left_dir / f'image_{timestamp:.6f}.png'
                else:
                    filename = self.right_dir / f'image_{timestamp:.6f}.png'
                # Faster PNG encoding (lower compression)
                cv2.imwrite(
                    str(filename),
                    cv_image,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1]
                )
            except Exception as e:
                self.get_logger().error(f'Error saving {side} image: {str(e)}')
            finally:
                self.image_queue.task_done()
    
    def left_image_callback(self, msg):
        try:
            # Decode JPEG compressed image (RGB color)
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                self.get_logger().error('Failed to decode left image')
                return
            
            # Generate timestamp
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            self.left_count += 1
            try:
                self.image_queue.put_nowait(('left', timestamp, cv_image))
            except queue.Full:
                self.dropped_images += 1
            if self.left_count % 10 == 0:
                self.get_logger().info(
                    f'Left images received: {self.left_count}, dropped: {self.dropped_images}'
                )
                
        except Exception as e:
            self.get_logger().error(f'Error processing left image: {str(e)}')
    
    def right_image_callback(self, msg):
        try:
            # Decode JPEG compressed image (RGB color)
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                self.get_logger().error('Failed to decode right image')
                return
            
            # Generate timestamp
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            self.right_count += 1
            try:
                self.image_queue.put_nowait(('right', timestamp, cv_image))
            except queue.Full:
                self.dropped_images += 1
            if self.right_count % 10 == 0:
                self.get_logger().info(
                    f'Right images received: {self.right_count}, dropped: {self.dropped_images}'
                )
                
        except Exception as e:
            self.get_logger().error(f'Error processing right image: {str(e)}')
    
    def imu_callback(self, msg):
        try:
            # Generate timestamp from message
            timestamp = msg.timestamp * 1e-6  # Convert microseconds to seconds
            
            # Prepare IMU data dictionary
            imu_data = {
                'timestamp': float(timestamp),
                'timestamp_sample': int(msg.timestamp_sample),
                'position': {
                    'x': float(msg.x),
                    'y': float(msg.y),
                    'z': float(msg.z)
                },
                'velocity': {
                    'vx': float(msg.vx),
                    'vy': float(msg.vy),
                    'vz': float(msg.vz)
                },
                'acceleration': {
                    'ax': float(msg.ax),
                    'ay': float(msg.ay),
                    'az': float(msg.az)
                },
                'heading': float(msg.heading),
                'delta_heading': float(msg.delta_heading),
                'xy_valid': bool(msg.xy_valid),
                'z_valid': bool(msg.z_valid),
                'v_xy_valid': bool(msg.v_xy_valid),
                'v_z_valid': bool(msg.v_z_valid)
            }
            
            # Save to YAML file
            filename = self.imu_dir / f'imu_{timestamp:.6f}.yaml'
            with open(filename, 'w') as f:
                yaml.dump(imu_data, f, default_flow_style=False)
            
            self.imu_count += 1
            if self.imu_count % 10 == 0:
                self.get_logger().info(f'IMU data saved: {self.imu_count}')
                
        except Exception as e:
            self.get_logger().error(f'Error saving IMU data: {str(e)}')
    
    def shutdown(self):
        self.get_logger().info('Shutting down recorder...')
        try:
            self.image_queue.put_nowait(None)
        except queue.Full:
            self.image_queue.put(None)
        self.image_worker.join(timeout=2.0)
        self.get_logger().info(f'Total saved - Left: {self.left_count}, Right: {self.right_count}, IMU: {self.imu_count}')


def main(args=None):
    rclpy.init(args=args)
    
    # Create recorder node with optional custom data directory
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    
    recorder = None
    try:
        recorder = StereoIMURecorder(data_dir=data_dir)
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        if recorder:
            recorder.get_logger().info('Keyboard interrupt detected')
    except Exception as e:
        if recorder:
            recorder.get_logger().error(f'Error: {str(e)}')
    finally:
        if recorder:
            recorder.shutdown()
            try:
                recorder.destroy_node()
            except:
                pass
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()

