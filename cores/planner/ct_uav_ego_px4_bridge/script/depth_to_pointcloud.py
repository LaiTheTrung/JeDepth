#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs_py import point_cloud2

class SimpleDepthToCloud(Node):
    def __init__(self):
        super().__init__('simple_depth_to_cloud')

        # --- CẤU HÌNH CAMERA ---
        self.fx = 128.0
        self.fy = 128.0
        self.cx = 128.0
        self.cy = 72.0

        self.subscription = self.create_subscription(
            Image,
            '/stereo/depth_map', 
            self.depth_callback,
            10
        )
        
        self.publisher = self.create_publisher(PointCloud2, '/test_point_cloud', 10)
        self.bridge = CvBridge()
        self.get_logger().info("Node Started: Depth -> PointCloud (Frame: map)")

    def depth_callback(self, msg):
        try:
            # 1. Convert ROS Image -> Numpy
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # 2. Tạo grid tọa độ
            height, width = depth_img.shape
            v, u = np.indices((height, width))

            # 3. Lọc nhiễu
            mask = (depth_img > 0.2) & (depth_img < 20.0) & np.isfinite(depth_img)
            
            z_valid = depth_img[mask] # Đây là Z trong hệ Camera (Depth)
            u_valid = u[mask]
            v_valid = v[mask]

            # 4. Tính toán tọa độ trong HỆ TRỤC CAMERA (Optical Frame)
            # Z = Hướng tới trước, X = Sang phải, Y = Hướng xuống
            x_cam = (u_valid - self.cx) * z_valid / self.fx
            y_cam = (v_valid - self.cy) * z_valid / self.fy
            z_cam = z_valid

            # Sử dụng HỆ TRỤC CAMERA (Z tới, X phải, Y xuống) ĐỀ RVIZ TỰ CHUYỂN QUA WORLD
            # Xóa đoạn code convert hệ quy chiếu ENU đi vì RViz có khả năng tự TF
            points = np.stack([x_cam, y_cam, z_cam], axis=1)

            # 6. Publish với frame_id là 'camera_link'
            header = Header()
            header.stamp = msg.header.stamp
            header.frame_id = "camera_link"  # <--- ĐỔI TỪ 'map' -> 'camera_link' ĐỂ RVIZ TỰ HIỂU CÂY TF

            pc2_msg = point_cloud2.create_cloud_xyz32(header, points)
            self.publisher.publish(pc2_msg)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SimpleDepthToCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()