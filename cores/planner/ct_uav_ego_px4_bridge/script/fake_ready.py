#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import time

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('force_ready_node')
    
    pub = node.create_publisher(Bool, '/bridge/force_ready', 10)
    node.get_logger().info("Đang gửi tín hiệu FORCE_READY...")
    
    # Đợi publisher kết nối
    time.sleep(0.5)
    
    msg = Bool()
    msg.data = True
    pub.publish(msg)
    
    node.get_logger().info("✅ Đã ép bridge sang state READY. Giờ bạn có thể dùng '2D Nav Goal' trên RViz!")
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
