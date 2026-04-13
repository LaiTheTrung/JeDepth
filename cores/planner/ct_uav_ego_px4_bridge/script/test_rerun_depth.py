#!/usr/bin/env python3
import numpy as np

# --- VÁ LỖI TƯƠNG THÍCH NUMPY < 2.0 VÀ RERUN 0.23.1 ---
_orig_asarray = np.asarray
def _patched_asarray(*args, **kwargs):
    kwargs.pop('copy', None)
    return _orig_asarray(*args, **kwargs)
np.asarray = _patched_asarray
# -----------------------------------------------------

import rerun as rr
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# --- Cấu hình Rerun ---
LAPTOP_IP = "10.5.10.67" 

class RerunDepthNode(Node):
    def __init__(self):
        super().__init__('rerun_depth_node')
        self.bridge = CvBridge()
        
        # 1. Khởi tạo Rerun
        rr.init("Ego_Stereo_Depth")
        rr.connect_grpc(f"rerun+http://{LAPTOP_IP}:9876/proxy")
        self.get_logger().info(f"✅ Đã kết nối Rerun Viewer tại: {LAPTOP_IP}:9876")
        
        # 2. Đăng ký subscribe topic depth
        # Dựa trên file launch, topic là: /stereo/depth_map
        self.sub_depth = self.create_subscription(
            Image,
            '/stereo/depth_map',
            self.depth_callback,
            10
        )
        self.get_logger().info("Đang lắng nghe topic: /stereo/depth_map (sẽ báo log khi nhận frame đầu tiên)...")
        self.first_frame = True
        self.last_pub_time = 0.0 # Lưu thời gian gửi frame cuối cùng

    def depth_callback(self, msg):
        import time
        try:
            # --- CHỐNG LAG MẠNG BẰNG CÁCH GIẢM FPS TRUYỀN TẢI ---
            # Node C++ có thể đang nhả depth liên tục ở 15-30 FPS.
            # Ảnh depth không thể được nén jpeg (hàng chục KB mỗi khung hình).
            # Gửi quá nhanh qua WiFi -> Mạng bị tắc nghẽn -> Gây độ trễ (latency).
            # Bóp lại 5 - 10 FPS cho viewer là đủ nhìn mượt mà không delay!
            current_time = time.time()
            if current_time - self.last_pub_time < 0.2: # Tối đa ~10 fps
                return
            self.last_pub_time = current_time
            
            # Chuyển đổi ROS Image sang Numpy array
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            if self.first_frame:
                self.get_logger().info(f"🎥 Đã nhận frame Depth đầu tiên! Kích thước: {cv_image.shape}, Type: {cv_image.dtype}")
                
                h, w = cv_image.shape[:2]
                fx = 187.8
                fy = 187.8
                cx = 128.0
                cy = 72.0

                # Khai báo máy ảnh Pinhole 1 LẦN DUY NHẤT (Static) để Rerun tự động tái tạo 3D Pointcloud
                # Việc gửi Pinhole ở MỌI FRAME sẽ gây thắt cổ chai gRPC rất nặng ở các bản Rerun cũ
                rr.log("world", rr.ViewCoordinates.FLU, static=True)
                rr.log("world/stereo_camera", rr.ViewCoordinates.RDF, static=True)
                
                rr.log(
                    "world/stereo_camera",
                    rr.Pinhole(
                        focal_length=[fx, fy],
                        principal_point=[cx, cy],
                        width=w, 
                        height=h
                    ),
                    static=True
                )
                self.first_frame = False

            # Tính toán meter
            if cv_image.dtype == np.uint16:
                meter_val = 0.001
            else:
                meter_val = 1.0

            # Cài đặt timeline (tuỳ chọn)
            try:
                msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if hasattr(rr, "set_time"):
                    rr.set_time("ros_time", duration=msg_time)
            except Exception:
                pass
                
            # Đẩy ảnh Depth lên nhánh của máy ảnh
            rr.log(
                "world/stereo_camera/depth", 
                rr.DepthImage(cv_image, meter=meter_val)
            )

        except Exception as e:
            self.get_logger().error(f"Lỗi khi xử lý depth map: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RerunDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
