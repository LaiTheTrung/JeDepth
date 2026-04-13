#!/usr/bin/env python3
"""
rerun_ego_viz.py
----------------
Rerun Viewer bridge cho EGO Planner tích hợp Web UI Gradio.
Chạy trên Jetson/Docker, stream toàn bộ dữ liệu qua gRPC về Laptop và hiển thị Web UI để set Goal.

Yêu cầu:
  pip install rerun-sdk==0.23.1 scipy gradio
  ROS2 environment sourced
"""

import numpy as np
import math
import threading
from scipy.spatial.transform import Rotation as R

# --- VÁ LỖI TƯƠNG THÍCH NUMPY < 2.0 VÀ RERUN 0.23.1 ---
_orig_asarray = np.asarray
def _patched_asarray(*args, **kwargs):
    kwargs.pop('copy', None)
    return _orig_asarray(*args, **kwargs)
np.asarray = _patched_asarray
# -------------------------------------------------------

import rerun as rr
import rclpy
from rclpy.node import Node
import time
import struct

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image as RosImage

import gradio as gr

# Cố gắng import traj_utils
try:
    from traj_utils.msg import Bspline
    HAVE_BSPLINE = True
except ImportError:
    HAVE_BSPLINE = False
    print("⚠️  Không tìm thấy traj_utils. Bỏ qua visualize trajectory.")

# ============================================================
#  CẤU HÌNH
# ============================================================

LAPTOP_IP    = "10.5.10.67"     # IP WiFi / Tailscale của Laptop
RERUN_PORT   = 9876
DRONE_ID     = 0

# Throttle: khoảng cách thời gian (giây) giữa 2 lần log
THROTTLE_MAP   = 0.15
THROTTLE_ODOM  = 0.05
THROTTLE_TRAJ  = 0.15
THROTTLE_GOAL  = 0.0
THROTTLE_DEPTH = 0.2

MAX_MAP_POINTS = 3000

# ============================================================
#  UI STATE LƯU TRẠNG THÁI GIAO DIỆN
# ============================================================

class UIState:
    def __init__(self):
        self.active = False
        self.x = 0.0
        self.y = 0.0
        self.z_base = 0.0
        self.z_offset = 0.0
        self.yaw_deg = 0.0

current_goal = UIState()

def update_ghost_marker():
    """Cập nhật bóng ma Marker trên Rerun ứng với tham số từ Gradio UI."""
    if not current_goal.active:
        return
    
    total_z = current_goal.z_base + current_goal.z_offset
    yaw_rad = math.radians(current_goal.yaw_deg)
    
    # Quaternion orientation
    try:
        q = R.from_euler('z', yaw_rad).as_quat() # [x,y,z,w]
    except AttributeError:
        # Fallback for some old scipy
        q = [0.0, 0.0, 0.0, 1.0]

    pos = [current_goal.x, current_goal.y, total_z]

    # Log vào nhánh riêng biệt cho UI (Không nhiễu data robot)
    rr.log(
        "ui/ghost_goal/pose",
        rr.Transform3D(translation=pos, rotation=rr.Quaternion(xyzw=q))
    )
    
    rr.log(
        "ui/ghost_goal/pose/marker",
        rr.Arrows3D(
            origins=[[0, 0, 0]], 
            vectors=[[1.5, 0, 0]],   # Dài 1.5m, chĩa thẳng mũi
            colors=[[0, 255, 255, 128]], # Màu cyan ghost
            radii=0.08
        )
    )

# ============================================================
#  UI CALLBACKS (GRADIO)
# ============================================================

def on_coord_change(x, y, z_base, z_off, yaw):
    current_goal.active = True
    current_goal.x = float(x)
    current_goal.y = float(y)
    current_goal.z_base = float(z_base)
    current_goal.z_offset = float(z_off)
    current_goal.yaw_deg = float(yaw)
    update_ghost_marker()

ros_node = None

def on_submit():
    if not current_goal.active:
        gr.Warning("Vui lòng click chọn mục tiêu trên bản đồ 3D trước!")
        return
    
    total_z = current_goal.z_base + current_goal.z_offset
    yaw_rad = math.radians(current_goal.yaw_deg)
    
    # Quaternion orientation
    try:
        q = R.from_euler('z', yaw_rad).as_quat() # [x,y,z,w]
    except AttributeError:
        # Fallback for some old scipy
        q = [0.0, 0.0, 0.0, 1.0]
    
    # Gửi qua ROS nếu node đã sẵn sàng
    if ros_node is not None:
        msg = PoseStamped()
        msg.header.stamp = ros_node.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(current_goal.x)
        msg.pose.position.y = float(current_goal.y)
        msg.pose.position.z = float(total_z)
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        
        ros_node.pub_goal.publish(msg)
        gr.Info(f"Đã gửi 3D Goal tới EGO Planner!")
    
    # IN RA SCREEN THEO DÕI
    print("\n" + "="*50)
    print("🎯 [HỆ THỐNG GIAO DIỆN] ĐÃ GỬI 3D GOAL QUA ROS:")
    print(f"   X     : {current_goal.x:.2f} m")
    print(f"   Y     : {current_goal.y:.2f} m")
    print(f"   Z tổng: {total_z:.2f} m  (Origin: {current_goal.z_base:.2f}  |  Offset: {current_goal.z_offset:.2f})")
    print(f"   Yaw   : {current_goal.yaw_deg:.1f} độ")
    print(f"   Topic : /drone_{DRONE_ID}_ego_planner_node/goal")
    print("="*50 + "\n")

def on_cancel():
    current_goal.active = False
    current_goal.x = 0.0
    current_goal.y = 0.0
    current_goal.z_base = 0.0
    current_goal.z_offset = 0.0
    current_goal.yaw_deg = 0.0
    
    # Dùng clear để xóa nhánh UI
    try:
        rr.log("ui/ghost_goal", rr.Clear.recursive())
    except AttributeError:
        # Fallback rerun versions
        rr.log("ui/ghost_goal", rr.Clear.flat())
        
    return 0.0, 0.0, 0.0, 0.0, 0.0

# ============================================================
#  DE BOOR B-SPLINE
# ============================================================

def deboor_evaluate(ctrl_pts: np.ndarray, knots: np.ndarray, order: int, t: float) -> np.ndarray:
    n, p = len(ctrl_pts), order
    span = np.clip(np.searchsorted(knots, t, side='right') - 1, p, n - 1)
    d = ctrl_pts[span - p: span + 1].copy().astype(np.float64)
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            i = j + span - p
            denom = knots[i + p - r + 1] - knots[i]
            alpha = 0.0 if abs(denom) < 1e-10 else (t - knots[i]) / denom
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[p]

def sample_bspline(ctrl_pts: np.ndarray, knots: np.ndarray, order: int, num_samples: int = 60) -> np.ndarray:
    t_vals = np.linspace(knots[order], knots[len(ctrl_pts)], num_samples, endpoint=False)
    pts = [deboor_evaluate(ctrl_pts, knots, order, t) for t in t_vals]
    return np.array(pts) if pts else np.zeros((0, 3))

# ============================================================
#  HELPER PC2
# ============================================================

def pointcloud2_to_xyz(msg: PointCloud2, max_pts: int = MAX_MAP_POINTS) -> np.ndarray:
    field_offsets = {f.name: f.offset for f in msg.fields}
    if not all(k in field_offsets for k in ('x', 'y', 'z')): return np.zeros((0, 3))
    
    ox, oy, oz = field_offsets['x'], field_offsets['y'], field_offsets['z']
    num_pts = msg.width * msg.height
    data = bytes(msg.data)
    
    if num_pts == 0: return np.zeros((0, 3))
    indices = range(num_pts) if num_pts <= max_pts else np.random.choice(num_pts, max_pts, replace=False)
    
    pts = []
    for i in indices:
        try:
            x = struct.unpack_from('f', data, i * msg.point_step + ox)[0]
            y = struct.unpack_from('f', data, i * msg.point_step + oy)[0]
            z = struct.unpack_from('f', data, i * msg.point_step + oz)[0]
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z): pts.append([x, y, z])
        except: pass
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)

def image_to_numpy(msg: RosImage):
    try:
        if msg.encoding in ['16UC1', 'mono16', '16uc1']:
            return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        elif msg.encoding in ['32FC1', '32fc1']:
            return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        elif msg.encoding in ['8UC1', 'mono8', '8uc1']:
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
    except Exception as e:
        pass
    return None

# ============================================================
#  ROS2 NODE
# ============================================================

class RerunEgoViz(Node):
    def __init__(self):
        super().__init__('rerun_ego_viz')

        rr.init("EgoPlanner_Viz")
        rr.connect_grpc(f"rerun+http://{LAPTOP_IP}:{RERUN_PORT}/proxy")
        self.get_logger().info(f"✅ Rerun kết nối tới: {LAPTOP_IP}:{RERUN_PORT}")

        rr.log("world", rr.ViewCoordinates.FLU, static=True)
        rr.log("world/camera", rr.ViewCoordinates.RDF, static=True)

        self._t_map = self._t_odom = self._t_traj = 0.0
        self._goal_pos = None

        self.sub_map = self.create_subscription(PointCloud2, f'/drone_{DRONE_ID}_ego_planner_node/grid_map/occupancy_inflate', self._map_callback, 5)
        self.sub_odom = self.create_subscription(Odometry, f'/drone_{DRONE_ID}_visual_slam/odom', self._odom_callback, 10)
        self.sub_goal = self.create_subscription(PoseStamped, f'/drone_{DRONE_ID}_ego_planner_node/goal', self._goal_callback, 5)
        
        # Nhận ảnh depth từ camera để map vào Rerun
        self.sub_depth = self.create_subscription(RosImage, '/stereo/depth_map', self._depth_callback, 3)

        # Trực tiếp kết nối xuống Ego Planner (bỏ qua filter 2D của move_base_simple)
        self.pub_goal = self.create_publisher(PoseStamped, f'/drone_{DRONE_ID}_ego_planner_node/goal', 5)

        if HAVE_BSPLINE:
            self.sub_bspline = self.create_subscription(Bspline, f'/drone_{DRONE_ID}_planning/bspline', self._bspline_callback, 5)

        self.get_logger().info("🚀 Sẵn sàng stream lên Rerun Viewer (và host Gradio)...")

    def _depth_callback(self, msg: RosImage):
        now = time.time()
        if not hasattr(self, '_t_depth'): self._t_depth = 0.0
        if now - self._t_depth < THROTTLE_DEPTH: return
        self._t_depth = now
        
        arr = image_to_numpy(msg)
        if arr is None: return
        
        meter_val = 1000.0 if msg.encoding in ['16UC1', 'mono16', '16uc1'] else 1.0
        
        try:
            rr.set_time("ros_time", duration=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        except: pass
        
        rr.log("world/camera/depth_map", rr.DepthImage(arr, meter=meter_val))

    def _map_callback(self, msg: PointCloud2):
        now = time.time()
        if now - self._t_map < THROTTLE_MAP: return
        self._t_map = now

        pts = pointcloud2_to_xyz(msg, max_pts=MAX_MAP_POINTS)
        if len(pts) == 0: return

        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
        ratios = ((pts[:, 2] - z_min) / max(z_max - z_min, 0.01)).reshape(-1, 1)
        colors = (np.clip(np.hstack([ratios, np.zeros_like(ratios), 1.0 - ratios]), 0, 1) * 255).astype(np.uint8)

        # Biểu diễn Map dưới dạng các khối Voxel lập phương thay vì các chấm tròn.
        # Ở EGO Planner, độ phân giải lưới grid nội bộ thường cài đặt mặc định quanh 0.1m
        voxel_size = 0.1
        half_sizes = np.full((len(pts), 3), voxel_size / 2.0)
        
        rr.log("world/local_map", rr.Boxes3D(centers=pts, half_sizes=half_sizes, colors=colors))

    def _odom_callback(self, msg: Odometry):
        now = time.time()
        if now - self._t_odom < THROTTLE_ODOM: return
        self._t_odom = now

        p = msg.pose.pose.position
        pos = [p.x, p.y, p.z]

        try:
            rr.set_time("ros_time", duration=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        except: pass

        rr.log("world/drone", rr.Points3D([pos], colors=[[0, 220, 100]], radii=0.18))
        rr.log("world/drone/transform", rr.Transform3D(translation=pos, rotation=rr.Quaternion(xyzw=[msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])))

        if self._goal_pos:
            rr.log("world/drone_to_goal", rr.LineStrips3D([[pos, self._goal_pos]], colors=[[255, 200, 0]], radii=0.01))

    def _goal_callback(self, msg: PoseStamped):
        p = msg.pose.position
        self._goal_pos = [p.x, p.y, p.z]
        rr.log("world/goal", rr.Points3D([self._goal_pos], colors=[[255, 50, 50]], radii=0.3))
        rr.log("world/goal/label", rr.TextDocument(f"GOAL\n({p.x:.1f}, {p.y:.1f}, {p.z:.1f})"))

    def _bspline_callback(self, msg):
        if not HAVE_BSPLINE: return
        now = time.time()
        if now - self._t_traj < THROTTLE_TRAJ: return
        self._t_traj = now

        try:
            ctrl_pts = np.array([[pt.x, pt.y, pt.z] for pt in msg.pos_pts], dtype=np.float64)
            knots = np.array(msg.knots, dtype=np.float64)
            traj_pts = sample_bspline(ctrl_pts, knots, int(msg.order), 60)
            
            rr.log("world/trajectory", rr.LineStrips3D([traj_pts.tolist()], colors=[[255, 230, 0]], radii=0.03))
            rr.log("world/trajectory/ctrl_pts", rr.Points3D(ctrl_pts, colors=[[255, 140, 0, 120]], radii=0.05))
        except: pass

# ============================================================
#  GRADIO UI BUILDER
# ============================================================

def build_ui():
    with gr.Blocks(title="3D Pose Goal Control Panel") as demo:
        gr.Markdown("## 🎮 Drone Goal Control Panel\nSử dụng cửa sổ này song song với Rerun Native App để chọn Goal.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📍 Tọa độ điểm tham chiếu")
                with gr.Row():
                    x_box = gr.Number(label="X (m)", interactive=True, value=0.0)
                    y_box = gr.Number(label="Y (m)", interactive=True, value=0.0)
                    z_box = gr.Number(label="Z_base (m)", interactive=True, value=0.0)
                
                gr.Markdown("### ⚙️ Tinh chỉnh Goal an toàn")
                with gr.Row():
                    slider_z = gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Z-Offset (m)")
                    slider_yaw = gr.Slider(-180, 180, value=0.0, step=1.0, label="Yaw Angle (Độ)")
                
                gr.Markdown("---")
                with gr.Row():
                    btn_send = gr.Button("✅ Kiểm tra & In 3D Goal", variant="primary")
                    btn_cancel = gr.Button("❌ Hủy/Xóa Goal", variant="stop")
        
        # --- Events ---
        ui_components = [x_box, y_box, z_box, slider_z, slider_yaw]
        
        for comp in [x_box, y_box, z_box, slider_z, slider_yaw]:
            comp.change(on_coord_change, inputs=ui_components)
        
        btn_send.click(on_submit)
        btn_cancel.click(on_cancel, outputs=ui_components)
    
    return demo

# ============================================================
#  MAIN
# ============================================================

def main(args=None):
    global ros_node
    rclpy.init(args=args)
    ros_node = RerunEgoViz()
    
    # 1. Chạy rclpy spin trên Thread nền để đảm bảo không kẹt luồng UI
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()
    
    try:
        # 2. Chạy Web UI Gradio trên Thread chính
        ui = build_ui()
        ros_node.get_logger().info("🌐 Đang khởi động Gradio Web UI, truy cập HTTP tại máy này port 7860...")
        ui.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft()) 
    except KeyboardInterrupt:
        ros_node.get_logger().info("🛑 Dừng Rerun Viz & Gradio.")
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
