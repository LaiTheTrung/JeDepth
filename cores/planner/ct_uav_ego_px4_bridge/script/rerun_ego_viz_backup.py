#!/usr/bin/env python3
"""
rerun_ego_viz.py
----------------
Rerun Viewer bridge cho EGO Planner.
Chạy trên Jetson/Docker, stream toàn bộ dữ liệu qua gRPC về Laptop.

Subscribe:
  - /drone_0_ego_planner_node/grid_map/occupancy_inflate  → Local map (PointCloud2)
  - /drone_0_visual_slam/odom                            → Drone pose (Odometry)
  - /move_base_simple/goal                               → Goal (PoseStamped)
  - /drone_0_planning/bspline                            → Trajectory (Bspline)

Usage:
  python3 rerun_ego_viz.py

Yêu cầu:
  pip install rerun-sdk==0.23.1 scipy
  ROS2 environment sourced
"""

import numpy as np

# --- VÁ LỖI TƯƠNG THÍCH NUMPY < 2.0 VÀ RERUN 0.23.1 ---
# Rerun SDK gọi np.asarray(copy=False) nhưng numpy cũ chưa hỗ trợ tham số này
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

# Cố gắng import traj_utils (chỉ khả dụng khi đã source ROS2 workspace)
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

# Throttle: khoảng cách thời gian tối thiểu (giây) giữa 2 lần log
THROTTLE_MAP   = 0.15   # Local map   → ~7 FPS
THROTTLE_ODOM  = 0.05   # Odom/TF     → ~20 FPS
THROTTLE_TRAJ  = 0.15   # Trajectory  → ~7 FPS
THROTTLE_GOAL  = 0.0    # Goal        → event-driven (không throttle)
THROTTLE_DEPTH = 0.2    # Depth       → ~5 FPS (optional)

# Số điểm tối đa của local map gửi lên mỗi frame (giảm băng thông)
MAX_MAP_POINTS = 3000

# ============================================================
#  DE BOOR B-SPLINE EVALUATION (Python, numpy-only)
# ============================================================

def deboor_evaluate(ctrl_pts: np.ndarray, knots: np.ndarray, order: int, t: float) -> np.ndarray:
    """Tính De Boor tại thời điểm t trên B-Spline bậc `order`."""
    n = len(ctrl_pts)
    p = order
    # Tìm knot span
    span = np.searchsorted(knots, t, side='right') - 1
    span = np.clip(span, p, n - 1)
    # De Boor recursion
    d = ctrl_pts[span - p: span + 1].copy().astype(np.float64)
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            i = j + span - p
            denom = knots[i + p - r + 1] - knots[i]
            if abs(denom) < 1e-10:
                alpha = 0.0
            else:
                alpha = (t - knots[i]) / denom
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[p]


def sample_bspline(ctrl_pts: np.ndarray, knots: np.ndarray, order: int, num_samples: int = 60) -> np.ndarray:
    """Sample `num_samples` điểm đều trên toàn bộ B-Spline."""
    p = order
    n = len(ctrl_pts)
    t_min = knots[p]
    t_max = knots[n]  # knots[n] là valid end
    t_vals = np.linspace(t_min, t_max, num_samples, endpoint=False)
    pts = []
    for t in t_vals:
        try:
            pt = deboor_evaluate(ctrl_pts, knots, p, t)
            pts.append(pt)
        except Exception:
            pass
    return np.array(pts) if pts else np.zeros((0, 3))


# ============================================================
#  HELPER: PointCloud2 → numpy
# ============================================================

def pointcloud2_to_xyz(msg: PointCloud2, max_pts: int = MAX_MAP_POINTS) -> np.ndarray:
    """
    Chuyển đổi sensor_msgs/PointCloud2 sang numpy array (N, 3).
    Hỗ trợ point step bất kỳ, đọc x/y/z từ fields.
    Áp dụng random downsample để giới hạn băng thông.
    """
    # Tìm offset của x, y, z trong fields
    field_offsets = {}
    for field in msg.fields:
        field_offsets[field.name] = field.offset

    if 'x' not in field_offsets or 'y' not in field_offsets or 'z' not in field_offsets:
        return np.zeros((0, 3))

    ox = field_offsets['x']
    oy = field_offsets['y']
    oz = field_offsets['z']

    point_step = msg.point_step
    num_pts    = msg.width * msg.height
    data       = bytes(msg.data)

    if num_pts == 0:
        return np.zeros((0, 3))

    # Đọc toàn bộ nếu dữ liệu nhỏ, hoặc random subsample nếu lớn
    if num_pts <= max_pts:
        indices = range(num_pts)
    else:
        indices = np.random.choice(num_pts, max_pts, replace=False)

    pts = []
    fmt = 'fff'  # 3 floats x, y, z
    for i in indices:
        base = i * point_step
        try:
            x = struct.unpack_from('f', data, base + ox)[0]
            y = struct.unpack_from('f', data, base + oy)[0]
            z = struct.unpack_from('f', data, base + oz)[0]
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                pts.append([x, y, z])
        except Exception:
            pass

    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)


# ============================================================
#  ROS2 NODE
# ============================================================

class RerunEgoViz(Node):
    def __init__(self):
        super().__init__('rerun_ego_viz')

        # --- Khởi tạo Rerun ---
        rr.init("EgoPlanner_Viz")
        rr.connect_grpc(f"rerun+http://{LAPTOP_IP}:{RERUN_PORT}/proxy")
        self.get_logger().info(f"✅ Rerun kết nối tới: {LAPTOP_IP}:{RERUN_PORT}")

        # --- Log static entities 1 lần duy nhất ---
        # Hệ toạ độ FLU (Forward-Left-Up) cho world → trục X là hướng tiến
        rr.log("world", rr.ViewCoordinates.FLU, static=True)
        rr.log("world/camera", rr.ViewCoordinates.RDF, static=True)

        # --- Throttle timestamps ---
        self._t_map   = 0.0
        self._t_odom  = 0.0
        self._t_traj  = 0.0
        self._t_depth = 0.0

        # --- Cache ---
        self._goal_pos = None

        # --- Lập subscriptions ---
        map_topic  = f'/drone_{DRONE_ID}_ego_planner_node/grid_map/occupancy_inflate'
        odom_topic = f'/drone_{DRONE_ID}_visual_slam/odom'
        goal_topic = f'/drone_{DRONE_ID}_ego_planner_node/goal'

        self.sub_map = self.create_subscription(
            PointCloud2, map_topic, self._map_callback, 5)
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self._odom_callback, 10)
        self.sub_goal = self.create_subscription(
            PoseStamped, goal_topic, self._goal_callback, 5)

        if HAVE_BSPLINE:
            bspline_topic = f'/drone_{DRONE_ID}_planning/bspline'
            self.sub_bspline = self.create_subscription(
                Bspline, bspline_topic, self._bspline_callback, 5)

        self.get_logger().info(
            f"📡 Subscribing:\n"
            f"   • {map_topic}\n"
            f"   • {odom_topic}\n"
            f"   • {goal_topic}\n"
            + (f"   • /drone_{DRONE_ID}_planning/bspline\n" if HAVE_BSPLINE else "")
        )
        self.get_logger().info("🚀 Sẵn sàng stream lên Rerun Viewer…")

    # ----------------------------------------------------------
    #  CALLBACK: Local Map (PointCloud2)
    # ----------------------------------------------------------
    def _map_callback(self, msg: PointCloud2):
        now = time.time()
        if now - self._t_map < THROTTLE_MAP:
            return
        self._t_map = now

        pts = pointcloud2_to_xyz(msg, max_pts=MAX_MAP_POINTS)
        if len(pts) == 0:
            return

        # Màu sắc: gradient theo chiều cao Z (xanh dương → đỏ)
        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
        z_range = max(z_max - z_min, 0.01)
        ratios   = ((pts[:, 2] - z_min) / z_range).reshape(-1, 1)
        colors   = np.hstack([ratios, np.zeros_like(ratios), 1.0 - ratios])  # R,G,B
        colors   = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

        rr.log(
            "world/local_map",
            rr.Points3D(pts, colors=colors, radii=0.05)
        )

    # ----------------------------------------------------------
    #  CALLBACK: Odometry / Pose Drone
    # ----------------------------------------------------------
    def _odom_callback(self, msg: Odometry):
        now = time.time()
        if now - self._t_odom < THROTTLE_ODOM:
            return
        self._t_odom = now

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        pos = [p.x, p.y, p.z]

        try:
            ros_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if hasattr(rr, "set_time"):
                rr.set_time("ros_time", duration=ros_time)
        except Exception:
            pass

        # Vị trí drone (điểm xanh lá)
        rr.log(
            "world/drone",
            rr.Points3D([pos], colors=[[0, 220, 100]], radii=0.18)
        )

        # Transform3D (dùng quaternion để vẽ đúng hướng mũi nhọn)
        rr.log(
            "world/drone/transform",
            rr.Transform3D(
                translation=pos,
                rotation=rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w])
            )
        )

        # Nếu có goal, vẽ đường thẳng nối drone → goal
        if self._goal_pos is not None:
            rr.log(
                "world/drone_to_goal",
                rr.LineStrips3D([[pos, self._goal_pos]],
                                colors=[[255, 200, 0]],
                                radii=0.01)
            )

    # ----------------------------------------------------------
    #  CALLBACK: Goal
    # ----------------------------------------------------------
    def _goal_callback(self, msg: PoseStamped):
        # Goal không throttle, luôn ghi ngay khi nhận
        p = msg.pose.position
        self._goal_pos = [p.x, p.y, p.z]

        self.get_logger().info(
            f"🎯 Goal: ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")

        # Điểm đỏ lớn để dễ nhìn
        rr.log(
            "world/goal",
            rr.Points3D([self._goal_pos], colors=[[255, 50, 50]], radii=0.3)
        )
        # Nhãn text trên goal
        rr.log(
            "world/goal/label",
            rr.TextDocument(f"GOAL\n({p.x:.1f}, {p.y:.1f}, {p.z:.1f})")
        )

    # ----------------------------------------------------------
    #  CALLBACK: B-Spline Trajectory
    # ----------------------------------------------------------
    def _bspline_callback(self, msg):
        if not HAVE_BSPLINE:
            return
        now = time.time()
        if now - self._t_traj < THROTTLE_TRAJ:
            return
        self._t_traj = now

        try:
            order = int(msg.order)
            knots_raw = list(msg.knots)
            ctrl_raw  = msg.pos_pts

            if len(ctrl_raw) < order + 1 or len(knots_raw) < 2:
                return

            ctrl_pts = np.array([[pt.x, pt.y, pt.z] for pt in ctrl_raw], dtype=np.float64)
            knots    = np.array(knots_raw, dtype=np.float64)

            # Sample đường traj từ B-Spline
            traj_pts = sample_bspline(ctrl_pts, knots, order, num_samples=60)

            if len(traj_pts) < 2:
                # Fallback: vẽ thẳng control points nếu sample thất bại
                traj_pts = ctrl_pts

            # Vẽ trajectory màu vàng
            rr.log(
                "world/trajectory",
                rr.LineStrips3D([traj_pts.tolist()],
                                colors=[[255, 230, 0]],
                                radii=0.03)
            )
            # Vẽ control points màu cam mờ
            rr.log(
                "world/trajectory/ctrl_pts",
                rr.Points3D(ctrl_pts, colors=[[255, 140, 0, 120]], radii=0.05)
            )

        except Exception as e:
            self.get_logger().warn(f"B-Spline callback error: {e}")


# ============================================================
#  MAIN
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = RerunEgoViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Dừng Rerun Viz.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
