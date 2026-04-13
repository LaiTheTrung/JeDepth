import numpy as np

# --- VÁ LỖI TƯƠNG THÍCH NUMPY < 2.0 VÀ RERUN 0.23.1 ---
# Rerun SDK phiên bản này tự động gọi hàm np.asarray() với tham số copy=False,
# nhưng Numpy < 2.0 chưa hỗ trợ tham số 'copy'. Để tránh báo Warning rác và bỏ qua log,
# ta sẽ chặn thông số này lại trước khi nó văng lỗi.
_orig_asarray = np.asarray
def _patched_asarray(*args, **kwargs):
    kwargs.pop('copy', None)
    return _orig_asarray(*args, **kwargs)
np.asarray = _patched_asarray
# -----------------------------------------------------

import rerun as rr
import time
import socket
import threading
from math import tau
from rerun import blueprint as rrb
from rerun.utilities import bounce_lerp, build_color_spiral


# --- Cấu hình ---
LAPTOP_IP = "10.5.10.67" 
UDP_PORT = 5005

# 1. Khởi tạo Rerun
rr.init("Ego_PX4_Bridge")
rr.connect_grpc(f"rerun+http://{LAPTOP_IP}:9876/proxy")
print(f"✅ Đã kết nối tới Laptop Rerun Viewer tại: {LAPTOP_IP}:9876")

# --- 2. Bộ nhận Goal (Giữ nguyên Pipeline UDP) ---
current_goal = [0, 0, 0]
def listen_goal():
    global current_goal
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("0.0.0.0", UDP_PORT))
        while True:
            data, _ = s.recvfrom(1024)
            coords = [float(x) for x in data.decode().split(',')]
            current_goal = coords
            rr.log("world/goal", rr.Points3D([current_goal], colors=[[255, 0, 0]], radii=0.2))

threading.Thread(target=listen_goal, daemon=True).start()

# --- 3. Cấu hình Blueprint cho cấu trúc DNA ---
def setup_blueprint():
    try:
        blueprint = rrb.Blueprint(
            rrb.Spatial3DView(
                origin="/",
                overrides={
                    "helix/structure/scaffolding/beads": rrb.VisibleTimeRanges(
                        timeline="stable_time",
                        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-0.3),
                        end=rrb.TimeRangeBoundary.cursor_relative(seconds=0.3),
                    ),
                },
            ),
        )
        rr.send_blueprint(blueprint)
    except AttributeError:
        print("⚠️ Phiên bản Rerun SDK của bạn không hỗ trợ Blueprint như code mẫu. Đã bỏ qua.")

# --- 4. Stream Loop: Kết hợp trajectory cũ và DNA ---
def run():
    DESCRIPTION = """
    # DNA + Ego_PX4_Bridge
    Đã kết hợp thành công mã mô phỏng DNA từ tutorial của Rerun
    và stream gửi qua gRPC.
    """
    rr.log("description", rr.TextDocument(DESCRIPTION.strip(), media_type=rr.MediaType.MARKDOWN), static=True)

    try:
        rr.set_time_seconds("stable_time", 0) # Set t=0 initial
    except AttributeError:
        pass # Rerun SDK quá cũ

    NUM_POINTS = 100
    points1, colors1 = build_color_spiral(NUM_POINTS)
    points2, colors2 = build_color_spiral(NUM_POINTS, angular_offset=tau * 0.5)
    
    rr.log("helix/structure/left", rr.Points3D(points1, colors=colors1, radii=0.08), static=True)
    rr.log("helix/structure/right", rr.Points3D(points2, colors=colors2, radii=0.08), static=True)
    rr.log(
        "helix/structure/scaffolding",
        rr.LineStrips3D(np.stack((points1, points2), axis=1), colors=[128, 128, 128]),
        static=True,
    )

    time_offsets = np.random.rand(NUM_POINTS)
    t = 0
    i = 0
    
    print("🚀 Bắt đầu thiết lập Blueprint và stream dữ liệu 3D...")
    setup_blueprint()

    path = []
    
    while True:
        # ---- CODE CŨ CỦA ROBOT TRAJECTORY ----
        pos = [np.cos(t), np.sin(t), 0.5]
        path.append(pos)
        if len(path) > 100:
            path.pop(0)
            
        rr.log("world/robot", rr.Points3D([pos], colors=[[0, 255, 0]], radii=0.1))
        if len(path) > 1:
            rr.log("world/path", rr.LineStrips3D([path]))
            
        t += 0.05
        
        # ---- CODE MỚI MÔ PHỎNG DNA CỦA RERUN ----
        time_dna = i * 0.01
        try:
            rr.set_time("stable_time", duration=time_dna)
        except Exception:
            pass
            
        times = np.repeat(time_dna, NUM_POINTS) + time_offsets
        beads = [bounce_lerp(points1[n], points2[n], times[n]) for n in range(NUM_POINTS)]
        # Màu DNA bounce lerp (mã màu Rerun tutorial)
        colors = [[int(bounce_lerp(80, 230, times[n] * 2))] for n in range(NUM_POINTS)]
        
        rr.log(
            "helix/structure/scaffolding/beads",
            rr.Points3D(beads, radii=0.06, colors=np.repeat(colors, 3, axis=-1)),
        )
        
        rr.log(
            "helix/structure",
            rr.Transform3D(rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=time_dna / 4.0 * tau)),
        )
        
        i += 1
        time.sleep(0.05) # chạy mượt ở 20 fps thay vì giật lag khi tải lượng lớn

if __name__ == "__main__":
    run()