import os
import math
import glob
from pathlib import Path

import cv2
import numpy as np
import imageio.v3 as iio
import torch
from scipy.spatial.transform import Rotation as R

# ==============================
# 1) Tham số camera & Cassini
# ==============================
mask_
MASK_PATHS = {
    # ví dụ: "cam2": "mask_cassini_cam2.png"
    "cam0": "mask_cassini.png",
    "cam1": "mask_cassini.png",
    "cam2": "mask_cassini.png",
    "cam3": "mask_cassini.png",
    "cam4": "mask_cassini.png",
    "cam5": "mask_cassini.png",
    "cam6": "mask_cassini.png",
    "cam7": "mask_cassini.png",
}

# Intrinsic từ FoV ngang (giữ đồng nhất cho tất cả cam nếu bạn muốn)
def K_from_fov(w, h, fov_deg):
    f = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K

# Ở đây giả định chúng tồn tại sẵn:
#   - gen_perspective2cassini_grid(h, w, focal_gen, H_out, W_out, R_mat)
#   - unwrap_along_x (cho disparity)
# Nếu chưa có unwrap_along_x, mình thêm một bản đơn giản:
def unwrap_along_x(phi_tensor):
    """
    Unwrap phi theo chiều x để loại seam. phi_tensor: torch[H,W]
    """
    # dùng numpy unwrap theo hàng, rồi ghép lại
    phi_np = phi_tensor.detach().cpu().numpy()
    phi_unwrapped = np.unwrap(phi_np, axis=1)
    return torch.from_numpy(phi_unwrapped).to(phi_tensor.device, dtype=phi_tensor.dtype)

# (đang giả định nằm trong cùng namespace)
def gen_perspective2cassini_grid(img_h, img_w, focal_length, cass_h, cass_w, R_mat = np.eye(3), trans = np.zeros((3,1)), depth = None):
    """Generate Cassini grid - same as before"""
    
    # Cassini angle grid (y-up convention)
    p = np.pi / cass_w
    th = 2 * np.pi / cass_h
    
    phi = np.array([-np.pi/2 + (i + 0.5)*p for i in range(cass_w)])
    theta = np.array([-np.pi + (i + 0.5)*th for i in range(cass_h)])
    
    phi_map, theta_map = np.meshgrid(phi, theta, sparse=False, indexing='xy')
    
    # 3D points (y-up)
    x = np.sin(phi_map)
    y = np.cos(phi_map) * np.sin(theta_map)
    z = np.cos(phi_map) * np.cos(theta_map)
    
    point3D = np.stack([x, y, z]).reshape(3, -1)
    point3D = R_mat @ point3D  # Apply rotation if any
    # Project to perspective
    x_3d, y_3d, z_3d = point3D[0], point3D[1], point3D[2]
    valid_mask = z_3d > 0
    
    cx, cy = img_w / 2, img_h / 2
    
    mapx = np.full_like(x_3d, -1.0, dtype=np.float32)
    mapy = np.full_like(y_3d, -1.0, dtype=np.float32)
    
    mapx[valid_mask] = focal_length * x_3d[valid_mask] / z_3d[valid_mask] + cx
    mapy[valid_mask] = focal_length * y_3d[valid_mask] / z_3d[valid_mask] + cy
    
    mapx = mapx.reshape(cass_h, cass_w)
    mapy = mapy.reshape(cass_h, cass_w)
    
    return mapx, mapy

def load_mask_for_cam(cam_name):
    mpath = MASK_PATHS.get(cam_name, None)
    if mpath is None or not Path(mpath).exists():
        return None
    m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    # chuẩn hoá về H_out x W_out nếu cần
    if (m.shape[0] != H_out) or (m.shape[1] != W_out):
        m = cv2.resize(m, (W_out, H_out), interpolation=cv2.INTER_LINEAR)
    return m

