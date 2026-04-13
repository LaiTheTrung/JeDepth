import os, math
import numpy as np
import cv2
from PIL import Image

IMAGES_ROOT = "data/images"
OUT_DIR     = "output"
FOV_H_DEG   = 140.0
FOV_V_DEG   = None
DEPTH_IS_RANGE = False   # True nếu DepthPerspective (range); False nếu DepthPlanar (Z-depth)
DEPTH_TRUNC = 50.0
VOXEL_SIZE  = 0.02
REMOVE_OUTLIERS = True
WRITE_ASCII = False
VISUALIZE   = True

def K_from_fov(W, H, fov_h_deg, fov_v_deg=None):
    fx = (W/2.0)/math.tan(math.radians(fov_h_deg)/2.0)
    fy = ((H/2.0)/math.tan(math.radians(fov_v_deg)/2.0)) if fov_v_deg is not None else fx*(H/W)
    cx, cy = W/2.0, H/2.0
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ===== Replaced: Open3D -> OpenCV/Numpy =====
def read_rgb(path):
    """
    Đọc ảnh màu HxWx3 uint8, trả về RGB (cv2 đọc BGR nên cần convert).
    Chấp nhận .jpg/.png/.bmp ...
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"RGB must be HxWx3 uint8: {path}")
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_depth_to_meters(path):
    """
    - Nếu .png uint16: giả định đơn vị mm -> trả về float32 mét
    - Nếu .exr float32 (OpenEXR): đọc IMREAD_ANYDEPTH; nếu có 3 kênh lấy kênh 0
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"Cannot read depth png: {path}")
        if arr.ndim != 2 or arr.dtype != np.uint16:
            raise TypeError(f"Depth PNG must be single-channel uint16 (mm): {path}")
        return (arr.astype(np.float32) / 1000.0)  # mm -> m
    elif ext == ".exr":
        arr = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if arr is None:
            raise FileNotFoundError(f"Cannot read EXR (ensure OpenCV built with OpenEXR): {path}")
        # arr có thể là HxW (1ch) hoặc HxWx3 (RGB float)
        if arr.ndim == 3:
            arr = arr[..., 0]  # lấy kênh đầu tiên làm depth
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr  # đã là mét
    else:
        raise TypeError(f"Unsupported depth ext: {ext}")

def resize_nearest(rgb, W, H):
    return np.array(Image.fromarray(rgb).resize((W,H), Image.NEAREST), dtype=np.uint8)

def range_to_z_depth_m(range_m, W, H, K, is_perspective=True):
    if is_perspective:  # perspective
        fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        xn = (uu - cx) / fx
        yn = (vv - cy) / fy
        return (range_m / np.sqrt(xn*xn + yn*yn + 1.0)).astype(np.float32)
    else:  # planar
        return range_m

def planar_to_range_m(planar, W, H, focal_len):
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)  # (H, W)
    cx = W / 2
    cy = H / 2
    x = (uu - cx) / focal_len
    y = (vv - cy) / focal_len
    range_m = planar * np.sqrt(x**2 + y**2 + 1.0)
    return range_m

def camera_look_at(cam_pos, target_pos, up=np.array([0, 0, 1], float)):
    """Create rotation matrix for camera looking at target (Z-up)"""
    fwd = target_pos - cam_pos
    fwd /= (np.linalg.norm(fwd) + 1e-12)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-12:
        cand = np.array([1, 0, 0], float) if abs(fwd[0]) < 0.9 else np.array([0, 1, 0], float)
        right = np.cross(fwd, cand)
    right /= (np.linalg.norm(right) + 1e-12)
    up2 = np.cross(right, fwd)
    up2 /= (np.linalg.norm(up2) + 1e-12)
    return np.column_stack([right, -up2, fwd])

def yaw_cv_for(cam, YAW_OFFSET_GLOBAL, YAW_SIGN, YAW_AIR, YAW_BIAS):
    return YAW_OFFSET_GLOBAL + YAW_SIGN * YAW_AIR[cam] + YAW_BIAS[cam]

def T_from_Rt(R, t):
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=t; return T

# ========== CONVERSION FUNCTION ==========
def rotation_matrix_from_yaw_zup(yaw_deg):
    """
    Create rotation matrix for yaw around Z-axis (Z-up convention).
    
    yaw = 0°   → camera looks along +X
    yaw = 90°  → camera looks along +Y
    yaw = 180° → camera looks along -X
    yaw = 270° → camera looks along -Y
    """
    yaw_rad = math.radians(yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    
    # Rotation around Z-axis
    R = np.array([
        [cos_y, -sin_y, 0],
        [sin_y,  cos_y, 0],
        [0,      0,     1]
    ], dtype=np.float32)
    
    return R

def T_c2w_fixed(cam, YAW_OFFSET_GLOBAL, YAW_SIGN, YAW_AIR, YAW_BIAS, POS_CM):
    """
    Camera-to-world transform (FIXED).
    
    Returns 4x4 matrix that transforms points from camera frame to world frame.
    Camera frame: +X right, +Y forward (camera looks this way), +Z up
    World frame: Z-up, yaw rotation around Z
    """
    # Get yaw angle
    yaw_cv = yaw_cv_for(cam, YAW_OFFSET_GLOBAL, YAW_SIGN, YAW_AIR, YAW_BIAS)
    
    # Create rotation matrix (yaw around Z)
    R = rotation_matrix_from_yaw_zup(yaw_cv)
    
    # Translation
    t_cm = POS_CM[cam]
    t = np.array([t_cm[0], -t_cm[1], t_cm[2]], dtype=np.float32) 
    
    return T_from_Rt(R, t)

def get_data(cam, stem, K, images_root=IMAGES_ROOT, depth_is_range=DEPTH_IS_RANGE):
    rgb_path   = os.path.join(images_root, cam, f"{stem}.jpg")
    if not os.path.isfile(rgb_path): rgb_path = os.path.join(images_root, cam, f"{stem}.png")
    depth_path = os.path.join(images_root, f"{cam}_depth", f"{stem}.png")
    if not os.path.isfile(depth_path): depth_path = os.path.join(images_root, f"{cam}_depth", f"{stem}.exr")
    if not (os.path.isfile(rgb_path) and os.path.isfile(depth_path)): 
        return None

    rgb = read_rgb(rgb_path)
    depth_m = read_depth_to_meters(depth_path)
    Hd, Wd = depth_m.shape
    Hc, Wc = rgb.shape[:2]

    # Nếu kích thước depth khác RGB, cân nhắc resize theo nearest:
    if (Hd != Hc) or (Wd != Wc):
        depth_m = cv2.resize(depth_m, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
        Hd, Wd = Hc, Wc

    depth_use = range_to_z_depth_m(depth_m, Wd, Hd, K, depth_is_range)
    return rgb, depth_use
