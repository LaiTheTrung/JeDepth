import torch
import numpy as np
"""
for erp:
use y-up conversion
--------Forward--------
phi = atan2(x, z)
theta = atan2(y, sqrt(x^2 + z^2))
r = sqrt(x^2 + y^2 + z^2)
------- Backward--------
x = sin(phi)*cos(theta)*r
y = sin(theta)*r
z = cos(phi)*cos(theta)*r


for cassini:
use y-up conversion
--------Forward--------
r = sqrt(x^2 + y^2 + z^2)
phi = asin(x/r)
theta = atan2(y, z)
------- Backward--------
x = sin(phi)*r
y = cos(phi)*sin(theta)*r
z = cos(phi)*cos(theta)*r

for pespective:
--------Forward--------
focal_length = 0.5*sensor_size/tan(0.5*fov)
fx = focal_length_x
fy = focal_length_y
u = fx * x / z + cx
v = fy * y / z + cy
------- Backward--------
x = (u - cx) * r / fx
y = (v - cy) * r / fy
z = r

for perspective:
use y-up conversion
------- Forward--------
x = (u-cx)/fx * z
y = (v-cy)/fy * z
z = fx * fy / sqrt( fy^2 + (u-cx)^2 + ( (v-cy)^2 * fx^2 )/ fy^2 )
------- Combine two equation------
u = fx * tan(phi) / cos(theta) + cx
v = fy * tan(theta) + cy
"""
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


def generate_perspective_to_erp_grid(img_h, img_w, focal_length, erp_h, erp_w, depth = None):
    """Generate Cassini grid - same as before"""
    
    # Cassini angle grid (y-up convention)
    p = 2 * np.pi / erp_w
    th = np.pi / erp_h

    phi = [-np.pi + (i + 0.5) * p for i in range(erp_w)]
    theta = [-np.pi / 2 + (i + 0.5) * th for i in range(erp_h)]

    phi_map, theta_map = np.meshgrid(phi, theta, sparse=False, indexing='xy')
    phi_map = phi_map.astype(np.float32)
    theta_map = theta_map.astype(np.float32)
    # 3D points (y-up)
    x = np.sin(phi_map) * np.cos(theta_map)
    y = np.sin(theta_map)
    z = np.cos(phi_map) * np.cos(theta_map)

    x_3d = x.reshape(-1)
    y_3d = y.reshape(-1)
    z_3d = z.reshape(-1)
    valid_mask = z_3d > 0
    
    cx, cy = img_w / 2, img_h / 2
    
    mapx = np.full_like(x_3d, -1.0, dtype=np.float32)
    mapy = np.full_like(y_3d, -1.0, dtype=np.float32)
    
    mapx[valid_mask] = focal_length * x_3d[valid_mask] / z_3d[valid_mask] + cx
    mapy[valid_mask] = focal_length * y_3d[valid_mask] / z_3d[valid_mask] + cy

    mapx = mapx.reshape(erp_h, erp_w).astype(np.float32)
    mapy = mapy.reshape(erp_h, erp_w).astype(np.float32)

    return mapx, mapy

def gen_erp2perspective_grid(img_h, img_w, focal_length, erp_h, erp_w):
    u = np.arange(img_w, dtype=np.float32)
    v = np.arange(img_h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    cx = img_w / 2
    cy = img_h / 2

    x = (uu - cx) / focal_length
    y = (vv - cy) / focal_length
    z = np.ones_like(x)

    mapx = np.full_like(x, -1.0, dtype=np.float32)
    mapy = np.full_like(y, -1.0, dtype=np.float32)

    mapx = np.arctan2(x, z)
    mapy = np.arctan2(y, np.sqrt(x**2 + z**2))

    mapx = np.clip((mapx + np.pi) / (2 * np.pi) * erp_w - 0.5, 0, erp_w - 1).astype(np.float32)
    mapy = np.clip((mapy + np.pi / 2) / np.pi * erp_h - 0.5, 0, erp_h - 1).astype(np.float32)

    return mapx, mapy

def spherical_grid(h, w):
    """ Generate meshgrid for equirectangular projection.

    Parameters
    ----------
    h : int
        height of the expected equirectangular image
    w : int
        width of the expected equirectangular image

    Returns
    -------
    phi_xy : numpy array
        phi value (-np.pi < phi < np.pi)
    theta_xy : numpy array
        theta value (-np.pi/2 < theta < np.pi/2)
    """
    p = 2 * np.pi / w
    th = np.pi / h
    phi = [-np.pi + (i + 0.5) * p for i in range(w)]
    theta = [-np.pi / 2 + (i + 0.5) * th for i in range(h)]
    phi_xy, theta_xy = np.meshgrid(phi, theta, sparse=False, indexing='xy')
    return phi_xy, theta_xy

def perspective_rays(h,w, focal_length):
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    cx = w / 2
    cy = h / 2

    x = (uu - cx) / focal_length
    y = (vv - cy) / focal_length
    z = np.ones_like(x)

    x_3d = x.reshape(-1)
    y_3d = y.reshape(-1)
    z_3d = z.reshape(-1)

    return x_3d, y_3d, z_3d

# ========== Z-UP VERSION ==========

import cv2
import math

def warp_erp_by_transform_full(src_erp_depth, T, camera_pos=None, 
                                 max_depth=100.0, invalid_depth=0.0):
    """
    Warp ERP depth map by full 4x4 transform (rotation + translation).
    
    This is the CORRECT way when you have translation:
    1. Convert ERP depth → 3D points in source camera frame
    2. Apply transform: points_world = T @ points_cam
    3. If camera_pos given: shift to new camera frame
    4. Convert back to spherical → ERP
    
    Args:
        src_erp_depth: (H, W) source ERP depth in meters
        T: 4x4 camera-to-world transform
        camera_pos: (3,) new camera position in world frame (optional)
                    If None, uses origin (0, 0, 0)
        max_depth: maximum valid depth
        invalid_depth: depth value for invalid pixels
    
    Returns:
        dst_erp_depth: (H, W) warped ERP depth
    """
    H, W = src_erp_depth.shape
    
    # Extract R and t from transform
    R = T[:3, :3]
    t = np.zeros(3, dtype=np.float32)
    
    # Default camera position is origin
    if camera_pos is None:
        camera_pos = np.zeros(3, dtype=np.float32)
    else:
        camera_pos = np.array(camera_pos, dtype=np.float32)
    
    # Build destination ERP grid
    u = np.arange(W, dtype=np.float32) + 0.5
    v = np.arange(H, dtype=np.float32) + 0.5
    uu, vv = np.meshgrid(u, v, indexing='xy')
    
    # Destination spherical angles (where we want to sample from)
    azimuth_dst = (uu / W) * (2 * np.pi) - np.pi
    elevation_dst = (np.pi / 2) - (vv / H) * np.pi
    
    # Destination unit directions (from new camera position)
    cos_e = np.cos(elevation_dst)
    sin_e = np.sin(elevation_dst)
    cos_a = np.cos(azimuth_dst)
    sin_a = np.sin(azimuth_dst)
    
    dirs_dst = np.stack([
        cos_e * cos_a,
        cos_e * sin_a,
        sin_e
    ], axis=0).reshape(3, -1)  # (3, H*W)
    
    # These are rays from new camera position
    # We need to find where they intersect the depth map in the SOURCE frame
    
    # Transform rays from new camera frame to source camera frame
    # Point in world: P_world = camera_pos + r * dir_dst
    # Point in source camera: P_src = R^T @ (P_world - t)
    #                               = R^T @ (camera_pos - t + r * dir_dst)
    
    Rinv = R.T
    
    # Ray origin in source frame
    ray_origin_src = Rinv @ (camera_pos - t)  # (3,)
    
    # Ray directions in source frame
    dirs_dst_src = Rinv @ dirs_dst  # (3, H*W)
    
    # Convert to spherical in source frame to find which ERP pixels to sample
    xs = dirs_dst_src[0, :]
    ys = dirs_dst_src[1, :]
    zs = dirs_dst_src[2, :]
    
    azimuth_src = np.arctan2(ys, xs)
    elevation_src = np.arctan2(zs, np.sqrt(xs**2 + ys**2))
    
    # Map to source ERP pixels
    u_src = (azimuth_src + np.pi) / (2 * np.pi) * W
    v_src = ((np.pi / 2) - elevation_src) / np.pi * H
    
    u_src = np.mod(u_src, W).astype(np.float32)
    v_src = np.clip(v_src, 0, H - 1e-3).astype(np.float32)
    
    # Sample source depth
    mapx = u_src.reshape(H, W)
    mapy = v_src.reshape(H, W)
    
    depth_sampled = cv2.remap(src_erp_depth, mapx, mapy,
                              interpolation=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_WRAP, borderValue=0).flatten()

    # Compute 3D points in source camera frame
    # P_src = ray_origin_src + depth_along_ray * dirs_dst_src
    # But depth_sampled is radial depth from source camera center
    # So we need: ||P_src|| = depth_sampled
    
    # For simplicity, if ray_origin_src ≈ 0 (colocated cameras),
    # then depth_along_ray ≈ depth_sampled
    
    # General case: solve ||ray_origin_src + t*dir|| = depth_sampled
    # This is quadratic in t, but for colocated cameras (ray_origin_src ≈ 0):
    # ray_origin_norm = np.linalg.norm(ray_origin_src)
    
    # if ray_origin_norm < 0.01:  # Colocated case
    #     # Simple: scale directions by depth
    #     points_src = dirs_dst_src * depth_sampled  # (3, H*W)
    # else:
    #     # Non-colocated: need to solve properly
    #     # ||ray_origin_src + t*dir||^2 = depth^2
    #     # This gets complex, for now approximate:
    #     points_src = ray_origin_src.reshape(3, 1) + dirs_dst_src * depth_sampled
    points_src = dirs_dst_src * depth_sampled
    # Transform to world frame
    points_world = R @ points_src + camera_pos.reshape(3, 1)  # (3, H*W)
    
    # Compute depth from new camera position
    # diff = points_world - camera_pos.reshape(3, 1)
    diff = points_world
    depth_dst = np.sqrt(np.sum(diff**2, axis=0))  # (H*W,)
    
    # Filter invalid depths
    valid = (depth_sampled > 0.1) & (depth_sampled < max_depth) & (depth_dst < max_depth)
    depth_dst[~valid] = invalid_depth
    
    return depth_dst.reshape(H, W)

