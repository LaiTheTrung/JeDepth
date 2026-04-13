from utils.perspective_processing import K_from_fov
import numpy as np

class CameraConfig:
    # Camera configuration parameters
    fov = 140.0
    w = 640
    h = 640
    H_out = 640
    W_out = 320
    crop_top = 0
    crop_bottom = 640
    crop_left = 0
    crop_right = 320
    K = K_from_fov(w, h, fov)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    focal_gen = fx
    
