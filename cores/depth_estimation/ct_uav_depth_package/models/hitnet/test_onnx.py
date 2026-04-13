#!/usr/bin/env python3
"""Simple ONNX inference test for HitNet stereo model.

This script loads a HitNet ONNX model, runs inference on a stereo
pair (left/right images), and saves the disparity/depth visualization
into the `images/` directory.
"""

import argparse
import os
from pathlib import Path
import math
import time

import cv2
import numpy as np
import onnxruntime as ort

from utils.evaluation import evaluate, eval_names
print("onnxruntime version:", ort.__version__)
print("ONNX Runtime providers:", ort.get_available_providers())

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    END = "\033[0m"


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")


def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")


def load_stereo_images(left_path: str, right_path: str, width: int, height: int):
    """Load and resize left/right grayscale images to (H, W)."""
    left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if left_img is None:
        raise ValueError(f"Failed to load left image: {left_path}")
    if right_img is None:
        raise ValueError(f"Failed to load right image: {right_path}")

    if left_img.shape[:2] != (height, width):
        left_img = cv2.resize(left_img, (width, height))
    if right_img.shape[:2] != (height, width):
        right_img = cv2.resize(right_img, (width, height))

    return left_img, right_img


def build_input_tensor(left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
    """Build ONNX input tensor from left/right images.

    Assumes ONNX expects shape (1, 1, 2*H, W) with [0,1] floats.
    If your model expects (1, 2, H, W), adjust here.
    """
    h, w = left_img.shape
    stacked = np.array([left_img, right_img])  # (2, H, W)
    tensor = stacked[np.newaxis, :, :, :].astype(np.float32) / 255.0
    return tensor


def disparity_to_depth(disparity: np.ndarray, baseline: float, focal_length: float) -> np.ndarray:
    """Convert disparity (pixels) to depth (meters)."""
    disp_safe = np.where(disparity > 0.1, disparity, 0.1)
    depth = (baseline * focal_length) / disp_safe
    depth[disparity <= 0.1] = 0
    return depth


def load_ground_truth_depth(depth_path: str) -> np.ndarray:
    """Load ground truth depth from uint16 PNG (millimeters) and convert to meters."""
    depth_mm = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth_mm is None:
        raise ValueError(f"Failed to load depth GT: {depth_path}")
    if depth_mm.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth GT, got {depth_mm.dtype}")
    depth_m = depth_mm.astype(np.float32) / 1000.0
    return depth_m


def compute_error_map(pred_depth: np.ndarray, gt_depth: np.ndarray, max_depth: float = 80.0):
    """Compute per-pixel absolute error map and simple metrics."""
    pred_2d = np.squeeze(pred_depth)
    gt_2d = np.squeeze(gt_depth)

    if pred_2d.shape != gt_2d.shape:
        gt_2d = cv2.resize(gt_2d, (pred_2d.shape[1], pred_2d.shape[0]), interpolation=cv2.INTER_NEAREST)

    valid = (gt_2d > 0) & (gt_2d < max_depth) & (pred_2d > 0) & (pred_2d < max_depth)
    if not np.any(valid):
        return None, {}

    diff = np.abs(pred_2d - gt_2d)
    err_map = np.zeros_like(pred_2d, dtype=np.float32)
    err_map[valid] = diff[valid]

    mae = diff[valid].mean()
    rmse = np.sqrt((diff[valid] ** 2).mean())
    return err_map, {"mae": float(mae), "rmse": float(rmse)}


def run_onnx_inference(args):
    onnx_path = args.onnx_path
    left_path = args.image_left
    right_path = args.image_right
    width = args.width
    height = args.height

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    print_info(f"Loading ONNX model: {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=[ "TensorrtExecutionProvider", "CUDAExecutionProvider","CPUExecutionProvider"])

    # Infer input/output names from model
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print_info(f"Input name: {input_name}")
    print_info(f"Output name: {output_name}")

    left_img, right_img = load_stereo_images(left_path, right_path, width, height)
    print_info(f"Loaded images: left={left_img.shape}, right={right_img.shape}")

    inp = build_input_tensor(left_img, right_img)
    print_info(f"Input tensor shape: {inp.shape}, range=({inp.min():.3f}, {inp.max():.3f})")

    # Compute focal lengths from FOVs
    fov_h_deg = args.fov_h
    fov_v_deg = args.fov_v
    fx = (width / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)
    fy = (height / 2.0) / math.tan(math.radians(fov_v_deg) / 2.0)
    focal_length = fx
    print_info(f"Computed focal lengths: fx={fx:.2f}, fy={fy:.2f}")

    # Run inference
    print_info("Running ONNX inference...")
    t0 = time.time()
    outputs = sess.run([output_name], {input_name: inp})
    dt = time.time() - t0
    print_success(f"Inference time: {dt*1000:.2f} ms  (FPS: {1.0/dt:.2f})")
    disp = outputs[0]

    # Expected shape: (1,1,H,W) or (1,H,W,1) depending on export
    disp = np.squeeze(disp)
    print_info(f"Disparity shape: {disp.shape}, range=({disp.min():.3f}, {disp.max():.3f})")

    depth = disparity_to_depth(disp, args.baseline, focal_length)
    depth_valid = depth[depth > 0]
    if depth_valid.size:
        print_info(f"Depth range (valid): {depth_valid.min():.3f} m .. {depth_valid.max():.3f} m")

    # Create output directory
    images_dir = Path("images")
    images_dir.mkdir(parents=True, exist_ok=True)

    # Save disparity visualization
    disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    disp_out = images_dir / "hitnet_onnx_disparity.png"
    cv2.imwrite(str(disp_out), disp_color)
    print_success(f"Saved disparity visualization to: {disp_out}")

    # Save depth visualization (0-10m)
    max_vis_depth = 10.0
    depth_vis = np.clip(depth / max_vis_depth * 255, 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    depth_out = images_dir / "hitnet_onnx_depth.png"
    cv2.imwrite(str(depth_out), depth_color)
    print_success(f"Saved depth visualization to: {depth_out}")

    # If ground truth provided, compute error map, metrics and save
    if args.gt_depth:
        gt_depth = load_ground_truth_depth(args.gt_depth)

        # Resize GT to match predicted depth resolution if needed
        if gt_depth.shape != depth.shape:
            print_info(f"Resizing GT depth from {gt_depth.shape} to {depth.shape}")
            gt_depth = cv2.resize(
                gt_depth,
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        err_map, stats = compute_error_map(depth, gt_depth)
        if err_map is not None:
            # Compute full set of metrics using evaluation utils
            mask = (gt_depth > 0) & (depth > 0) & np.isfinite(gt_depth) & np.isfinite(depth)
            if np.any(mask):
                metrics_vec = evaluate(depth, gt_depth, mask)
                print("\n" + "="*60)
                print("DEPTH ESTIMATION METRICS (ONNX)")
                print("="*60)
                for name, val in zip(eval_names, metrics_vec):
                    if name.startswith("δ"):
                        print(f"{name}: {val:.2f} %")
                    elif name == "SILog":
                        print(f"{name}: {val:.4f}")
                    else:
                        print(f"{name}: {val:.4f} m")
                print("="*60)
                print_info(f"GT metrics - MAE: {metrics_vec[0]:.4f} m, RMSE: {metrics_vec[1]:.4f} m")
            err_vis = np.clip(err_map / 1.0 * 255, 0, 255).astype(np.uint8)  # 0-1m error
            err_color = cv2.applyColorMap(err_vis, cv2.COLORMAP_JET)
            err_out = images_dir / "hitnet_onnx_error_map.png"
            cv2.imwrite(str(err_out), err_color)
            print_success(f"Saved error map to: {err_out}")


def main():
    parser = argparse.ArgumentParser(description="Test HitNet ONNX inference")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to HitNet ONNX model")
    parser.add_argument("--image_left", type=str, required=True, help="Path to left image")
    parser.add_argument("--image_right", type=str, required=True, help="Path to right image")
    parser.add_argument("--gt_depth", type=str, default=None, help="Path to ground truth depth PNG (uint16, mm)")
    parser.add_argument("--width", type=int, default=320, help="Input width (default: 320)")
    parser.add_argument("--height", type=int, default=240, help="Input height (default: 240)")
    parser.add_argument("--baseline", type=float, default=0.1, help="Stereo baseline in meters (default: 0.1)")
    parser.add_argument("--fov_h", type=float, default=90.0, help="Horizontal FOV in degrees (default: 90.0)")
    parser.add_argument("--fov_v", type=float, default=90.0, help="Vertical FOV in degrees (default: 90.0)")

    args = parser.parse_args()
    run_onnx_inference(args)


if __name__ == "__main__":
    main()
