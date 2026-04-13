#!/usr/bin/env python3
"""
Test file for HitNet TensorRT implementation
Tests initialization, inference, and performance
"""

import os
import sys
import argparse
import time
import math
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List

from hitnet import HitnetTRT
from utils.evaluation import evaluate, eval_names


def save_metrics_to_file(metrics: dict, filepath: str):
    """Save metrics to JSON file"""
    import json
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print_success(f"Saved metrics to: {filepath}")


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")


def print_error(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")


def test_hitnet_initialization(onnx_path: str, engine_path: str, stream_number: int = 2) -> HitnetTRT:
    """
    Test HitNet TensorRT initialization
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to TensorRT engine
        stream_number: Number of concurrent streams
        
    Returns:
        Initialized HitnetTRT instance
    """
    print("\n" + "="*60)
    print("TEST 1: HitNet TensorRT Initialization")
    print("="*60)
    
    try:
        # Check if files exist
        if not os.path.exists(onnx_path):
            print_error(f"ONNX model not found at: {onnx_path}")
            return None
        print_success(f"ONNX model found: {onnx_path}")
        
        # Initialize HitNet
        print_info(f"Initializing HitNet with {stream_number} streams...")
        start_time = time.time()
        
        hitnet_trt = HitnetTRT(show_info=True)
        result = hitnet_trt.init(onnx_path, engine_path, stream_number)
        
        init_time = time.time() - start_time
        
        # Check initialization result
        assert result == 0, f"HitnetTRT initialization failed with code {result}"
        print_success(f"HitNet initialized successfully in {init_time:.2f}s")
        
        # Verify executors
        assert len(hitnet_trt.executors) == stream_number, \
            f"Expected {stream_number} executors, got {len(hitnet_trt.executors)}"
        print_success(f"Created {len(hitnet_trt.executors)} executors")
        
        # Verify streams
        assert len(hitnet_trt.streams) == stream_number, \
            f"Expected {stream_number} streams, got {len(hitnet_trt.streams)}"
        print_success(f"Created {len(hitnet_trt.streams)} CUDA streams")
        
        # Check engine
        assert hitnet_trt.engine is not None, "Engine is None"
        print_success("TensorRT engine loaded successfully")
        
        # Print engine info
        print_info(f"Engine inputs: {hitnet_trt.executors[0].input_shape}")
        print_info(f"Engine outputs: {hitnet_trt.executors[0].output_shape}")
        
        print_success("Initialization test PASSED")
        return hitnet_trt
        
    except Exception as e:
        print_error(f"Initialization test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def prepare_test_images(image_left_path: str, image_right_path: str, 
                       width: int = 320, height: int = 240) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare stereo image pair
    
    Args:
        image_left_path: Path to left image
        image_right_path: Path to right image
        width: Target width
        height: Target height
        
    Returns:
        Tuple of (left_image, right_image) as numpy arrays
    """
    print_info(f"Loading images: {width}x{height}")
    
    # Load images
    left_img = cv2.imread(image_left_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(image_right_path, cv2.IMREAD_GRAYSCALE)
    
    if left_img is None:
        raise ValueError(f"Failed to load left image: {image_left_path}")
    if right_img is None:
        raise ValueError(f"Failed to load right image: {image_right_path}")
    
    print_success(f"Loaded left image: {left_img.shape}")
    print_success(f"Loaded right image: {right_img.shape}")
    
    # Resize if necessary
    if left_img.shape != (height, width):
        left_img = cv2.resize(left_img, (width, height))
        print_info(f"Resized left image to {width}x{height}")
    
    if right_img.shape != (height, width):
        right_img = cv2.resize(right_img, (width, height))
        print_info(f"Resized right image to {width}x{height}")
    
    return left_img, right_img


def create_synthetic_stereo_pair(width: int = 320, height: int = 240, 
                                disparity: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic stereo pair for testing
    
    Args:
        width: Image width
        height: Image height
        disparity: Horizontal shift in pixels
        
    Returns:
        Tuple of (left_image, right_image)
    """
    print_info(f"Creating synthetic stereo pair: {width}x{height}, disparity={disparity}px")
    
    # Create base image with random patterns
    base = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    
    # Add some geometric shapes
    cv2.circle(base, (width//4, height//4), 30, 255, -1)
    cv2.rectangle(base, (width//2, height//4), (width//2 + 50, height//4 + 50), 200, -1)
    cv2.circle(base, (3*width//4, 3*height//4), 40, 180, -1)
    
    # Create right image by shifting left
    right = np.zeros_like(base)
    if disparity > 0:
        right[:, disparity:] = base[:, :-disparity]
    
    left = base
    
    print_success("Created synthetic stereo pair")
    
    return left, right


def test_single_inference(hitnet_trt: HitnetTRT, left_img: np.ndarray, 
                         right_img: np.ndarray, gt_depth: np.ndarray = None,
                         baseline: float = 0.1,
                         fov_h_deg: float = 90.0,
                         fov_v_deg: float = 90.0,
                         visualize: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Test single stereo pair inference with optional ground truth evaluation
    
    Args:
        hitnet_trt: Initialized HitNet instance
        left_img: Left image (H, W)
        right_img: Right image (H, W)
        gt_depth: Optional ground truth depth (H, W) in meters
        baseline: Stereo baseline in meters
        focal_length: Focal length in pixels
        visualize: Whether to show results
        
    Returns:
        Tuple of (disparity_map, metrics_dict)
    """
    print("\n" + "="*60)
    print("TEST 2: Single Inference")
    print("="*60)
    
    try:
        h, w = left_img.shape
        print_info(f"Input image size: {w}x{h}")

        # Compute focal lengths from FOVs
        fx = (w / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)
        fy = (h / 2.0) / math.tan(math.radians(fov_v_deg) / 2.0)
        focal_length = fx  # use horizontal for depth calculation
        print_info(f"Computed focal lengths: fx={fx:.2f}, fy={fy:.2f}")
        
        if gt_depth is not None:
            print_info(f"Ground truth depth loaded: {gt_depth.shape}")
        
        # Stack left and right vertically
        stacked = np.array([left_img, right_img])  # (2, H, W)
        print_info(f"Stacked shape: {stacked.shape}")
        
        # Add batch and channel dimensions
        tensor = stacked[np.newaxis, :, :, :].astype(np.float32)  # (1, 2, H, W)
        
        # Normalize to [0, 1]
        tensor = tensor / 255.0
        print_info(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        print_info(f"Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        
        # Prepare input (single stereo pair, so we need stream_number inputs)
        inputs = [tensor] * hitnet_trt.stream_number

        # Warm-up
        print_info("Warming up TensorRT engine (10 runs)...")
        for _ in range(10):
            hitnet_trt.do_inference(inputs)

        # Run inference (timed)
        print_info("Running timed inference...")
        start_time = time.time()

        result = hitnet_trt.do_inference(inputs)

        inference_time = time.time() - start_time
        
        assert result == 0, f"Inference failed with code {result}"
        print_success(f"Inference completed in {inference_time*1000:.2f}ms")
        print_info(f"FPS: {1/inference_time:.2f}")
        
        # Get output
        outputs = hitnet_trt.get_output()
        disparity = outputs[0]  # First output
        
        print_success(f"Output disparity shape: {disparity.shape}")
        print_info(f"Disparity range: [{disparity.min():.2f}, {disparity.max():.2f}]")
        print_info(f"Disparity mean: {disparity.mean():.2f}, std: {disparity.std():.2f}")
        
        # Convert disparity to depth
        pred_depth = disparity_to_depth(disparity, baseline, focal_length)
        print_success(f"Converted disparity to depth")
        print_info(f"Predicted depth range: [{pred_depth[pred_depth>0].min():.2f}, {pred_depth[pred_depth>0].max():.2f}] m")
        
        # Evaluate against ground truth if available
        metrics = None
        if gt_depth is not None:
            print_info(f"Stereo parameters: baseline={baseline}m, fov_h={fov_h_deg}deg, fov_v={fov_v_deg}deg, focal_length={focal_length}px")
            
            # Ensure both are 2D and same shape
            pred_depth_2d = np.squeeze(pred_depth)
            gt_depth_2d = np.squeeze(gt_depth)

            if pred_depth_2d.shape != gt_depth_2d.shape:
                print_warning(f"Resizing prediction from {pred_depth_2d.shape} to {gt_depth_2d.shape}")
                pred_depth_eval = cv2.resize(pred_depth_2d, (gt_depth_2d.shape[1], gt_depth_2d.shape[0]))
            else:
                pred_depth_eval = pred_depth_2d
            
            # Build mask and use evaluation utils (NumPy)
            mask = (gt_depth_2d > 0) & (pred_depth_eval > 0) & np.isfinite(gt_depth_2d) & np.isfinite(pred_depth_eval)
            if np.any(mask):
                metrics_vec = evaluate(pred_depth_eval, gt_depth_2d, mask)
                print("\n" + "="*60)
                print("DEPTH ESTIMATION METRICS")
                print("="*60)
                for name, val in zip(eval_names, metrics_vec):
                    if name.startswith("δ"):
                        print(f"{name}: {val:.2f} %")
                    elif name == "SILog":
                        print(f"{name}: {val:.4f}")
                    else:
                        print(f"{name}: {val:.4f} m")
                print("="*60)

                metrics = {k: float(v) for k, v in zip(eval_names, metrics_vec)}

                # Quality assessment based on δ1
                a1 = metrics_vec[5] / 100.0
                if a1 > 0.9:
                    print_success("Excellent depth estimation quality!")
                elif a1 > 0.7:
                    print_success("Good depth estimation quality")
                elif a1 > 0.5:
                    print_warning("Moderate depth estimation quality")
                else:
                    print_error("Poor depth estimation quality")
        
        # Visualize
        if visualize:
            visualize_results(left_img, right_img, disparity, gt_depth, pred_depth)
        
        print_success("Single inference test PASSED")
        return disparity, metrics
        
    except Exception as e:
        print_error(f"Single inference test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def test_batch_inference(hitnet_trt: HitnetTRT, num_pairs: int = 4, 
                        width: int = 320, height: int = 240):
    """
    Test batch inference with multiple stereo pairs
    
    Args:
        hitnet_trt: Initialized HitNet instance
        num_pairs: Number of stereo pairs
        width: Image width
        height: Image height
    """
    print("\n" + "="*60)
    print("TEST 3: Batch Inference")
    print("="*60)
    
    try:
        # Create multiple synthetic stereo pairs
        print_info(f"Creating {num_pairs} synthetic stereo pairs...")
        inputs = []
        
        for i in range(num_pairs):
            left, right = create_synthetic_stereo_pair(width, height, disparity=15+i*5)
            stacked = np.vstack([left, right])
            tensor = stacked[np.newaxis, np.newaxis, :, :].astype(np.float32) / 255.0
            inputs.append(tensor)
        
        print_success(f"Created {len(inputs)} input tensors")
        
        # Run batch inference
        print_info("Running batch inference...")
        start_time = time.time()
        
        result = hitnet_trt.do_inference(inputs)
        
        inference_time = time.time() - start_time
        
        assert result == 0, f"Batch inference failed with code {result}"
        print_success(f"Batch inference completed in {inference_time*1000:.2f}ms")
        print_info(f"Average time per pair: {inference_time/num_pairs*1000:.2f}ms")
        print_info(f"Throughput: {num_pairs/inference_time:.2f} pairs/sec")
        
        # Get outputs
        outputs = hitnet_trt.get_output()
        print_success(f"Got {len(outputs)} output disparities")
        
        for i, disp in enumerate(outputs):
            print_info(f"Disparity {i}: shape={disp.shape}, range=[{disp.min():.2f}, {disp.max():.2f}]")
        
        print_success("Batch inference test PASSED")
        
    except Exception as e:
        print_error(f"Batch inference test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


def test_performance_benchmark(hitnet_trt: HitnetTRT, iterations: int = 100, 
                              width: int = 320, height: int = 240):
    """
    Benchmark inference performance
    
    Args:
        hitnet_trt: Initialized HitNet instance
        iterations: Number of iterations
        width: Image width
        height: Image height
    """
    print("\n" + "="*60)
    print("TEST 4: Performance Benchmark")
    print("="*60)
    
    try:
        print_info(f"Running {iterations} iterations...")
        
        # Create test data
        num_pairs = hitnet_trt.stream_number
        inputs = []
        for i in range(num_pairs):
            left, right = create_synthetic_stereo_pair(width, height)
            stacked = np.vstack([left, right])
            tensor = stacked[np.newaxis, np.newaxis, :, :].astype(np.float32) / 255.0
            inputs.append(tensor)
        
        # Warm-up
        print_info("Warming up (10 iterations)...")
        for _ in range(10):
            hitnet_trt.do_inference(inputs)
        
        # Benchmark
        print_info("Benchmarking...")
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            result = hitnet_trt.do_inference(inputs)
            inference_time = time.time() - start_time
            
            if result == 0:
                times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print_info(f"Progress: {i+1}/{iterations}")
        
        # Statistics
        times = np.array(times)
        mean_time = times.mean()
        std_time = times.std()
        min_time = times.min()
        max_time = times.max()
        
        print("\n" + "-"*60)
        print("Performance Statistics:")
        print("-"*60)
        print(f"Mean inference time:   {mean_time*1000:.2f} ms")
        print(f"Std deviation:         {std_time*1000:.2f} ms")
        print(f"Min time:              {min_time*1000:.2f} ms")
        print(f"Max time:              {max_time*1000:.2f} ms")
        print(f"Mean FPS:              {1/mean_time:.2f}")
        print(f"Throughput:            {num_pairs/mean_time:.2f} pairs/sec")
        print("-"*60)
        
        print_success("Performance benchmark PASSED")
        
    except Exception as e:
        print_error(f"Performance benchmark FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


def load_ground_truth_depth(depth_path: str) -> np.ndarray:
    """
    Load ground truth depth map from uint16 PNG (millimeter unit)
    
    Args:
        depth_path: Path to depth PNG file
        
    Returns:
        Depth map in meters as float32
    """
    print_info(f"Loading ground truth depth: {depth_path}")
    
    # Load uint16 PNG
    depth_mm = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    if depth_mm is None:
        raise ValueError(f"Failed to load depth map: {depth_path}")
    
    if depth_mm.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth map, got {depth_mm.dtype}")
    
    print_success(f"Loaded depth map: shape={depth_mm.shape}, dtype={depth_mm.dtype}")
    print_info(f"Depth range (mm): [{depth_mm.min()}, {depth_mm.max()}]")
    
    # Convert millimeters to meters
    depth_m = depth_mm.astype(np.float32) / 1000.0
    
    # Handle invalid depths (0 values)
    valid_mask = depth_mm > 0
    num_valid = valid_mask.sum()
    total_pixels = depth_mm.size
    
    print_info(f"Valid pixels: {num_valid}/{total_pixels} ({100*num_valid/total_pixels:.1f}%)")
    
    return depth_m


def disparity_to_depth(disparity: np.ndarray, baseline: float, focal_length: float) -> np.ndarray:
    """
    Convert disparity to depth using stereo geometry
    
    Args:
        disparity: Disparity map (pixels)
        baseline: Stereo baseline distance (meters)
        focal_length: Focal length (pixels)
        
    Returns:
        Depth map (meters)
    """
    # Avoid division by zero
    disparity_safe = np.where(disparity > 0.1, disparity, 0.1)
    
    # Depth = (baseline * focal_length) / disparity
    depth = (baseline * focal_length) / disparity_safe
    
    # Mask invalid disparities
    depth[disparity <= 0.1] = 0
    
    return depth


def compute_depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, 
                         max_depth: float = 80.0) -> dict:
    """Backward compatibility stub (no longer used)."""
    print_info("compute_depth_metrics is deprecated; using utils.evaluation.evaluate instead.")
    mask = (gt_depth > 0) & (gt_depth < max_depth) & \
           (pred_depth > 0) & (pred_depth < max_depth) & \
           np.isfinite(pred_depth) & np.isfinite(gt_depth)
    if not np.any(mask):
        return None
    metrics_vec = evaluate(pred_depth, gt_depth, mask)
    return {k: float(v) for k, v in zip(eval_names, metrics_vec)}


def print_metrics(metrics: dict):
    """Backward compatibility stub to print metrics dict from evaluate()."""
    print("\n" + "="*60)
    print("DEPTH ESTIMATION METRICS")
    print("="*60)
    for name in eval_names:
        val = metrics.get(name, None)
        if val is None:
            continue
        if name.startswith("δ"):
            print(f"{name}: {val:.2f} %")
        elif name == "SILog":
            print(f"{name}: {val:.4f}")
        else:
            print(f"{name}: {val:.4f} m")
    print("="*60)


def visualize_results(left_img: np.ndarray, right_img: np.ndarray, 
                     disparity: np.ndarray, gt_depth: np.ndarray = None,
                     pred_depth: np.ndarray = None, save_path: str = None):
    """
    Visualize stereo results with optional ground truth
    
    Args:
        left_img: Left image
        right_img: Right image
        disparity: Disparity map
        gt_depth: Optional ground truth depth
        pred_depth: Optional predicted depth
        save_path: Optional path to save visualization
    """
    print_info("Visualizing results...")
    
    # Normalize disparity for visualization
    disp_2d = np.squeeze(disparity)
    disp_vis = cv2.normalize(disp_2d, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = disp_vis.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    
    # Convert grayscale to BGR
    if len(left_img.shape) == 2:
        left_bgr = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
    else:
        left_bgr = left_img
    
    if len(right_img.shape) == 2:
        right_bgr = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
    else:
        right_bgr = right_img
    
    # Create rows
    rows = []
    
    # Row 1: Left and Right images
    top_row = np.hstack([left_bgr, right_bgr])
    cv2.putText(top_row, "Left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(top_row, "Right", (left_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    rows.append(top_row)
    
    # Row 2: Disparity
    disp_row = np.hstack([disp_color, disp_color])
    cv2.putText(disp_row, "Predicted Disparity", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    rows.append(disp_row)
    
    # Row 3: Depth comparison (if ground truth available)
    if gt_depth is not None and pred_depth is not None:
        # Ensure both are 2D
        pred_depth_2d = np.squeeze(pred_depth)
        gt_depth_2d = np.squeeze(gt_depth)

        # Resize to match
        if gt_depth_2d.shape != pred_depth_2d.shape:
            gt_depth_vis = cv2.resize(gt_depth_2d, (pred_depth_2d.shape[1], pred_depth_2d.shape[0]))
        else:
            gt_depth_vis = gt_depth_2d
        
        # Normalize for visualization (0-10m range)
        max_vis_depth = 10.0
        pred_vis = np.clip(pred_depth_2d / max_vis_depth * 255, 0, 255).astype(np.uint8)
        gt_vis = np.clip(gt_depth_vis / max_vis_depth * 255, 0, 255).astype(np.uint8)
        
        # Apply colormap
        pred_color = cv2.applyColorMap(pred_vis, cv2.COLORMAP_TURBO)
        gt_color = cv2.applyColorMap(gt_vis, cv2.COLORMAP_TURBO)
        
        # Error map
        valid_mask = (gt_depth_vis > 0) & (pred_depth_2d > 0)
        error_map = np.zeros_like(pred_depth_2d)
        error_map[valid_mask] = np.abs(pred_depth_2d[valid_mask] - gt_depth_vis[valid_mask])
        error_vis = np.clip(error_map / 1.0 * 255, 0, 255).astype(np.uint8)  # 0-1m error range
        error_color = cv2.applyColorMap(error_vis, cv2.COLORMAP_JET)
        
        # Depth row
        depth_row = np.hstack([pred_color, gt_color])
        cv2.putText(depth_row, "Predicted Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(depth_row, "Ground Truth", (pred_depth.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        rows.append(depth_row)
        
        # Error row
        error_row = np.hstack([error_color, error_color])
        cv2.putText(error_row, "Absolute Error (m)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        rows.append(error_row)
    
    # Stack all rows
    result = np.vstack(rows)

    # Display
    # cv2.imshow("HitNet Results", result)
    # print_info("Press any key to continue...")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, result)
        print_success(f"Saved visualization to: {save_path}")

    # Additionally save raw outputs to images directory if gt/pred provided
    images_dir = Path("images")
    try:
        images_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        images_dir = Path(".")

    # Save disparity visualization
    disp_out_path = images_dir / "hitnet_disparity.png"
    cv2.imwrite(str(disp_out_path), disp_color)
    print_success(f"Saved disparity visualization to: {disp_out_path}")

    if gt_depth is not None and pred_depth is not None:
        # Save predicted depth and ground truth depth as 8-bit color maps
        depth_pred_path = images_dir / "hitnet_pred_depth.png"
        depth_gt_path = images_dir / "hitnet_gt_depth.png"
        cv2.imwrite(str(depth_pred_path), pred_color)
        cv2.imwrite(str(depth_gt_path), gt_color)
        print_success(f"Saved predicted depth to: {depth_pred_path}")
        print_success(f"Saved ground truth depth to: {depth_gt_path}")

        # Save error map
        error_map_path = images_dir / "hitnet_error_map.png"
        cv2.imwrite(str(error_map_path), error_color)
        print_success(f"Saved error map to: {error_map_path}")


def run_all_tests(args):
    """Run all tests"""
    print("\n" + "="*60)
    print("HITNET TENSORRT TEST SUITE")
    print("="*60)
    
    # Test 1: Initialization
    hitnet_trt = test_hitnet_initialization(
        args.onnx_path, 
        args.engine_path, 
        args.stream_number
    )
    
    if hitnet_trt is None:
        print_error("Initialization failed, stopping tests")
        return
    
    # Load ground truth depth if provided
    gt_depth = None
    if args.gt_depth:
        try:
            gt_depth = load_ground_truth_depth(args.gt_depth)
            print_success(f"Loaded ground truth depth: {gt_depth.shape}")
        except Exception as e:
            print_error(f"Failed to load ground truth: {e}")
            gt_depth = None
    
    # Test 2: Single Inference
    if args.image_left and args.image_right:
        # Use provided images
        try:
            left_img, right_img = prepare_test_images(
                args.image_left, 
                args.image_right,
                args.width,
                args.height
            )
            
            # Resize ground truth to match if needed
            if gt_depth is not None and gt_depth.shape[:2] != (args.height, args.width):
                print_info(f"Resizing ground truth from {gt_depth.shape} to ({args.height}, {args.width})")
                gt_depth = cv2.resize(gt_depth, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
            
            disparity, metrics = test_single_inference(
                hitnet_trt, left_img, right_img, 
                gt_depth=gt_depth,
                baseline=args.baseline,
                fov_h_deg=args.fov_h,
                fov_v_deg=args.fov_v,
                visualize=args.visualize
            )
            
            # Save metrics to file if requested
            if metrics and args.save_metrics:
                save_metrics_to_file(metrics, args.save_metrics)
                
        except Exception as e:
            print_error(f"Failed to load images: {e}")
            print_warning("Using synthetic images instead")
            left_img, right_img = create_synthetic_stereo_pair(args.width, args.height)
            disparity, metrics = test_single_inference(
                hitnet_trt, left_img, right_img, 
                baseline=args.baseline,
                fov_h_deg=args.fov_h,
                fov_v_deg=args.fov_v,
                visualize=args.visualize
            )
    else:
        # Use synthetic images
        print_warning("No images provided, using synthetic stereo pair")
        left_img, right_img = create_synthetic_stereo_pair(args.width, args.height)
        disparity, metrics = test_single_inference(
            hitnet_trt, left_img, right_img,
            baseline=args.baseline,
            fov_h_deg=args.fov_h,
            fov_v_deg=args.fov_v,
            visualize=args.visualize
        )
    
    # Test 3: Batch Inference
    if args.test_batch:
        test_batch_inference(hitnet_trt, args.stream_number, args.width, args.height)
    
    # Test 4: Performance Benchmark
    if args.benchmark:
        test_performance_benchmark(hitnet_trt, args.benchmark_iterations, 
                                  args.width, args.height)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test HitNet TensorRT Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with ONNX model (will build engine)
  python test_hitnet.py --onnx_path hitnet.onnx --engine_path hitnet.trt
  
  # Test with real images and ground truth depth
  python test_hitnet.py --onnx_path hitnet.onnx --engine_path hitnet.trt \\
      --image_left left.png --image_right right.png \\
      --gt_depth depth_gt.png --visualize
  
  # Test with calibration parameters
  python test_hitnet.py --onnx_path hitnet.onnx --engine_path hitnet.trt \\
      --image_left left.png --image_right right.png --gt_depth depth_gt.png \\
      --baseline 0.12 --focal_length 450.0 --visualize
  
  # Run performance benchmark
  python test_hitnet.py --onnx_path hitnet.onnx --engine_path hitnet.trt \\
      --benchmark --benchmark_iterations 200
  
  # Full test suite with metrics saved
  python test_hitnet.py --onnx_path hitnet.onnx --engine_path hitnet.trt \\
      --image_left left.png --image_right right.png --gt_depth depth_gt.png \\
      --test_batch --benchmark --visualize --save_metrics results.json
        """
    )
    
    # Required arguments
    parser.add_argument('--onnx_path', type=str, required=True,
                       help='Path to the ONNX model file')
    parser.add_argument('--engine_path', type=str, required=True,
                       help='Path to the TensorRT engine file (will be created if not exists)')
    
    # Optional image inputs
    parser.add_argument('--image_left', type=str, default=None,
                       help='Path to the left image for inference test')
    parser.add_argument('--image_right', type=str, default=None,
                       help='Path to the right image for inference test')
    parser.add_argument('--gt_depth', type=str, default=None,
                       help='Path to ground truth depth PNG (uint16, millimeter unit)')
    
    # Stereo calibration parameters
    parser.add_argument('--baseline', type=float, default=0.1,
                       help='Stereo baseline in meters (default: 0.1)')
    parser.add_argument('--fov_h', type=float, default=90.0,
                       help='Horizontal field of view in degrees (default: 90.0)')
    parser.add_argument('--fov_v', type=float, default=90.0,
                       help='Vertical field of view in degrees (default: 90.0)')
    
    # Configuration
    parser.add_argument('--stream_number', type=int, default=4,
                       help='Number of streams to initialize (default: 4)')
    parser.add_argument('--width', type=int, default=320,
                       help='Image width (default: 320)')
    parser.add_argument('--height', type=int, default=240,
                       help='Image height (default: 240)')
    
    # Test options
    parser.add_argument('--test_batch', action='store_true',
                       help='Run batch inference test')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--benchmark_iterations', type=int, default=100,
                       help='Number of iterations for benchmark (default: 100)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize results')
    parser.add_argument('--save_metrics', type=str, default=None,
                       help='Save metrics to JSON file')
    
    args = parser.parse_args()
    
    # Run tests
    run_all_tests(args)