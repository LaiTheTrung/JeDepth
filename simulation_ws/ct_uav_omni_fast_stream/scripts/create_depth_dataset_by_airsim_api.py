#!/usr/bin/env python3

import cosysairsim as airsim
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import time

def quaternion_to_airsim(qw, qx, qy, qz):
    """Convert quaternion to AirSim Quaternionr"""
    return airsim.Quaternionr(x_val=qx, y_val=qy, z_val=qz, w_val=qw)

def is_depth_camera(cam_name):
    """Check if camera is a depth camera"""
    return 'depth' in cam_name.lower()

def save_rgb_image(image_response, output_path):
    """Save RGB image as JPEG"""
    if image_response.image_data_uint8:
        img_1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            cv2.imwrite(str(output_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return img_bgr
    return None

def save_depth_image(image_response, output_path):
    """
    Save depth image as PNG (uint16).
    AirSim returns depth in meters as float32, we convert to millimeters as uint16.
    Returns the depth image in meters (float32) for further processing.
    """
    if len(image_response.image_data_float) > 0:
        # Depth data as float32 array (in meters)
        depth_meters = np.array(image_response.image_data_float, dtype=np.float32)
        depth_meters = depth_meters.reshape(image_response.height, image_response.width)
        
        # Convert meters to millimeters and clip to uint16 range
        depth_mm = depth_meters * 1000.0
        depth_mm = np.clip(depth_mm, 0, 65535)
        depth_uint16 = depth_mm.astype(np.uint16)
        
        # Save as PNG
        cv2.imwrite(str(output_path), depth_uint16, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return depth_meters  # Return depth in meters for ERP fusion
    return None

def create_depth_visualization(depth_image_path, viz_path):
    """Create colormap visualization of depth image"""
    depth_image = cv2.imread(str(depth_image_path), cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        return False
    
    # Normalize depth for visualization (0-5000mm range)
    depth_normalized = np.clip(depth_image, 0, 5000) / 5000.0 * 255
    depth_normalized = depth_normalized.astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    cv2.imwrite(str(viz_path), depth_colormap)
    return True

def setup_output_directories(output_dir, camera_names):
    """Create output directory structure"""
    output_path = Path(output_dir)
    
    # Create main directories
    images_dir = output_path / 'images'
    video_dir = output_path / 'video'
    viz_dir = output_path / 'depth_visualization_samples'
    fused_depth_dir = output_path / 'fused_depth'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    fused_depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each camera
    camera_dirs = {}
    for cam_name in camera_names:
        cam_dir = images_dir / cam_name
        cam_dir.mkdir(exist_ok=True)
        camera_dirs[cam_name] = cam_dir
    
    return output_path, images_dir, video_dir, viz_dir, fused_depth_dir, camera_dirs

def capture_images(client, vehicle_name, camera_names, camera_dirs, timestamp, image_paths):
    """
    Capture RGB and depth images from all cameras.
    Returns (success_count, depth_data_dict, rgb_data_dict) where depth_data_dict contains depth in meters.
    """
    # Prepare image requests
    image_requests = []
    for cam_name in camera_names:
        if is_depth_camera(cam_name):
            # Request depth image (DepthPlanar type)
            base_cam = cam_name[:-len('_depth')]
            image_requests.append(
                airsim.ImageRequest(base_cam, airsim.ImageType.DepthPlanar, pixels_as_float=True)
            )
        else:
            # Request RGB image (compressed PNG)
            image_requests.append(
                airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, True)
            )
    
    # Capture all images at once
    responses = client.simGetImages(image_requests, vehicle_name)
    
    # Save images and collect depth data
    success_count = 0
    depth_data_dict = {}  # {cam_name: depth_meters}
    rgb_data_dict = {}    # {cam_name: rgb_image}
    
    for idx, (cam_name, response) in enumerate(zip(camera_names, responses)):
        if is_depth_camera(cam_name):
            # Save depth as PNG (uint16)
            filename = f"{timestamp:019d}.png"
            filepath = camera_dirs[cam_name] / filename
            depth_meters = save_depth_image(response, filepath)
            
            if depth_meters is not None:
                image_paths[cam_name].append(str(filepath.relative_to(camera_dirs[cam_name].parent.parent)))
                depth_data_dict[cam_name] = depth_meters
                success_count += 1
            else:
                image_paths[cam_name].append(None)
        else:
            # Save RGB as JPEG
            filename = f"{timestamp:019d}.jpg"
            filepath = camera_dirs[cam_name] / filename
            rgb_img = save_rgb_image(response, filepath)
            
            if rgb_img is not None:
                image_paths[cam_name].append(str(filepath.relative_to(camera_dirs[cam_name].parent.parent)))
                rgb_data_dict[cam_name] = rgb_img
                success_count += 1
            else:
                image_paths[cam_name].append(None)
    
    return success_count, depth_data_dict, rgb_data_dict

def create_videos(output_dir, camera_names, image_paths, fps=30):
    """Create videos for RGB cameras only"""
    video_dir = output_dir / 'video'
    video_paths = {}
    video_writers = {}
    frame_counts = defaultdict(int)
    
    # Get RGB cameras only
    rgb_cameras = [cam for cam in camera_names if not is_depth_camera(cam)]
    
    print(f"\nCreating RGB videos in {video_dir}...")
    
    for cam_name in tqdm(rgb_cameras, desc="Creating videos"):
        video_path = video_dir / f"{cam_name}.avi"
        video_paths[cam_name] = str(video_path)
        
        # Get all valid image paths for this camera
        valid_paths = [p for p in image_paths[cam_name] if p is not None]
        
        if not valid_paths:
            print(f"Warning: No images found for {cam_name}")
            continue
        
        # Read first image to get dimensions
        first_img_path = output_dir / valid_paths[0]
        first_img = cv2.imread(str(first_img_path))
        
        if first_img is None:
            print(f"Warning: Could not read first image for {cam_name}")
            continue
        
        height, width = first_img.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print(f"Warning: Failed to open video writer for {cam_name}")
            continue
        
        # Write all frames
        for img_path in valid_paths:
            full_path = output_dir / img_path
            img = cv2.imread(str(full_path))
            if img is not None:
                writer.write(img)
                frame_counts[cam_name] += 1
        
        writer.release()
    
    print(f"✓ Created {len(rgb_cameras)} RGB videos")
    for cam_name in rgb_cameras:
        if cam_name in frame_counts:
            print(f"  {cam_name}.avi: {frame_counts[cam_name]} frames")
    
    return video_paths

def save_csv(output_dir, timestamps, image_paths, camera_names, fused_depth_paths):
    """Save synchronized data to CSV file"""
    csv_path = output_dir / 'synchronized_data.csv'
    
    print(f"\nSaving CSV to {csv_path}...")
    
    # Prepare CSV data
    rows = []
    for idx, timestamp in enumerate(timestamps):
        row = {'timestamp': timestamp}
        
        for cam_name in camera_names:
            if idx < len(image_paths[cam_name]) and image_paths[cam_name][idx]:
                row[cam_name] = image_paths[cam_name][idx]
            else:
                row[cam_name] = None
        
        # Add fused depth path
        if idx < len(fused_depth_paths):
            row['fused_depth'] = fused_depth_paths[idx]
        else:
            row['fused_depth'] = None
        
        rows.append(row)
    
    # Write to CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Saved CSV with {len(rows)} rows")
    print(f"  Columns: {', '.join(df.columns)}")
    
    return csv_path

def create_depth_visualization_samples(output_dir, camera_names, image_paths, sample_count=10):
    """Create sample depth visualizations for verification"""
    viz_dir = output_dir / 'depth_visualization_samples'
    depth_cameras = [cam for cam in camera_names if is_depth_camera(cam)]
    
    if not depth_cameras:
        return
    
    print(f"\nCreating depth visualization samples in {viz_dir}...")
    
    # Sample evenly distributed frames
    total_frames = len(image_paths[depth_cameras[0]])
    sample_indices = np.linspace(0, total_frames - 1, min(sample_count, total_frames), dtype=int)
    
    for idx in tqdm(sample_indices, desc="Creating depth visualizations"):
        for cam_name in depth_cameras:
            if idx >= len(image_paths[cam_name]) or image_paths[cam_name][idx] is None:
                continue
            
            img_path = output_dir / image_paths[cam_name][idx]
            timestamp = int(Path(img_path).stem)
            
            viz_filename = f"{cam_name}_{timestamp:019d}_viz.jpg"
            viz_path = viz_dir / viz_filename
            
            create_depth_visualization(img_path, viz_path)
    
    print(f"✓ Created {len(sample_indices)} depth visualization samples per camera")

from utils.generate_grid import generate_perspective_to_erp_grid, warp_erp_by_transform_full
from utils.perspective_utils import planar_to_range_m, T_c2w_fixed
from utils.pointCloud_utils import soft_zbuffer_gating

# Yaw mapping AirSim (NED, +Z down) -> CV (Z up)
YAW_SIGN = 1.0
YAW_OFFSET_GLOBAL = 0.0
YAW_BIAS = {"cam0":0.0,"cam1":0.0,"cam2":0.0,"cam3":0.0,"cam4":0.0,"cam5":0.0,"cam6":0.0,"cam7":0.0}

# Camera settings from AirSim
YAW_AIR = {"cam0":0.0,"cam1":90.0,"cam2":90.0,"cam3":180.0,"cam4":180.0,"cam5":270.0,"cam6":270.0,"cam7":0.0}
POS_CM = {"cam0":[ 0.4,  0.4,0], "cam1":[ 0.4,  0.4,0], "cam2":[-0.4,  0.4,0], "cam3":[-0.4,  0.4,0],
          "cam4":[-0.4, -0.4,0], "cam5":[-0.4, -0.4,0], "cam6":[ 0.4, -0.4,0], "cam7":[ 0.4, -0.4,0]}

def depthPlanar_images_to_erp_depth(camera_names, depth_dict, K, erp_h=512, erp_w=1024):
    """
    Convert depth planar images to equirectangular depth maps and fuse them.
    
    Args:
        camera_names: list of camera names (without '_depth' suffix)
        depth_dict: dict of {camera_name_depth: depth_image (H x W) in meters}
        K: intrinsic matrix (3x3)
        erp_h: output ERP height
        erp_w: output ERP width
    
    Returns:
        fused_depth: fused equirectangular depth map (H x W) in meters
    """
    if not depth_dict:
        return None
    
    # Get image dimensions from first depth image
    first_depth = list(depth_dict.values())[0]
    img_h, img_w = first_depth.shape[:2]
    
    erp_depth_dict = {}
    
    for cam_name in camera_names:
        depth_cam_name = f"{cam_name}_depth"
        
        if depth_cam_name not in depth_dict:
            continue
        
        depth_data = depth_dict[depth_cam_name]
        
        # Validate dimensions
        if depth_data.shape[0] != img_h or depth_data.shape[1] != img_w:
            print(f"Warning: {depth_cam_name} has different dimensions, skipping")
            continue
        
        # Convert planar depth to range depth
        depth_use = planar_to_range_m(depth_data, img_w, img_h, K[0,0])
        
        # Generate mapping grid for this camera
        mapx, mapy = generate_perspective_to_erp_grid(img_h, img_w, K[0,0], erp_h, erp_w, depth_use)
        
        # Warp to ERP
        erp_depth_cv = cv2.remap(depth_use, mapx, mapy, 
                                  interpolation=cv2.INTER_NEAREST, 
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=0).reshape(erp_h, erp_w)
        
        # Apply camera transformation
        T_cam = T_c2w_fixed(cam_name, YAW_OFFSET_GLOBAL, YAW_SIGN, YAW_AIR, YAW_BIAS, POS_CM)
        rot_erp = warp_erp_by_transform_full(erp_depth_cv, T_cam, -T_cam[:3, 3])
        
        erp_depth_dict[cam_name] = rot_erp
    
    if not erp_depth_dict:
        return None
    
    # Stack all ERP depth maps
    depth_stack = np.array([erp_depth_dict[cam_name] for cam_name in camera_names 
                            if cam_name in erp_depth_dict])
    
    if depth_stack.size == 0:
        return None
    
    # Apply soft z-buffer gating for fusion
    gates = soft_zbuffer_gating(depth_stack, alpha=0.03, beta=0.01, gamma0=0.02, gamma1=0.01)
    masked = np.where(gates, depth_stack, np.inf)
    fused_depth = masked.min(axis=0)
    fused_depth[~np.isfinite(fused_depth)] = 0.0
    
    return fused_depth

def save_fused_depth(fused_depth, output_path):
    """
    Save fused ERP depth as PNG (uint16, in millimeters).
    
    Args:
        fused_depth: fused depth in meters (H x W)
        output_path: output file path
    
    Returns:
        True if successful, False otherwise
    """
    if fused_depth is None:
        return False
    
    # Convert meters to millimeters
    depth_mm = fused_depth * 1000.0
    depth_mm = np.clip(depth_mm, 0, 65535)
    depth_uint16 = depth_mm.astype(np.uint16)
    
    # Save as PNG
    cv2.imwrite(str(output_path), depth_uint16, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return True

def create_fused_depth_visualization_samples(output_dir, fused_depth_paths, sample_count=10):
    """Create sample fused depth visualizations"""
    viz_dir = output_dir / 'depth_visualization_samples'
    
    if not fused_depth_paths:
        return
    
    print(f"\nCreating fused depth visualization samples...")
    
    # Sample evenly distributed frames
    total_frames = len(fused_depth_paths)
    sample_indices = np.linspace(0, total_frames - 1, min(sample_count, total_frames), dtype=int)
    
    for idx in tqdm(sample_indices, desc="Creating fused depth visualizations"):
        if fused_depth_paths[idx] is None:
            continue
        
        img_path = output_dir / fused_depth_paths[idx]
        timestamp = int(Path(img_path).stem)
        
        viz_filename = f"fused_depth_{timestamp:019d}_viz.jpg"
        viz_path = viz_dir / viz_filename
        
        create_depth_visualization(img_path, viz_path)
    
    print(f"✓ Created {len(sample_indices)} fused depth visualization samples")

def main():
    parser = argparse.ArgumentParser(
        description='Create RGB-D dataset from AirSim using recorded trajectory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python create_depth_dataset_airsim_api.py trajectory.txt -o dataset
  
Note: 
  - RGB images are saved as JPEG
  - Depth images are saved as PNG (uint16, in millimeters)
  - Fused ERP depth is saved in fused_depth/ folder
  - Only RGB videos are created (depth not suitable for video format)
  - The trajectory file should have columns: timestamp, x, y, z, qw, qx, qy, qz
        """
    )
    parser.add_argument('trajectory_file', help='Path to trajectory file (txt/csv from logger)')
    parser.add_argument('-o', '--output', default='dataset', help='Output directory (default: dataset)')
    parser.add_argument('-f', '--fps', type=int, default=10, help='Video FPS (default: 10)')
    parser.add_argument('--host', default='10.42.0.244', help='AirSim host IP (default: 10.42.0.244)')
    parser.add_argument('--min_depth', type=float, default=0.5, help = "minimum of depth value")
    parser.add_argument('--max_depth', type=float, default=50, help = "minimum of depth value")
    parser.add_argument('--port', type=int, default=41451, help='AirSim port (default: 41451)')
    parser.add_argument('--vehicle', default='survey', help='Vehicle name (default: survey)')
    parser.add_argument('--cameras', nargs='+', 
                       default=['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6', 'cam7',
                                'cam0_depth', 'cam1_depth', 'cam2_depth', 'cam3_depth',
                                'cam4_depth', 'cam5_depth', 'cam6_depth', 'cam7_depth'],
                       help='Camera names (default: cam0-cam7 + depth)')
    parser.add_argument('--fov', type=float, default=140.0, help='Camera FOV in degrees (default: 140.0)')
    parser.add_argument('--erp-height', type=int, default=512, help='ERP output height (default: 512)')
    parser.add_argument('--erp-width', type=int, default=1024, help='ERP output width (default: 1024)')
    parser.add_argument('--no-viz', action='store_true', 
                       help='Skip creating depth visualization samples')
    parser.add_argument('--no-video', action='store_true',
                       help='Skip creating video files')
    
    args = parser.parse_args()
    
    # ========================================================================
    # STEP 1: Load trajectory
    # ========================================================================
    print("="*60)
    print("STEP 1: Loading trajectory")
    print("="*60)
    
    trajectory_path = Path(args.trajectory_file)
    if not trajectory_path.exists():
        print(f"Error: Trajectory file not found: {trajectory_path}")
        return 1
    
    print(f"Loading trajectory from: {trajectory_path}")
    
    try:
        # Load trajectory data
        df = pd.read_csv(trajectory_path, sep='\t')
        print(f"✓ Loaded {len(df)} poses")
        print(f"  Columns: {', '.join(df.columns)}")
        
        # Validate required columns
        required_cols = ['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return 1
        
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return 1
    
    # ========================================================================
    # STEP 2: Connect to AirSim and setup cameras
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 2: Connecting to AirSim and setting up cameras")
    print("="*60)
    
    try:
        print(f"Connecting to AirSim at {args.host}:{args.port}...")
        client = airsim.MultirotorClient(ip=args.host, port=args.port)
        client.confirmConnection()
        print("✓ Connected to AirSim")
        
        # Set camera FOV for RGB cameras
        rgb_cams = [c for c in args.cameras if not is_depth_camera(c)]
        print(f"Setting camera FOV to {args.fov}° for {len(rgb_cams)} RGB cameras...")
        for cam_name in rgb_cams:
            print('cam_name:', cam_name)
            client.simSetCameraFov(cam_name, args.fov)
        print("✓ Camera FOV configured")
        
        # Get camera info to extract intrinsics
        cam_info = client.simGetCameraInfo(rgb_cams[0])
        # Assume square pixels and calculate focal length from FOV
        img_width = 640  # Default, will be updated from first capture
        img_height = 480
        fov_rad = np.deg2rad(args.fov)
        focal_length = (img_width / 2.0) / np.tan(fov_rad / 2.0)
        
        # Create intrinsic matrix
        K = np.array([
            [focal_length, 0, img_width / 2.0],
            [0, focal_length, img_height / 2.0],
            [0, 0, 1]
        ])
        print(f"✓ Camera intrinsics estimated: f={focal_length:.2f}")
        
    except Exception as e:
        print(f"Error connecting to AirSim: {e}")
        return 1
    
    # Setup output directories
    print(f"\nSetting up output directories in: {args.output}")
    output_dir, images_dir, video_dir, viz_dir, fused_depth_dir, camera_dirs = setup_output_directories(
        args.output, args.cameras
    )
    print(f"✓ Output directory: {output_dir.absolute()}")
    
    # ========================================================================
    # STEP 3: Process trajectory and capture images
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 3: Processing trajectory and capturing images")
    print("="*60)
    print(f"Cameras: {', '.join(args.cameras)}")
    print(f"Total poses: {len(df)}")
    print(f"ERP output: {args.erp_height}x{args.erp_width}")
    print()
    
    image_paths = defaultdict(list)
    fused_depth_paths = []
    timestamps = []
    failed_captures = 0
    
    # Get base camera names (without _depth suffix)
    base_camera_names = [c for c in args.cameras if not is_depth_camera(c)]
    
    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Capturing images"):
            # Extract pose from dataframe
            timestamp = int(row['timestamp']*1000)  # Convert to milliseconds
            x, y, z = row['x'], row['y'], row['z']
            qw, qx, qy, qz = row['qw'], row['qx'], row['qy'], row['qz']
            
            # Create AirSim pose
            pose = airsim.Pose()
            pose.position = airsim.Vector3r(x_val=x, y_val=y, z_val=z)
            pose.orientation = quaternion_to_airsim(qw, qx, qy, qz)
            
            # Set vehicle pose
            client.simSetVehiclePose(pose, True, args.vehicle)
            
            # Small delay to ensure pose is set
            time.sleep(0.01)
            
            # Pause simulation for stable capture
            client.simPause(True)
            
            # Capture images from all cameras
            success_count, depth_data_dict, rgb_data_dict = capture_images(
                client, args.vehicle, args.cameras, 
                camera_dirs, timestamp, image_paths
            )
            
            # Update image dimensions from first capture
            if idx == 0 and rgb_data_dict:
                first_rgb = list(rgb_data_dict.values())[0]
                img_height, img_width = first_rgb.shape[:2]
                # Recalculate intrinsics with actual image size
                from utils.perspective_utils import K_from_fov
                K = K_from_fov(img_width, img_height, args.fov)
                focal_length = K[0,0]
                print(f"Updated intrinsics from first capture: {img_width}x{img_height}, f={focal_length:.2f}")
            
            # Fuse depth images to ERP
            if depth_data_dict:
                fused_depth = depthPlanar_images_to_erp_depth(
                    base_camera_names, depth_data_dict, K, 
                    args.erp_height, args.erp_width
                )
                
                if fused_depth is not None:
                    # Save fused depth
                    fused_filename = f"{timestamp:019d}.png"
                    fused_filepath = fused_depth_dir / fused_filename
                    if save_fused_depth(fused_depth, fused_filepath):
                        fused_depth_paths.append(str(fused_filepath.relative_to(output_dir)))
                    else:
                        fused_depth_paths.append(None)
                else:
                    fused_depth_paths.append(None)
            else:
                fused_depth_paths.append(None)
            
            timestamps.append(timestamp)
            
            # Resume simulation
            client.simPause(False)

            if success_count < len(args.cameras):
                failed_captures += 1
            
            # Log progress every 50 frames
            if (idx + 1) % 50 == 0:
                tqdm.write(f"Progress: {idx + 1}/{len(df)} | "
                          f"Position: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        print(f"\n✓ Captured {len(timestamps)} image sets")
        if failed_captures > 0:
            print(f"  Warning: {failed_captures} frames had incomplete captures")
        
        # Print capture statistics
        print(f"\nCapture statistics:")
        for cam_name in args.cameras:
            valid_images = sum(1 for p in image_paths[cam_name] if p is not None)
            img_type = "Depth (PNG, uint16)" if is_depth_camera(cam_name) else "RGB (JPEG)"
            print(f"  {cam_name}: {valid_images}/{len(timestamps)} images ({img_type})")
        
        valid_fused = sum(1 for p in fused_depth_paths if p is not None)
        print(f"  fused_depth: {valid_fused}/{len(timestamps)} images (ERP PNG, uint16)")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        print(f"Captured {len(timestamps)}/{len(df)} image sets before interruption")
    except Exception as e:
        print(f"\nError during capture: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Resume simulation
        print("\nResuming simulation...")
        client.simPause(False)
    
    if len(timestamps) == 0:
        print("Error: No images were captured!")
        return 1
    
    # ========================================================================
    # STEP 4: Post-processing
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 4: Post-processing")
    print("="*60)
    
    # Create videos
    if not args.no_video:
        create_videos(output_dir, args.cameras, image_paths, args.fps)
    
    # Save CSV
    csv_path = save_csv(output_dir, timestamps, image_paths, args.cameras, fused_depth_paths)
    
    # Create depth visualizations
    if not args.no_viz:
        create_depth_visualization_samples(output_dir, args.cameras, image_paths)
        create_fused_depth_visualization_samples(output_dir, fused_depth_paths)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*60)
    print("✓ PROCESSING COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nContents:")
    print(f"  📁 images/          - {len(timestamps)} synchronized image sets")
    
    rgb_cams = [c for c in args.cameras if not is_depth_camera(c)]
    depth_cams = [c for c in args.cameras if is_depth_camera(c)]
    
    if rgb_cams:
        print(f"     ├─ RGB cameras ({len(rgb_cams)}):  JPEG format")
        for cam in rgb_cams:
            print(f"        • {cam}/")
    
    if depth_cams:
        print(f"     └─ Depth cameras ({len(depth_cams)}): PNG format (uint16, millimeters)")
        for cam in depth_cams:
            print(f"        • {cam}/")
    
    valid_fused = sum(1 for p in fused_depth_paths if p is not None)
    print(f"  📁 fused_depth/     - {valid_fused} fused ERP depth images (PNG, uint16)")
    
    if not args.no_video:
        print(f"  📁 video/           - {len(rgb_cams)} RGB video files")
    print(f"  📄 synchronized_data.csv - Metadata with {len(timestamps)} rows")
    if not args.no_viz:
        print(f"  📁 depth_visualization_samples/ - Sample depth colormaps")
    print("="*60)
    print("\nDepth Image Info:")
    print("  - Format: 16-bit unsigned integer (uint16)")
    print("  - Values: Depth in millimeters (0-65535mm)")
    print("  - Compression: PNG level 3")
    print("  - To read: depth_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)")
    print("  - Convert to meters: depth_meters = depth_img.astype(float) / 1000.0")
    print("\nFused ERP Depth Info:")
    print(f"  - Resolution: {args.erp_height}x{args.erp_width}")
    print("  - Format: Equirectangular projection")
    print("  - Fusion method: Soft z-buffer gating")
    print("  - Saved in: fused_depth/ folder")
    print("="*60)
    
    return 0

if __name__ == '__main__':
    exit(main())