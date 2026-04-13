#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import math
import struct

# Helper function for Log Odds
def logit(p):
    return math.log(p / (1 - p))

class EgoRaycastSimulation(Node):
    def __init__(self):
        super().__init__('ego_raycast_sim')

        # --- Parameters (Mimicking plan_env/grid_map.h) ---
        # Camera Intrinsics
        self.declare_parameter('fx', 387.229248046875) # RealSense example or sim
        self.declare_parameter('fy', 387.229248046875)
        self.declare_parameter('cx', 321.0465393066406)
        self.declare_parameter('cy', 243.44969177246094)
        
        # GridMap Parameters
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('max_ray_length', 4.5)
        self.declare_parameter('min_ray_length', 0.1)
        
        # Probabilities for Raycast (Log Odds)
        self.declare_parameter('p_hit', 0.70)
        self.declare_parameter('p_miss', 0.35)
        self.declare_parameter('p_min', 0.12)
        self.declare_parameter('p_max', 0.97)
        self.declare_parameter('p_occ', 0.80)

        # Performance
        self.declare_parameter('skip_pixel', 10) # Higher skip for Python performance!
        self.declare_parameter('k_depth_scaling_factor', 1000.0) # Often 1000 for mm -> m

        # TF
        self.declare_parameter('frame_id', 'map')

        # Load Params
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.resolution = self.get_parameter('resolution').value
        self.max_ray_length = self.get_parameter('max_ray_length').value
        self.min_ray_length = self.get_parameter('min_ray_length').value
        self.skip = self.get_parameter('skip_pixel').value
        self.scale = self.get_parameter('k_depth_scaling_factor').value
        self.frame_id = self.get_parameter('frame_id').value

        # Probability Constants
        self.p_hit = self.get_parameter('p_hit').value
        self.p_miss = self.get_parameter('p_miss').value
        self.p_min = self.get_parameter('p_min').value
        self.p_max = self.get_parameter('p_max').value
        self.p_occ = self.get_parameter('p_occ').value

        self.prob_hit_log = logit(self.p_hit)
        self.prob_miss_log = logit(self.p_miss)
        self.clamp_min_log = logit(self.p_min)
        self.clamp_max_log = logit(self.p_max)
        self.min_occupancy_log = logit(self.p_occ)

        # Map Storage: Dictionary {(x_idx, y_idx, z_idx): log_odds_value}
        self.voxel_map = {}
        
        # Subscribers
        self.subscription = self.create_subscription(
            Image,
            '/stereo/depth_map', # Adjust topic as needed
            self.depth_callback,
            10
        )
        self.publisher = self.create_publisher(PointCloud2, '/ego_raycast_occupancy', 10)
        
        self.bridge = CvBridge()
        self.get_logger().info("Ego Raycast Simulation Node Started (Full Raycast Logic)")
        self.get_logger().info(f"Resolution: {self.resolution}m, Skip: {self.skip} (Python optimized)")

    def pos_to_index(self, pos):
        """ Convert float position to integer voxel index based on resolution """
        return np.floor(pos / self.resolution).astype(int)

    def index_to_pos(self, idx):
        """ Convert integer voxel index to float center position """
        return (idx + 0.5) * self.resolution

    def update_voxel(self, idx_tuple, is_hit):
        """ Update log odds for a single voxel """
        current_val = self.voxel_map.get(idx_tuple, self.clamp_min_log) # Default to free space logic usually, or unknown
        
        # In GridMap, initialization is often unknown or free. 
        # Here we assume min_log (free) if unknown for visual clarity in sparse map
        
        update = self.prob_hit_log if is_hit else self.prob_miss_log
        new_val = current_val + update
        
        # Clamp
        if new_val > self.clamp_max_log:
            new_val = self.clamp_max_log
        elif new_val < self.clamp_min_log:
            new_val = self.clamp_min_log
            
        self.voxel_map[idx_tuple] = new_val

    def raycast_bresenham_3d(self, p1, p2):
        """
        Implementation of 3D Ray Traversal (Bresenham/DDA style).
        Yields voxel indices from p1 to p2 (excluding p1, usually we raycast FROM camera TO point).
        The C++ GridMap raycaster steps from 'start' to 'end'.
        """
        x1, y1, z1 = int(p1[0]), int(p1[1]), int(p1[2])
        x2, y2, z2 = int(p2[0]), int(p2[1]), int(p2[2])

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        
        xs = 1 if x2 > x1 else -1
        ys = 1 if y2 > y1 else -1
        zs = 1 if z2 > z1 else -1

        # Driving axis
        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while x1 != x2:
                x1 += xs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dx
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
                yield (x1, y1, z1)
                
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while y1 != y2:
                y1 += ys
                if p1 >= 0:
                    x1 += xs
                    p1 -= 2 * dy
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
                yield (x1, y1, z1)
                
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while z1 != z2:
                z1 += zs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dz
                if p2 >= 0:
                    x1 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx
                yield (x1, y1, z1)

    def depth_callback(self, msg):
        try:
            # --- 1. PROJECTION (Same as before) ---
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
            if self.scale != 1.0:
                depth_img = depth_img / self.scale

            height, width = depth_img.shape
            
            # Reduce Resolution for Python Performance
            u_indices = np.arange(0, width, self.skip)
            v_indices = np.arange(0, height, self.skip)
            u_grid, v_grid = np.meshgrid(u_indices, v_indices)
            
            depths = depth_img[v_grid, u_grid]
            
            # Flatten for processing
            depths_flat = depths.flatten()
            u_flat = u_grid.flatten()
            v_flat = v_grid.flatten()
            
            # --- 2. RAYCAST PROCESS (Logic from GridMap.cpp) ---
            # We must process explicit rays to do traversal
            
            # Filter invalids blindly first to save time? 
            # Note: C++ uses min/max filtering in projectDepthImage, but handles Max Range clamping in Raycast.
            # We will follow that.
            
            mask = depths_flat > self.min_ray_length
            depths_valid = depths_flat[mask]
            u_valid = u_flat[mask]
            v_valid = v_flat[mask]
            
            # Camera Frame Projection
            z_cam = depths_valid
            x_cam = (u_valid - self.cx) * z_cam / self.fx
            y_cam = (v_valid - self.cy) * z_cam / self.fy
            
            # Body/Map Frame Transformation (Identity/Rotated as per Ego)
            # Map X (Front) = Cam Z
            # Map Y (Left)  = -Cam X
            # Map Z (Up)    = -Cam Y
            x_map = z_cam
            y_map = -x_cam
            z_map = -y_cam
            
            points_map = np.stack([x_map, y_map, z_map], axis=1)
            
            # Camera Position in Map Frame (Assumed 0,0,0 for this standard node, 
            # In real system this comes from Odometry)
            camera_pos = np.array([0.0, 0.0, 0.0])
            camera_idx = self.pos_to_index(camera_pos)

            # Per-frame visited cache (to prevent double counting in one frame like C++)
            visited_voxels = set()
            
            for i in range(len(points_map)):
                pt_w = points_map[i]
                length = np.linalg.norm(pt_w - camera_pos)
                
                is_hit = True
                
                # Logic: Clamp to Max Range
                if length > self.max_ray_length:
                    # Scale vector to max_ray_length
                    pt_w = (pt_w - camera_pos) / length * self.max_ray_length + camera_pos
                    is_hit = False # It's a MISS (clearing free space up to max range)
                
                # Endpoint Index
                end_idx = self.pos_to_index(pt_w)
                end_idx_tuple = tuple(end_idx)
                
                # Update Endpoint
                if end_idx_tuple not in visited_voxels:
                    self.update_voxel(end_idx_tuple, is_hit) # 1 if Hit, 0 if Miss (Clear)
                    visited_voxels.add(end_idx_tuple)

                # --- RAY TRAVERSAL (CLEARING FREE SPACE) ---
                # Iterate from Camera to Endpoint
                # Note: We convert to Index space for Bresenham
                
                # We stop before the endpoint to avoid clearing the obstacle we just marked!
                # C++ Logic: "if (vox_idx != INVALID_IDX) ... while (raycaster.step(ray_pt)) ... setCacheOccupancy(tmp, 0)"
                # The step loop traverses the path.
                
                # Python Generator
                ray_path = self.raycast_bresenham_3d(camera_idx, end_idx)
                
                for vox_idx in ray_path:
                    # In Bresenham, if we reach end_idx, we stop?
                    # My implementation yields points BETWEEN start And end.
                    # We must ensure we don't clear the 'end_idx' if it was a HIT.
                    if vox_idx == end_idx_tuple:
                        continue
                        
                    if vox_idx not in visited_voxels:
                        self.update_voxel(vox_idx, False) # Mark as MISS (Free space)
                        visited_voxels.add(vox_idx)

            # --- 3. PUBLISH OCCUPANCY ---
            # Extract occupied voxels
            
            points_out = []
            for idx, log_val in self.voxel_map.items():
                if log_val > self.min_occupancy_log:
                    # Convert back to point
                    pt = self.index_to_pos(np.array(idx))
                    points_out.append(pt)
            
            if not points_out:
                return

            points_out = np.array(points_out, dtype=np.float32)
            
            # Create PointCloud2 msg manually or via helper
            # Standard helper might be slow for big arrays, using raw struct packing is faster or standard lib
            # Using header
            header = Header()
            header.stamp = msg.header.stamp
            header.frame_id = self.frame_id # 'map'
            
            # Simple conversion using sensor_msgs_py
            # Note: Need explicit import inside method or global if available
            from sensor_msgs_py import point_cloud2
            pc2_msg = point_cloud2.create_cloud_xyz32(header, points_out)
            
            self.publisher.publish(pc2_msg)
            
            # Optional: Periodic cleanup of map to prevent infinite growth in this demo
            if len(self.voxel_map) > 100000:
                self.voxel_map.clear()
                self.get_logger().warn("Map cleared to prevent memory overflow in demo script.")

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = EgoRaycastSimulation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
