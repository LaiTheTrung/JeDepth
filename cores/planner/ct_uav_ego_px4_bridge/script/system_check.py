#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude, VehicleStatus
from sensor_msgs.msg import Image, Imu
import time
import psutil
import os
import sys
import numpy as np
from collections import deque

class SystemCheckNode(Node):
    def __init__(self):
        super().__init__('system_check_node')

        # --- CONFIGURATION ---
        self.declare_parameter('depth_topic', '/stereo/depth_map')
        self.depth_topic = self.get_parameter('depth_topic').value

        # --- METRICS STORAGE ---
        self.topics = {
            'PX4 Position': {'topic': '/fmu/out/vehicle_local_position', 'count': 0, 'hz': 0.0, 'last_ts': 0, 'status': 'WAITING'},
            'PX4 Attitude': {'topic': '/fmu/out/vehicle_attitude', 'count': 0, 'hz': 0.0, 'last_ts': 0, 'status': 'WAITING'},
            'PX4 Status':   {'topic': '/fmu/out/vehicle_status', 'count': 0, 'hz': 0.0, 'last_ts': 0, 'status': 'WAITING'},
            'Stereo Depth': {'topic': self.depth_topic, 'count': 0, 'hz': 0.0, 'last_ts': 0, 'status': 'WAITING'},
        }
        
        # QoS for PX4 (Best Effort is critical)
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- SUBSCRIBERS ---
        self.create_subscription(VehicleLocalPosition, self.topics['PX4 Position']['topic'], 
                                 lambda msg: self.topic_callback('PX4 Position', msg), qos_best_effort)
        
        self.create_subscription(VehicleAttitude, self.topics['PX4 Attitude']['topic'], 
                                 lambda msg: self.topic_callback('PX4 Attitude', msg), qos_best_effort)

        self.create_subscription(VehicleStatus, self.topics['PX4 Status']['topic'], 
                                 lambda msg: self.topic_callback('PX4 Status', msg), qos_best_effort)

        self.create_subscription(Image, self.topics['Stereo Depth']['topic'], 
                                 lambda msg: self.topic_callback('Stereo Depth', msg), 10)

        # --- TIMER ---
        self.create_timer(1.0, self.display_status) # Update UI every 1s
        self.start_time = time.time()
        
        print("\033[2J") # Clear screen

    def topic_callback(self, name, msg):
        now = time.time()
        data = self.topics[name]
        data['count'] += 1
        data['last_ts'] = now
        
        # Specific Checks
        if name == 'PX4 Position':
            if not np.isfinite([msg.x, msg.y, msg.z]).all():
                data['status'] = 'INVALID (NaN)'
            else:
                data['status'] = 'OK'
                
        elif name == 'Stereo Depth':
            if msg.data:
                data['status'] = f'OK ({msg.width}x{msg.height})'
            else:
                data['status'] = 'EMPTY'

    def display_status(self):
        # Move cursor to top-left
        print("\033[H", end="")
        
        print("="*60)
        print(f"🚀 SYSTEM PRE-FLIGHT CHECK | Time: {time.strftime('%H:%M:%S')}")
        print("="*60)
        
        # 1. TOPIC HEALTH
        print(f"{'TOPIC NAME':<20} | {'FREQ (Hz)':<10} | {'STATUS':<20}")
        print("-" * 60)
        
        all_good = True
        
        for name, data in self.topics.items():
            # Calculate Hz
            data['hz'] = data['count'] # Since timer is 1s, count is Hz
            data['count'] = 0 # Reset count
            
            # Check timeout
            if time.time() - data['last_ts'] > 2.0:
                data['status'] = 'TIMEOUT (>2s)'
                color = "\033[91m" # Red
                all_good = False
            elif data['hz'] < 5.0:
                color = "\033[93m" # Yellow
                if data['status'] == 'OK': data['status'] = 'LOW RATE'
            else:
                color = "\033[92m" # Green
            
            print(f"{name:<20} | {color}{data['hz']:<10.1f}\033[0m | {color}{data['status']:<20}\033[0m")

        print("-" * 60)

        # 2. SYSTEM RESOURCES
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        
        cpu_color = "\033[91m" if cpu > 85 else "\033[92m"
        mem_color = "\033[91m" if mem > 85 else "\033[92m"
        
        print(f"💻 SYSTEM LOAD:")
        print(f"   CPU: {cpu_color}{cpu}%\033[0m")
        print(f"   RAM: {mem_color}{mem}%\033[0m")
        
        # 3. OVERALL VERDICT
        print("="*60)
        if all_good:
            print(f"\033[92m✅ SYSTEM READY FOR FLIGHT\033[0m")
        else:
            print(f"\033[91m❌ SYSTEM NOT READY - CHECK ERRORS ABOVE\033[0m")
        print("="*60)
        print("Press Ctrl+C to exit...")

    def check_status(self):
        """Returns True if all systems are OK"""
        for name, data in self.topics.items():
            # Check timeout (>2s is bad)
            if time.time() - data['last_ts'] > 2.0:
                return False
            # Check frequency (<5Hz is warning/bad depending on strictness, let's say bad for critical)
            # Actually, let's just check status string
            if 'TIMEOUT' in data['status'] or 'INVALID' in data['status']:
                return False
        
        # Check CPU
        if psutil.cpu_percent() > 90:
            return False
            
        return True

import argparse

def main(args=None):
    rclpy.init(args=args)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='System Check Node')
    parser.add_argument('--timeout', type=float, default=0.0, help='Duration to run check in seconds (0 = infinite)')
    parsed_args, _ = parser.parse_known_args()
    
    node = SystemCheckNode()
    
    try:
        start_time = time.time()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            
            # Check timeout
            if parsed_args.timeout > 0:
                elapsed = time.time() - start_time
                if elapsed > parsed_args.timeout:
                    # Final Check
                    if node.check_status():
                        print(f"\n\033[92m[SUCCESS] System check passed after {parsed_args.timeout}s.\033[0m")
                        sys.exit(0)
                    else:
                        print(f"\n\033[91m[FAILURE] System check failed after {parsed_args.timeout}s.\033[0m")
                        sys.exit(1)
                        
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
