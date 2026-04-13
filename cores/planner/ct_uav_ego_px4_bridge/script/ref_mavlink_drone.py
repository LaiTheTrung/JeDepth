from pymavlink import mavutil
import time
import logging

logger = logging.getLogger(__name__)

import threading
import math

class MAVLinkDrone:
    """
    MAVLink Drone Controller for PX4/ArduPilot
    Supports OFFBOARD mode, velocity control, and basic flight operations
    """
    
    def __init__(self, connection_string='udpin:127.0.0.1:14552', serial_enable=False, timeout=30):
        self.connection_string = connection_string
        self.serial_enable = serial_enable
        self.timeout = timeout
        self.mav = None
        self.connected = False
        
        # Connect automatically on init
        self.connect()

        # # ============================================================
        # # START HEADING READER THREAD
        # # ============================================================
        # self.current_heading = 0.0
        # self.heading_lock = threading.Lock()
        # self.heading_thread = None
        # self.heading_thread_running = False
        # self.start_heading_thread()

    def connect(self):
        """Establish MAVLink connection and wait for heartbeat"""
        try:
            logger.info(f"Connecting to drone: {self.connection_string}")
            if self.serial_enable:
                self.mav = mavutil.mavlink_connection('/dev/serial0', baud=115200)
            else:
                self.mav = mavutil.mavlink_connection(self.connection_string)   
            
            logger.info("Waiting for heartbeat...")
            self.mav.wait_heartbeat(timeout=self.timeout)
            
            logger.info(f"✓ Connected to system {self.mav.target_system}, "
                       f"component {self.mav.target_component}")
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False

    def start_heading_thread(self):
        """Khởi động thread đọc heading"""
        self.heading_thread_running = True
        self.heading_thread = threading.Thread(target=self._heading_reader_loop, daemon=True)
        self.heading_thread.start()
        print("✓ Heading reader thread started")
    
    def _heading_reader_loop(self):
        """Loop chạy trong thread riêng để đọc heading liên tục"""
        print("📡 Heading reader thread running...")
        
        while self.heading_thread_running:
            try:
                # Đọc ATTITUDE message
                msg = self.mav.recv_match(type='ATTITUDE', blocking=True, timeout=0.5)
                
                if msg:
                    yaw = math.degrees(msg.yaw)
                    # Chuyển yaw sang heading (0-360)
                    heading = yaw if yaw >= 0 else yaw + 360
                    
                    # Cập nhật heading (thread-safe)
                    with self.heading_lock:
                        self.current_heading = heading
                        
            except Exception as e:
                print(f"⚠️ Error reading heading: {e}")
                time.sleep(0.1)
        
        print("📡 Heading reader thread stopped")
    
    def get_current_heading(self):
        """Lấy heading hiện tại (thread-safe)"""
        with self.heading_lock:
            return self.current_heading

    def global_to_drone_ned(self, vx_global, vy_global, vz_global, heading=None):
        """
        Chuyển đổi vận tốc từ tọa độ Global (North-East-Down) 
        sang tọa độ Local của drone
        
        Args:
            vx_global: Vận tốc theo hướng Bắc (m/s)
            vy_global: Vận tốc theo hướng Đông (m/s)
            vz_global: Vận tốc xuống (m/s)
            heading: Góc heading (độ), None = lấy từ thread
            
        Returns:
            tuple: (vx_local, vy_local, vz_local)
        """
        if heading is None:
            # heading = self.get_current_heading()
            heading = self.current_heading
        
        heading_rad = math.radians(heading)
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        
        vx_local = vx_global * cos_h + vy_global * sin_h
        vy_local = -vx_global * sin_h + vy_global * cos_h
        vz_local = vz_global
        
        return vx_local, vy_local, vz_local
    
    def stop_heading_thread(self):
        """Dừng thread đọc heading"""
        if self.heading_thread_running:
            self.heading_thread_running = False
            if self.heading_thread:
                self.heading_thread.join(timeout=2.0)
            print("✓ Heading reader thread stopped")
    
    def arm(self, force=False):
        """
        Arm the drone
        
        Args:
            force: Force arming (bypass safety checks)
        
        Returns:
            bool: True if command sent successfully
        """
        if not self.connected:
            logger.error("Cannot arm - not connected")
            return False
        
        try:
            arm_value = 1 if not force else 21196  # Magic number for force arm
            
            self.mav.mav.command_long_send(
                self.mav.target_system,
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # confirmation
                arm_value,  # param1: 1=arm, 0=disarm, 21196=force arm
                0, 0, 0, 0, 0, 0
            )
            
            logger.info("✓ ARM command sent" + (" (FORCE)" if force else ""))
            return True
            
        except Exception as e:
            logger.error(f"Arm failed: {e}")
            return False
    
    def disarm(self, force=False):
        """
        Disarm the drone
        
        Args:
            force: Force disarm (emergency)
        
        Returns:
            bool: True if command sent successfully
        """
        if not self.connected:
            logger.error("Cannot disarm - not connected")
            return False
        
        try:
            disarm_value = 0 if not force else 21196
            
            self.mav.mav.command_long_send(
                self.mav.target_system,
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                disarm_value,  # param1: 0=disarm, 21196=force disarm
                0, 0, 0, 0, 0, 0
            )
            
            logger.info("✓ DISARM command sent" + (" (FORCE)" if force else ""))
            return True
            
        except Exception as e:
            logger.error(f"Disarm failed: {e}")
            return False
    
    def set_mode(self, mode):
        """
        Set flight mode (high-level wrapper)
        
        Args:
            mode: Flight mode name ('OFFBOARD', 'LAND', 'HOLD', etc.)
        
        Returns:
            bool: True if command sent successfully
        """
        mode_mapping = {
            'OFFBOARD': (mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0),
            'LAND': (mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, 6),
            'HOLD': (mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, 3),
            'STABILIZED': (mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 0, 0),
        }
        
        if mode.upper() not in mode_mapping:
            logger.error(f"Unknown mode: {mode}")
            return False
        
        base_mode, main_mode, sub_mode = mode_mapping[mode.upper()]
        return self.set_custom_mode(base_mode, main_mode, sub_mode)
    
    def set_custom_mode(self, base_mode, main_mode, sub_mode=0):
        """
        Set custom flight mode (low-level)
        
        Args:
            base_mode: Base mode flag
            main_mode: PX4 main mode (0-7)
            sub_mode: PX4 sub mode (optional)
        
        Returns:
            bool: True if command sent successfully
        """
        if not self.connected:
            logger.error("Cannot set mode - not connected")
            return False
        
        try:
            self.mav.mav.command_long_send(
                self.mav.target_system,
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,  # confirmation
                base_mode,   # param1: base mode
                main_mode,   # param2: custom main mode
                sub_mode,    # param3: custom sub mode
                0, 0, 0, 0
            )
            
            logger.info(f"✓ Mode command sent (base={base_mode}, main={main_mode}, sub={sub_mode})")
            return True
            
        except Exception as e:
            logger.error(f"Set mode failed: {e}")
            return False
    
    def set_offboard_mode(self):
        return self.set_mode('OFFBOARD')
    
    def set_land_mode(self):
        return self.set_mode('LAND')
    
    def start_offboard(self):
        """Initialize OFFBOARD mode"""
        self.arm()
        time.sleep(1)
        self.set_offboard_mode()
    
    def send_velocity_local_ned(self, vx, vy, vz, yaw_rate=0.0):
        if not self.connected:
            logger.warning("Cannot send velocity - not connected")
            return False
        
        try:
            time_boot_ms = int((time.time() * 1000) % 4294967295)
            
            # Type mask to ignore position and acceleration
            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            )
            
            # If yaw_rate is 0, ignore it
            if abs(yaw_rate) < 0.001:
                type_mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
            
            self.mav.mav.set_position_target_local_ned_send(
                time_boot_ms,
                self.mav.target_system,
                self.mav.target_component,
                # mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                type_mask,
                0, 0, 0,      # position ignored
                vx, vy, vz,   # velocity (m/s)
                0, 0, 0,      # accel ignored
                0, 0
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Send velocity failed: {e}")
            return False
    
    def stop_movement(self):
        return self.send_velocity_local_ned(0, 0, 0)
    
    def takeoff(self, altitude):
        if not self.connected:
            logger.error("Cannot takeoff - not connected")
            return False
        
        try:
            self.mav.mav.command_long_send(
                self.mav.target_system,
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0, 0, 0, 0, 0, 0,
                float(altitude)  # param7: altitude
            )
            
            logger.info(f"✓ Takeoff command sent (altitude: {altitude}m)")
            return True
            
        except Exception as e:
            logger.error(f"Takeoff failed: {e}")
            return False
    
    def land(self):
        """Land at current position"""
        return self.set_land_mode()
    
    def get_heartbeat(self, timeout=5):
        if not self.connected:
            return None
        
        try:
            msg = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
            return msg
        except Exception as e:
            logger.error(f"Get heartbeat failed: {e}")
            return None
    
    def get_position(self, timeout=1):
        if not self.connected:
            return None
        
        try:
            msg = self.mav.recv_match(
                type='LOCAL_POSITION_NED', 
                blocking=True, 
                timeout=timeout
            )
            
            if msg:
                return {
                    'x': msg.x,
                    'y': msg.y,
                    'z': msg.z,
                    'vx': msg.vx,
                    'vy': msg.vy,
                    'vz': msg.vz
                }
            return None
            
        except Exception as e:
            logger.error(f"Get position failed: {e}")
            return None
    
    def close(self):
        """Close MAVLink connection"""
        if self.mav:
            try:
                self.mav.close()
                logger.info("✓ Connection closed")
            except:
                pass
        self.connected = False
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.close()