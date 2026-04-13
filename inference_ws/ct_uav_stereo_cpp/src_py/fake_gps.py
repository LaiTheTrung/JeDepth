#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry
import math


class FakeGPSPublisher(Node):
    def __init__(self):
        super().__init__('fake_gps_publisher')
        
        # Publisher cho topic /fmu/in/vehicle_mocap_odometry
        self.publisher_ = self.create_publisher(
            VehicleOdometry,
            '/fmu/in/vehicle_mocap_odometry',
            10
        )
        
        # Timer để publish với tần số 10Hz
        self.timer = self.create_timer(0.1, self.publish_odometry)
        
        # Tọa độ GPS gốc (Can Tho University)
        self.lat_origin = 10.7778617
        self.lon_origin = 106.6899343
        self.alt_origin = 0.0  # meters above sea level
        
        # Earth radius
        self.EARTH_RADIUS = 6378137.0  # meters
        
        # Biến để tạo dữ liệu giả
        self.counter = 0
        
        self.get_logger().info(f'Fake GPS Publisher started at LAT: {self.lat_origin}, LON: {self.lon_origin}')

    def geodetic_to_ned(self, lat, lon, alt):
        """Chuyển đổi tọa độ GPS (lat, lon, alt) sang NED frame"""
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        lat_origin_rad = math.radians(self.lat_origin)
        lon_origin_rad = math.radians(self.lon_origin)
        
        # Calculate differences
        dlat = lat_rad - lat_origin_rad
        dlon = lon_rad - lon_origin_rad
        
        # Convert to NED (North-East-Down)
        north = dlat * self.EARTH_RADIUS
        east = dlon * self.EARTH_RADIUS * math.cos(lat_origin_rad)
        down = -(alt - self.alt_origin)  # Down is negative of altitude
        
        return north, east, down

    def publish_odometry(self):
        msg = VehicleOdometry()
        
        # Timestamp (microseconds)
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.timestamp_sample = msg.timestamp
        
        # Tạo chuyển động nhỏ xung quanh điểm gốc (bán kính ~10m)
        radius_meters = 10.0
        angular_velocity = 0.05  # rad/s
        angle = self.counter * angular_velocity * 0.1  # slow movement
        altitude = 5.0  # 5 meters above ground
        
        # Tính offset từ điểm gốc (in meters)
        north_offset = radius_meters * math.cos(angle)
        east_offset = radius_meters * math.sin(angle)
        
        # Convert offset to lat/lon
        lat_rad = math.radians(self.lat_origin)
        dlat = north_offset / self.EARTH_RADIUS
        dlon = east_offset / (self.EARTH_RADIUS * math.cos(lat_rad))
        
        current_lat = self.lat_origin + math.degrees(dlat)
        current_lon = self.lon_origin + math.degrees(dlon)
        current_alt = altitude
        
        # Convert to NED
        north, east, down = self.geodetic_to_ned(current_lat, current_lon, current_alt)
        
        # Position (NED frame)
        msg.position[0] = north
        msg.position[1] = east
        msg.position[2] = down
        
        # Velocity (m/s) - chuyển động tròn
        msg.velocity[0] = -radius_meters * angular_velocity * 0.1 * math.sin(angle)
        msg.velocity[1] = radius_meters * angular_velocity * 0.1 * math.cos(angle)
        msg.velocity[2] = 0.0
        
        # Orientation (quaternion - identity = không xoay)
        msg.q[0] = 1.0
        msg.q[1] = 0.0
        msg.q[2] = 0.0
        msg.q[3] = 0.0
        
        # Pose frame
        msg.pose_frame = VehicleOdometry.POSE_FRAME_NED
        
        # Publish message
        self.publisher_.publish(msg)
        
        self.counter += 1
        
        if self.counter % 10 == 0:
            self.get_logger().info(
                f'GPS: LAT={current_lat:.7f}, LON={current_lon:.7f}, ALT={current_alt:.2f}m | '
                f'NED: N={north:.2f}, E={east:.2f}, D={down:.2f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = FakeGPSPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()