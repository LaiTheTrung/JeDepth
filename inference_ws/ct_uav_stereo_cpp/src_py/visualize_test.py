"""
This code aim to visualize the data for debuging:
topic list: 
- /fmu/in/obstacle_distance

đọc data của nó là chuỗi uint16 [72] và hiển thị dưới dạng biểu đồ  tròn. Đối với giá trị nào >= 10m thì hiển thị lớn nhất của những giá trị < 10m.
"""

import rclpy
from rclpy.node import Node
from px4_msgs.msg import ObstacleDistance
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class ObstacleVisualizer(Node):
    def __init__(self):
        super().__init__('obstacle_visualizer')
        
        # QoS for PX4 topics
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        # Subscribe to obstacle distance
        self.subscription = self.create_subscription(
            ObstacleDistance,
            '/fmu/in/obstacle_distance',
            self.obstacle_callback,
            qos_sensor
        )
        
        # Store latest data
        self.distances = np.full(72, 65535, dtype=np.uint16)
        self.max_display = 1000  # 10m in cm
        
        self.get_logger().info('Obstacle Visualizer started')
        self.get_logger().info('Subscribed to: /fmu/in/obstacle_distance')
    
    def obstacle_callback(self, msg):
        """Callback for obstacle distance messages"""
        self.distances = np.array(msg.distances, dtype=np.uint16)
    
    def get_display_distances(self):
        """Process distances for display"""
        # Convert to float for processing
        distances = self.distances.astype(np.float32)
        
        # Find max value among valid measurements < 10m
        valid_distances = distances[distances < self.max_display]
        if len(valid_distances) > 0:
            max_valid = np.max(valid_distances)
        else:
            max_valid = self.max_display
        
        # Replace values >= 10m with max_valid
        display_distances = distances.copy()
        display_distances[distances >= self.max_display] = self.max_display
        
        # Convert to meters for display
        return display_distances / 100.0


def main():
    rclpy.init()
    
    visualizer = ObstacleVisualizer()
    
    # Setup polar plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Number of sectors (72 sectors, 5 degrees each)
    num_sectors = 72
    theta = np.linspace(0, 2 * np.pi, num_sectors, endpoint=False)
    width = 2 * np.pi / num_sectors
    
    # Initialize bars
    bars = ax.bar(theta, np.zeros(num_sectors), width=width, bottom=0.0)
    
    # Styling
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 10)
    ax.set_ylabel('Distance (m)', labelpad=30)
    ax.set_title('Obstacle Distance Visualization', pad=20)
    
    # Color map
    colors = plt.cm.RdYlGn(np.linspace(0, 1, 10))
    
    def update(frame):
        """Update plot with new data"""
        rclpy.spin_once(visualizer, timeout_sec=0.01)
        
        # Get processed distances
        distances = visualizer.get_display_distances()
        
        # Update bars
        for bar, distance in zip(bars, distances):
            bar.set_height(distance)
            
            # Color based on distance (red = close, green = far)
            if distance < 1.0:
                color = colors[0]  # Red
            elif distance < 2.0:
                color = colors[2]  # Orange
            elif distance < 5.0:
                color = colors[5]  # Yellow
            else:
                color = colors[9]  # Green
            
            bar.set_color(color)
            bar.set_alpha(0.8)
        
        return bars
    
    # Create animation
    ani = FuncAnimation(fig, update, interval=50, blit=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

