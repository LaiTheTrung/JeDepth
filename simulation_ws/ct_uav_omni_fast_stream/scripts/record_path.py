import cosysairsim as airsim
import time
import os
from datetime import datetime

"""
Code này record lại trajectory của UAV trong AirSim. Sử dụng API của airsim.
Ý tưởng chính là lưu lại vị trí (x,y,z) và hướng (quaternion) của UAV tại các thời điểm khác nhau.
Dữ liệu được lưu vào file text với định dạng tab-separated, có thể dễ dàng load bằng pandas để phân tích sau này.

Dữ liệu này có thể dùng để phát lại đường bay đã ghi (playback) khi muốn record lại dataset mà không cần phải điều khiển UAV trong AirSim nhiều lần.
Khi playback, ta có thể sử dụng hàm simSetVehiclePose để đặt UAV về đúng vị trí và hướng đã ghi lại. Sau đó UAV sẽ dừng lại cho tới khi ta thu được
đầy đủ dữ liệu cần thiết (hình ảnh, pointcloud, v.v...).

"""
class TrajectoryLogger:
    """
    Simple UAV trajectory logger - logs position and orientation at fixed intervals.
    Data saved in tab-separated text file (loadable with pandas).
    """
    
    def __init__(self, client, vehicle_name="", output_dir="trajectory_logs"):
        """
        Initialize the trajectory logger.
        
        Args:
            client: AirSim MultirotorClient instance
            vehicle_name: Name of the vehicle (empty string for default)
            output_dir: Directory to save log files
        """
        self.client = client
        self.vehicle_name = vehicle_name
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        self.log_file = None
        self.sample_count = 0
        self.start_time = None
        
    def start(self, filename=None):
        """
        Open log file and write header.
        
        Args:
            filename: Optional custom filename. If None, auto-generates timestamp-based name.
        """
        # Generate filename
        if filename is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp_str}.txt"
        
        self.log_filename = os.path.join(self.output_dir, filename)
        
        # Open log file and write header
        self.log_file = open(self.log_filename, 'w')
        self.log_file.write("timestamp\tx\ty\tz\tqw\tqx\tqy\tqz\n")
        self.log_file.flush()
        
        self.sample_count = 0
        self.start_time = time.time()
        
        print(f"Logging started. Saving to: {self.log_filename}")
        return self.log_filename
    
    def log_sample(self):
        """Log one sample of position and orientation"""
        if self.log_file is None:
            raise RuntimeError("Logger not started. Call start() first.")
        
        # Get vehicle pose
        pose = self.client.simGetVehiclePose(self.vehicle_name)
        
        # Position (NED coordinates)
        x = pose.position.x_val
        y = pose.position.y_val
        z = pose.position.z_val
        
        # Orientation (quaternion)
        qw = pose.orientation.w_val
        qx = pose.orientation.x_val
        qy = pose.orientation.y_val
        qz = pose.orientation.z_val
        
        # Get timestamp
        timestamp = time.time()
        
        # Write to file
        self.log_file.write(
            f"{timestamp:.6f}\t{x:.6f}\t{y:.6f}\t{z:.6f}\t"
            f"{qw:.6f}\t{qx:.6f}\t{qy:.6f}\t{qz:.6f}\n"
        )
        self.log_file.flush()
        
        self.sample_count += 1
        
        return timestamp, x, y, z, qw, qx, qy, qz
    
    def stop(self):
        """Close the log file"""
        if self.log_file is not None:
            self.log_file.close()
            elapsed = time.time() - self.start_time
            print(f"\nLogging stopped. Total samples: {self.sample_count}")
            print(f"Duration: {elapsed:.2f} seconds")
            print(f"Data saved to: {self.log_filename}")
            self.log_file = None


def load_trajectory(filename):
    """
    Load trajectory data using pandas.
    
    Args:
        filename: Path to the trajectory file
        
    Returns:
        pandas DataFrame with trajectory data
    """
    import pandas as pd
    df = pd.read_csv(filename, sep='\t')
    print(f"Loaded {len(df)} samples from {filename}")
    return df


# --- Example 1: Simple logging with manual control ---
def example_manual_logging():
    """Example: Log data manually at your own pace"""
    HOST_IP = "10.42.0.244"
    HOST_PORT = 41451
    VEHICLE_NAME = "survey"
    
    # Connect
    client = airsim.MultirotorClient(ip=HOST_IP, port=HOST_PORT)
    client.confirmConnection()
    print("Connected to AirSim\n")
    
    # Create logger
    logger = TrajectoryLogger(client, vehicle_name=VEHICLE_NAME)
    logger.start()
    
    try:
        # Log 10 samples
        for i in range(10):
            timestamp, x, y, z, qw, qx, qy, qz = logger.log_sample()
            print(f"Sample {i+1}: Position ({x:.2f}, {y:.2f}, {z:.2f})")
            time.sleep(0.5)  # Wait 0.5 seconds between samples
            
    finally:
        logger.stop()


# --- Example 2: Logging at fixed frequency ---
def example_fixed_frequency():
    """Example: Log data at a fixed frequency (Hz)"""
    # HOST_IP = "10.42.0.244"
nvidia    HOST_PORT = 41451
    VEHICLE_NAME = "survey"
    LOG_FREQUENCY = 5.0  # Hz
    DURATION = 400.0  # seconds (2.000 samples at 5Hz -> 400s)
    
    # Connect
    client = airsim.MultirotorClient(ip=HOST_IP, port=HOST_PORT)
    client.confirmConnection()
    print("Connected to AirSim\n")
    
    # Create logger
    logger = TrajectoryLogger(client, vehicle_name=VEHICLE_NAME)
    logger.start()
    
    log_interval = 1.0 / LOG_FREQUENCY
    end_time = time.time() + DURATION
    
    print(f"Logging at {LOG_FREQUENCY} Hz for {DURATION} seconds...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        while time.time() < end_time:
            loop_start = time.time()
            
            # Log one sample
            timestamp, x, y, z, qw, qx, qy, qz = logger.log_sample()
            
            # Print progress every second
            if logger.sample_count % LOG_FREQUENCY == 0:
                print(f"Logged {logger.sample_count} samples | "
                      f"Position: ({x:.2f}, {y:.2f}, {z:.2f})")
            
            # Sleep to maintain frequency
            elapsed = time.time() - loop_start
            sleep_time = log_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        logger.stop()
    
    # Load and display the data
    print("\n" + "="*50)
    print("Loading data with pandas...")
    print("="*50)
    
    try:
        import pandas as pd
        df = load_trajectory(logger.log_filename)
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nBasic statistics:")
        print(df[['x', 'y', 'z']].describe())
    except ImportError:
        print("Install pandas to load data: pip install pandas")


# --- Example 3: Integration with existing control loop ---
def example_with_control_loop():
    """Example: Logging while controlling the drone"""
    HOST_IP = "10.42.0.244"
    HOST_PORT = 41451
    VEHICLE_NAME = "survey"
    
    # Connect
    client = airsim.MultirotorClient(ip=HOST_IP, port=HOST_PORT)
    client.confirmConnection()
    print("Connected to AirSim\n")
    
    # Create logger
    logger = TrajectoryLogger(client, vehicle_name=VEHICLE_NAME)
    logger.start(filename="teleport_test.txt")
    
    try:
        # Get initial pose
        pose = client.simGetVehiclePose(VEHICLE_NAME)
        logger.log_sample()
        print("Logged initial position")
        time.sleep(1)
        
        # Teleport +10m in x-direction
        print("\nTeleporting +10m in x-direction...")
        pose.position.x_val += 10
        client.simSetVehiclePose(pose, True, VEHICLE_NAME)
        
        # Log new position
        time.sleep(0.5)
        logger.log_sample()
        timestamp, x, y, z, _, _, _, _ = logger.log_sample()
        print(f"Logged new position: ({x:.2f}, {y:.2f}, {z:.2f})")
        time.sleep(1)
        
        # Teleport back
        print("\nTeleporting back...")
        pose.position.x_val -= 10
        client.simSetVehiclePose(pose, True, VEHICLE_NAME)
        
        # Log final position
        time.sleep(0.5)
        logger.log_sample()
        timestamp, x, y, z, _, _, _, _ = logger.log_sample()
        print(f"Logged final position: ({x:.2f}, {y:.2f}, {z:.2f})")
        
    finally:
        logger.stop()


# --- Run example ---
if __name__ == "__main__":
    # Choose which example to run:
    # example_manual_logging()
    example_fixed_frequency()
    # example_with_control_loop()