import cosysairsim as airsim
import numpy as np
import cv2
import sys
import time

# --- Equirectangular to Fisheye Converter ---
class EquirectToFisheye:
    def __init__(self, input_width, input_height, output_size=800, h_fov=200, v_fov=200, yaw=0, pitch=0):
        """
        Initialize the converter with mapping matrices.
        
        Args:
            input_width: Width of input equirectangular image
            input_height: Height of input equirectangular image
            output_size: Size of output fisheye image (square)
            h_fov: Horizontal field of view in degrees
            v_fov: Vertical field of view in degrees
            yaw: Yaw rotation in degrees
            pitch: Pitch rotation in degrees
        """
        self.input_width = input_width
        self.input_height = input_height
        self.output_size = output_size
        self.h_fov = np.deg2rad(h_fov)
        self.v_fov = np.deg2rad(v_fov)
        self.yaw = np.deg2rad(yaw)
        self.pitch = np.deg2rad(pitch)
        
        # Pre-compute mapping matrices for remap
        self.mapx, self.mapy = self._create_mapping()
        
    def _create_mapping(self):
        """Create the mapping matrices for cv2.remap"""
        mapx = np.zeros((self.output_size, self.output_size), dtype=np.float32)
        mapy = np.zeros((self.output_size, self.output_size), dtype=np.float32)
        
        # Center of the fisheye image
        center = self.output_size / 2.0
        
        # Maximum radius for the given FOV
        max_radius = center
        
        for y in range(self.output_size):
            for x in range(self.output_size):
                # Calculate distance from center
                dx = x - center
                dy = y - center
                r = np.sqrt(dx*dx + dy*dy)
                
                # Skip pixels outside the fisheye circle
                if r > max_radius:
                    mapx[y, x] = -1
                    mapy[y, x] = -1
                    continue
                
                # Normalize radius (0 to 1)
                r_norm = r / max_radius
                
                # Fisheye to 3D direction
                # theta is the angle from the fisheye center
                theta = r_norm * (self.h_fov / 2.0)
                
                # phi is the azimuthal angle
                phi = np.arctan2(dy, dx)
                
                # Convert to 3D unit vector
                sin_theta = np.sin(theta)
                x3d = sin_theta * np.cos(phi)
                y3d = sin_theta * np.sin(phi)
                z3d = np.cos(theta)
                
                # Apply pitch rotation
                if self.pitch != 0:
                    cos_p = np.cos(self.pitch)
                    sin_p = np.sin(self.pitch)
                    y3d_new = y3d * cos_p - z3d * sin_p
                    z3d_new = y3d * sin_p + z3d * cos_p
                    y3d = y3d_new
                    z3d = z3d_new
                
                # Apply yaw rotation
                if self.yaw != 0:
                    cos_y = np.cos(self.yaw)
                    sin_y = np.sin(self.yaw)
                    x3d_new = x3d * cos_y - z3d * sin_y
                    z3d_new = x3d * sin_y + z3d * cos_y
                    x3d = x3d_new
                    z3d = z3d_new
                
                # Convert 3D direction to equirectangular coordinates
                longitude = np.arctan2(x3d, z3d)
                latitude = np.arcsin(np.clip(y3d, -1.0, 1.0))
                
                # Map to pixel coordinates
                u = (longitude / (2 * np.pi) + 0.5) * self.input_width
                v = (0.5 - latitude / np.pi) * self.input_height
                
                mapx[y, x] = u
                mapy[y, x] = v
        
        return mapx, mapy
    
    def convert(self, equirect_img):
        """
        Convert equirectangular image to fisheye projection.
        
        Args:
            equirect_img: Input equirectangular image (numpy array)
            
        Returns:
            Fisheye projected image
        """
        # Apply the mapping using remap
        fisheye_img = cv2.remap(equirect_img, self.mapx, self.mapy, 
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Optional: Create circular mask for cleaner output
        mask = np.zeros((self.output_size, self.output_size), dtype=np.uint8)
        center = (self.output_size // 2, self.output_size // 2)
        cv2.circle(mask, center, self.output_size // 2, 255, -1)
        fisheye_img = cv2.bitwise_and(fisheye_img, fisheye_img, mask=mask)
        
        return fisheye_img


# --- 1. Configuration ---
host_ip = "10.42.0.244"
host_port = 41451
# AirSim camera details
CAMERA_NAME = "cam0"
IMAGE_TYPE = 12  # 12 = Equirectangular panorama

# Fisheye parameters
OUTPUT_SIZE = 800  # Output fisheye image size (square)
H_FOV = 200  # Horizontal field of view
V_FOV = 200  # Vertical field of view
YAW = 0
PITCH = 0

# Desired framerate for the output stream
OUTPUT_FPS = 30

# --- 2. Connect to AirSim ---
try:
    client = airsim.VehicleClient(ip=host_ip, port=host_port)
    client.confirmConnection()
    print("Connected to AirSim.")
except Exception as e:
    print(f"Error: Could not connect to AirSim.")
    print("Please make sure AirSim (and your UE5/Unity project) is running.")
    print(f"Details: {e}")
    sys.exit(1)


# --- 3. Get first image to find dimensions ---
print("Fetching initial image from AirSim to get dimensions...")
try:
    # Request a PNG compressed image
    image_request = airsim.ImageRequest(CAMERA_NAME, IMAGE_TYPE, False, True) 
    image_response = client.simGetImages([image_request])[0]
    
    if not image_response.image_data_uint8:
        print("Error: No image data received from AirSim.")
        print(f"Camera: '{CAMERA_NAME}', Type: {IMAGE_TYPE}. Is this camera and type active?")
        print(f"Message from AirSim: {image_response.message}")
        sys.exit(1)

    # Decode the PNG image
    img_1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    img_360 = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)
    
    if img_360 is None:
        print("Error: Could not decode image from AirSim.")
        print("The data received may not be a valid PNG.")
        sys.exit(1)
        
    INPUT_HEIGHT, INPUT_WIDTH, _ = img_360.shape
    print(f"Success. Input dimensions detected: {INPUT_WIDTH}x{INPUT_HEIGHT}")

except Exception as e:
    print(f"Error while getting initial image: {e}")
    sys.exit(1)

# --- 4. Initialize Converter ---
print(f"Initializing fisheye converter (FOV: {H_FOV}°x{V_FOV}°)...")
converter = EquirectToFisheye(
    input_width=INPUT_WIDTH,
    input_height=INPUT_HEIGHT,
    output_size=OUTPUT_SIZE,
    h_fov=H_FOV,
    v_fov=V_FOV,
    yaw=YAW,
    pitch=PITCH
)
print("Converter initialized successfully!")

# --- 5. Start Display Windows ---
print("\n--- Starting OpenCV Display (Fisheye 200° FOV) ---")
print("Press 'q' to quit the stream")
print("-------------------------\n")

# Initialize display windows
cv2.namedWindow('Input Panorama (1600x800)', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Input Panorama (1600x800)', 1280, 640)

cv2.namedWindow('Fisheye Camera (200° FOV)', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Fisheye Camera (200° FOV)', 800, 800)

# --- 6. Main Streaming Loop ---
try:
    # FPS tracking
    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0
    
    while True:
        start_time = time.time()

        # Request the PNG compressed image
        image_response = client.simGetImages([image_request])[0]
        
        if image_response.image_data_uint8:
            # Decode PNG/JPG to raw BGR
            img_1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
            img_360 = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)
            
            if img_360 is not None:
                # Make a copy for display with FPS
                img_360_display = img_360.copy()
                
                # Update FPS counter
                frame_count += 1
                elapsed_fps_time = time.time() - fps_start_time
                if elapsed_fps_time >= 1.0:
                    current_fps = frame_count / elapsed_fps_time
                    frame_count = 0
                    fps_start_time = time.time()
                
                # Add FPS text to the input frame
                cv2.putText(img_360_display, f'Input FPS: {current_fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the input panorama
                cv2.imshow('Input Panorama (1600x800)', img_360_display)
                
                # Convert to fisheye using OpenCV
                img_fisheye = converter.convert(img_360)
                
                # Add FPS text to the filtered frame
                cv2.putText(img_fisheye, f'Output FPS: {current_fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the fisheye frame
                cv2.imshow('Fisheye Camera (200° FOV)', img_fisheye)
                
                # Check for 'q' key press to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Stream stopped by user (pressed 'q').")
                    break
        
        # Calculate time to sleep to maintain the desired FPS
        elapsed = time.time() - start_time
        sleep_time = (1.0 / OUTPUT_FPS) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nStream stopped by user (Ctrl+C).")
except Exception as e:
    print(f"An error occurred during the streaming loop: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Clean up
    print("Cleaning up and closing display...")
    cv2.destroyAllWindows()
    print("Display closed.")