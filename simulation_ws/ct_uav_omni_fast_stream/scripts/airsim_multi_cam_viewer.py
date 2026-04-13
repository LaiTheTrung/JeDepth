"""
Multi-camera AirSim viewer with UDP streaming

- Spawns one thread per camera index (0..3)
- Each thread fetches im            # Get latest frame for this camera
            with frame_locks[camera_index]:
                frame_info = latest_frames[camera_index]
            
            if frame_info is None:
                time.sleep(0.01)
                continue
            
            img, inst, avg, timestamp_ns = frame_info
            
            # Encode to JPEG (no resizing, keep original resolution)
            ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                time.sleep(0.005)
                continue
            
            # Base64 encode JPEG
            jpeg_b64 = base64.b64encode(buffer)
            
            # Prepend timestamp (as string) followed by a separator and the JPEG data
            timestamp_str = str(timestamp_ns).encode('utf-8')
            frame_data = timestamp_str + b'|' + jpeg_b64ith compress=False (raw BGR bytes)
- Per-camera FPS counters (instant + moving average)
- Main thread displays each camera window with FPS overlay
- Designed to avoid delays between cameras (no artificial sleeps)

UDP Streaming:
- Each camera streams to its own UDP port (5001-5004 for cameras 0-3)
- Combined view streams to port 5005
- Clients send 'Hello' to register and receive base64-encoded JPEG frames

Usage:
    python airsim_multi_cam_viewer.py
    # To enable UDP streaming, call start_udp_streaming() in main()

Requires: airsim, opencv-python, numpy
"""

import airsim
import cv2
import numpy as np
import threading
import time
import socket
import base64
from collections import deque

CAMERA_INDICES = [0, 1, 2, 3]
IMAGE_TYPE = airsim.ImageType.Scene

# Perfect sync configuration
PERFECT_SYNC = True  # Set to True to enable perfect sync with sim pause/unpause

# UDP streaming configuration
UDP_BASE_PORT = 5001  # Cameras 0-3 use ports 5001-5004
UDP_COMBINED_PORT = 5005  # Combined view uses port 5005
UDP_IP = "0.0.0.0"
BUFF_SIZE = 262144

# Shared buffers
latest_frames = {i: None for i in CAMERA_INDICES}
frame_locks = {i: threading.Lock() for i in CAMERA_INDICES}
latest_composite = None
composite_lock = threading.Lock()
stop_event = threading.Event()

# Synchronization state - using Condition variables for instant signaling
capture_lock = threading.Lock()
capture_cv = threading.Condition(capture_lock)
capture_counter = 0  # Increments each cycle to signal new capture
cameras_ready_counter = 0  # Tracks how many cameras are ready to capture
cameras_done_counter = 0  # Tracks how many cameras have completed capture

# Coordinator FPS tracking
coordinator_fps_counter = None  # Will be initialized in main()

# Display FPS tracking (measures how often display shows NEW frames)
display_fps_counters = {i: None for i in CAMERA_INDICES}  # One per camera

# UDP streaming state
udp_sockets = {}
connected_clients = {i: set() for i in CAMERA_INDICES}
connected_clients['combined'] = set()
client_locks = {i: threading.Lock() for i in CAMERA_INDICES}
client_locks['combined'] = threading.Lock()

class FPSCounter:
    def __init__(self, avg_size=30):
        self._last = None
        self._deltas = deque(maxlen=avg_size)
        self._lock = threading.Lock()

    def tick(self):
        now = time.time()
        with self._lock:
            if self._last is None:
                self._last = now
                return 0.0, 0.0
            dt = now - self._last
            self._last = now
            if dt <= 0:
                inst = 0.0
            else:
                inst = 1.0 / dt
            self._deltas.append(inst)
            avg = sum(self._deltas) / len(self._deltas)
            return inst, avg

    def reset(self):
        with self._lock:
            self._last = None
            self._deltas.clear()


def udp_listener_thread(camera_index, port):
    """Listen for client registration messages ('Hello') for a specific camera stream."""
    global connected_clients, client_locks, stop_event
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
    sock.bind((UDP_IP, port))
    sock.settimeout(1.0)
    
    cam_key = camera_index if camera_index != 'combined' else 'combined'
    print(f"[UDP Listener {cam_key}] Listening on port {port}")
    
    while not stop_event.is_set():
        try:
            data, client_addr = sock.recvfrom(1024)
            if data == b'Hello':
                with client_locks[cam_key]:
                    if client_addr not in connected_clients[cam_key]:
                        connected_clients[cam_key].add(client_addr)
                        print(f"[UDP Listener {cam_key}] Client connected: {client_addr}")
                sock.sendto(b'ACK', client_addr)
        except socket.timeout:
            continue
        except Exception as e:
            if not stop_event.is_set():
                print(f"[UDP Listener {cam_key}] Error: {e}")
    
    sock.close()
    print(f"[UDP Listener {cam_key}] Stopped")


def udp_sender_thread(camera_index, port):
    """Send frames for a specific camera to connected clients."""
    global latest_frames, frame_locks, connected_clients, client_locks, stop_event
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFF_SIZE * 8)
    
    print(f"[UDP Sender {camera_index}] Starting on port {port}")
    
    while not stop_event.is_set():
        try:
            # Get latest frame for this camera
            with frame_locks[camera_index]:
                frame_info = latest_frames[camera_index]
            
            if frame_info is None:
                time.sleep(0.01)
                continue
            
            img, inst, avg, timestamp_ns = frame_info
            
            # Encode to JPEG (no resizing, keep original resolution)
            ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                time.sleep(0.005)
                continue
            
            # Base64 encode JPEG
            jpeg_b64 = base64.b64encode(buffer)
            
            # Prepend timestamp (as string) followed by a separator and the JPEG data
            timestamp_str = str(timestamp_ns).encode('utf-8')
            frame_data = timestamp_str + b'|' + jpeg_b64
            
            # Send to all connected clients
            with client_locks[camera_index]:
                disconnected = set()
                for client_addr in connected_clients[camera_index]:
                    try:
                        sock.sendto(frame_data, client_addr)
                    except Exception as e:
                        disconnected.add(client_addr)
                
                for client_addr in disconnected:
                    connected_clients[camera_index].remove(client_addr)
                    print(f"[UDP Sender {camera_index}] Client disconnected: {client_addr}")
            
            time.sleep(0.01)  # ~30 FPS
            
        except Exception as e:
            if not stop_event.is_set():
                print(f"[UDP Sender {camera_index}] Error: {e}")
            time.sleep(0.01)
    
    sock.close()
    print(f"[UDP Sender {camera_index}] Stopped")


def udp_combined_sender_thread(port):
    """Send combined composite view to connected clients."""
    global latest_composite, composite_lock, connected_clients, client_locks, stop_event
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFF_SIZE * 8)
    
    print(f"[UDP Combined Sender] Starting on port {port}")
    
    while not stop_event.is_set():
        try:
            # Get latest composite frame
            with composite_lock:
                composite = latest_composite
            
            if composite is None:
                time.sleep(0.01)
                continue
            
            # Get current timestamp
            timestamp_ns = time.time_ns()
            
            # Encode to JPEG (no resizing)
            ret, buffer = cv2.imencode('.jpg', composite, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                time.sleep(0.005)
                continue
            
            # Base64 encode JPEG
            jpeg_b64 = base64.b64encode(buffer)
            
            # Prepend timestamp (as string) followed by a separator and the JPEG data
            timestamp_str = str(timestamp_ns).encode('utf-8')
            frame_data = timestamp_str + b'|' + jpeg_b64
            
            # Send to all connected clients
            with client_locks['combined']:
                disconnected = set()
                for client_addr in connected_clients['combined']:
                    try:
                        sock.sendto(frame_data, client_addr)
                    except Exception as e:
                        disconnected.add(client_addr)
                
                for client_addr in disconnected:
                    connected_clients['combined'].remove(client_addr)
                    print(f"[UDP Combined Sender] Client disconnected: {client_addr}")
            
            time.sleep(0.01)  # ~30 FPS
            
        except Exception as e:
            if not stop_event.is_set():
                print(f"[UDP Combined Sender] Error: {e}")
            time.sleep(0.01)
    
    sock.close()
    print(f"[UDP Combined Sender] Stopped")


def start_udp_streaming():
    """
    Start UDP streaming threads for all cameras and combined view.
    
    Ports:
    - Camera 0: 5001
    - Camera 1: 5002
    - Camera 2: 5003
    - Camera 3: 5004
    - Combined: 5005
    """
    threads = []
    
    # Start listener and sender for each camera
    for i in CAMERA_INDICES:
        port = UDP_BASE_PORT + i
        
        listener = threading.Thread(target=udp_listener_thread, args=(i, port), daemon=True)
        listener.start()
        threads.append(listener)
        
        sender = threading.Thread(target=udp_sender_thread, args=(i, port), daemon=True)
        sender.start()
        threads.append(sender)
    
    # Start combined stream
    combined_listener = threading.Thread(target=udp_listener_thread, args=('combined', UDP_COMBINED_PORT), daemon=True)
    combined_listener.start()
    threads.append(combined_listener)
    
    combined_sender = threading.Thread(target=udp_combined_sender_thread, args=(UDP_COMBINED_PORT,), daemon=True)
    combined_sender.start()
    threads.append(combined_sender)
    
    print(f"[UDP Streaming] Started {len(threads)} streaming threads")
    return threads


def synchronized_capture_coordinator():
    """
    Coordinator thread - CORRECTED WORKFLOW:
    
    Loop: PAUSE → CAPTURE ALL 4 CAMERAS → UNPAUSE → stream to display
    
    1. Pause simulation
    2. Broadcast capture signal to all 4 camera threads
    3. Wait for ALL 4 cameras to COMPLETE their capture (not just start)
    4. Unpause simulation
    5. Camera threads update display buffers
    6. Loop repeats immediately
    
    Uses Condition variables for instant synchronization with no delays.
    """
    global capture_lock, capture_cv, capture_counter, cameras_done_counter, stop_event, coordinator_fps_counter
    
    # Create dedicated client for pause/unpause control
    sim_client = None
    if PERFECT_SYNC:
        try:
            sim_client = airsim.MultirotorClient()
            sim_client.confirmConnection()
            print("[Coordinator] Perfect sync mode ENABLED - using sim.pause/unpause")
            print("[Coordinator] Workflow: PAUSE → GET 4 CAMERAS → UNPAUSE → stream to display")
        except Exception as e:
            print(f"[Coordinator] Failed to create sim control client: {e}")
            print("[Coordinator] Falling back to non-perfect-sync mode")
    else:
        print("[Coordinator] Perfect sync mode DISABLED - capturing without pause")
    
    print("[Coordinator] Starting synchronized capture (ZERO DELAY mode)")
    
    while not stop_event.is_set():
        # === START LOOP ===
        
        # Track coordinator loop FPS (complete cycle time)
        if coordinator_fps_counter is not None:
            coordinator_fps_counter.tick()
        
        # STEP 1: PAUSE simulation BEFORE capture
        if PERFECT_SYNC and sim_client is not None:
            try:
                sim_client.simPause(True)
            except Exception as e:
                print(f"[Coordinator] Error pausing simulation: {e}")
        
        # STEP 2: SEND CAPTURE MESSAGE - Instant broadcast to ALL 4 cameras
        with capture_cv:
            cameras_done_counter = 0  # Reset done counter
            capture_counter += 1  # Increment to signal new capture cycle
            capture_cv.notify_all()  # Wake up ALL camera threads instantly
        
        # STEP 3: WAIT for ALL 4 CAMERAS to COMPLETE capture (while sim is paused)
        with capture_cv:
            timeout_time = time.time() + 2.0  # 2 second timeout
            while cameras_done_counter < len(CAMERA_INDICES):
                remaining = timeout_time - time.time()
                if remaining <= 0:
                    print(f"[Coordinator] Timeout: Only {cameras_done_counter}/{len(CAMERA_INDICES)} cameras completed")
                    break
                capture_cv.wait(timeout=remaining)
        
        # STEP 4: UNPAUSE simulation AFTER all captures are done
        if PERFECT_SYNC and sim_client is not None:
            try:
                sim_client.simPause(False)
            except Exception as e:
                print(f"[Coordinator] Error unpausing simulation: {e}")
            
            # Give simulation a brief moment to run and advance to new state
            # Without this, we pause again immediately and get the same frame
            # This tiny delay allows the simulation to progress between captures
            time.sleep(0.001)  # 1ms - just enough for simulation to advance
        
        # === END LOOP === 
        # Loop repeats: unpause → brief run → pause → capture
        # This ensures each capture gets a NEW simulation state


def camera_thread_fn(camera_index, _client_placeholder, fps_counter: FPSCounter):
    """
    Camera capture thread - CORRECTED WORKFLOW:
    
    1. Wait for capture signal from coordinator
    2. Capture image from AirSim (sim is PAUSED at this point)
    3. Signal coordinator that capture is DONE
    4. Update display buffer (after coordinator unpauses)
    
    This ensures coordinator doesn't unpause until ALL captures are complete.
    """
    global latest_frames, frame_locks, stop_event
    global capture_lock, capture_cv, capture_counter, cameras_done_counter
    
    # Create a dedicated client per thread to avoid IOLoop conflicts
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
    except Exception as e:
        print(f"[Camera {camera_index}] Failed to create local AirSim client: {e}")
        client = None

    req = airsim.ImageRequest(camera_name=str(camera_index), image_type=IMAGE_TYPE, pixels_as_float=False, compress=False)
    
    print(f"[Camera {camera_index}] Ready for synchronized capture")
    
    last_capture = 0  # Track last capture counter we processed
    
    while not stop_event.is_set():
        try:
            # STEP 1: WAIT for capture signal from coordinator
            with capture_cv:
                while capture_counter == last_capture and not stop_event.is_set():
                    capture_cv.wait(timeout=1.0)
                
                if stop_event.is_set():
                    break
                    
                last_capture = capture_counter  # Got new capture signal
            
            # STEP 2: CAPTURE image from AirSim (sim is PAUSED now)
            
            if client is None:
                # try to (re)create client
                try:
                    client = airsim.MultirotorClient()
                    client.confirmConnection()
                except Exception as e:
                    print(f"[Camera {camera_index}] Reconnect client error: {e}")
                    # Signal completion even on error
                    with capture_cv:
                        cameras_done_counter += 1
                        capture_cv.notify_all()
                    time.sleep(0.01)
                    continue
            
            # Capture frame from AirSim (timestamp captured at capture time)
            timestamp_ns = time.time_ns()
            res = client.simGetImages([req])
            
            if not res:
                with capture_cv:
                    cameras_done_counter += 1
                    capture_cv.notify_all()
                continue
            r = res[0]
            if not r.image_data_uint8:
                with capture_cv:
                    cameras_done_counter += 1
                    capture_cv.notify_all()
                continue

            # Safely create a contiguous, writable copy before reshape
            data = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
            expected = int(r.height) * int(r.width) * 3
            img = None
            try:
                if data.size == expected:
                    # make a copy to avoid "existing exports" or readonly buffer issues
                    data_copy = np.copy(data)
                    img = data_copy.reshape(r.height, r.width, 3)
                else:
                    # fallback: decode compressed image
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception as e:
                # decoding/reshape error — fallback to decode from bytes
                try:
                    img = cv2.imdecode(np.frombuffer(r.image_data_uint8, dtype=np.uint8), cv2.IMREAD_COLOR)
                except Exception as e2:
                    print(f"[Camera {camera_index}] Decode error fallback: {e2}")
                    img = None

            if img is None:
                with capture_cv:
                    cameras_done_counter += 1
                    capture_cv.notify_all()
                continue

            # STEP 3: Signal coordinator that THIS camera's capture is COMPLETE
            # This must happen BEFORE updating display buffer so coordinator
            # knows all captures are done before unpausing
            with capture_cv:
                cameras_done_counter += 1
                capture_cv.notify_all()  # Wake up coordinator instantly
            
            # STEP 4: Update display buffer (after signaling completion)
            # This happens after coordinator unpauses, but that's OK
            inst, avg = fps_counter.tick()
            
            with frame_locks[camera_index]:
                latest_frames[camera_index] = (img.copy(), inst, avg, timestamp_ns)
            
        except Exception as e:
            print(f"[Camera {camera_index}] Error: {e}")
            # Signal completion even on error
            with capture_cv:
                cameras_done_counter += 1
                capture_cv.notify_all()
            time.sleep(0.01)


def main():
    global coordinator_fps_counter, display_fps_counters
    
    client = airsim.MultirotorClient()
    client.confirmConnection()

    fps_counters = {i: FPSCounter() for i in CAMERA_INDICES}
    coordinator_fps_counter = FPSCounter()  # Track coordinator loop FPS
    
    # Track display FPS per camera (measures when NEW frames are shown)
    for i in CAMERA_INDICES:
        display_fps_counters[i] = FPSCounter()
    
    threads = []

    # Start coordinator thread for synchronized capture
    coordinator = threading.Thread(target=synchronized_capture_coordinator, daemon=True)
    coordinator.start()
    threads.append(coordinator)
    print("[Main] Started synchronized capture coordinator")
    if PERFECT_SYNC:
        print("[Main] PERFECT SYNC MODE: Simulation will pause/unpause for each capture cycle")

    # start camera threads
    for i in CAMERA_INDICES:
        t = threading.Thread(target=camera_thread_fn, args=(i, client, fps_counters[i]), daemon=True)
        t.start()
        threads.append(t)

    # Start UDP streaming
    udp_threads = start_udp_streaming()
    threads.extend(udp_threads)

    window_names = {i: f"Camera {i}" for i in CAMERA_INDICES}

    try:
        # Composite window dimensions: each camera native width × height (user said 1280x800 each)
        # Final window: 4 * width x height -> 5120 x 800
        # We'll detect the first valid frame to get dimensions; otherwise assume 1280x800
        first_width = None
        first_height = None
        
        # Track last seen timestamp per camera to detect NEW frames
        last_timestamps = {i: None for i in CAMERA_INDICES}

        while True:
            # Get coordinator FPS for display
            coord_inst_fps = 0.0
            coord_avg_fps = 0.0
            if coordinator_fps_counter is not None:
                with coordinator_fps_counter._lock:
                    if coordinator_fps_counter._deltas:
                        coord_inst_fps = coordinator_fps_counter._deltas[-1] if coordinator_fps_counter._deltas else 0.0
                        coord_avg_fps = sum(coordinator_fps_counter._deltas) / len(coordinator_fps_counter._deltas)
            
            # gather images (keep order 0..3)
            imgs = []
            for i in CAMERA_INDICES:
                with frame_locks[i]:
                    frame_info = latest_frames[i]
                if frame_info is not None:
                    img, inst, avg, timestamp_ns = frame_info
                    
                    # Track display FPS - only tick when we get a NEW frame
                    if last_timestamps[i] != timestamp_ns:
                        last_timestamps[i] = timestamp_ns
                        display_fps_counters[i].tick()
                    
                    # draw overlay on a local copy so we don't mutate producer buffer
                    try:
                        img_disp = img.copy()
                    except Exception:
                        img_disp = np.copy(img)

                    # overlay FPS info
                    try:
                        cv2.putText(img_disp, f"CAM {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                        if PERFECT_SYNC:
                            # Use coordinator cycle FPS (measures complete pause→capture→unpause cycle)
                            cv2.putText(img_disp, f"CYCLE FPS: {coord_avg_fps:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                            cv2.putText(img_disp, f"(Pause/Unpause cycle time)", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                        else:
                            cv2.putText(img_disp, f"CAM FPS: {avg:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                        cv2.putText(img_disp, f"TIME: {timestamp_ns}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                    except Exception:
                        pass

                    imgs.append(img_disp)
                    if first_width is None:
                        first_height, first_width = img_disp.shape[:2]
                else:
                    # placeholder of expected size
                    if first_width is None:
                        # assume 1280x800 until we know better
                        ph, pw = 800, 1280
                    else:
                        ph, pw = first_height, first_width
                    blank = np.zeros((ph, pw, 3), dtype=np.uint8)
                    cv2.putText(blank, f"CAM {i} - No frame", (10, ph//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                    imgs.append(blank)

            # Ensure all images have the same height before horizontal concatenation
            # Resize if necessary (maintain width per-camera if already matching)
            heights = [img.shape[0] for img in imgs]
            target_h = max(heights)
            norm_imgs = []
            for img in imgs:
                h, w = img.shape[:2]
                if h != target_h:
                    # scale to target height, keep width proportional
                    scale = target_h / h
                    new_w = int(w * scale)
                    img = cv2.resize(img, (new_w, target_h))
                norm_imgs.append(img)

            # Concatenate horizontally
            try:
                composite = cv2.hconcat(norm_imgs)
            except Exception:
                # fallback: place images manually onto a black canvas with equal widths
                widths = [img.shape[1] for img in norm_imgs]
                total_w = sum(widths)
                composite = np.zeros((target_h, total_w, 3), dtype=np.uint8)
                x = 0
                for img in norm_imgs:
                    h, w = img.shape[:2]
                    composite[0:h, x:x+w] = img
                    x += w

            # If user requested a fixed composite size (5120x800), resize final composite to that width while keeping height
            try:
                # only resize if composite height matches expected 800; if not, keep current size
                if composite.shape[0] != 800:
                    composite = cv2.resize(composite, (composite.shape[1], 800))
                # If composite width differs from 5120, scale horizontally to 5120 while preserving height
                if composite.shape[1] != 5120:
                    composite = cv2.resize(composite, (5120, 800))
            except Exception:
                pass
            
            # Add perfect sync indicator to composite
            if PERFECT_SYNC:
                try:
                    cv2.putText(composite, "PERFECT SYNC: ON", (composite.shape[1] - 400, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    cv2.putText(composite, f"Cycle FPS: {coord_avg_fps:.1f}", (composite.shape[1] - 400, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                except Exception:
                    pass

            # Store composite for UDP streaming
            with composite_lock:
                global latest_composite
                latest_composite = composite.copy()

            cv2.imshow('AllCameras', composite)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
