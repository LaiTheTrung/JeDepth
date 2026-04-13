# Ego PX4 Bridge - Real World Deployment Guide

This package bridges the gap between **PX4 Autopilot** (running on the flight controller) and **Ego Planner** (running on the Jetson Companion Computer). It handles coordinate transformation, time synchronization, and safety checks.

## 📂 Structure

- `config/drone_params.yaml`: Configuration for drone ID, height, and **Camera Intrinsics**.
- `launch/bridge.launch.py`: Launch file to start the bridge with parameters.
- `ego_px4_bridge/bridge_real.py`: The main optimized node for real-world flying.

## 🚀 How to Run

### 1. Pre-flight Checks (CRITICAL)

Before every flight, verify the following:

- **Hardware**:

  - [ ] Propellers are tight.
  - [ ] Battery is charged.
  - [ ] RC Transmitter is ON and connected.
  - [ ] **Kill Switch** on RC is configured and tested.

- **Software**:
  - [ ] **Camera Calibration**: Ensure `fx, fy, cx, cy` in `drone_params.yaml` match your actual stereo camera calibration.
  - [ ] **PX4 Connection**: `MicroXRCEAgent` must be running and connected.
    ```bash
    # Check connection
    ros2 topic hz /fmu/out/vehicle_local_position_v1
    # Should be > 30Hz
    ```
  - [ ] **Performance Mode**:
    ```bash
    sudo nvpmodel -m 0
    sudo jetson_clocks
    ```

### 2. Launching

**Step 1: Start Camera & Inference**
Start your stereo depth node first to ensure depth data is available.

```bash
ros2 run ct_uav_stereo_cpp stereo_depth_node
```

**Step 2: Start the Bridge**

```bash
ros2 launch ego_px4_bridge bridge.launch.py
```

_The drone will NOT take off yet. It waits in `INIT` state._

**Step 3: Start Ego Planner**
Launch your Ego Planner node.

### 3. Flight Sequence

1.  **Arming**: When the bridge detects the drone is on the ground (Altitude < 0.5m) and receives valid data, it will automatically **ARM** and **TAKEOFF** to `takeoff_height` (default 1.2m).
2.  **Handover**: After 15 seconds of hovering, it switches to `MISSION` mode and listens to Ego Planner commands.
3.  **Emergency**: If Ego Planner stops sending commands for > 0.5s, the drone will **HOLD POSITION**.

## ⚠️ Safety Notes

1.  **RC Override**: Always keep your finger on the Flight Mode switch. If the drone behaves erratically, switch to **Position Mode** or **Stabilized** on the RC to regain manual control.
2.  **Lighting**: Stereo cameras fail in low light or on textureless surfaces (white walls). Fly in well-lit areas with texture.
3.  **Geofence**: The bridge has a software geofence (default 2000m). Ensure this is safe for your flying area.

## 🔧 Tuning

Edit `src/ego_px4_bridge/config/drone_params.yaml` to adjust:

- `takeoff_height`: Height to hover before starting mission.
- `camera`: Intrinsics (Must be accurate for obstacle avoidance!).
