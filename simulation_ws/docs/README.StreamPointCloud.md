# UAV Point Cloud Stream & Collision Prevention

ROS2 node for streaming depth camera point clouds from AirSim and publishing PX4 obstacle distance messages.

## Quick Start

### 1. Launch AirSim Simulation
```powershell
cd D:\trung_Nav_team\Simulation_31_10_CityPark\CityPark\Windows
.\CityPark.exe
```

### 2. Start PX4 SITL
```powershell
px4s
```

### 3. Run Collision Prevention Node
```bash
wsl
ros2 run ct_uav_omni_fast_stream test_2cam_depth_collision_preventation.py \
  --ros-args \
  -p host_ip:=${PX4_SIM_HOST_ADDR} \
  -p vehicle_name:='survey' \
  -p fov_deg:=85.0 \
  -p max_depth_m:=60.0 \
  -p publish_pc:=true
```

### 4. (Optional) Enable PX4-ROS2 Bridge
```powershell
wsl 
MicroXRCEAgent udp4 -p 8888
```

### 5. Visualize in RViz2
```bash
uav_rviz
```

## Features

- **Dual depth camera fusion** → 72-sector obstacle distance (5° resolution)
- **TF tree**: `world → base_link → camera_optical_frame`
- **Point clouds** published in world frame with automatic transform
- **Latency-compensated timestamps** from AirSim
- **PX4 attitude/position integration** (NED → ENU conversion)

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/fmu/in/obstacle_distance` | `ObstacleDistance` | 72-sector collision prevention |
| `/cam{N}/depth/points` | `PointCloud2` | Depth point cloud (world frame) |
| `/cam{N}/depth_cropped` | `Image` | Cropped depth visualization |
| `/airsim/orientation` | `QuaternionStamped` | UAV orientation from AirSim |
| `/tf` | `TFMessage` | Dynamic `world→base_link` transform |