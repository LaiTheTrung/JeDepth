# Camera Driver for CSI Stereo Cameras

ROS2 stereo node for CSI stereo cameras using GStreamer with CUDA acceleration on Jetson platforms.

## Features

- Dual CSI camera support (stereo configuration)
- GStreamer pipeline with CUDA acceleration (nvarguscamerasrc + nvvidconv)
- Configurable resolution, framerate, and flip method
- Synchronized image publishing to ROS2 topics
- Low latency camera capture and processing
x
## Topics Published

- `/cam0/image_raw` (sensor_msgs/Image) - Left camera raw images
- `/cam1/image_raw` (sensor_msgs/Image) - Right camera raw images


## Usage

### Run camera node directly
```bash
ros2 run ct_uav_stereo_cpp camera_driver_node --ros-args \
    -p left_sensor_id:=0 \
    -p right_sensor_id:=1 \
    -p width:=1280 \
    -p height:=720 \
    -p framerate:=60
```

### Run stereo pipeline 
```bash
ros2 launch ct_uav_stereo_cpp stereo_depth.launch.py \
    left_sensor_id:=0 \
    right_sensor_id:=1 \
    model_width:=320 \
    model_height:=256
```
## GStreamer Pipeline

The driver uses the following GStreamer pipeline for each camera:
```
nvarguscamerasrc sensor-id=<ID> ! 
video/x-raw(memory:NVMM), width=<W>, height=<H>, format=NV12, framerate=<FPS>/1 ! 
nvvidconv flip-method=<FLIP> ! 
video/x-raw, format=BGRx ! 
videoconvert ! 
video/x-raw, format=BGR ! 
appsink drop=1
```

This pipeline leverages NVIDIA hardware acceleration for optimal performance on Jetson platforms.

## Requirements

- ROS2 (tested on Humble)
- OpenCV with GStreamer support
- NVIDIA Jetson platform with CSI cameras
- GStreamer plugins: nvarguscamerasrc, nvvidconv

## Troubleshooting

If cameras fail to open:
1. Check camera connections: `ls /dev/video*`
2. Test GStreamer pipeline manually:
   ```bash
   gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! nvvidconv ! autovideosink
   ```
3. Verify correct sensor IDs for your camera setup
4. Check permissions: user must be in `video` group
