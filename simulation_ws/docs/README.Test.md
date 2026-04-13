# SETUP SIMULATION 
This document provides instructions on how to run simulations using the provided codebase.
Note: this tutorial support Windows OS only.
## Step 1: 
first check the WSL_IP address by running the following command in your Powershell CMD:
```bash
ipconfig 
#Check the WSL IP address which is (ROSBridge)
```
## Step 2:
Move to the Simulation folder, edit the settings.json file by update the Localhostip with the WSL_IP address you got from step 1.

## Step 3:
Open your powershell and run
```powershell
wsl 
nano ~/.bashrc
# change the PX4_SIM_HOST_ADDR to your WSL_IP address  
# Ctrl + S and Ctrl + X to save and exit
px4s 
```
This will start PX4 simulation automatically.

# TEST SIMULATION ALGORITHMS
Open another powershell and run
```powershell
wsl
ros2 launch ct_uav_omni_fast_stream wrapper_airsim.launch.py host_ip:=${PX4_SIM_HOST_ADDR} inter_publish:=false
ros2 launch ct_uav_omni_fast_stream wrapper_airsim_eight.launch.py host_ip:=${PX4_SIM_HOST_ADDR} inter_publish:=true
ros2 launch ct_uav_omni_fast_stream wrapper_airsim_stereo.launch.py host_ip:=${PX4_SIM_HOST_ADDR} inter_publish:=false
ros2 run ct_uav_omni_fast_stream test_2cam_depth_collision_preventation.py \
  --ros-args \
  -p host_ip:=${PX4_SIM_HOST_ADDR} \
  -p vehicle_name:='survey' \
  -p fov_deg:=85.0 \
  -p max_depth_m:=50.0 \
  -p publish_pc:=true
  
  ```
This will start the ROS2 wrapper to communicate with AirSim simulator.

## Step 5:
If you want to publish the px4 topics to ROS2-DDS, open another powershell and run
```powershell
wsl
MicroXRCEAgent udp4 -p 8888
```

## Step 6: 
Run rviz for visulization point clound
```powershell
ros2 run rviz2 rviz2 -d /mnt/d/trung_Nav_team/Cosys-AirSim/ros2/src/ct_uav_omni_fast_stream/rviz/uav_collision_prevention.rviz
```