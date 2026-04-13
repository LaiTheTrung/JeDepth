# RUN SIMULATION 
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

## Step 4:
Open another powershell and run
```powershell
wsl
ros2 launch ct_uav_omni_fast_stream wrapper_airsim.launch.py host_ip:=${PX4_SIM_HOST_ADDR} inter_publish:=false
ros2 launch ct_uav_omni_fast_stream wrapper_airsim_eight.launch.py host_ip:=${PX4_SIM_HOST_ADDR} inter_publish:=true
```
This will start the ROS2 wrapper to communicate with AirSim simulator.

## Step 5:
If you want to publish the px4 topics to ROS2-DDS, open another powershell and run
```powershell
wsl
MicroXRCEAgent udp4 -p 8888
```
