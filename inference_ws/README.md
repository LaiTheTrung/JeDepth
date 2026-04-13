This code only test on jetpack 6.2

# For efficient building and inference

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks --fan
```

# Build docker image

```bash
make up
```

# Run Application

Open a termnial

```bash
make exec # Auto enable GUI if ssh
# cd /usr/src/jetson_multimedia_api/argus/cmake
# sudo cmake ..
# sudo make install
source /workspace/ros2_third_party/install/setup.bash
cd /workspace/ros_packages
source install/setup.bash
export ROS_DOMAIN_ID=96
ros2 launch ct_uav_stereo_cpp stereo_obstacle_avoidance.launch.py
colcon build # if ls do not see build install log
#else: source/install/setup.bash

```

# Down the container

```bash
make down
```

# To install argus for stereo cpp

```bash
sudo apt-get install -y nvidia-l4t-jetson-multimedia-api cmake build-essential cuda pkg-config libgtk-3-dev
cd /usr/src/jetson_multimedia_api/argus/cmake
sudo cmake ..
sudo make
sudo make install
```

# record data

```bash
source /workspace/ros2_third_party/install/setup.bash
cd /workspace/ros_packages
source install/setup.bash
export ROS_DOMAIN_ID=96
ros2 launch ct_uav_stereo_cpp stereo_camera_with_record.launch.py

#open other termnial
make exec
source /workspace/ros2_third_party/install/setup.bash
cd /workspace/ros_packages
source install/setup.bash
export ROS_DOMAIN_ID=96
ros2 service call /start_recording std_srvs/srv/Trigger

```
