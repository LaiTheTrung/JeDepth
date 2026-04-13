# Run simulation
```bash
cd ros_packages
colcon build --packages-select ct_uav_depth_package
source install/setup.bash
# Demo1: two stereo run on the same time in motion direction
ros2 launch ct_uav_depth_package demo_one_hitnet.launch.py update_rate_hz:=12.0 publish_depth:=true 