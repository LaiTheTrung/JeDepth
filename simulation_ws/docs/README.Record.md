# In order to record the data, need to run:
```bash
 ros2 launch ct_uav_omni_fast_stream composable_recorder.launch.py inter_publish:=false # for record images concatnated (as fast as fast stream node)
 ros2 launch ct_uav_omni_fast_stream composable_seg_recorder.launch.py inter_publish:=false record_mode:=0 jpeg_quality:=30 # for record 2 images/ 2 segmentation
 ros2 launch ct_uav_omni_fast_stream composable_seg_recorder.launch.py inter_publish:=false record_mode:=1 # for record 4 images
 ros2 launch ct_uav_omni_fast_stream composable_seg_recorder.launch.py inter_publish:=false record_mode:=2 # for record 4 images

# For stable recording, you should set the inter_publish to true (publish compressed topic) which help reducing the topics drop and size of the bag dataset
 # control the UAV to position you want, then start recording by
 ros2 service call /start_recording std_srvs/srv/Trigger
 # stop record when you need (do not shutdown the running code)
 ros2 service call /stop_recording std_srvs/srv/Trigger
```
The output data will be saved in the record_bags folder. 
```bash
# To replay the recorded data
ros2 bag play <data_file>.db3
```
In order to use it, you need to convert the bag data to normal format ('jpg', 'png', ...)
```bash
cd scripts
#==========Convert the concatenated images data =====================
python3 convert_ros2_db3_to_data.py  /mnt/d/trung_Nav_team/Cosys-AirSim/ros2/src/ct_uav_omni_fast_stream/recorded_bags/bag_raw2025-10-30-16-09-15/bag_raw2025-10-30-16-09-15_0.db3 -t /raw_record_node/raw_image -o video_record.mp4 -f 30 
# ============For the record mode = 0 =============
# fps video output: -f 30(FPS), time sync range between topics: -t 30(ms)
python3 create_segment_dataset.py /path/to/bag -o data_seg -f 30 -t 30
#=====================For the record mode = 2 ========================
python3 create_depth_dataset.py /path/to/bag -o data_depth -f 30 -t 30
#=====================For record 8 cameras/depth ========================
#We need to record the path first
python3 record_path.py
#Check the trajectory_logs folder created inside the scripts folder
python3 create_depth_dataset_by_airsim_api.py trajectory_logs/trajectory_sim_31_10.txt -o my_dataset --host 172.30.80.1 --port 41451 --vehicle survey --erp-height 1024 --erp-width 2048 --fov 140.0 --no-viz --no-video

```