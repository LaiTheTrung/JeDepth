cmake_minimum_required(VERSION 3.12.0)
project(ct_uav_omni_fast_stream)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geographic_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(tf2 REQUIRED)
find_package(image_transport REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)
find_package(rosidl_default_runtime REQUIRED)
find_package(airsim_interfaces REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rcutils REQUIRED)

set(AIRSIM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../)

add_subdirectory("${AIRSIM_ROOT}/cmake/rpclib_wrapper" rpclib_wrapper)
add_subdirectory("${AIRSIM_ROOT}/cmake/AirLib" AirLib)
add_subdirectory("${AIRSIM_ROOT}/cmake/MavLinkCom" MavLinkCom)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Wnoexcept -Wstrict-null-sentinel")
set(CXX_EXP_LIB "-nostdinc++ -I/usr/include/c++/8 -I/usr/include/x86_64-linux-gnu/c++/8 -nodefaultlibs 
-l/usr/lib/x86_64-linux-gnu/libc++.so -l/usr/lib/x86_64-linux-gnu/libc++abi.so
-lm -lc -lgcc_s -lgcc
-lstdc++fs -fmax-errors=10")
set(RPC_LIB_INCLUDES " ${AIRSIM_ROOT}/external/rpclib/rpclib-2.3.1/include")
set(RPC_LIB rpc)
message(STATUS "found RPC_LIB_INCLUDES=${RPC_LIB_INCLUDES}")

set(INCLUDE_DIRS include 
  ${AIRSIM_ROOT}/AirLib/deps/eigen3
  ${AIRSIM_ROOT}/AirLib/include
  ${RPC_LIB_INCLUDES}
  ${AIRSIM_ROOT}/MavLinkCom/include
  ${AIRSIM_ROOT}/MavLinkCom/common_utils
  ${OpenCV_INCLUDE_DIRS}
  )
include_directories(${INCLUDE_DIRS})
#==========instal python package =========
ament_python_install_package(${PROJECT_NAME})

# ========== Build Fast Stream Node Executable ==========
add_executable(fast_stream_node src/fast_stream_node.cpp)
add_executable(fast_stream_node_eight src/fast_stream_node_cassini.cpp)
add_executable(fast_stream_node_stereo src/fast_stream_stereo.cpp)
add_executable(data_record_node src/record_data.cpp)

ament_target_dependencies(fast_stream_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  rclcpp_components
  OpenCV
)

ament_target_dependencies(fast_stream_node_eight
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  rclcpp_components
  OpenCV
)

ament_target_dependencies(fast_stream_node_stereo
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  rclcpp_components
  OpenCV
)

ament_target_dependencies(data_record_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  rclcpp_components
  OpenCV
)

target_link_libraries(fast_stream_node
  AirLib
  MavLinkCom
  rpc
  ${OpenCV_LIBS}
  pthread
)


target_link_libraries(fast_stream_node_eight
  AirLib
  MavLinkCom
  rpc
  ${OpenCV_LIBS}
  pthread
)

target_link_libraries(fast_stream_node_stereo
  AirLib
  MavLinkCom
  rpc
  ${OpenCV_LIBS}
  pthread
)

target_link_libraries(data_record_node
  AirLib
  MavLinkCom
  rpc
  ${OpenCV_LIBS}
  pthread
)

# ========== Build Composable Component Library ==========
add_library(ct_uav_fast_gray_stream_component SHARED src/fast_stream_node.cpp)
add_library(ct_uav_record_component SHARED src/record_data.cpp)

ament_target_dependencies(ct_uav_fast_gray_stream_component
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  cv_bridge
  OpenCV
)

ament_target_dependencies(ct_uav_record_component
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  cv_bridge
  OpenCV
)

target_link_libraries(ct_uav_fast_gray_stream_component
  AirLib
  MavLinkCom
  rpc
  ${OpenCV_LIBS}
  pthread
)

target_link_libraries(ct_uav_record_component
  AirLib
  MavLinkCom
  rpc
  ${OpenCV_LIBS}
  pthread
)

# Register component and create executable
rclcpp_components_register_node(
  ct_uav_fast_gray_stream_component
  PLUGIN "ct_uav_omni::FastStreamNode"
  EXECUTABLE ct_uav_fast_gray_stream
)

rclcpp_components_register_node(
  ct_uav_record_component
  PLUGIN "ct_uav_omni::DataRecorderNode"
  EXECUTABLE ct_uav_record
)


# ========== Install ==========
install(TARGETS
  fast_stream_node
  fast_stream_node_eight
  fast_stream_node_stereo
  data_record_node
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  src_py/test_2cam_depth_collision_preventation.py
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  ct_uav_fast_gray_stream_component
  ct_uav_record_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES README.md DESTINATION share/${PROJECT_NAME})

install(DIRECTORY launch rviz DESTINATION share/${PROJECT_NAME})



ament_export_dependencies(rclcpp)
ament_export_dependencies(nav_msgs)
ament_export_dependencies(geographic_msgs)
ament_export_dependencies(std_srvs)
ament_export_dependencies(tf2_ros)
ament_export_dependencies(tf2_sensor_msgs)
ament_export_dependencies(rclpy)
ament_export_dependencies(tf2)
ament_export_dependencies(image_transport)
ament_export_dependencies(geometry_msgs)
ament_export_dependencies(cv_bridge)
ament_export_dependencies(tf2_geometry_msgs)
ament_export_dependencies(sensor_msgs)
ament_export_dependencies(mavros_msgs)
ament_export_dependencies(rosidl_default_generators)
ament_export_dependencies(ament_cmake)
ament_export_dependencies(std_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_export_dependencies(airsim_interfaces)
ament_export_dependencies(pcl_conversions)
ament_export_dependencies(PCL)


ament_export_include_directories(${INCLUDE_DIRS})
ament_export_libraries(airsim_settings_parser pd_position_controller_simple
  airsim_ros)

ament_package()
