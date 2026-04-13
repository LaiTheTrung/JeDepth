cmake_minimum_required(VERSION 3.8)
project(odom_visualization)

# 使用 C++14 标准
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -Wall -g")

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
include_directories(${nav_msgs_INCLUDE_DIRS})
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Armadillo REQUIRED)
find_package(quadrotor_msgs REQUIRED)
# find_package(pose_utils REQUIRED) # Removed as per instruction

set(dependencies
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2_msgs
  tf2_ros
  quadrotor_msgs
  # pose_utils # Removed
)

# Add pose_utils as subdirectory
add_subdirectory(../pose_utils pose_utils_build)

# 包含目录
include_directories(
  include
  ${ARMADILLO_INCLUDE_DIRS}
  ../pose_utils/include
  /opt/ros/humble/include
  /opt/ros/humble/include/nav_msgs
)

# 添加可执行文件
add_executable(odom_visualization src/odom_visualization.cpp)

# 链接依赖
ament_target_dependencies(odom_visualization ${dependencies})

target_link_libraries(odom_visualization
  ${ARMADILLO_LIBRARIES}
  pose_utils
)

# 安装目标
install(TARGETS
  odom_visualization
  DESTINATION lib/${PROJECT_NAME}
)

# 安装其他文件
install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
