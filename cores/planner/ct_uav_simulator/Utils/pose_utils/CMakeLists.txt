cmake_minimum_required(VERSION 3.5)
project(pose_utils)

# ========================
# Build options
# ========================
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_BUILD_TYPE Release)
add_compile_options(-O3 -Wall)

# ========================
# Dependencies
# ========================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Armadillo REQUIRED)

# ========================
# Library
# ========================
add_library(pose_utils SHARED
  src/pose_utils.cpp
)

target_include_directories(pose_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(pose_utils
  ${ARMADILLO_LIBRARIES}
)

ament_target_dependencies(pose_utils
  rclcpp
)

# ========================
# Install (ROS2 CORRECT)
# ========================
install(
  TARGETS pose_utils
  EXPORT pose_utilsTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

# ========================
# Export
# ========================
ament_export_targets(pose_utilsTargets HAS_LIBRARY_TARGET)
ament_export_libraries(pose_utils)
ament_export_include_directories(include)

ament_package()
