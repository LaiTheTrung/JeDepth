cmake_minimum_required(VERSION 3.8)
project(ct_uav_stereo_cpp LANGUAGES CXX CUDA)

# Set CUDA architecture for Jetson Orin (compute capability 8.7)
set(CMAKE_CUDA_ARCHITECTURES 87)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies

# Jetson Multimedia API paths
list(APPEND CMAKE_MODULE_PATH "/usr/src/jetson_multimedia_api/argus/cmake")
list(APPEND ARGUS_INCLUDE_DIR "/usr/src/jetson_multimedia_api/argus/include")
set(JETSON_MULTIMEDIA_API_DIR /usr/src/jetson_multimedia_api)
set(ARGUS_DIR /usr/src/jetson_multimedia_api/argus)
set(ENV{ASAN_OPTIONS} "alloc_dealloc_mismatch=0")



find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV 4.10 REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rosbag2_composable_recorder REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(EGL REQUIRED)
find_package(OpenGLES REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(Argus REQUIRED)
find_package(OpenGL REQUIRED)
find_package(yaml-cpp REQUIRED)

pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-video-1.0 gstreamer-app-1.0 glib-2.0)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

# Find TensorRT
set(TENSORRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT root directory")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)


# Include directories
include_directories(
  include
  ${CUDA_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${TENSORRT_LIBRARY_INFER}
  ${TENSORRT_LIBRARY_ONNXPARSER}
  ${GSTREAMER_INCLUDE_DIRS}
  ${GTK3_INCLUDE_DIRS}
  ${X11_INCLUDE_DIRS}  # Add X11 include directories
  ${TENSORRT_INCLUDE_DIR}
  ${ARGUS_INCLUDE_DIR}
  ${ARGUS_DIR}/samples/utils
  ${JETSON_MULTIMEDIA_API_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/libs/libSGM/include
)

ament_export_include_directories("include/${PROJECT_NAME}")

# Build libSGM library with OpenCV wrapper support
set(BUILD_OPENCV_WRAPPER ON CACHE BOOL "Build OpenCV wrapper for LibSGM" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build libSGM as shared library" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(libs/libSGM)

# Build camera driver library
add_library(${PROJECT_NAME}_lib SHARED
  libs/stereo_camera_driver.cpp
  libs/stereo_calibration.cpp
  libs/depth_filter.cpp
  libs/cu_color_cvt.cu
  libs/histogram.cu
  libs/infer_fastACV.cpp
  libs/infer_fastACV_kernel.cu
  libs/infer_fastFS.cpp
  libs/infer_fastFS_kernel.cu
  libs/infer_sgm.cpp
  libs/infer_hitnet.cpp
  ${ARGUS_DIR}/samples/utils/ArgusHelpers.cpp
  ${ARGUS_DIR}/samples/utils/CUDAHelper.cpp
  ${ARGUS_DIR}/samples/utils/EGLGlobal.cpp
  ${ARGUS_DIR}/samples/utils/Observed.cpp
  ${ARGUS_DIR}/samples/utils/Thread.cpp
  ${ARGUS_DIR}/samples/utils/CommonOptions.cpp
  ${ARGUS_DIR}/samples/utils/Options.cpp
  ${ARGUS_DIR}/samples/utils/gtk/GuiElement.cpp
  ${ARGUS_DIR}/samples/utils/gtk/Window.cpp
  ${ARGUS_DIR}/samples/utils/PreviewConsumer.cpp
  ${ARGUS_DIR}/samples/utils/RectUtils.cpp
  ${ARGUS_DIR}/samples/utils/NativeBuffer.cpp
  ${ARGUS_DIR}/samples/utils/WindowBase.cpp
  ${ARGUS_DIR}/samples/utils/GLContext.cpp
)

set_target_properties(${PROJECT_NAME}_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_ARCHITECTURES native
)

ament_target_dependencies( ${PROJECT_NAME}_lib
  rclcpp
  OpenCV
  cv_bridge
  image_transport
  rosbag2_composable_recorder
  sensor_msgs
  rclcpp_components
)

target_link_libraries(${PROJECT_NAME}_lib
  ${ARGUS_LIBRARIES}
  ${GSTREAMER_LIBRARIES}
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIB}
  ${TENSORRT_LIBRARY_INFER}
  ${TENSORRT_LIBRARY_ONNXPARSER}
  cuda
  EGL
  GLESv2 #Required for Argus
  ${GTK3_LIBRARIES}
  ${X11_LIBRARIES}
  sgm
  yaml-cpp
)

# Build FastACV inference library with CUDA

add_executable(stereo_depth_node
  src/stereo_depth_node.cpp
)

add_executable(stereo_camera_node
  src/stereo_camera_node.cpp
)
# Composable node component library
add_library(stereo_camera_node_component SHARED
  src/stereo_camera_node.cpp
)
ament_target_dependencies(stereo_camera_node_component
  rclcpp
  rclcpp_components
  OpenCV
  cv_bridge
  sensor_msgs
  std_msgs
)
target_link_libraries(stereo_camera_node_component
  ${PROJECT_NAME}_lib
)
rclcpp_components_register_node(stereo_camera_node_component
  PLUGIN "CtUAVStereoCpp::StereoCameraNode"
  EXECUTABLE stereo_camera_node_composed
)
add_executable(obstacle_prevention_node
  src/obstacle_prevenation_node.cpp
)
set(SRCS_LIBSGM_COMMON 
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/libSGM/sample/sample_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/libSGM/sample/sample_common.h)

target_link_libraries(stereo_depth_node
  ${PROJECT_NAME}_lib
)
ament_target_dependencies(stereo_depth_node
  rclcpp
  OpenCV
  cv_bridge
  image_transport
  sensor_msgs
  std_msgs
)

ament_target_dependencies(stereo_camera_node
  rclcpp
  rclcpp_components
  OpenCV
  cv_bridge
  image_transport
  sensor_msgs
  std_msgs
)
target_link_libraries(stereo_camera_node
  ${PROJECT_NAME}_lib
)

target_link_libraries(obstacle_prevention_node
  ${OpenCV_LIBS}
)

ament_target_dependencies(obstacle_prevention_node
  rclcpp
  OpenCV
  cv_bridge
  sensor_msgs
  px4_msgs
)

# Install targets
install(TARGETS
  ${PROJECT_NAME}_lib
  stereo_depth_node
  stereo_camera_node
  stereo_camera_node_component
  stereo_camera_node_composed
  obstacle_prevention_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install header files
install(DIRECTORY
  include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
