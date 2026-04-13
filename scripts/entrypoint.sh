#!/bin/bash
set -e

# One-time setup: Build and install Jetson Multimedia API (Argus)
FLAG_FILE="/workspace/.container_flags/argus_installed"
if [ ! -f "$FLAG_FILE" ]; then
    echo "First-time setup: Building Jetson Multimedia API (Argus)..."
    mkdir -p /workspace/.container_flags
    cd /usr/src/jetson_multimedia_api/argus/cmake
    cmake ..
    make install
    
    # Create flag file to indicate installation is complete
    touch "$FLAG_FILE"
    echo "Argus installation complete."
else
    echo "Argus already installed, skipping build."
fi
source /opt/ros/humble/setup.bash
source /workspace/ros2_third_party/install/setup.bash
exec "$@"