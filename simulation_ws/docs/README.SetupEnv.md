# 🧭 PX4 + AirSim + ROS2 + WSL Setup Guide

A clean, end‑to‑end guide for running **PX4**, **AirSim (Unreal Engine)**, and **ROS 2 Humble** on **WSL 2 (Ubuntu 22.04)** — including bridged networking for external communication.

---

## Table of Contents

* [Prerequisites](#prerequisites)
* [Install & Verify WSL 2](#install--verify-wsl-2)
* [PX4 / AirSim Setup](#px4--airsim-setup)
* [QGroundControl Setup](#qgroundcontrol-setup)
* [ROS 2 / AirSim Bridge](#ros-2--airsim-bridge)
* [External Networking (Bridged WSL2)](#external-networking-bridged-wsl2)
* [Final Verification Checklist](#final-verification-checklist)
* [References](#references)

---

## Prerequisites

* **Docker Desktop** — install and sign in.
* **Visual Studio Code** — install and sign in.

---

## Install & Verify WSL 2

Open **Windows Command Prompt** (CMD) and run:

```bash
mkdir C:\WSL\Ubuntu-22.04
wsl --install -d Ubuntu-22.04 --web-download
wsl --set-default Ubuntu-22.04
wsl --list --verbose
```

**Expected output:**

```
wsl --list --verbose
  NAME              STATE           VERSION
* Ubuntu-22.04      Running         2
  docker-desktop    Running         2
```

Open WSL and verify Ubuntu version:

```bash
cat /etc/lsb-release
```

**Output:**

```
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=22.04
DISTRIB_CODENAME=jammy
DISTRIB_DESCRIPTION="Ubuntu 22.04.5 LTS"
```

> 💡 **Optional:** Speed up WSL 2 using this guide: [https://www.ceos3c.com/linux/wsl2-performance-optimization-speed-up-your-linux-experience/](https://www.ceos3c.com/linux/wsl2-performance-optimization-speed-up-your-linux-experience/)

---

## PX4 / AirSim Setup

### Step 1 — Get PX4

Navigate to your workspace and clone PX4:

```bash
git clone https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot/Tools/setup
bash ubuntu.sh
```

### Step 2 — Configure AirSim & Windows Firewall

Edit `settings.json` in your **AirSim** folder (see official examples). Allow PX4 UDP port via **PowerShell (Admin)**:

```powershell
New-NetFirewallRule -DisplayName "Allow PX4 UDP 14580" -Direction Inbound -Protocol UDP -LocalPort 14580 -Action Allow
```

> ❌ **Not recommended:** Disable firewall entirely
>
> ```powershell
> Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False
> ```

### Step 3 — Configure PX4 environment in WSL

Open your shell config:

```bash
nano ~/.bashrc
```

Append:

```bash
export PX4_SIM_HOST_ADDR=172.21.32.1
alias px4s="cd ~/PX4-Autopilot && make px4_sitl none_iris"
source /opt/ros/humble/setup.bash
```

> 🔎 Replace `172.21.32.1` with your **actual WSL IP**, found via `ipconfig` in Windows.

### Step 4 — Build PX4

From the PX4 root (in WSL):

```bash
sudo apt update
sudo apt install -y make cmake python3-pip
python3 -m pip install kconfiglib
python3 -m pip install empy==3.3.4
bash ./Tools/setup/ubuntu.sh
make px4_sitl none_iris
```

> ⏳ First build may take several minutes.
>
> 🛠️ If build fails, ensure `empy==3.3.4` is installed.

---

## QGroundControl Setup

1. Download and install the latest **QGroundControl** for Windows: [https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl-installer.exe](https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl-installer.exe)
2. Open **Application Settings → Comm Links**.
3. Create a new link using **UDP**, **Port `14580`**.
4. Set **Target Address** to your **WSL IP** (from `ipconfig`).
5. Return to the main screen and **manually connect** (top‑left).

> 🎬 When PX4 SITL and AirSim are running, QGC should display telemetry.

---

## ROS 2 / AirSim Bridge

Instead of using AirSim’s ROS 2 plugin for control, this setup **reads images via the C++ API** and **controls via PX4**.

### Install ROS 2 (from source)

Follow the official Humble source installation guide:

* [https://docs.ros.org/en/humble/Installation/Alternatives/Ubuntu-Development-Setup.html](https://docs.ros.org/en/humble/Installation/Alternatives/Ubuntu-Development-Setup.html)

### Install Px4 ROS2 requirements 
```bash
pip install --user -U empy==3.3.4 pyros-genmsg setuptools
cd <workspace>
git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/
#Start the agent with settings for connecting to the uXRCE-DDS client running on the simulator:
MicroXRCEAgent udp4 -p 8888
```

In order to read the msg from px4 uXRCE-DDS, we need to build the ros2 message of these type of sensors.
```bash
cd <workspace>
mkdir -p ws_sensor_combined/src/
cd ws_sensor_combined/src/
git clone https://github.com/PX4/px4_msgs.git
git clone https://github.com/PX4/px4_ros_com.git
cd ..
source /opt/ros/humble/setup.bash
colcon build
```
Add auto‑sourcing (optional):

```bash
nano ~/.bashrc
```

Append:

```bash
# Auto-source Cosys-AirSim ROS 2 overlay
source /mnt/d/trung_Nav_team/ws_sensor_combined/install/setup.bash
```

### Install Cosys‑AirSim (UE 5.5 supported)

```bash
git clone https://github.com/Cosys-Lab/Cosys-AirSim.git
cd Cosys-AirSim
./setup.sh
./build.sh
```

### Create / prepare the ROS 2 workspace

```bash
cd ros2/src
sudo apt install -y ros-humble-geographic-msgs ros-humble-angles

git clone -b humble https://github.com/ros-perception/vision_opencv.git
cd ..
sudo rosdep init
rosdep update
rosdep install --from-paths src -y --ignore-src \
  --skip-keys pcl \
  --skip-keys message_runtime \
  --skip-keys message_generation 


colcon build --symlink-install --packages-select cv_bridge
source install/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Add auto‑sourcing (optional):

```bash
nano ~/.bashrc
```

Append:

```bash
# Auto-source Cosys-AirSim ROS 2 overlay
source /mnt/d/trung_Nav_team/Cosys-AirSim/ros2/install/setup.bash
```

### Test the image streaming pipeline

```bash
cd <your_workspace>/Cosys-AirSim/ros2/src
mkdir ct_uav_omni_fast_stream
# Use --recurse-submodules to clone submodules
git clone --recurse-submodules -b dev_simulation \
  git@gitlab.ctgroupvietnam.com:ct-uav/uaalab/obstacle-avoidance.git
cd ../..
colcon build --packages-above ct_uav_omni_fast_stream
```

---

## External Networking (Bridged WSL2)

Enable bridged networking so WSL2 can communicate with external ROS 2 systems.

### Step 1 — Create `.wslconfig`

Create/edit:

```
C:\Users\<your_username>\.wslconfig
```

Add:

```ini
[wsl2]
networkingMode=bridged
vmSwitch=ROSBridge
```

### Step 2 — Enable Hyper‑V

Open **PowerShell (Admin)**:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Hyper-V /all /norestart
# Reboot Windows after enabling
```

### Step 3 — Create a Hyper‑V External Virtual Switch

1. Launch **Hyper‑V Manager** as Administrator.
2. Go to **Virtual Switch Manager → New virtual network switch → External**.
3. Rename it **ROSBridge**.
4. Bind it to your **physical Ethernet adapter** (Intel/Realtek/etc.).
5. **Apply → OK**.

### Step 4 — Restart WSL

```bash
wsl --shutdown
wsl
```

---

## Final Verification Checklist

| Component      | Command / Action          | Expected Result                      |
| -------------- | ------------------------- | ------------------------------------ |
| WSL networking | `ping <external_ip>`      | Successful ping                      |
| PX4 SITL       | `make px4_sitl none_iris` | PX4 simulator launches               |
| AirSim         | Launch Unreal project     | Connection/activity visible          |
| QGroundControl | Connect UDP `14580`       | Telemetry visible                    |
| ROS 2          | `ros2 topic list`         | AirSim + PX4 topics are discoverable |

---

## References

* PX4 Docs — [https://docs.px4.io/main/en/](https://docs.px4.io/main/en/)
* AirSim GitHub — [https://github.com/microsoft/AirSim](https://github.com/microsoft/AirSim)
* ROS 2 Docs — [https://docs.ros.org/en/humble/](https://docs.ros.org/en/humble/)
* Cosys‑AirSim — [https://github.com/Cosys-Lab/Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim)

---

**Author:** Trung Lai
**Last Updated:** October 2025
**License:** MIT
