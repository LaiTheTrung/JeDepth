import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
import time
import numpy as np

class MotorTestNode(Node):
    def __init__(self):
        super().__init__('motor_test_node')

        # QoS for PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.offboard_ctrl_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10)

        # Subscribers
        self.pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.pos_cb, qos_profile)
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.status_cb, qos_profile)

        # State
        self.current_pos = None
        self.nav_state = None
        self.arming_state = None
        self.test_phase = 0 # 0: Init, 1: Offboard, 2: Arm, 3: Wait, 4: Disarm, 5: Done
        self.start_arm_time = None
        self.offboard_set_time = None

        # Timer (20Hz)
        self.create_timer(0.05, self.timer_cb)
        self.get_logger().info("Motor Test Node Started. Waiting for PX4 connection...")

    def pos_cb(self, msg):
        self.current_pos = msg

    def status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        self.offboard_ctrl_pub.publish(msg)

    def publish_hold_position(self):
        if self.current_pos is None: return
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        # Hold current position
        msg.position = [self.current_pos.x, self.current_pos.y, self.current_pos.z]
        msg.yaw = float('nan')
        self.traj_pub.publish(msg)

    def send_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.from_external = True
        self.vehicle_command_pub.publish(msg)

    def timer_cb(self):
        self.publish_offboard_heartbeat()
        self.publish_hold_position()

        if self.current_pos is None or self.nav_state is None:
            if self.test_phase == 0:
                self.get_logger().info("Waiting for position lock...", throttle_duration_sec=1.0)
            return

        # Phase 0: Ready
        if self.test_phase == 0:
            self.get_logger().info("Position locked. Switching to OFFBOARD...")
            self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
            self.offboard_set_time = time.time()
            self.test_phase = 1

        # Phase 1: Wait for Offboard
        elif self.test_phase == 1:
            if self.nav_state == 14: # Offboard
                self.get_logger().info("Offboard Mode Active. ARMING in 2 seconds...")
                time.sleep(2.0) # Safety pause
                self.test_phase = 2
            elif time.time() - self.offboard_set_time > 5.0:
                self.get_logger().error("Failed to enter Offboard mode. Retrying...")
                self.test_phase = 0

        # Phase 2: Arm
        elif self.test_phase == 2:
            self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
            if self.arming_state == 2: # Armed
                self.get_logger().info("MOTORS SPINNING! (Waiting 1s)")
                self.start_arm_time = time.time()
                self.test_phase = 3
        
        # Phase 3: Wait 1s
        elif self.test_phase == 3:
            if time.time() - self.start_arm_time > 1.0:
                self.get_logger().info("Time up. DISARMING!")
                self.test_phase = 4

        # Phase 4: Disarm
        elif self.test_phase == 4:
            self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
            if self.arming_state != 2: # Disarmed
                self.get_logger().info("TEST COMPLETE. MOTORS STOPPED.")
                self.test_phase = 5
                raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    node = MotorTestNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        print("Done.")
    except KeyboardInterrupt:
        pass
    finally:
        # Emergency Disarm on exit
        node.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
