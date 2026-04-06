import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ─── Motor Pin Config (BCM numbering) ───────────────────────────
# Adjust these to match your physical wiring on the Jetson Orin
MOTOR_A_IN1 = 17   # Left motor  - direction pin 1
MOTOR_A_IN2 = 18   # Left motor  - direction pin 2
MOTOR_A_ENA = 27   # Left motor  - PWM enable

MOTOR_B_IN1 = 22   # Right motor - direction pin 1
MOTOR_B_IN2 = 23   # Right motor - direction pin 2
MOTOR_B_ENB = 24   # Right motor - PWM enable
# ────────────────────────────────────────────────────────────────

# Try to import GPIO — gracefully degrade if not on Jetson hardware
try:
    import Jetson.GPIO as GPIO
    HARDWARE_AVAILABLE = True
except ImportError:
    try:
        import RPi.GPIO as GPIO
        HARDWARE_AVAILABLE = True
    except ImportError:
        GPIO = None
        HARDWARE_AVAILABLE = False
        print("⚠️  GPIO not available — running in simulation mode (no motors will move)")


class MotorController(Node):
    """
    ROS 2 node to control robot motors from Twist messages.
    Subscribes to /cmd_vel and drives L298N motor driver via GPIO.
    """

    def __init__(self):
        super().__init__("motor_controller")

        if HARDWARE_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            for pin in [MOTOR_A_IN1, MOTOR_A_IN2, MOTOR_A_ENA,
                        MOTOR_B_IN1, MOTOR_B_IN2, MOTOR_B_ENB]:
                GPIO.setup(pin, GPIO.OUT)

            self.pwm_a = GPIO.PWM(MOTOR_A_ENA, 100)
            self.pwm_b = GPIO.PWM(MOTOR_B_ENB, 100)
            self.pwm_a.start(0)
            self.pwm_b.start(0)
            self.get_logger().info("✅ GPIO initialized on hardware")
        else:
            self.pwm_a = None
            self.pwm_b = None
            self.get_logger().warn("⚠️  Running in simulation mode — no GPIO available")

        self.create_subscription(
            Twist,
            "/cmd_vel",
            self.drive_callback,
            10
        )
        self.get_logger().info("✅ Motor controller ready — subscribed to /cmd_vel")

    def drive_callback(self, msg: Twist):
        """
        Receives a Twist message and translates it to motor speeds.
        linear.x  : forward/backward speed (0.0 to 1.0)
        angular.z : left/right turn rate (-1.0 to 1.0)
        """
        speed = msg.linear.x    # 0.0 to 1.0
        turn  = msg.angular.z   # -1.0 (left) to 1.0 (right)

        # Differential drive math
        left_speed  = max(-1.0, min(1.0, speed - turn))
        right_speed = max(-1.0, min(1.0, speed + turn))

        self.get_logger().debug(
            f"L={left_speed:.2f}  R={right_speed:.2f}"
        )

        if HARDWARE_AVAILABLE:
            self._set_motor(MOTOR_A_IN1, MOTOR_A_IN2, self.pwm_a, left_speed)
            self._set_motor(MOTOR_B_IN1, MOTOR_B_IN2, self.pwm_b, right_speed)

    def _set_motor(self, in1: int, in2: int, pwm, speed: float):
        """
        Sets one motor direction and duty cycle.
        speed > 0 → forward
        speed < 0 → backward
        speed = 0 → stop
        """
        if speed > 0.0:
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
        elif speed < 0.0:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.HIGH)
        else:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.LOW)

        pwm.ChangeDutyCycle(abs(speed) * 100)

    def emergency_stop(self):
        """Immediately halt all motors."""
        self.get_logger().warn("🛑 EMERGENCY STOP")
        if HARDWARE_AVAILABLE:
            for pin in [MOTOR_A_IN1, MOTOR_A_IN2, MOTOR_B_IN1, MOTOR_B_IN2]:
                GPIO.output(pin, GPIO.LOW)
            self.pwm_a.ChangeDutyCycle(0)
            self.pwm_b.ChangeDutyCycle(0)

    def destroy_node(self):
        """Clean up GPIO on shutdown."""
        self.emergency_stop()
        if HARDWARE_AVAILABLE:
            self.pwm_a.stop()
            self.pwm_b.stop()
            GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    motor_node = MotorController()
    try:
        rclpy.spin(motor_node)
    except KeyboardInterrupt:
        pass
    finally:
        motor_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
