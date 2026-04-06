import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import os
import sys

# Add core to path to import DepthAI
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from depth_ai import DepthAI
from ultralytics import YOLO

class IndianRoadRobotNode(Node):
    """
    Main robot brain node.
    Subscribes : /carla/ego_vehicle/rgb_front/image  (camera)
    Publishes  : /carla/ego_vehicle/vehicle_control  (drive commands)
    """

    def __init__(self):
        super().__init__("indian_road_robot")
        self.bridge = CvBridge()

        # ── Load AI Models ─────────────────────────────────────────
        self.get_logger().info("Loading Vision AI (YOLOv8)...")
        # Assume model is in standard location
        model_path = "/Users/yashmogare/robot-ai/llm/robot-ai/models/vision/best.pt"
        if not os.path.exists(model_path):
            self.get_logger().warn(f"Best model not found at {model_path}. Trying fallback...")
            model_path = "yolov8m.pt"
            
        self.vision_ai = YOLO(model_path)

        self.get_logger().info("Loading Depth AI...")
        self.depth_ai = DepthAI(model_size="small")

        self.get_logger().info("✅ All models loaded")

        # ── ROS 2 Pub/Sub ──────────────────────────────────────────
        self.image_sub = self.create_subscription(
            Image,
            "/carla/ego_vehicle/rgb_front/image",
            self.camera_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            "/carla/ego_vehicle/twist",
            10
        )

        # State
        self.last_detections = []
        self.emergency_stop  = False

    def camera_callback(self, msg: Image):
        """Called every frame — run full AI pipeline here"""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # ── 1. Vision AI — detect objects ─────────────────────────
        results     = self.vision_ai(frame, conf=0.45, verbose=False)
        detections  = results[0].boxes
        class_names = results[0].names

        detected_labels = []
        for box in detections:
            cls_name = class_names[int(box.cls[0])]
            conf     = float(box.conf[0])
            detected_labels.append((cls_name, conf))

        # ── 2. Depth AI — measure distances ───────────────────────
        depth_map     = self.depth_ai.estimate(frame)
        center_dist   = self.depth_ai.get_obstacle_distance(depth_map, "center")
        left_dist     = self.depth_ai.get_obstacle_distance(depth_map, "left")
        right_dist    = self.depth_ai.get_obstacle_distance(depth_map, "right")

        # ── 3. Decision Logic ─────────────────────────────────────
        cmd = Twist()

        # Emergency stop conditions (including cows and rickshaws)
        emergency_classes = {"cow", "dog", "person", "child", "auto_rickshaw"}
        stop_detected = any(lbl in emergency_classes for lbl, _ in detected_labels)

        if stop_detected or center_dist < 0.25:
            # STOP
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0
            self.get_logger().warn(f"🛑 STOP — Predicted: {detected_labels}")

        elif center_dist < 0.5:
            # SLOW DOWN
            cmd.linear.x  = 0.3
            cmd.angular.z = 0.0

        else:
            # NAVIGATE
            cmd.linear.x  = 0.8
            if left_dist < right_dist:
                cmd.angular.z = -0.3   # steer right
            elif right_dist < left_dist:
                cmd.angular.z = 0.3    # steer left
            else:
                cmd.angular.z = 0.0

        # Pothole detected → slow down
        if any(lbl == "pothole" for lbl, _ in detected_labels):
            cmd.linear.x = min(cmd.linear.x, 0.4)
            self.get_logger().info("⚠️  Pothole detected — slowing")

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = IndianRoadRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
