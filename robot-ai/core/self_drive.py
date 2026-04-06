"""
self_drive.py — Main self-driving loop.

PHASE 1 (current): Runs inside CARLA simulator.
  - Connects to a running CARLA server
  - Spawns a vehicle + camera sensor
  - Runs Vision AI + Depth AI on each camera frame
  - Sends the fused WorldModel through the state machine
  - Publishes throttle/steer/brake back to CARLA
  - Shows the live dashboard HUD

PHASE 2 (later): Swap CARLA for real USB camera + GPIO motors.
  All Phase-2 changes are marked with # [HARDWARE] comments.

Usage:
  # Terminal 1: ./CarlaUE4.sh -quality-level=Low -fps=20
  # Terminal 2: python3 core/self_drive.py
  # Press Q in the dashboard window to stop.
"""

import sys
import os
import time
import math
import threading
import numpy as np
import cv2

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─── AI Models ────────────────────────────────────────────────────
from ultralytics import YOLO
from core.depth_ai      import DepthAI
from core.sensor_fusion import SensorFusion
from core.state_machine import DrivingStateMachine, DriveState
from core.dashboard     import Dashboard
from core.data_logger   import DataLogger

# ─── Config ───────────────────────────────────────────────────────
CARLA_HOST     = "localhost"
CARLA_PORT     = 2000
CARLA_TIMEOUT  = 10.0
IMAGE_W        = 640
IMAGE_H        = 480
TARGET_FPS     = 15

YOLO_MODEL     = "models/vision/best.pt"
YOLO_FALLBACK  = "yolov8n.pt"       # use nano if best.pt not trained yet
YOLO_CONF      = 0.40

DEPTH_SIZE     = "small"            # "small"=fast, "base"=balanced, "large"=best

ENABLE_DASHBOARD = True
ENABLE_LOGGING   = True

# ─── CARLA imports (graceful if not installed) ────────────────────
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    print("⚠️  CARLA not installed — running in DEMO mode (static frame)")


class SelfDriveSystem:
    """
    Orchestrates the full self-driving pipeline inside CARLA.
    """

    def __init__(self):
        # ── Load AI models ──────────────────────────────────────
        model_path = YOLO_MODEL if os.path.exists(YOLO_MODEL) else YOLO_FALLBACK
        print(f"🔭 Loading Vision AI: {model_path}")
        self.vision   = YOLO(model_path)

        print(f"📐 Loading Depth AI ({DEPTH_SIZE})...")
        self.depth_ai = DepthAI(model_size=DEPTH_SIZE)

        self.fusion   = SensorFusion()
        self.fsm      = DrivingStateMachine()
        self.dashboard = Dashboard() if ENABLE_DASHBOARD else None
        self.logger    = DataLogger()  if ENABLE_LOGGING   else None

        # ── CARLA state ─────────────────────────────────────────
        self._client   = None
        self._world    = None
        self._vehicle  = None
        self._camera   = None
        self._actor_list = []

        # Shared camera frame (written by sensor callback, read by main loop)
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray = np.zeros((IMAGE_H, IMAGE_W, 3), np.uint8)

        self._running = False

    # ─── Lifecycle ────────────────────────────────────────────────

    def start(self):
        if CARLA_AVAILABLE:
            self._setup_carla()
        else:
            print("⚠️  CARLA not found — demo mode: using blank frame")

        self.fsm.start()
        self._running = True
        print("✅ Self-drive system started! Press Q to quit dashboard.")
        self._run_loop()

    def stop(self):
        self._running = False
        self._cleanup()
        if self.logger:
            self.logger.close()
        if self.dashboard:
            self.dashboard.close()
        print("🛑 Self-drive system stopped.")

    # ─── CARLA Setup ──────────────────────────────────────────────

    def _setup_carla(self):
        import random
        print(f"🚗 Connecting to CARLA at {CARLA_HOST}:{CARLA_PORT}...")
        self._client = carla.Client(CARLA_HOST, CARLA_PORT)
        self._client.set_timeout(CARLA_TIMEOUT)
        self._world  = self._client.get_world()
        bp_lib       = self._world.get_blueprint_library()

        # Spawn vehicle
        vehicle_bp   = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_pts    = self._world.get_map().get_spawn_points()
        self._vehicle = self._world.spawn_actor(vehicle_bp, random.choice(spawn_pts))
        self._actor_list.append(self._vehicle)

        # Spawn front RGB camera
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov", "90")
        cam_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
        self._camera  = self._world.spawn_actor(cam_bp, cam_transform, attach_to=self._vehicle)
        self._camera.listen(self._on_camera_frame)
        self._actor_list.append(self._camera)

        # Randomise Indian weather
        self._set_weather()
        self._world.tick()
        print("✅ CARLA vehicle + camera spawned")

    def _set_weather(self):
        """Pick a random Indian weather preset."""
        import random
        presets = [
            carla.WeatherParameters(cloudiness=10,  precipitation=0,  sun_altitude_angle=75),
            carla.WeatherParameters(cloudiness=85,  precipitation=70, wetness=80, wind_intensity=35),
            carla.WeatherParameters(cloudiness=45,  fog_density=20,   sun_altitude_angle=50),
            carla.WeatherParameters(cloudiness=20,  precipitation=0,  sun_altitude_angle=-5),
        ]
        self._world.set_weather(random.choice(presets))

    def _on_camera_frame(self, image):
        """CARLA sensor callback — runs in a background thread."""
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]  # BGRA→BGR
        with self._frame_lock:
            self._latest_frame = arr.copy()

    # [HARDWARE] Replace _setup_carla() with:
    #   self._cap = cv2.VideoCapture(0)  ← USB camera
    # Replace _on_camera_frame with:
    #   ret, frame = self._cap.read()   ← read in main loop

    # ─── Main Loop ────────────────────────────────────────────────

    def _run_loop(self):
        fps_timer = time.time()
        fps = 0.0

        while self._running:
            t_start = time.time()

            # 1. Grab latest frame
            with self._frame_lock:
                frame = self._latest_frame.copy()

            if frame.sum() == 0:
                time.sleep(0.05)
                continue

            # 2. Vision AI → detections
            results    = self.vision(frame, conf=YOLO_CONF, verbose=False)
            raw_dets   = []
            for box in results[0].boxes:
                raw_dets.append({
                    "cls"  : results[0].names[int(box.cls[0])],
                    "conf" : float(box.conf[0]),
                    "bbox" : box.xyxy[0].tolist(),
                })

            # 3. Depth AI → depth map
            depth_map = self.depth_ai.estimate(frame)

            # 4. Sensor fusion → WorldModel
            world = self.fusion.fuse(
                frame      = frame,
                detections = raw_dets,
                depth_map  = depth_map,
                depth_ai   = self.depth_ai,
                timestamp  = time.time(),
            )

            # 5. State machine → DriveCommand
            cmd = self.fsm.tick(world)

            # 6. Apply command to CARLA (or hardware)
            self._apply_command(cmd)

            # 7. Log
            if self.logger:
                self.logger.log(world, cmd)

            # 8. Dashboard HUD
            if self.dashboard:
                t_now = time.time()
                fps   = 1.0 / max(t_now - fps_timer, 1e-6)
                fps_timer = t_now
                alive = self.dashboard.show(frame, depth_map, world, cmd, fps)
                if not alive:
                    break   # user pressed Q

            # Tick CARLA world forward
            if CARLA_AVAILABLE and self._world:
                self._world.tick()

            # Frame rate cap
            elapsed = time.time() - t_start
            sleep   = max(0.0, (1.0 / TARGET_FPS) - elapsed)
            time.sleep(sleep)

        self.stop()

    def _apply_command(self, cmd):
        """Send DriveCommand to CARLA vehicle."""
        if not CARLA_AVAILABLE or self._vehicle is None:
            return   # demo mode

        control = carla.VehicleControl(
            throttle = float(np.clip(cmd.throttle, 0.0, 1.0)),
            steer    = float(np.clip(cmd.steer,    -1.0, 1.0)),
            brake    = float(np.clip(cmd.brake,    0.0, 1.0)),
        )
        self._vehicle.apply_control(control)

    # [HARDWARE] Replace _apply_command with motor_control.py's drive_callback()

    def _cleanup(self):
        for actor in self._actor_list:
            try:
                actor.destroy()
            except Exception:
                pass
        self._actor_list = []


# ─── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    system = SelfDriveSystem()
    try:
        system.start()
    except KeyboardInterrupt:
        system.stop()
