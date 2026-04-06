"""
lidar_sensor.py — LiDAR integration for the Bento Robot.

PHASE 1 (Simulation): Uses CARLA's ray-cast LiDAR sensor.
PHASE 2 (Hardware):   Uses RPLidar A1/A2 via rplidar-robotics library.

Output: A 360° distance scan as a numpy array (angles × distances).
The sensor_fusion.py uses this to supplement DepthAnything's depth map
with hard, accurate distance readings — especially useful for:
  - Detecting objects in blind spots (sides/rear)
  - Precise stopping distance at < 1m
  - Night-time / bad-weather when camera fails
"""

import numpy as np
import math
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LidarScan:
    """A single 360° LiDAR scan snapshot."""
    angles_deg:   np.ndarray = field(default_factory=lambda: np.array([]))
    distances_m:  np.ndarray = field(default_factory=lambda: np.array([]))
    timestamp:    float = 0.0

    # Pre-computed zone minimums (meters)
    front_min:  float = 99.0   # 330°–30° (forward cone)
    left_min:   float = 99.0   # 60°–120°
    right_min:  float = 99.0   # 240°–300°
    rear_min:   float = 99.0   # 150°–210°

    # Danger thresholds
    front_danger: bool = False
    side_danger:  bool = False

    STOP_DIST  = 0.50   # meters — full stop
    SLOW_DIST  = 1.20   # meters — slow down


# ─── CARLA LiDAR (Phase 1 — Simulation) ─────────────────────────
class CarlaLidar:
    """
    Attaches a ray-cast LiDAR sensor to a CARLA vehicle.
    Produces a LidarScan every tick.
    """

    def __init__(self, world, vehicle, channels: int = 1,
                 range_m: float = 15.0, points_per_sec: int = 10000):
        bp_lib  = world.get_blueprint_library()
        lid_bp  = bp_lib.find("sensor.lidar.ray_cast")
        lid_bp.set_attribute("channels", str(channels))
        lid_bp.set_attribute("range", str(range_m))
        lid_bp.set_attribute("points_per_second", str(points_per_sec))
        lid_bp.set_attribute("rotation_frequency", "20")

        import carla
        transform = carla.Transform(carla.Location(x=0.0, z=1.8))
        self._sensor = world.spawn_actor(lid_bp, transform, attach_to=vehicle)
        self._latest: Optional[LidarScan] = None
        self._lock   = threading.Lock()
        self._sensor.listen(self._on_scan)

    def _on_scan(self, raw):
        """CARLA callback — parse raw point cloud into LidarScan."""
        data    = np.frombuffer(raw.raw_data, dtype=np.float32)
        data    = data.reshape(-1, 4)   # x, y, z, intensity
        xs, ys  = data[:, 0], data[:, 1]

        # Convert XY to polar (angle from forward axis)
        angles  = np.degrees(np.arctan2(ys, xs)) % 360
        dists   = np.sqrt(xs**2 + ys**2)

        scan = LidarScan(
            angles_deg  = angles,
            distances_m = dists,
            timestamp   = time.time(),
        )
        scan = self._compute_zones(scan)

        with self._lock:
            self._latest = scan

    def get_scan(self) -> Optional[LidarScan]:
        with self._lock:
            return self._latest

    def destroy(self):
        self._sensor.destroy()

    @staticmethod
    def _compute_zones(scan: LidarScan) -> LidarScan:
        a, d = scan.angles_deg, scan.distances_m

        def zone_min(lo, hi):
            if lo <= hi:
                mask = (a >= lo) & (a <= hi)
            else:
                mask = (a >= lo) | (a <= hi)
            pts = d[mask]
            return float(pts.min()) if len(pts) > 0 else 99.0

        scan.front_min = zone_min(330, 30)
        scan.left_min  = zone_min(60,  120)
        scan.right_min = zone_min(240, 300)
        scan.rear_min  = zone_min(150, 210)

        scan.front_danger = scan.front_min < LidarScan.STOP_DIST
        scan.side_danger  = min(scan.left_min, scan.right_min) < LidarScan.SLOW_DIST

        return scan


# ─── Real RPLidar (Phase 2 — Hardware) ──────────────────────────
class RPLidarSensor:
    """
    Wraps the RPLidar A1/A2 serial sensor.
    Requires: pip install rplidar-robotics
    Connect via USB → /dev/ttyUSB0 on Jetson.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200):
        try:
            from rplidar import RPLidar
            self._lidar  = RPLidar(port, baudrate=baudrate)
            self._latest: Optional[LidarScan] = None
            self._lock   = threading.Lock()
            self._thread = threading.Thread(target=self._scan_loop, daemon=True)
            self._thread.start()
            print(f"✅ RPLidar connected on {port}")
        except ImportError:
            print("⚠️  rplidar-robotics not installed: pip install rplidar-robotics")
            self._lidar  = None
        except Exception as e:
            print(f"⚠️  RPLidar not found on {port}: {e}")
            self._lidar  = None

    def _scan_loop(self):
        if self._lidar is None:
            return
        for scan in self._lidar.iter_scans():
            angles = np.array([pt[1] for pt in scan], dtype=np.float32)
            dists  = np.array([pt[2] / 1000.0 for pt in scan], dtype=np.float32)  # mm→m

            snap = LidarScan(
                angles_deg  = angles,
                distances_m = dists,
                timestamp   = time.time(),
            )
            snap = CarlaLidar._compute_zones(snap)
            with self._lock:
                self._latest = snap

    def get_scan(self) -> Optional[LidarScan]:
        with self._lock:
            return self._latest

    def destroy(self):
        if self._lidar:
            self._lidar.stop()
            self._lidar.disconnect()


# ─── Unified interface ────────────────────────────────────────────
class LidarSystem:
    """
    Unified LiDAR manager.
    Auto-selects CARLA or RPLidar based on context.
    """

    def __init__(self, mode: str = "carla", **kwargs):
        """
        mode: "carla"   → CARLA simulation LiDAR
              "rplidar" → Real RPLidar A1/A2 on Jetson
              "mock"    → Always-clear mock (for testing without hardware)
        """
        self.mode = mode

        if mode == "carla":
            self._sensor = CarlaLidar(**kwargs)
        elif mode == "rplidar":
            self._sensor = RPLidarSensor(**kwargs)
        else:
            self._sensor = None   # mock mode

    def get_scan(self) -> LidarScan:
        if self._sensor is None:
            # Mock: return a clear scan
            mock = LidarScan(timestamp=time.time())
            mock.front_min  = 5.0
            mock.left_min   = 5.0
            mock.right_min  = 5.0
            mock.rear_min   = 5.0
            return mock

        scan = self._sensor.get_scan()
        return scan if scan is not None else LidarScan(timestamp=time.time())

    def destroy(self):
        if self._sensor:
            self._sensor.destroy()
