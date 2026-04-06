"""
imu_sensor.py — IMU (Inertial Measurement Unit) for the Bento Robot.

Hardware: MPU6050 or ICM-42688-P via I2C on Jetson Orin NX
CARLA sim: Uses vehicle physics API (get_transform + get_acceleration)

Key uses for Indian roads:
  • Pothole/bump detection via Z-axis shock spikes
  • Road tilt / slope measurement (uphill/downhill)
  • Skid/slip detection (sudden lateral G-force)
  • Vibration quality score — rough road vs smooth

Install (hardware):  pip install mpu6050-raspberrypi smbus2

Wiring (Jetson I2C-1):
  VCC → 3.3V
  GND → GND
  SDA → Pin 3
  SCL → Pin 5
"""

import time
import math
import threading
import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import Optional


@dataclass
class IMUReading:
    """One snapshot from the IMU."""
    accel_x: float = 0.0   # m/s² — forward/backward
    accel_y: float = 0.0   # m/s² — left/right
    accel_z: float = 9.81  # m/s² — vertical (gravity at rest)
    gyro_x:  float = 0.0   # °/s — pitch rate
    gyro_y:  float = 0.0   # °/s — roll rate
    gyro_z:  float = 0.0   # °/s — yaw rate
    timestamp: float = 0.0

    # Derived values
    tilt_deg:      float = 0.0   # road slope in degrees
    vibration:     float = 0.0   # RMS vibration (0=smooth, 1=very rough)
    pothole_shock: bool  = False  # True if sudden Z-spike detected
    lateral_g:     float = 0.0   # lateral G-force (skid indicator)

    POTHOLE_THRESHOLD = 3.0   # m/s² Z-spike to declare pothole
    SKID_THRESHOLD    = 5.0   # m/s² lateral G for skid alert


class IMUSensor:
    """
    IMU sensor manager.
    Reads acceleration + gyro, computes road quality metrics.
    """

    def __init__(self, mode: str = "mock"):
        """
        mode: "mpu6050"  → real MPU-6050 on I2C
              "carla"    → data from CARLA vehicle physics
              "mock"     → synthetic sine-wave for testing
        """
        self.mode   = mode
        self._mpu   = None
        self._lock  = threading.Lock()
        self._latest = IMUReading(timestamp=time.time())

        # Rolling window for vibration RMS (last 50 readings)
        self._z_history = deque(maxlen=50)
        self._running   = False

        if mode == "mpu6050":
            self._init_mpu6050()
        elif mode == "mock":
            self._start_mock_thread()

        print(f"✅ IMU sensor ready (mode={mode})")

    def _init_mpu6050(self):
        try:
            from mpu6050 import mpu6050
            self._mpu = mpu6050(0x68)
            self._running = True
            t = threading.Thread(target=self._hw_read_loop, daemon=True)
            t.start()
        except ImportError:
            print("⚠️  mpu6050 lib not found — pip install mpu6050-raspberrypi")
            self.mode = "mock"
            self._start_mock_thread()
        except Exception as e:
            print(f"⚠️  MPU-6050 not found: {e} — using mock")
            self.mode = "mock"
            self._start_mock_thread()

    def _hw_read_loop(self):
        while self._running:
            try:
                a = self._mpu.get_accel_data()
                g = self._mpu.get_gyro_data()
                r = IMUReading(
                    accel_x = a["x"], accel_y = a["y"], accel_z = a["z"],
                    gyro_x  = g["x"], gyro_y  = g["y"], gyro_z  = g["z"],
                    timestamp = time.time(),
                )
                self._process(r)
            except Exception:
                pass
            time.sleep(0.02)   # 50 Hz

    def _start_mock_thread(self):
        """Generates synthetic IMU data with occasional simulated potholes."""
        self._running = True
        t = threading.Thread(target=self._mock_loop, daemon=True)
        t.start()

    def _mock_loop(self):
        t0 = time.time()
        while self._running:
            t = time.time() - t0
            # Gentle driving vibration + occasional pothole at every 12 s
            pothole = (int(t) % 12 == 0) and (t % 1.0 < 0.05)
            r = IMUReading(
                accel_x   = 0.2 * math.sin(t * 0.5),
                accel_y   = 0.1 * math.sin(t * 1.1),
                accel_z   = 9.81 + 0.3 * math.sin(t * 5) + (4.5 if pothole else 0),
                gyro_x    = 0.5 * math.sin(t * 0.3),
                gyro_y    = 0.3 * math.sin(t * 0.7),
                gyro_z    = 0.1 * math.sin(t * 0.2),
                timestamp = time.time(),
            )
            self._process(r)
            time.sleep(0.02)

    def update_from_carla(self, vehicle):
        """
        Called each CARLA tick to pull physics from the simulated vehicle.
        vehicle: carla.Actor
        """
        try:
            a = vehicle.get_acceleration()
            v = vehicle.get_angular_velocity()
            t = vehicle.get_transform()
            r = IMUReading(
                accel_x   = a.x, accel_y = a.y, accel_z = a.z + 9.81,
                gyro_x    = v.x, gyro_y  = v.y, gyro_z  = v.z,
                timestamp = time.time(),
            )
            # Add road tilt from pitch angle
            r.tilt_deg = t.rotation.pitch
            self._process(r)
        except Exception:
            pass

    def _process(self, r: IMUReading):
        """Compute derived metrics (vibration, pothole, tilt, lateral-G)."""
        # Vertical vibration RMS
        self._z_history.append(r.accel_z - 9.81)
        arr       = np.array(self._z_history)
        r.vibration     = float(np.sqrt(np.mean(arr ** 2))) / 5.0
        r.vibration     = min(1.0, r.vibration)

        # Pothole: sudden Z-spike
        r.pothole_shock = abs(r.accel_z - 9.81) > IMUReading.POTHOLE_THRESHOLD

        # Lateral G
        r.lateral_g = abs(r.accel_y)

        # Road tilt from accelerometer (if not already set from CARLA)
        if r.tilt_deg == 0.0:
            r.tilt_deg = math.degrees(math.atan2(r.accel_x, 9.81))

        with self._lock:
            self._latest = r

    def get(self) -> IMUReading:
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
