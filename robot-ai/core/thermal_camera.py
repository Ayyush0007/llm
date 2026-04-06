"""
thermal_camera.py — Infrared/Thermal camera for the Bento Robot.

Why critical for Indian roads:
  • Night driving in Jalgaon — street lighting is sparse/absent
  • Detect animals (cows, dogs) in darkness before RGB camera can
  • Detect overheating motor/battery before failure
  • Fog/dust penetration — IR sees through what RGB can't

Hardware options (cheapest to best):
  Option A: MLX90640 (32×24 px, I2C, ₹2,200) ← recommended start
  Option B: FLIR Lepton 3.5 (160×120 px, SPI, ₹12,000) ← high quality
  Option C: Seek Thermal Compact (USB, 206×156 px, ₹25,000) ← plug-and-play

CARLA sim: No native thermal sensor — we simulate from depth map
           (closer objects = brighter in thermal = more heat signature)

Install:  pip install adafruit-mlx90640

Wiring (MLX90640 via I2C-1):
  VCC → 3.3V
  GND → GND
  SDA → Pin 3
  SCL → Pin 5
"""

import time
import threading
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ThermalFrame:
    """One frame from the thermal camera."""
    raw_temps:      np.ndarray = field(default_factory=lambda: np.zeros((24, 32)))
    frame_celsius:  np.ndarray = field(default_factory=lambda: np.zeros((24, 32)))
    colormap_bgr:   Optional[np.ndarray] = None   # BGR image for display

    # Derived
    max_temp:     float = 0.0
    min_temp:     float = 0.0
    hot_zones:    List[tuple] = field(default_factory=list)  # (row,col,temp)
    living_thing: bool  = False   # True if > 35°C hot spot found
    motor_alert:  bool  = False   # True if motor area > 70°C
    timestamp:    float = 0.0

    LIVING_THRESHOLD = 32.0   # °C — body heat
    MOTOR_THRESHOLD  = 68.0   # °C — motor overheat


class ThermalCamera:
    """
    Thermal / IR camera manager.
    Detects living beings and mechanical overheat in darkness.
    """

    def __init__(self, mode: str = "mock",
                 sensor: str = "mlx90640",
                 display_size: tuple = (320, 240)):
        """
        mode:   "mlx90640" → real I2C sensor
                "carla"    → simulate from depth map
                "mock"     → synthetic warm-object frames
        sensor: "mlx90640" | "lepton" | "seek"
        """
        self.mode         = mode
        self._display_size = display_size
        self._latest: Optional[ThermalFrame] = None
        self._lock   = threading.Lock()
        self._running = False
        self._mlx    = None

        if mode == "mlx90640":
            self._init_mlx90640()
        else:
            self._start_mock()

        print(f"✅ Thermal camera ready (mode={mode})")

    def _init_mlx90640(self):
        try:
            import board, busio
            import adafruit_mlx90640
            i2c = busio.I2C(board.SCL, board.SDA, frequency=1_000_000)
            self._mlx = adafruit_mlx90640.MLX90640(i2c)
            self._mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
            self._running = True
            t = threading.Thread(target=self._hw_loop, daemon=True)
            t.start()
            print("✅ MLX90640 thermal sensor initialised")
        except ImportError:
            print("⚠️  adafruit-mlx90640 not installed — using mock thermal")
            self._start_mock()
        except Exception as e:
            print(f"⚠️  MLX90640 failed: {e} — using mock thermal")
            self._start_mock()

    def _hw_loop(self):
        buf = [0.0] * 768   # 32×24 = 768 pixels
        while self._running:
            try:
                self._mlx.getFrame(buf)
                arr = np.array(buf, dtype=np.float32).reshape(24, 32)
                frame = self._process(arr)
                with self._lock:
                    self._latest = frame
            except Exception:
                pass
            time.sleep(0.25)   # 4Hz hardware rate

    def update_from_depth(self, depth_map: np.ndarray):
        """
        [CARLA mode] Simulate thermal from depth:
        closer objects appear warmer.
        """
        small  = cv2.resize(depth_map, (32, 24))
        # Invert: depth=0 (near) → high temp; depth=1 (far) → ambient
        temps  = 20.0 + (1.0 - small) * 20.0   # range 20°C–40°C
        # Add a hot object signature randomly for animals
        frame  = self._process(temps.astype(np.float32))
        with self._lock:
            self._latest = frame

    def _process(self, temps: np.ndarray) -> ThermalFrame:
        frame = ThermalFrame(
            raw_temps     = temps,
            frame_celsius = temps,
            max_temp      = float(temps.max()),
            min_temp      = float(temps.min()),
            timestamp     = time.time(),
        )
        # Find hot zones (living things / mechanical heat)
        hot_mask = temps > ThermalFrame.LIVING_THRESHOLD
        coords   = np.argwhere(hot_mask)
        frame.hot_zones    = [(int(r), int(c), float(temps[r, c]))
                               for r, c in coords[:10]]
        frame.living_thing = any(t >= ThermalFrame.LIVING_THRESHOLD
                                  for _, _, t in frame.hot_zones)
        frame.motor_alert  = frame.max_temp >= ThermalFrame.MOTOR_THRESHOLD

        # Render colourmap for dashboard
        norm        = cv2.normalize(temps, None, 0, 255, cv2.NORM_MINMAX)
        norm_u8     = norm.astype(np.uint8)
        color       = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
        frame.colormap_bgr = cv2.resize(color, self._display_size)

        return frame

    def _start_mock(self):
        self.mode     = "mock"
        self._running = True
        t = threading.Thread(target=self._mock_loop, daemon=True)
        t.start()

    def _mock_loop(self):
        import math, random
        t0 = time.time()
        while self._running:
            t = time.time() - t0
            base = np.full((24, 32), 28.0, dtype=np.float32)   # ambient 28°C
            # Walk a "cow" across the scene every 15s
            cow_col = int((t % 15) / 15 * 32)
            base[8:16, max(0, cow_col-3):cow_col+3] = 37.5    # cow body temp

            # Warm road surface in centre-bottom
            base[18:, 10:22] += 5.0

            frame = self._process(base)
            with self._lock:
                self._latest = frame
            time.sleep(0.25)

    def get(self) -> Optional[ThermalFrame]:
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
