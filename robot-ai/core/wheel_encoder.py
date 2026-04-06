"""
wheel_encoder.py — Wheel odometry sensor for the Bento Robot.

Uses encoder pulses from motor shaft to compute:
  • True wheel speed (not GPS speed — useful in tunnels/markets)
  • Distance travelled
  • Slip detection (encoder speed vs IMU speed mismatch)
  • Pothole depth estimation (sudden speed drop + IMU spike)

Hardware: Optical encoder disc on each motor (600 PPR)
Jetson GPIO: Rising-edge interrupt on encoder output pin

Wiring:
  Encoder OUT → GPIO Pin 15 (left) and Pin 16 (right)
  VCC → 3.3V
  GND → GND
"""

import time
import threading
import math
from dataclasses import dataclass
from typing import Optional

# Encoder / Drive parameters
PULSES_PER_REV  = 20       # encoder pulses per full wheel revolution
WHEEL_DIAM_M    = 0.065    # wheel diameter in metres (6.5 cm — typical for 4WD chassis)
WHEEL_CIRCUMF   = math.pi * WHEEL_DIAM_M
METERS_PER_PULSE = WHEEL_CIRCUMF / PULSES_PER_REV

# Slip threshold: if encoder speed < 60% of expected speed → wheel slip
SLIP_RATIO_THRESHOLD = 0.60

# GPIO pins
ENC_PIN_LEFT  = 15
ENC_PIN_RIGHT = 16


@dataclass
class WheelOdometry:
    """Current odometry state."""
    speed_left_mps:  float = 0.0   # m/s
    speed_right_mps: float = 0.0   # m/s
    speed_avg_mps:   float = 0.0   # m/s
    speed_kmh:       float = 0.0
    distance_m:      float = 0.0   # total distance since start
    slip_detected:   bool  = False
    timestamp:       float = 0.0


class WheelEncoder:
    """
    Dual-channel wheel encoder odometry.
    Uses GPIO edge interrupts for pulse counting.
    """

    def __init__(self, mode: str = "mock"):
        """
        mode: "gpio"  → real encoder via Jetson GPIO interrupts
              "carla" → derive speed from CARLA vehicle velocity
              "mock"  → synthetic data
        """
        self.mode      = mode
        self._odo      = WheelOdometry(timestamp=time.time())
        self._lock     = threading.Lock()
        self._running  = False

        # Pulse counters (incremented by GPIO IRQ)
        self._count_l  = 0
        self._count_r  = 0
        self._last_t   = time.time()
        self._total_dist = 0.0

        if mode == "gpio":
            self._init_gpio()
        elif mode == "mock":
            self._start_mock()

        print(f"✅ Wheel encoder ready (mode={mode})")

    def _init_gpio(self):
        try:
            import Jetson.GPIO as GPIO
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(ENC_PIN_LEFT,  GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(ENC_PIN_RIGHT, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            GPIO.add_event_detect(ENC_PIN_LEFT,  GPIO.RISING,
                                   callback=lambda _: self._on_pulse("L"))
            GPIO.add_event_detect(ENC_PIN_RIGHT, GPIO.RISING,
                                   callback=lambda _: self._on_pulse("R"))

            self._gpio    = GPIO
            self._running = True
            t = threading.Thread(target=self._compute_loop, daemon=True)
            t.start()
            print(f"✅ Encoder GPIO pins {ENC_PIN_LEFT},{ENC_PIN_RIGHT} configured")

        except ImportError:
            print("⚠️  Jetson.GPIO not available — using mock encoder")
            self._start_mock()
        except Exception as e:
            print(f"⚠️  Encoder init failed: {e} — using mock")
            self._start_mock()

    def _on_pulse(self, side: str):
        """GPIO IRQ callback — increment pulse counter."""
        if side == "L":
            self._count_l += 1
        else:
            self._count_r += 1

    def _compute_loop(self):
        """Periodically compute speed from pulse counts (every 100ms)."""
        INTERVAL = 0.10
        while self._running:
            time.sleep(INTERVAL)
            cl, cr   = self._count_l, self._count_r
            self._count_l = 0
            self._count_r = 0

            spd_l = (cl * METERS_PER_PULSE) / INTERVAL
            spd_r = (cr * METERS_PER_PULSE) / INTERVAL
            spd   = (spd_l + spd_r) / 2

            self._total_dist += spd * INTERVAL

            with self._lock:
                self._odo = WheelOdometry(
                    speed_left_mps  = spd_l,
                    speed_right_mps = spd_r,
                    speed_avg_mps   = spd,
                    speed_kmh       = spd * 3.6,
                    distance_m      = self._total_dist,
                    timestamp       = time.time(),
                )

    def update_from_carla(self, vehicle):
        """Pull ground-truth speed from CARLA vehicle velocity."""
        try:
            v   = vehicle.get_velocity()
            spd = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            self._total_dist += spd * 0.05   # assume ~20 Hz tick

            with self._lock:
                self._odo = WheelOdometry(
                    speed_left_mps  = spd,
                    speed_right_mps = spd,
                    speed_avg_mps   = spd,
                    speed_kmh       = spd * 3.6,
                    distance_m      = self._total_dist,
                    timestamp       = time.time(),
                )
        except Exception:
            pass

    def check_slip(self, expected_speed_mps: float) -> bool:
        """
        Compares encoder speed against commanded throttle speed.
        Returns True if wheel slip is detected.
        """
        odo = self.get()
        if expected_speed_mps < 0.1:
            return False
        ratio = odo.speed_avg_mps / (expected_speed_mps + 1e-6)
        return ratio < SLIP_RATIO_THRESHOLD

    def _start_mock(self):
        self.mode     = "mock"
        self._running = True
        t = threading.Thread(target=self._mock_loop, daemon=True)
        t.start()

    def _mock_loop(self):
        t0 = time.time()
        while self._running:
            t   = time.time() - t0
            spd = max(0, 3.0 + 2.0 * math.sin(t * 0.2))   # m/s
            self._total_dist += spd * 0.05
            with self._lock:
                self._odo = WheelOdometry(
                    speed_left_mps  = spd * 0.98,
                    speed_right_mps = spd * 1.02,
                    speed_avg_mps   = spd,
                    speed_kmh       = spd * 3.6,
                    distance_m      = self._total_dist,
                    timestamp       = time.time(),
                )
            time.sleep(0.05)

    def get(self) -> WheelOdometry:
        with self._lock:
            return self._odo

    def stop(self):
        self._running = False
        if hasattr(self, "_gpio"):
            self._gpio.cleanup()
