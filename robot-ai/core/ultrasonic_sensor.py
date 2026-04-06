"""
ultrasonic_sensor.py — Short-range bumper ultrasonic sensors (HC-SR04).

4 sensors mounted on the robot chassis:
  front_center, front_left, front_right, rear_center

Why ultrasonic for Indian roads:
  • LiDAR has a blind zone < 15cm — HC-SR04 covers 2cm–4m
  • Very cheap (₹80 each, total ₹320 for 4)
  • Detects low-lying objects: stray dogs under the chassis, kerbstones
  • Works in dust/fog where camera fails at < 30cm

PHASE 1 (CARLA): Uses CARLA radar/collision sensor as proxy.
PHASE 2 (Jetson): Uses HC-SR04 via GPIO trigger/echo pins.

Wiring (one HC-SR04 sensor):
  VCC   → 5V
  GND   → GND
  TRIG  → any GPIO output pin
  ECHO  → GPIO input (use 1kΩ + 2kΩ voltage divider — Jetson is 3.3V!)
"""

import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


# BCM pin assignments for 4 sensors on Jetson
# Adjust to your actual wiring
GPIO_PINS = {
    "front_center": {"trig": 29, "echo": 31},
    "front_left":   {"trig": 33, "echo": 35},
    "front_right":  {"trig": 37, "echo": 38},
    "rear_center":  {"trig": 40, "echo": 36},
}

# Speed of sound at 35°C (hot Indian summer) in cm/µs
SPEED_OF_SOUND_CM_US = 0.03504


@dataclass
class UltrasonicReading:
    """Distance readings from all 4 bumper sensors."""
    front_center: float = 4.0   # metres
    front_left:   float = 4.0
    front_right:  float = 4.0
    rear_center:  float = 4.0
    timestamp:    float = 0.0

    # Flags
    front_too_close:  bool = False   # < 0.30m
    rear_too_close:   bool = False   # < 0.20m

    FRONT_STOP_M = 0.30
    REAR_STOP_M  = 0.20


class UltrasonicSystem:
    """
    Multi-sensor HC-SR04 manager.
    Polls all 4 sensors in parallel threads.
    """

    def __init__(self, mode: str = "mock"):
        """
        mode: "gpio"  → real HC-SR04 sensors via Jetson GPIO
              "carla" → CARLA collision/radar sensor proxy
              "mock"  → randomised readings (for testing)
        """
        self.mode = mode
        self._readings: Dict[str, float] = {k: 4.0 for k in GPIO_PINS}
        self._lock    = threading.Lock()
        self._running = False
        self._gpio    = None

        if mode == "gpio":
            self._init_gpio()
        elif mode == "mock":
            self._start_mock()

        print(f"✅ Ultrasonic system ready (mode={mode}, 4 sensors)")

    def _init_gpio(self):
        try:
            import Jetson.GPIO as GPIO
            self._gpio = GPIO
            GPIO.setmode(GPIO.BOARD)
            for name, pins in GPIO_PINS.items():
                GPIO.setup(pins["trig"], GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(pins["echo"], GPIO.IN)

            self._running = True
            for name in GPIO_PINS:
                t = threading.Thread(target=self._poll_sensor, args=(name,), daemon=True)
                t.start()
            print("✅ HC-SR04 GPIO sensors initialised")

        except ImportError:
            print("⚠️  Jetson.GPIO not available — using mock ultrasonic")
            self._start_mock()
        except Exception as e:
            print(f"⚠️  HC-SR04 init failed: {e} — using mock")
            self._start_mock()

    def _poll_sensor(self, name: str):
        """Continuously poll one HC-SR04 sensor."""
        GPIO  = self._gpio
        pins  = GPIO_PINS[name]
        trig  = pins["trig"]
        echo  = pins["echo"]

        while self._running:
            # Send 10µs trigger pulse
            GPIO.output(trig, GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(trig, GPIO.LOW)

            # Measure echo pulse width
            t_start = time.time()
            while GPIO.input(echo) == 0 and time.time() - t_start < 0.05:
                pass
            pulse_start = time.time()
            while GPIO.input(echo) == 1 and time.time() - pulse_start < 0.05:
                pass
            pulse_end = time.time()

            duration_us = (pulse_end - pulse_start) * 1_000_000
            distance_m  = (duration_us * SPEED_OF_SOUND_CM_US) / 2 / 100

            # Clamp to sensor range [0.02m, 4.0m]
            distance_m = max(0.02, min(4.0, distance_m))

            with self._lock:
                self._readings[name] = distance_m

            time.sleep(0.06)   # ~16 Hz per sensor

    def _start_mock(self):
        self.mode     = "mock"
        self._running = True
        t = threading.Thread(target=self._mock_loop, daemon=True)
        t.start()

    def _mock_loop(self):
        import math
        t0 = time.time()
        while self._running:
            t = time.time() - t0
            # Simulate approaching obstacle on front_center every 20s
            approach = max(0.05, 2.0 - (t % 20) * 0.1) if (t % 20) < 20 else 2.0
            with self._lock:
                self._readings["front_center"] = approach
                self._readings["front_left"]   = 1.5 + 0.5 * math.sin(t * 0.3)
                self._readings["front_right"]  = 2.0 + 0.4 * math.sin(t * 0.5)
                self._readings["rear_center"]  = 3.0
            time.sleep(0.05)

    def update_from_carla(self, collision_events: list):
        """Proxy: treat CARLA collision history as a front obstacle."""
        if collision_events:
            with self._lock:
                self._readings["front_center"] = 0.10   # very close
        else:
            with self._lock:
                self._readings["front_center"] = max(
                    self._readings["front_center"], 2.0
                )

    def get(self) -> UltrasonicReading:
        with self._lock:
            r = UltrasonicReading(
                front_center = self._readings["front_center"],
                front_left   = self._readings["front_left"],
                front_right  = self._readings["front_right"],
                rear_center  = self._readings["rear_center"],
                timestamp    = time.time(),
            )
        r.front_too_close = r.front_center < UltrasonicReading.FRONT_STOP_M
        r.rear_too_close  = r.rear_center  < UltrasonicReading.REAR_STOP_M
        return r

    def stop(self):
        self._running = False
        if self._gpio:
            self._gpio.cleanup()
