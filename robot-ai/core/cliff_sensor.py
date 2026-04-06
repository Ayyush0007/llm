"""
cliff_sensor.py — Edge / Drop-off detector for Bento Robot.

Why critical for Indian roads:
  • Open manholes, missing gutter covers, and sudden broken road edges are very common.
  • Prevents the robot from falling into drains off the side of the road.
  • Failsafe that instantly overrides all other driving commands (Stop + Back up).

Hardware: TCRT5000 IR Line Tracking / Obstacle Avoidance module (₹40 each)
Requires 2-4 sensors mounted pointing straight down at the front/sides of the chassis.

Wiring:
  VCC → 3.3V
  GND → GND
  Sensor 1 (Front Left)  D0 → GPIO 29
  Sensor 2 (Front Right) D0 → GPIO 31
  Sensor 3 (Rear Left)   D0 → GPIO 33
  Sensor 4 (Rear Right)  D0 → GPIO 35
"""

import time
import threading
from dataclasses import dataclass
from typing import List

# GPIO Pins for 4 downward-facing IR sensors
CLIFF_PINS = [29, 31, 33, 35]

@dataclass
class CliffReading:
    cliff_detected: bool = False
    sensors_triggered: List[int] = None
    timestamp: float = 0.0


class CliffSensor:
    """
    Detects sudden drop-offs using downward-facing IR sensors.
    Digital HIGH = no surface detected (cliff)
    Digital LOW  = surface detected (safe)
    """

    def __init__(self, mode: str = "mock"):
        self.mode = mode
        self._latest = CliffReading(sensors_triggered=[], timestamp=time.time())
        self._lock   = threading.Lock()
        self._running = False

        if mode == "gpio":
            self._init_gpio()
        else:
            self._start_mock()

        print(f"✅ Cliff edge sensor ready (mode={mode})")

    def _init_gpio(self):
        try:
            import Jetson.GPIO as GPIO
            GPIO.setmode(GPIO.BOARD)
            for pin in CLIFF_PINS:
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                
            self._gpio = GPIO
            self._running = True
            t = threading.Thread(target=self._hw_loop, daemon=True)
            t.start()
            print(f"✅ TCRT5000 Cliff sensors on pins {CLIFF_PINS}")
        except ImportError:
            print("⚠️  Jetson.GPIO not available — using mock cliff sensor")
            self._start_mock()
        except Exception as e:
            print(f"⚠️  Cliff sensor init failed: {e} — using mock")
            self._start_mock()

    def _hw_loop(self):
        while self._running:
            try:
                triggered = []
                for i, pin in enumerate(CLIFF_PINS):
                    # TCRT5000 module usually outputs HIGH when IR is NOT reflected (cliff)
                    if self._gpio.input(pin) == self._gpio.HIGH:
                        triggered.append(i)
                        
                with self._lock:
                    self._latest = CliffReading(
                        cliff_detected = len(triggered) > 0,
                        sensors_triggered = triggered,
                        timestamp = time.time()
                    )
            except Exception:
                pass
            time.sleep(0.05)  # 20 Hz (needs fast reaction to stop)

    def update_from_carla(self, vehicle, raycast_sensors):
        """
        Simulate cliff detection in CARLA using 4 short downward raycasts.
        (Needs separate downward sensors attached to vehicle in CARLA)
        """
        # Simplified: Check Z-velocity. If dropping fast, we fell off something.
        try:
            v_z = vehicle.get_velocity().z
            triggered = [0, 1] if v_z < -1.5 else []
            with self._lock:
                self._latest = CliffReading(
                    cliff_detected = len(triggered) > 0,
                    sensors_triggered = triggered,
                    timestamp = time.time()
                )
        except Exception:
            pass

    def _start_mock(self):
        self.mode = "mock"
        self._running = True
        t = threading.Thread(target=self._mock_loop, daemon=True)
        t.start()
        
    def _mock_loop(self):
        import math, random
        t0 = time.time()
        while self._running:
            t = time.time() - t0
            # Manhole simulation! Trigger briefly every 40 seconds
            if 39.0 < (t % 40.0) < 39.5:
                triggered = [0] if random.random() > 0.5 else [1]
            else:
                triggered = []
                
            with self._lock:
                self._latest = CliffReading(
                    cliff_detected = len(triggered) > 0,
                    sensors_triggered = triggered,
                    timestamp = time.time()
                )
            time.sleep(0.05)

    def get(self) -> CliffReading:
        with self._lock:
            return self._latest
            
    def stop(self):
        self._running = False
        if hasattr(self, "_gpio"):
            self._gpio.cleanup()
