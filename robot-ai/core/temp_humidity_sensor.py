"""
temp_humidity_sensor.py — Environment + Jetson thermals for Bento Robot.

Why critical for Indian roads:
  • Jalgaon summer temperatures easily cross 45°C
  • Jetson Orin NX will aggressively thermal-throttle without active cooling monitoring
  • Humidity (monsoon) checks for condensation inside the chassis enclosure
  • Robot can autonomously slow down to reduce internal heat generation

Hardware: BME280 Environment Sensor (I2C 0x76, ₹300)
Jetson Internal: reads `/sys/class/thermal/thermal_zone0/temp`

Install:  pip install smbus2 RPi.bme280

Wiring (BME280 via I2C-1):
  VCC → 3.3V
  GND → GND
  SDA → Pin 3
  SCL → Pin 5
"""

import time
import threading
import os
from dataclasses import dataclass
from enum import Enum, auto


class ThermalState(Enum):
    NORMAL   = auto()   # CPU < 65°C
    WARM     = auto()   # CPU 65-80°C (Throttle speed slightly)
    HOT      = auto()   # CPU > 80°C (Throttle heavily, start cooling fan)
    CRITICAL = auto()   # CPU > 90°C (Emergency stop, shutdown impending)


@dataclass
class TempHumidReading:
    env_temp_c:    float = 0.0    # Ambient external temperature (BME280)
    env_humid_p:   float = 0.0    # Ambient external humidity % (BME280)
    cpu_temp_c:    float = 0.0    # Jetson SoC temperature
    state:         ThermalState = ThermalState.NORMAL
    
    # Adaptive outputs
    speed_cap:     float = 1.0    # Soft throttle limit to prevent overheat
    timestamp:     float = 0.0


class TempHumiditySensor:
    """
    Monitors external environment (BME280) and internal CPU thermals.
    """

    def __init__(self, mode: str = "mock"):
        self.mode = mode
        self._latest = TempHumidReading(timestamp=time.time())
        self._lock   = threading.Lock()
        self._running = False
        
        # BME280 configuration
        self._port = 1
        self._address = 0x76
        self._bus = None
        self._calib = None

        if mode == "bme280":
            self._init_bme280()
        else:
            self._start_mock()

        print(f"✅ Temp/Humidity sensor ready (mode={mode})")

    def _init_bme280(self):
        try:
            import smbus2
            import bme280
            self._bus = smbus2.SMBus(self._port)
            self._calib = bme280.load_calibration_params(self._bus, self._address)
            self._running = True
            t = threading.Thread(target=self._hw_loop, daemon=True)
            t.start()
            print("✅ BME280 sensor initialised")
        except ImportError:
            print("⚠️  RPi.bme280 not installed — using mock temp")
            self._start_mock()
        except Exception as e:
            print(f"⚠️  BME280 init failed: {e} — using mock temp")
            self._start_mock()

    def _read_cpu_temp(self) -> float:
        """Reads Jetson thermal zone. Falls back to mock if not on linux."""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read().strip()) / 1000.0
        except FileNotFoundError:
            return 55.0  # Mock CPU temp

    def _hw_loop(self):
        import bme280
        while self._running:
            try:
                data = bme280.sample(self._bus, self._address, self._calib)
                env_t = data.temperature
                env_h = data.humidity
                cpu_t = self._read_cpu_temp()
                self._update(env_t, env_h, cpu_t)
            except Exception:
                cpu_t = self._read_cpu_temp()
                self._update(30.0, 50.0, cpu_t)  # fallback
            time.sleep(2.0)

    def update_from_carla(self, weather_params):
        """No temperature in CARLA weather, just mock based on sun altitude."""
        try:
            alt = getattr(weather_params, 'sun_altitude_angle', 45.0)
            env_t = 25.0 + (alt / 90.0) * 20.0  # Peak 45°C at midday
            env_h = 40.0
            cpu_t = env_t + 25.0  # Base logic: SoC runs hotter than ambient
            self._update(env_t, env_h, cpu_t)
        except Exception:
            pass

    def _start_mock(self):
        self.mode = "mock"
        self._running = True
        t = threading.Thread(target=self._mock_loop, daemon=True)
        t.start()

    def _mock_loop(self):
        import math
        t0 = time.time()
        while self._running:
            t = time.time() - t0
            # Simulating Jalgaon summer heat over 5 mins
            phase = (t % 300) / 300.0
            
            # Ambient temp swings from 30°C to 46°C
            env_t = 38.0 + 8.0 * math.sin(phase * 2 * math.pi - math.pi/2)
            env_h = 30.0
            
            # CPU temp follows ambient + some random load heat
            cpu_t = env_t + 35.0 + (5.0 * math.sin(t * 0.1))
            
            self._update(env_t, env_h, cpu_t)
            time.sleep(2.0)

    def _update(self, env_t: float, env_h: float, cpu_t: float):
        if cpu_t > 90.0:
            state = ThermalState.CRITICAL
            cap   = 0.0    # Forcibly stop the robot
        elif cpu_t > 80.0:
            state = ThermalState.HOT
            cap   = 0.4    # Max 40% speed
        elif cpu_t > 65.0:
            state = ThermalState.WARM
            cap   = 0.8    # Max 80% speed
        else:
            state = ThermalState.NORMAL
            cap   = 1.0
            
        with self._lock:
            self._latest = TempHumidReading(
                env_temp_c = round(env_t, 1),
                env_humid_p = round(env_h, 1),
                cpu_temp_c = round(cpu_t, 1),
                state      = state,
                speed_cap  = cap,
                timestamp  = time.time()
            )

    def get(self) -> TempHumidReading:
        with self._lock:
            return self._latest
            
    def stop(self):
        self._running = False
