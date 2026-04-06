"""
air_quality_sensor.py — Dust / PM2.5 / Air Quality for the Bento Robot.

Why critical for Indian roads:
  • Construction dust on NH-6 / Jalgaon → severely reduces camera range
  • High AQI areas → reduce camera confidence, use LiDAR + ultrasonic more
  • Detects construction zones ahead (sudden AQI spike + camera blur)
  • Data logged for future route planning (avoid dusty roads)

Hardware options:
  Option A: MQ-135 gas sensor (I2C/analog, ₹150) ← basic air quality
  Option B: SDS011 PM2.5 (UART, ₹1,200) ← accurate PM readings
  Option C: BME688 (I2C, gas + temp + humidity, ₹900) ← best all-round

CARLA sim: Approximates dust from rain weather + particle effects.

Install:  pip install pyserial smbus2

Wiring (SDS011 via UART):
  TX → Jetson RX (Pin 10)
  RX → Jetson TX (Pin 8)  [optional: only TX needed for read]
  VCC → 5V
  GND → GND
"""

import time
import threading
import struct
from dataclasses import dataclass
from enum import Enum, auto


class VisibilityLevel(Enum):
    CLEAR       = auto()   # AQI < 50 — full camera confidence
    DUSTY       = auto()   # AQI 50–150 — reduce camera trust slightly
    VERY_DUSTY  = auto()   # AQI 150–300 — major visibility loss
    HAZARDOUS   = auto()   # AQI > 300 — near-zero visibility (construction zone)


@dataclass
class AirQualityReading:
    pm25:          float = 0.0    # µg/m³ PM2.5
    pm10:          float = 0.0    # µg/m³ PM10
    aqi:           float = 0.0    # calculated AQI index (0–500)
    visibility:    VisibilityLevel = VisibilityLevel.CLEAR
    timestamp:     float = 0.0

    # Adaptive outputs
    camera_conf_scale: float = 1.0    # multiply YOLO confidence
    lidar_weight:      float = 0.5    # how much to trust LiDAR vs camera
    speed_cap:         float = 1.0


def pm25_to_aqi(pm25: float) -> float:
    """EPA standard AQI breakpoints for PM2.5."""
    bp = [
        (0.0,  12.0,   0,   50),
        (12.1, 35.4,  51,  100),
        (35.5, 55.4, 101,  150),
        (55.5,150.4, 151,  200),
        (150.5,250.4,201, 300),
        (250.5,500.4,301, 500),
    ]
    for lo_c, hi_c, lo_i, hi_i in bp:
        if lo_c <= pm25 <= hi_c:
            return (hi_i - lo_i) / (hi_c - lo_c) * (pm25 - lo_c) + lo_i
    return 500.0


class AirQualitySensor:
    """
    Dust / PM2.5 sensor manager.
    Outputs adaptive AI parameters based on visibility conditions.
    """

    def __init__(self, mode: str = "mock",
                 port: str = "/dev/ttyUSB2", baud: int = 9600):
        self.mode    = mode
        self._latest = AirQualityReading(timestamp=time.time())
        self._lock   = threading.Lock()
        self._running = False
        self._serial = None

        if mode == "sds011":
            self._init_sds011(port, baud)
        else:
            self._start_mock()

        print(f"✅ Air quality sensor ready (mode={mode})")

    def _init_sds011(self, port, baud):
        try:
            import serial
            self._serial  = serial.Serial(port, baud, timeout=2)
            self._running = True
            t = threading.Thread(target=self._uart_loop, daemon=True)
            t.start()
            print(f"✅ SDS011 PM2.5 sensor on {port}")
        except ImportError:
            print("⚠️  pyserial not installed — using mock air quality")
            self._start_mock()
        except Exception as e:
            print(f"⚠️  SDS011 not found: {e} — using mock")
            self._start_mock()

    def _uart_loop(self):
        """SDS011 outputs 10-byte packets at 1Hz."""
        while self._running:
            try:
                raw = self._serial.read(10)
                if len(raw) == 10 and raw[0] == 0xAA and raw[9] == 0xAB:
                    pm25 = struct.unpack("<H", raw[2:4])[0] / 10.0
                    pm10 = struct.unpack("<H", raw[4:6])[0] / 10.0
                    self._update(pm25, pm10)
            except Exception:
                pass

    def _start_mock(self):
        self.mode     = "mock"
        self._running = True
        t = threading.Thread(target=self._mock_loop, daemon=True)
        t.start()

    def _mock_loop(self):
        """Simulate: clean morning → dusty afternoon → construction zone spike."""
        import math, random
        t0 = time.time()
        while self._running:
            t     = (time.time() - t0) % 300   # 5-min cycle
            phase = t / 300
            # Base PM2.5: 15µg morning, 80µg afternoon peak
            base_pm25 = 15 + 65 * math.sin(math.pi * phase)
            # Construction zone spike at phase 0.7–0.8
            if 0.70 < phase < 0.80:
                base_pm25 += random.uniform(150, 300)
            pm25  = max(0, base_pm25 + random.uniform(-5, 5))
            pm10  = pm25 * 1.8
            self._update(pm25, pm10)
            time.sleep(3.0)

    def update_from_carla(self, weather_params):
        """
        Approximate AQI from CARLA weather precipitation + dust.
        """
        try:
            # High precipitation → wash particles → low AQI
            rain    = getattr(weather_params, "precipitation", 0)
            dust    = getattr(weather_params, "dust_storm",    0)
            pm25    = max(5.0, 20 + dust * 3 - rain * 0.5)
            self._update(pm25, pm25 * 1.8)
        except Exception:
            pass

    def _update(self, pm25: float, pm10: float):
        aqi = pm25_to_aqi(pm25)

        if aqi < 50:
            vis = VisibilityLevel.CLEAR
            conf, lidar_w, cap = 1.0, 0.40, 1.0
        elif aqi < 150:
            vis = VisibilityLevel.DUSTY
            conf, lidar_w, cap = 0.80, 0.55, 0.80
        elif aqi < 300:
            vis = VisibilityLevel.VERY_DUSTY
            conf, lidar_w, cap = 0.60, 0.70, 0.55
        else:
            vis = VisibilityLevel.HAZARDOUS
            conf, lidar_w, cap = 0.40, 0.90, 0.30   # almost stop, trust LiDAR

        with self._lock:
            self._latest = AirQualityReading(
                pm25               = round(pm25, 1),
                pm10               = round(pm10, 1),
                aqi                = round(aqi, 0),
                visibility         = vis,
                camera_conf_scale  = conf,
                lidar_weight       = lidar_w,
                speed_cap          = cap,
                timestamp          = time.time(),
            )

    def get(self) -> AirQualityReading:
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        if self._serial:
            self._serial.close()
