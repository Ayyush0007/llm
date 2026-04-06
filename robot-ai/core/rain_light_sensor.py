"""
rain_light_sensor.py — Rain + Ambient Light sensor for the Bento Robot.

Why critical for Indian roads:
  • Monsoon season (June–Sept) in Jalgaon — heavy rain floods roads
  • Rain reduces camera visibility → lower confidence thresholds
  • Night driving → switch to IR-sensitive mode
  • Dust storms → similar visibility reduction to rain

Detects:
  • Rain intensity: NONE / LIGHT / HEAVY (via raindrop sensor resistivity)
  • Ambient light:  BRIGHT_DAY / OVERCAST / DUSK / NIGHT
  • Adaptive outputs: YOLO conf, speed cap, depth model size

Hardware:
  • FC-37 rain sensor → GPIO analog (via MCP3008 ADC) or digital threshold pin
  • BH1750 ambient light sensor → I2C (address 0x23)

Install:  pip install smbus2
Wiring:
  BH1750 SDA → Pin 3 (I2C-1)
  BH1750 SCL → Pin 5
  FC-37 D0   → any GPIO input pin (digital threshold)
"""

import time
import threading
import math
from dataclasses import dataclass
from enum import Enum, auto


class RainLevel(Enum):
    NONE   = auto()   # dry
    LIGHT  = auto()   # drizzle
    HEAVY  = auto()   # monsoon downpour


class LightLevel(Enum):
    BRIGHT_DAY = auto()   # > 10000 lux
    OVERCAST   = auto()   # 1000–10000 lux
    DUSK       = auto()   # 50–1000 lux
    NIGHT      = auto()   # < 50 lux


@dataclass
class EnvironmentReading:
    """Combined environment snapshot."""
    rain:        RainLevel  = RainLevel.NONE
    light:       LightLevel = LightLevel.BRIGHT_DAY
    lux:         float = 50000.0
    rain_raw:    float = 0.0    # 0.0 = dry, 1.0 = soaking wet

    # Adaptive AI parameters
    yolo_conf_scale:  float = 1.0    # multiply base YOLO confidence by this
    speed_cap:        float = 1.0    # cap max speed to this fraction
    depth_model_hint: str   = "small" # suggest depth model to use
    timestamp:        float = 0.0


class RainLightSensor:
    """
    Combined rain + ambient light sensor.
    Outputs adaptive AI parameters based on current weather conditions.
    """

    def __init__(self, mode: str = "mock",
                 rain_pin: int = 11, i2c_bus: int = 1):
        """
        mode: "gpio"  → real FC-37 + BH1750 hardware
              "mock"  → simulated day/night + monsoon cycle
        """
        self.mode   = mode
        self._latest = EnvironmentReading(timestamp=time.time())
        self._lock   = threading.Lock()
        self._running = False

        if mode == "gpio":
            self._init_gpio(rain_pin, i2c_bus)
        else:
            self._start_mock()

        print(f"✅ Rain/Light sensor ready (mode={mode})")

    def _init_gpio(self, rain_pin, i2c_bus):
        try:
            import Jetson.GPIO as GPIO
            import smbus2
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(rain_pin, GPIO.IN)
            self._rain_pin = rain_pin
            self._bus = smbus2.SMBus(i2c_bus)
            # BH1750 power on
            self._bus.write_byte(0x23, 0x01)
            self._gpio = GPIO
            self._running = True
            t = threading.Thread(target=self._hw_loop, daemon=True)
            t.start()
        except Exception as e:
            print(f"⚠️  Sensor init failed: {e} — using mock")
            self._start_mock()

    def _hw_loop(self):
        while self._running:
            try:
                # Rain (digital pin: LOW = rain detected)
                rain_digital = not self._gpio.input(self._rain_pin)
                rain_raw = 0.8 if rain_digital else 0.0

                # Light (BH1750 one-time measurement)
                data = self._bus.read_i2c_block_data(0x23, 0x20, 2)
                lux  = (data[0] << 8 | data[1]) / 1.2

                self._update(rain_raw, lux)
            except Exception:
                pass
            time.sleep(2.0)

    def _start_mock(self):
        self.mode     = "mock"
        self._running = True
        t = threading.Thread(target=self._mock_loop, daemon=True)
        t.start()

    def _mock_loop(self):
        """Simulates a full Indian day: sunrise → hot day → dusk → night + monsoon."""
        t0 = time.time()
        while self._running:
            t     = (time.time() - t0) % 86400   # 24h cycle (compressed to 5min)
            phase = (t / 300) % 1.0               # 0→1 over 5 minutes

            # Lux cycle
            lux = 50000 * math.sin(math.pi * phase) if phase > 0 else 0
            lux = max(0, lux)

            # Monsoon rain happens at phase 0.6–0.75 (afternoon shower)
            rain_raw = 0.9 if 0.6 < phase < 0.75 else 0.0

            self._update(rain_raw, lux)
            time.sleep(3.0)

    def _update(self, rain_raw: float, lux: float):
        """Convert raw values to categorized reading + AI parameters."""
        # Rain level
        if rain_raw > 0.6:
            rain = RainLevel.HEAVY
        elif rain_raw > 0.2:
            rain = RainLevel.LIGHT
        else:
            rain = RainLevel.NONE

        # Light level
        if lux > 10000:
            light = LightLevel.BRIGHT_DAY
        elif lux > 1000:
            light = LightLevel.OVERCAST
        elif lux > 50:
            light = LightLevel.DUSK
        else:
            light = LightLevel.NIGHT

        # Adaptive AI parameters
        if rain == RainLevel.HEAVY:
            conf_scale  = 0.65    # lower YOLO confidence threshold (more false pos tolerated)
            speed_cap   = 0.40    # max 40% speed in heavy rain
            depth_hint  = "base"  # use bigger depth model if available
        elif rain == RainLevel.LIGHT or light in (LightLevel.DUSK, LightLevel.NIGHT):
            conf_scale  = 0.75
            speed_cap   = 0.60
            depth_hint  = "small"
        else:
            conf_scale  = 1.0    # full confidence in good conditions
            speed_cap   = 1.0
            depth_hint  = "small"

        with self._lock:
            self._latest = EnvironmentReading(
                rain             = rain,
                light            = light,
                lux              = lux,
                rain_raw         = rain_raw,
                yolo_conf_scale  = conf_scale,
                speed_cap        = speed_cap,
                depth_model_hint = depth_hint,
                timestamp        = time.time(),
            )

    def get(self) -> EnvironmentReading:
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
