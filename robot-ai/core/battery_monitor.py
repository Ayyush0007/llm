"""
battery_monitor.py — Battery / Power Monitor for Bento Robot.

Why critical for Indian roads:
  • Prevent sudden power death mid-road (very dangerous in traffic)
  • Monitor motor current spikes (motors straining in potholes/mud)
  • Safely power down Jetson to prevent SD card corruption

Hardware: INA219 or INA3221 voltage/current sensor via I2C (₹350)
CARLA sim: Simulates battery drain over time based on speed + motors

Install:  pip install pi-ina219
Wiring:
  SDA → Jetson Pin 3 (I2C-1)
  SCL → Jetson Pin 5 (I2C-1)
  VIN+, VIN- → Series with battery output
"""

import time
import threading
from dataclasses import dataclass
from enum import Enum, auto


class BatteryState(Enum):
    HEALTHY  = auto()   # > 20%
    LOW      = auto()   # 10%-20% (Robot should head home / beep)
    CRITICAL = auto()   # < 10%   (Robot must stop safely)


@dataclass
class BatteryReading:
    voltage_v:    float = 12.0
    current_a:    float = 0.0
    power_w:      float = 0.0
    percent:      float = 100.0
    state:        BatteryState = BatteryState.HEALTHY
    timestamp:    float = 0.0
    
    # Motor stall detection
    motor_stall:  bool  = False   # True if current > max safe limit


class BatteryMonitor:
    """
    Monitors system voltage, current drain, and remaining capacity.
    """
    
    MAX_VOLTAGE = 12.6  # 3S LiPo fully charged
    MIN_VOLTAGE = 9.6   # 3S LiPo empty cut-off
    STALL_CURRENT_A = 15.0  # Motors drawing too much current (stuck in pothole)

    def __init__(self, mode: str = "mock"):
        """
        mode: "ina219" → Real INA219 I2C sensor
              "carla"  → Simulated drain linked to CARLA vehicle
              "mock"   → Simple time-based drain
        """
        self.mode = mode
        self._latest = BatteryReading(timestamp=time.time())
        self._lock   = threading.Lock()
        self._running = False
        self._sensor = None
        
        # For simulation
        self._sim_capacity_ah = 5.0   # 5000 mAh battery
        self._sim_drawn_ah    = 0.0

        if mode == "ina219":
            self._init_ina219()
        else:
            self._start_mock()

        print(f"✅ Battery monitor ready (mode={mode})")

    def _init_ina219(self):
        try:
            from ina219 import INA219
            # 0.1 ohm shunt, max 3.2A (Note: need external shunt for high motor currents, 
            # but standard INA219 module is good for Jetson power monitoring)
            self._sensor = INA219(shunt_ohms=0.1, max_expected_amps=3.1)
            self._sensor.configure()
            self._running = True
            t = threading.Thread(target=self._hw_loop, daemon=True)
            t.start()
            print("✅ INA219 battery sensor initialised")
        except ImportError:
            print("⚠️  pi-ina219 not installed — using mock battery")
            self._start_mock()
        except Exception as e:
            print(f"⚠️  INA219 init failed: {e} — using mock battery")
            self._start_mock()

    def _hw_loop(self):
        while self._running:
            try:
                v = self._sensor.voltage()
                i = self._sensor.current() / 1000.0  # A
                p = self._sensor.power() / 1000.0    # W
                self._update(v, i, p)
            except Exception:
                pass
            time.sleep(1.0)  # 1 Hz

    def update_from_carla(self, vehicle):
        """Simulate battery drain based on vehicle speed/throttle."""
        try:
            import math
            v = vehicle.get_velocity()
            spd = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            
            # Base Jetson power (1.5A) + Motor power proportional to speed
            current = 1.5 + (spd * 0.5)
            self._sim_drawn_ah += (current * 0.05) / 3600.0  # assume 20Hz tick
            
            pct = max(0.0, 1.0 - (self._sim_drawn_ah / self._sim_capacity_ah))
            volts = self.MIN_VOLTAGE + (self.MAX_VOLTAGE - self.MIN_VOLTAGE) * pct
            power = volts * current
            
            self._update(volts, current, power)
        except Exception:
            pass

    def _start_mock(self):
        self.mode = "mock"
        self._running = True
        t = threading.Thread(target=self._mock_loop, daemon=True)
        t.start()
        
    def _mock_loop(self):
        t0 = time.time()
        while self._running:
            t = time.time() - t0
            # Drain battery linearly over 1 hour (for testing, speed it up to 10 mins)
            pct = max(0.0, 1.0 - (t / 600.0))
            v = self.MIN_VOLTAGE + (self.MAX_VOLTAGE - self.MIN_VOLTAGE) * pct
            i = 2.0  # 2 Amps draw
            p = v * i
            self._update(v, i, p)
            time.sleep(1.0)

    def _update(self, vol: float, cur: float, pwr: float):
        # Calculate percentage based on voltage curve (linear approximation)
        pct = (vol - self.MIN_VOLTAGE) / (self.MAX_VOLTAGE - self.MIN_VOLTAGE)
        pct = max(0.0, min(1.0, pct)) * 100.0
        
        if pct <= 10.0:
            state = BatteryState.CRITICAL
        elif pct <= 20.0:
            state = BatteryState.LOW
        else:
            state = BatteryState.HEALTHY
            
        stall = cur > self.STALL_CURRENT_A
        
        with self._lock:
            self._latest = BatteryReading(
                voltage_v = round(vol, 2),
                current_a = round(cur, 2),
                power_w   = round(pwr, 2),
                percent   = round(pct, 1),
                state     = state,
                motor_stall = stall,
                timestamp = time.time()
            )

    def get(self) -> BatteryReading:
        with self._lock:
            return self._latest
            
    def stop(self):
        self._running = False
