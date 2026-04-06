"""
gps_sensor.py — GPS/GNSS sensor for the Bento Robot.

Hardware: u-blox NEO-M8N or BN-880 GPS module via UART on Jetson
CARLA sim: Uses vehicle transform (lat/lon from OpenDRIVE map)

Key uses for Indian roads:
  • Real-time location on Jalgaon city map
  • Route progress tracking
  • Google Maps / OSM API integration for road data
  • Speed-over-ground validation

Install: pip install pyserial pynmea2 requests

Wiring (Jetson UART):
  TX → Jetson RX (Pin 10)
  RX → Jetson TX (Pin 8)
  VCC → 3.3V
  GND → GND
"""

import time
import threading
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPSFix:
    """One GPS position fix."""
    lat:       float = 0.0
    lon:       float = 0.0
    altitude:  float = 0.0
    speed_kmh: float = 0.0   # from GPS Doppler
    heading:   float = 0.0   # degrees from north
    accuracy:  float = 99.0  # horizontal accuracy in metres
    fix_valid: bool  = False
    timestamp: float = 0.0

    # Derived
    city:      str = ""
    road_name: str = ""


# Jalgaon city bounding box — used to detect "home ground"
JALGAON_LAT = (20.85, 21.10)
JALGAON_LON = (75.40, 75.75)


def is_in_jalgaon(lat: float, lon: float) -> bool:
    return (JALGAON_LAT[0] <= lat <= JALGAON_LAT[1] and
            JALGAON_LON[0] <= lon <= JALGAON_LON[1])


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distance between two GPS coordinates in metres."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class GPSSensor:
    """
    GPS sensor manager.
    Supports real UART GPS, CARLA simulation, and mock mode.
    """

    def __init__(self, mode: str = "mock", port: str = "/dev/ttyUSB1",
                 baud: int = 9600, osm_lookup: bool = False):
        """
        mode: "uart"   → real NMEA GPS via serial port
              "carla"  → position from CARLA vehicle transform
              "mock"   → static Jalgaon coordinates (for testing)
        osm_lookup: If True, reverse-geocode to get road name (needs internet)
        """
        self.mode       = mode
        self._osm       = osm_lookup
        self._latest    = GPSFix(timestamp=time.time())
        self._lock      = threading.Lock()
        self._running   = False
        self._serial    = None

        if mode == "uart":
            self._init_uart(port, baud)
        elif mode == "mock":
            self._start_mock()

        print(f"✅ GPS sensor ready (mode={mode})")

    def _init_uart(self, port, baud):
        try:
            import serial
            import pynmea2
            self._serial  = serial.Serial(port, baud, timeout=1)
            self._pynmea2 = pynmea2
            self._running = True
            t = threading.Thread(target=self._uart_read_loop, daemon=True)
            t.start()
            print(f"✅ GPS UART opened on {port} @ {baud} baud")
        except ImportError:
            print("⚠️  pyserial/pynmea2 not installed — pip install pyserial pynmea2")
            self._start_mock()
        except Exception as e:
            print(f"⚠️  GPS UART failed: {e} — using mock")
            self._start_mock()

    def _uart_read_loop(self):
        while self._running:
            try:
                line = self._serial.readline().decode("ascii", errors="replace").strip()
                msg  = self._pynmea2.parse(line)

                if msg.sentence_type in ("GGA", "RMC"):
                    fix = GPSFix(
                        lat       = msg.latitude  if hasattr(msg, "latitude")  else 0.0,
                        lon       = msg.longitude if hasattr(msg, "longitude") else 0.0,
                        speed_kmh = getattr(msg, "spd_over_grnd", 0.0) * 1.852,
                        fix_valid = True,
                        timestamp = time.time(),
                    )
                    if self._osm:
                        fix.road_name = self._reverse_geocode(fix.lat, fix.lon)
                    with self._lock:
                        self._latest = fix
            except Exception:
                pass

    def _start_mock(self):
        """Simulates movement within Jalgaon city."""
        self.mode    = "mock"
        self._running = True
        t = threading.Thread(target=self._mock_loop, daemon=True)
        t.start()

    def _mock_loop(self):
        t0   = time.time()
        lat0 = 20.9350   # Jalgaon city centre
        lon0 = 75.5607
        while self._running:
            t = time.time() - t0
            fix = GPSFix(
                lat       = lat0 + 0.001 * math.sin(t * 0.05),
                lon       = lon0 + 0.001 * math.cos(t * 0.05),
                speed_kmh = 25 + 10 * math.sin(t * 0.1),
                heading   = (t * 5) % 360,
                accuracy  = 2.5,
                fix_valid = True,
                timestamp = time.time(),
            )
            fix.city = "Jalgaon, MH" if is_in_jalgaon(fix.lat, fix.lon) else "Unknown"
            with self._lock:
                self._latest = fix
            time.sleep(0.5)

    def update_from_carla(self, vehicle):
        """Pull GPS-equivalent position from CARLA's world transform."""
        try:
            loc = vehicle.get_transform().location
            # CARLA doesn't have real GPS, map X/Y to mock lat/lon
            fix = GPSFix(
                lat       = 20.9350 + loc.y / 100_000,
                lon       = 75.5607 + loc.x / 100_000,
                altitude  = loc.z,
                fix_valid = True,
                timestamp = time.time(),
            )
            v = vehicle.get_velocity()
            fix.speed_kmh = math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6
            with self._lock:
                self._latest = fix
        except Exception:
            pass

    def _reverse_geocode(self, lat: float, lon: float) -> str:
        """Get road name from OSM Nominatim (free, no API key)."""
        try:
            import requests
            url  = "https://nominatim.openstreetmap.org/reverse"
            r    = requests.get(url, params={"lat": lat, "lon": lon,
                                             "format": "json"},
                                headers={"User-Agent": "BentoRobotAI/1.0"},
                                timeout=2)
            data = r.json()
            addr = data.get("address", {})
            return addr.get("road", addr.get("suburb", "unknown road"))
        except Exception:
            return ""

    def get(self) -> GPSFix:
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        if self._serial:
            self._serial.close()
