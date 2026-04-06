"""
lcd_display.py — LCD / OLED display controller for the Bento Robot.

Supports two display types:
  • HD44780 16×2 LCD   → via I2C backpack (PCF8574) on Jetson GPIO
  • SSD1306 128×64 OLED → via I2C on address 0x3C

Shows robot state, speed, danger level, and detection alerts.

Hardware wiring (Jetson Orin NX):
  SDA → Pin 3 (I2C-1)
  SCL → Pin 5 (I2C-1)
  VCC → 3.3V or 5V
  GND → GND

Install deps:
  pip install adafruit-circuitpython-ssd1306 RPLCD pillow smbus2
"""

import time
import threading
from dataclasses import dataclass
from core.state_machine import DriveState


# Try hardware imports — gracefully degrade to console simulation
try:
    import board
    import busio
    from adafruit_ssd1306 import SSD1306_I2C
    from PIL import Image, ImageDraw, ImageFont
    OLED_AVAILABLE = True
except ImportError:
    OLED_AVAILABLE = False

try:
    from RPLCD.i2c import CharLCD
    LCD_AVAILABLE = True
except ImportError:
    LCD_AVAILABLE = False


# Display modes
DISPLAY_OLED = "oled"
DISPLAY_LCD  = "lcd"
DISPLAY_MOCK = "mock"   # console output, no hardware needed


class LCDDisplaySystem:
    """
    Physical display controller.
    Falls back to console print if hardware not present.
    """

    def __init__(self, mode: str = DISPLAY_MOCK,
                 i2c_address: int = 0x3C):
        self.mode = mode
        self._lock   = threading.Lock()
        self._screen = None

        if mode == DISPLAY_OLED and OLED_AVAILABLE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self._screen = SSD1306_I2C(128, 64, i2c, addr=i2c_address)
                self._screen.fill(0)
                self._screen.show()
                print("✅ OLED display initialised (128×64)")
            except Exception as e:
                print(f"⚠️  OLED init failed: {e} — falling back to console")
                self.mode = DISPLAY_MOCK

        elif mode == DISPLAY_LCD and LCD_AVAILABLE:
            try:
                self._screen = CharLCD(
                    i2c_expander="PCF8574",
                    address=i2c_address,
                    port=1,
                    cols=16, rows=2,
                    backlight_enabled=True,
                )
                self._screen.clear()
                print("✅ 16×2 LCD initialised")
            except Exception as e:
                print(f"⚠️  LCD init failed: {e} — falling back to console")
                self.mode = DISPLAY_MOCK

        else:
            if mode != DISPLAY_MOCK:
                print(f"⚠️  Display hardware libs not installed — using console mode")
            self.mode = DISPLAY_MOCK

    def update(self, state: DriveState, speed: float,
               danger: float, detections: list,
               lidar_front: float = 99.0):
        """
        Update display with current robot status.
        Called once per frame from the self-drive loop.
        """
        with self._lock:
            if self.mode == DISPLAY_OLED:
                self._render_oled(state, speed, danger, detections, lidar_front)
            elif self.mode == DISPLAY_LCD:
                self._render_lcd(state, speed, danger, detections, lidar_front)
            else:
                self._render_console(state, speed, danger, detections, lidar_front)

    # ─── OLED Renderer (128×64 pixels) ───────────────────────────
    def _render_oled(self, state, speed, danger, detections, lidar_front):
        img  = Image.new("1", (128, 64), 0)
        draw = ImageDraw.Draw(img)

        # Row 1: State + speed
        draw.text((0,  0), f"STATE: {state.name}", fill=1)
        draw.text((0, 12), f"SPD:{speed:.2f}  LIDAR:{lidar_front:.1f}m", fill=1)

        # Row 2: Danger bar
        draw.text((0, 24), "DANGER:", fill=1)
        bar_w = int(danger * 80)
        draw.rectangle([(48, 26), (128, 34)], fill=0, outline=1)
        draw.rectangle([(48, 26), (48 + bar_w, 34)], fill=1)

        # Row 3: Detections
        det_txt = ", ".join(detections[:3]) if detections else "clear"
        draw.text((0, 38), f"SEE: {det_txt[:20]}", fill=1)

        # Row 4: Bento logo
        draw.text((0, 52), "BENTO ROBOT AI v1", fill=1)

        self._screen.image(img)
        self._screen.show()

    # ─── 16×2 LCD Renderer ────────────────────────────────────────
    def _render_lcd(self, state, speed, danger, detections, lidar_front):
        state_str = state.name[:6].ljust(6)
        det_str   = detections[0][:8] if detections else "CLEAR   "
        spd_str   = f"{speed:.1f}"

        line1 = f"{state_str} S:{spd_str}"     # e.g. "CRUISE S:0.7"
        line2 = f"L:{lidar_front:.1f}m {det_str}"  # e.g. "L:1.2m COW     "

        self._screen.clear()
        self._screen.write_string(line1[:16].ljust(16))
        self._screen.cursor_pos = (1, 0)
        self._screen.write_string(line2[:16].ljust(16))

    # ─── Console Fallback ──────────────────────────────────────────
    def _render_console(self, state, speed, danger, detections, lidar_front):
        # Only print when something changes (reduce spam)
        det_str = ", ".join(detections[:2]) if detections else "none"
        bar     = "█" * int(danger * 10) + "░" * (10 - int(danger * 10))
        print(
            f"\r📟 [{state.name:8s}] spd:{speed:.2f}  "
            f"lidar:{lidar_front:.1f}m  danger:[{bar}]  "
            f"see:{det_str[:20]:<20}",
            end="", flush=True,
        )

    def clear(self):
        """Blank the screen."""
        if self.mode == DISPLAY_LCD and self._screen:
            self._screen.clear()
        elif self.mode == DISPLAY_OLED and self._screen:
            self._screen.fill(0)
            self._screen.show()

    def shutdown(self):
        if self.mode == DISPLAY_LCD and self._screen:
            self._screen.clear()
            self._screen.write_string("Robot shutdown.")
        elif self.mode == DISPLAY_OLED and self._screen:
            self._screen.fill(0)
            self._screen.show()
        print("\n📟 Display cleared")
