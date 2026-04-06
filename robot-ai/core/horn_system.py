"""
horn_system.py — Audio alert / horn system for the Bento Robot.

Plays different horn patterns based on the robot's situation:
  • SHORT beep  → acknowledged a known person
  • DOUBLE beep → obstacle detected, slowing down
  • LONG beep   → emergency stop (cow/person directly ahead)
  • ALARM       → unknown person / intruder detected
  • STARTUP     → robot booted and ready
  • SHUTDOWN    → robot powering down

PHASE 1 (CARLA / Mac): Uses system sound (beep / pygame)
PHASE 2 (Jetson):      Drives a piezo buzzer via GPIO PWM

Hardware wiring (Jetson):
  Piezo buzzer (+) → GPIO Pin 32 (PWM-capable)
  Piezo buzzer (-) → GND

Install:
  pip install pygame
"""

import time
import threading
import os
from enum import Enum, auto


class HornPattern(Enum):
    STARTUP   = auto()   # Boot jingle
    SHUTDOWN  = auto()   # Power-off tone
    BEEP      = auto()   # Short single beep
    DOUBLE    = auto()   # Two short beeps
    LONG      = auto()   # Long warning beep
    ALARM     = auto()   # Rapid alarm (intruder / emergency)


# Buzzer patterns: list of (frequency_hz, duration_sec, pause_sec)
PATTERNS = {
    HornPattern.STARTUP  : [(800, 0.08, 0.05), (1000, 0.08, 0.05), (1200, 0.12, 0.0)],
    HornPattern.SHUTDOWN : [(1200, 0.08, 0.05), (800, 0.2, 0.0)],
    HornPattern.BEEP     : [(900,  0.10, 0.0)],
    HornPattern.DOUBLE   : [(900,  0.08, 0.08), (900, 0.08, 0.0)],
    HornPattern.LONG     : [(700,  0.60, 0.0)],
    HornPattern.ALARM    : [(1100, 0.07, 0.05)] * 6,
}


class HornSystem:
    """
    Non-blocking horn/buzzer controller.
    Plays patterns in a background thread so the main loop isn't blocked.
    """

    def __init__(self, mode: str = "pygame", gpio_pin: int = 32):
        """
        mode: "pygame"  → software audio via pygame (works on Mac/Linux)
              "gpio"    → real piezo buzzer on Jetson GPIO
              "mock"    → print to console only (no sound)
        """
        self.mode      = mode
        self._pin      = gpio_pin
        self._pygame   = None
        self._pwm      = None
        self._busy     = False
        self._lock     = threading.Lock()

        if mode == "pygame":
            self._init_pygame()
        elif mode == "gpio":
            self._init_gpio()
        else:
            print("📢 Horn system: console-only mode")

    def _init_pygame(self):
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=256)
            self._pygame = pygame
            print("✅ Horn system: pygame audio ready")
        except ImportError:
            print("⚠️  pygame not installed (pip install pygame) — using mock mode")
            self.mode = "mock"
        except Exception as e:
            print(f"⚠️  pygame audio init failed: {e} — using mock mode")
            self.mode = "mock"

    def _init_gpio(self):
        try:
            import Jetson.GPIO as GPIO
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(self._pin, GPIO.OUT)
            self._pwm = GPIO.PWM(self._pin, 1000)
            self._pwm.start(0)
            print(f"✅ Horn system: GPIO buzzer on pin {self._pin}")
        except ImportError:
            print("⚠️  Jetson.GPIO not available — using mock mode")
            self.mode = "mock"

    # ── Public API ────────────────────────────────────────────────

    def play(self, pattern: HornPattern, block: bool = False):
        """Play a horn pattern. Non-blocking by default."""
        if self._busy:
            return   # Don't interrupt an ongoing pattern

        if block:
            self._play_pattern(pattern)
        else:
            t = threading.Thread(target=self._play_pattern, args=(pattern,), daemon=True)
            t.start()

    def beep(self):   self.play(HornPattern.BEEP)
    def double(self): self.play(HornPattern.DOUBLE)
    def alert(self):  self.play(HornPattern.LONG)
    def alarm(self):  self.play(HornPattern.ALARM)

    # ── Pattern Execution ─────────────────────────────────────────

    def _play_pattern(self, pattern: HornPattern):
        with self._lock:
            self._busy = True
            steps = PATTERNS[pattern]

            for freq, duration, pause in steps:
                if self.mode == "pygame":
                    self._play_tone_pygame(freq, duration)
                elif self.mode == "gpio":
                    self._play_tone_gpio(freq, duration)
                else:
                    print(f"\n📢 HORN [{pattern.name}] {freq}Hz {duration}s")

                if pause > 0:
                    time.sleep(pause)

            self._busy = False

    def _play_tone_pygame(self, freq: int, duration: float):
        """Synthesise a sine-wave tone via pygame."""
        import numpy as np
        sample_rate = 44100
        n_samples   = int(sample_rate * duration)
        t           = np.linspace(0, duration, n_samples, endpoint=False)
        wave        = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        sound       = self._pygame.sndarray.make_sound(wave)
        sound.play()
        time.sleep(duration)
        sound.stop()

    def _play_tone_gpio(self, freq: int, duration: float):
        """Drive the piezo buzzer at `freq` Hz for `duration` seconds."""
        if self._pwm:
            self._pwm.ChangeFrequency(freq)
            self._pwm.ChangeDutyCycle(50)
            time.sleep(duration)
            self._pwm.ChangeDutyCycle(0)

    # ── Auto-trigger helpers (called from self_drive.py) ──────────

    def handle_world_events(self, world, prev_state, curr_state):
        """
        Auto-play the correct horn based on state transitions.
        Call this once per tick from the main self-drive loop.
        """
        from core.state_machine import DriveState

        # Entering emergency stop
        if curr_state == DriveState.STOP and prev_state != DriveState.STOP:
            self.alert()

        # Starting to avoid
        elif curr_state in (DriveState.AVOID_L, DriveState.AVOID_R) \
                and prev_state == DriveState.CRUISE:
            self.double()

        # Detected a person/obstacle
        elif world.emergency_stop:
            self.alarm()

    def shutdown(self):
        self.play(HornPattern.SHUTDOWN, block=True)
        if self.mode == "gpio" and self._pwm:
            self._pwm.stop()
            import Jetson.GPIO as GPIO
            GPIO.cleanup()
