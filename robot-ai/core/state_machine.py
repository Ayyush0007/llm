"""
state_machine.py — Finite State Machine for self-driving behavior.

States:
  IDLE      → Robot is stationary, waiting to start
  CRUISE    → Moving forward at target speed, road is clear
  SLOW      → Obstacle ahead, reducing speed
  STOP      → Full stop (emergency or obstruction)
  AVOID_L   → Steering left to go around obstacle on right
  AVOID_R   → Steering right to go around obstacle on left

Transitions are driven by the WorldModel produced by SensorFusion.
"""

from enum import Enum, auto
from dataclasses import dataclass
import time


class DriveState(Enum):
    IDLE     = auto()
    CRUISE   = auto()
    SLOW     = auto()
    STOP     = auto()
    AVOID_L  = auto()   # steer left
    AVOID_R  = auto()   # steer right


@dataclass
class DriveCommand:
    """Output of the state machine — sent to motors."""
    throttle: float = 0.0    # 0.0 to 1.0
    steer:    float = 0.0    # -1.0 (left) to +1.0 (right)
    brake:    float = 0.0    # 0.0 to 1.0
    state:    DriveState = DriveState.IDLE
    reason:   str = ""


# Speed targets per state
SPEED_MAP = {
    DriveState.CRUISE  : 0.70,
    DriveState.SLOW    : 0.35,
    DriveState.STOP    : 0.00,
    DriveState.AVOID_L : 0.40,
    DriveState.AVOID_R : 0.40,
    DriveState.IDLE    : 0.00,
}

STEER_AVOID = 0.45   # steering angle during avoidance


class DrivingStateMachine:
    """
    Takes a WorldModel each tick and returns a DriveCommand.
    Applies hysteresis to prevent jittery state changes.
    """

    def __init__(self):
        self.state = DriveState.IDLE
        self._state_entry_time = time.time()
        self._stop_hold_secs   = 1.5   # minimum time in STOP before resuming
        self._avoid_hold_secs  = 2.0   # minimum time in AVOID before re-assessing
        self.started = False

    # ── Public API ────────────────────────────────────────────────

    def start(self):
        """Call once to begin driving."""
        self._transition(DriveState.CRUISE, "System started")
        self.started = True

    def tick(self, world) -> DriveCommand:
        """
        Update state based on current world model.
        Returns the DriveCommand to execute.
        """
        if not self.started:
            return DriveCommand(state=DriveState.IDLE, reason="Not started")

        new_state, reason = self._evaluate(world)

        # Only transition if new state is different
        if new_state != self.state:
            # Hysteresis: don't leave STOP or AVOID too quickly
            if self.state in (DriveState.STOP, DriveState.AVOID_L, DriveState.AVOID_R):
                hold = self._stop_hold_secs if self.state == DriveState.STOP else self._avoid_hold_secs
                if time.time() - self._state_entry_time < hold:
                    new_state = self.state   # stay
                    reason    = f"Holding {self.state.name} (hysteresis)"
                else:
                    self._transition(new_state, reason)
            else:
                self._transition(new_state, reason)

        return self._build_command(world)

    # ── Private ───────────────────────────────────────────────────

    def _evaluate(self, world):
        """Determine what state we should be in given the world model."""

        # Emergency override — always wins
        if world.emergency_stop:
            return DriveState.STOP, f"EMERGENCY: {[d.cls for d in world.detections]}"

        # Center totally blocked
        if world.depth_center < 0.20:
            return DriveState.STOP, f"Center blocked (depth={world.depth_center:.2f})"

        # Pothole — slow down
        if world.pothole_ahead and world.depth_center < 0.50:
            return DriveState.SLOW, "Pothole ahead"

        # Based on best path from sensor fusion
        if world.best_path == "stop":
            return DriveState.STOP, "No clear path"

        elif world.best_path == "left":
            return DriveState.AVOID_L, f"Obstacle RIGHT (depth_r={world.depth_right:.2f})"

        elif world.best_path == "right":
            return DriveState.AVOID_R, f"Obstacle LEFT (depth_l={world.depth_left:.2f})"

        # Center clear but moderately close
        elif world.depth_center < 0.45:
            return DriveState.SLOW, f"Caution (depth_c={world.depth_center:.2f})"

        else:
            return DriveState.CRUISE, "Road clear"

    def _build_command(self, world) -> DriveCommand:
        speed = SPEED_MAP[self.state]

        steer = 0.0
        brake = 0.0

        if self.state == DriveState.STOP:
            brake = 1.0
            speed = 0.0

        elif self.state == DriveState.AVOID_L:
            steer = -STEER_AVOID   # negative = left

        elif self.state == DriveState.AVOID_R:
            steer = +STEER_AVOID   # positive = right

        elif self.state == DriveState.CRUISE:
            # Gentle center-tracking: nudge away from nearer wall
            if world.depth_left < world.depth_right - 0.1:
                steer = +0.15   # too close left, nudge right
            elif world.depth_right < world.depth_left - 0.1:
                steer = -0.15   # too close right, nudge left

        return DriveCommand(
            throttle=speed,
            steer=steer,
            brake=brake,
            state=self.state,
            reason=self._last_reason,
        )

    def _transition(self, new_state: DriveState, reason: str):
        self.state            = new_state
        self._state_entry_time = time.time()
        self._last_reason      = reason

    @property
    def _last_reason(self):
        return getattr(self, "__last_reason", "")

    @_last_reason.setter
    def _last_reason(self, v):
        self.__last_reason = v
