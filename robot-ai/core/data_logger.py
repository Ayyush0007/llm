"""
data_logger.py — Logs every robot decision to a CSV file for later analysis.

Useful for:
  - Reviewing why the robot made a bad decision
  - Building a dataset of edge cases for re-training
  - Plotting reward / state distribution over time

Output CSV columns:
  frame_id, timestamp, state, throttle, steer, brake,
  depth_l, depth_c, depth_r, danger_c, emergency,
  pothole, detections, reason
"""

import csv
import os
import time
from pathlib import Path
from core.state_machine import DriveCommand
from core.sensor_fusion import WorldModel


class DataLogger:
    """Records every tick to a CSV log file."""

    COLUMNS = [
        "frame_id", "wall_time",
        "state", "throttle", "steer", "brake",
        "depth_l", "depth_c", "depth_r",
        "danger_l", "danger_c", "danger_r",
        "emergency_stop", "pothole_ahead",
        "best_path", "detections", "reason",
    ]

    def __init__(self, log_dir: str = "logs/self_drive"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(log_dir, f"run_{ts}.csv")
        self._file = open(self._path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()
        print(f"📝 Logging to {self._path}")

    def log(self, world: WorldModel, cmd: DriveCommand):
        det_str = "|".join(f"{d.cls}:{d.confidence:.2f}" for d in world.detections)
        self._writer.writerow({
            "frame_id"     : world.frame_id,
            "wall_time"    : f"{world.timestamp:.3f}",
            "state"        : cmd.state.name,
            "throttle"     : f"{cmd.throttle:.2f}",
            "steer"        : f"{cmd.steer:.2f}",
            "brake"        : f"{cmd.brake:.2f}",
            "depth_l"      : f"{world.depth_left:.3f}",
            "depth_c"      : f"{world.depth_center:.3f}",
            "depth_r"      : f"{world.depth_right:.3f}",
            "danger_l"     : f"{world.danger_left:.3f}",
            "danger_c"     : f"{world.danger_center:.3f}",
            "danger_r"     : f"{world.danger_right:.3f}",
            "emergency_stop": world.emergency_stop,
            "pothole_ahead" : world.pothole_ahead,
            "best_path"    : world.best_path,
            "detections"   : det_str,
            "reason"       : cmd.reason,
        })
        self._file.flush()

    def close(self):
        self._file.close()
        print(f"✅ Log saved to {self._path}")
