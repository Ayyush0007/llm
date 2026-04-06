"""
sensor_fusion.py — Fuses Vision AI + Depth AI into a unified WorldModel.

The WorldModel is a single snapshot of "what the robot sees right now":
  - What objects are detected (class, confidence, bounding box)
  - How far away each zone (left/center/right) is
  - Danger level per zone (0.0 = safe, 1.0 = emergency stop)
  - Overall scene risk score
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np
import cv2


@dataclass
class Detection:
    cls: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized)


@dataclass
class WorldModel:
    """Fused snapshot of the robot's perception at one instant."""
    # Raw inputs
    detections: List[Detection] = field(default_factory=list)
    depth_left: float = 1.0     # 0=near, 1=far
    depth_center: float = 1.0
    depth_right: float = 1.0

    # Derived danger scores (0.0=safe, 1.0=critical)
    danger_left: float = 0.0
    danger_center: float = 0.0
    danger_right: float = 0.0

    # High-level flags
    emergency_stop: bool = False   # person/cow/animal directly ahead
    pothole_ahead: bool = False
    clear_to_go: bool = True

    # Recommended direction: "left", "center", "right", "stop"
    best_path: str = "center"

    # Debug info
    frame_id: int = 0
    timestamp: float = 0.0


# Classes that trigger full emergency stop
EMERGENCY_CLASSES = {"person", "cow", "dog", "child", "auto_rickshaw"}

# Classes that trigger slow-down
CAUTION_CLASSES = {"bicycle", "motorcycle", "pothole", "speed_bump", "pedestrian_crossing"}

# Depth thresholds (lower value = closer)
DEPTH_STOP  = 0.20   # < 20% → STOP
DEPTH_SLOW  = 0.45   # < 45% → SLOW DOWN
DEPTH_CLEAR = 0.60   # > 60% → all clear


class SensorFusion:
    """
    Fuses raw detections and depth maps into a single WorldModel.
    """

    def __init__(self):
        self._frame_id = 0

    def fuse(
        self,
        frame: np.ndarray,
        detections: List[dict],         # from YOLO: [{cls, conf, bbox}, ...]
        depth_map: np.ndarray,          # from DepthAI.estimate()
        depth_ai,                       # DepthAI instance for zone queries
        timestamp: float = 0.0,
    ) -> WorldModel:

        self._frame_id += 1
        model = WorldModel(frame_id=self._frame_id, timestamp=timestamp)

        # ── 1. Parse detections ────────────────────────────────────
        model.detections = [
            Detection(
                cls=d["cls"],
                confidence=d["conf"],
                bbox=tuple(d["bbox"])
            )
            for d in detections
        ]

        detected_classes = {d.cls for d in model.detections}

        # ── 2. Depth per zone ──────────────────────────────────────
        model.depth_left   = depth_ai.get_obstacle_distance(depth_map, "left")
        model.depth_center = depth_ai.get_obstacle_distance(depth_map, "center")
        model.depth_right  = depth_ai.get_obstacle_distance(depth_map, "right")

        # ── 3. Danger scores ───────────────────────────────────────
        model.danger_left   = self._depth_to_danger(model.depth_left)
        model.danger_center = self._depth_to_danger(model.depth_center)
        model.danger_right  = self._depth_to_danger(model.depth_right)

        # Boost danger_center if emergency object detected
        if detected_classes & EMERGENCY_CLASSES:
            model.danger_center = max(model.danger_center, 0.9)
            model.emergency_stop = True

        # ── 4. Flags ────────────────────────────────────────────────
        model.pothole_ahead = "pothole" in detected_classes
        model.clear_to_go   = model.danger_center < DEPTH_SLOW and not model.emergency_stop

        # ── 5. Best path decision ──────────────────────────────────
        model.best_path = self._choose_path(model)

        return model

    def _depth_to_danger(self, depth: float) -> float:
        """Maps a depth [0,1] to danger [0,1]. Lower depth = higher danger."""
        if depth < DEPTH_STOP:
            return 1.0
        elif depth < DEPTH_SLOW:
            # Linear interpolation between STOP and SLOW thresholds
            return 1.0 - (depth - DEPTH_STOP) / (DEPTH_SLOW - DEPTH_STOP) * 0.5
        elif depth < DEPTH_CLEAR:
            return 0.3
        else:
            return 0.0

    def _choose_path(self, model: WorldModel) -> str:
        """Pick the least-dangerous direction."""
        if model.emergency_stop or model.depth_center < DEPTH_STOP:
            return "stop"

        dangers = {
            "left"  : model.danger_left,
            "center": model.danger_center,
            "right" : model.danger_right,
        }
        return min(dangers, key=dangers.get)
