"""
camera_system.py — Multi-camera manager for the Bento Robot.

Manages up to 4 cameras simultaneously in background threads:
  front  → primary navigation camera (used by Vision AI + Depth AI)
  left   → side obstacle detection
  right  → side obstacle detection
  rear   → reversing / parking safety

PHASE 1 (CARLA):  Each camera = one CARLA RGB sensor actor.
PHASE 2 (Hardware): Each camera = one USB/CSI camera via cv2.VideoCapture.

Camera index / V4L2 device IDs (typical Jetson setup):
  front  → /dev/video0  (or CARLA: front RGB)
  left   → /dev/video2
  right  → /dev/video4
  rear   → /dev/video6
"""

import cv2
import numpy as np
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class CameraFrame:
    """A single frame from one camera."""
    name:      str
    frame:     Optional[np.ndarray] = None
    timestamp: float = 0.0
    ok:        bool  = False


class CameraChannel:
    """
    One background-threaded camera stream.
    Wraps cv2.VideoCapture — works for USB, CSI, RTSP.
    """

    def __init__(self, name: str, source: int | str, width=640, height=480):
        self.name    = name
        self._source = source
        self._cap    = None
        self._latest = CameraFrame(name=name)
        self._lock   = threading.Lock()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._running = False
        self._width  = width
        self._height = height

    def start(self):
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            print(f"⚠️  Camera '{self.name}' not available at {self._source}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._running = True
        self._thread.start()
        print(f"✅ Camera '{self.name}' started ({self._width}×{self._height})")
        return True

    def _capture_loop(self):
        while self._running:
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    with self._lock:
                        self._latest = CameraFrame(
                            name      = self.name,
                            frame     = frame,
                            timestamp = time.time(),
                            ok        = True,
                        )
            time.sleep(0.005)   # ~200 Hz read attempt, capped by camera FPS

    def get_frame(self) -> CameraFrame:
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()


class CameraSystem:
    """
    Multi-camera manager.
    Provides the latest frame from each camera direction.

    PHASE 1 (CARLA): Inject frames manually via inject_carla_frame().
    PHASE 2 (Hardware): Call start_all(), then get_frame("front") etc.
    """

    DIRECTIONS = ["front", "left", "right", "rear"]

    def __init__(
        self,
        mode      : str = "carla",     # "carla" | "usb" | "mock"
        width     : int = 640,
        height    : int = 480,
        usb_indices: dict = None,       # {"front": 0, "left": 2, "right": 4, "rear": 6}
    ):
        self.mode   = mode
        self._width = width
        self._height = height

        # Maps direction → CameraChannel (hardware mode only)
        self._channels: Dict[str, CameraChannel] = {}

        # Maps direction → latest frame (all modes)
        self._frames: Dict[str, CameraFrame] = {
            d: CameraFrame(name=d) for d in self.DIRECTIONS
        }
        self._lock = threading.Lock()

        if mode == "usb":
            indices = usb_indices or {"front": 0, "left": 2, "right": 4}
            for direction, idx in indices.items():
                ch = CameraChannel(direction, idx, width, height)
                self._channels[direction] = ch

    def start_all(self):
        """Start all hardware cameras. Only needed in 'usb' mode."""
        if self.mode != "usb":
            return
        for ch in self._channels.values():
            ch.start()

    def inject_carla_frame(self, direction: str, frame: np.ndarray):
        """
        Inject a frame from a CARLA camera sensor.
        Called from the CARLA sensor callback thread.
        """
        with self._lock:
            self._frames[direction] = CameraFrame(
                name      = direction,
                frame     = frame,
                timestamp = time.time(),
                ok        = True,
            )

    def get_frame(self, direction: str = "front") -> CameraFrame:
        """Get the latest frame from a camera direction."""
        if self.mode == "usb" and direction in self._channels:
            return self._channels[direction].get_frame()
        with self._lock:
            return self._frames.get(direction, CameraFrame(name=direction))

    def get_all_frames(self) -> Dict[str, CameraFrame]:
        """Get frames from all directions at once."""
        return {d: self.get_frame(d) for d in self.DIRECTIONS}

    def make_grid(self, size: int = 320) -> np.ndarray:
        """
        Compose a 2×2 grid of all camera views for the dashboard.
        Directions: TL=front, TR=right, BL=left, BR=rear.
        """
        def get_tile(direction):
            cf = self.get_frame(direction)
            if cf.ok and cf.frame is not None:
                return cv2.resize(cf.frame, (size, size))
            # placeholder tile
            tile = np.zeros((size, size, 3), dtype=np.uint8)
            cv2.putText(tile, direction.upper(), (size//4, size//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
            return tile

        top    = np.hstack([get_tile("front"), get_tile("right")])
        bottom = np.hstack([get_tile("left"),  get_tile("rear")])
        grid   = np.vstack([top, bottom])

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        for label, pos in [("FRONT",(10,20)), ("RIGHT",(size+10,20)),
                            ("LEFT", (10,size+20)), ("REAR",(size+10,size+20))]:
            cv2.putText(grid, label, pos, font, 0.55, (200, 200, 200), 1)

        return grid

    def stop_all(self):
        for ch in self._channels.values():
            ch.stop()
        print("📷 All cameras stopped")
