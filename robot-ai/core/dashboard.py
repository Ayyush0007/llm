"""
dashboard.py — Live HUD that visualises what the robot sees and thinks.

Shows in one window:
  ┌─────────────────────────┬──────────────────┐
  │   RGB Camera Feed       │  Depth Map       │
  │   + YOLO detections     │  (MAGMA colour)  │
  ├─────────────────────────┴──────────────────┤
  │   State: CRUISE   Path: CENTER             │
  │   L:0.82  C:0.76  R:0.91   Speed: 0.70    │
  │   [■■■░░] Danger  [■░░░░] Steer            │
  └────────────────────────────────────────────┘
"""

import cv2
import numpy as np
from core.state_machine import DriveCommand, DriveState
from core.sensor_fusion import WorldModel

# State color map
STATE_COLORS = {
    DriveState.IDLE    : (120, 120, 120),  # grey
    DriveState.CRUISE  : (0,   200, 80),   # green
    DriveState.SLOW    : (0,   200, 220),  # cyan
    DriveState.STOP    : (0,   0,   255),  # red
    DriveState.AVOID_L : (0,   140, 255),  # orange
    DriveState.AVOID_R : (0,   140, 255),
}


class Dashboard:
    """Builds and displays a real-time HUD from WorldModel + DriveCommand."""

    def __init__(self, width: int = 1280, height: int = 480):
        self.width  = width
        self.height = height
        self._window_name = "🤖 Bento Robot — Self-Drive"
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, width, height)

    def show(
        self,
        rgb_frame: np.ndarray,
        depth_map: np.ndarray,
        world: WorldModel,
        cmd: DriveCommand,
        fps: float = 0.0,
    ) -> bool:
        """
        Render the HUD. Returns False if user pressed 'q' to quit.
        """
        h, w = rgb_frame.shape[:2]
        target_h = self.height - 120    # leave bottom panel for stats

        # ── Left panel: camera + detections ────────────────────────
        cam = cv2.resize(rgb_frame, (self.width // 2, target_h))
        cam = self._draw_detections(cam, world, (self.width // 2, target_h))
        cam = self._draw_zones(cam, world, (self.width // 2, target_h))

        # ── Right panel: depth map ──────────────────────────────────
        depth_u8     = (depth_map * 255).astype(np.uint8)
        depth_color  = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
        depth_resized = cv2.resize(depth_color, (self.width // 2, target_h))
        depth_resized = self._draw_depth_zones(depth_resized, world, (self.width // 2, target_h))

        # ── Combine top row ─────────────────────────────────────────
        top_row  = np.hstack([cam, depth_resized])

        # ── Bottom panel: stats ─────────────────────────────────────
        bottom   = self._build_stats_panel(world, cmd, fps)

        # ── Merge ───────────────────────────────────────────────────
        canvas   = np.vstack([top_row, bottom])
        cv2.imshow(self._window_name, canvas)

        return cv2.waitKey(1) & 0xFF != ord("q")

    # ── Helpers ────────────────────────────────────────────────────

    def _draw_detections(self, img, world: WorldModel, size):
        w, h = size
        font = cv2.FONT_HERSHEY_SIMPLEX
        for det in world.detections:
            x1, y1, x2, y2 = det.bbox
            # bbox may be pixel coords or normalized — handle both
            if x1 < 1.0:
                x1, x2 = int(x1 * w), int(x2 * w)
                y1, y2 = int(y1 * h), int(y2 * h)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            color = (0, 0, 220) if det.cls in {"cow","person","dog"} else (0, 180, 60)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{det.cls} {det.confidence:.2f}",
                        (x1, max(y1 - 6, 12)), font, 0.5, color, 1)
        return img

    def _draw_zones(self, img, world: WorldModel, size):
        """Overlay L / C / R zone borders on camera feed."""
        w, h = size
        third = w // 3
        alpha = 0.25

        def zone_color(danger):
            if danger > 0.7: return (0, 0, 180)
            if danger > 0.3: return (0, 160, 200)
            return (0, 180, 0)

        overlay = img.copy()
        for i, (danger, label) in enumerate([
            (world.danger_left,   "L"),
            (world.danger_center, "C"),
            (world.danger_right,  "R"),
        ]):
            x0, x1 = i * third, (i + 1) * third
            color   = zone_color(danger)
            cv2.rectangle(overlay, (x0, h//3), (x1, 2*h//3), color, -1)
            cv2.putText(img, f"{label}:{danger:.2f}",
                        (x0 + 5, h//3 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img

    def _draw_depth_zones(self, img, world: WorldModel, size):
        """Draw zone depth values on depth map."""
        w, h = size
        for i, (depth, label) in enumerate([
            (world.depth_left,   "L"),
            (world.depth_center, "C"),
            (world.depth_right,  "R"),
        ]):
            x = (i * w // 3) + 8
            text = f"{label}:{depth:.2f}"
            color = (0, 0, 255) if depth < 0.3 else (255, 255, 255)
            cv2.putText(img, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return img

    def _build_stats_panel(self, world: WorldModel, cmd: DriveCommand, fps: float):
        h_panel = 120
        panel   = np.zeros((h_panel, self.width, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        state_color = STATE_COLORS.get(cmd.state, (200, 200, 200))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Row 1: State + FPS
        cv2.putText(panel, f"STATE: {cmd.state.name}",
                    (20, 35), font, 1.0, state_color, 2)
        cv2.putText(panel, f"PATH: {world.best_path.upper()}",
                    (300, 35), font, 0.8, (200, 200, 200), 1)
        cv2.putText(panel, f"{fps:.1f} FPS",
                    (self.width - 130, 35), font, 0.8, (150, 150, 150), 1)

        # Row 2: Reason
        cv2.putText(panel, f"Reason: {cmd.reason[:80]}",
                    (20, 65), font, 0.5, (160, 160, 160), 1)

        # Row 3: Throttle / Steer bars
        def bar(label, val, x, y, max_val=1.0, color=(80, 200, 80)):
            cv2.putText(panel, label, (x, y), font, 0.5, (200,200,200), 1)
            bar_w = 150
            filled = int(abs(val) / max_val * bar_w)
            cv2.rectangle(panel, (x+60, y-12), (x+60+bar_w, y+2), (60,60,60), -1)
            cv2.rectangle(panel, (x+60, y-12), (x+60+filled, y+2), color, -1)
            cv2.putText(panel, f"{val:.2f}", (x+60+bar_w+5, y), font, 0.45, (200,200,200), 1)

        bar("THROTTLE", cmd.throttle, 20,  105, color=(0, 200, 80))
        bar("STEER",    cmd.steer,    250, 105, max_val=1.0, color=(0, 160, 255))
        bar("BRAKE",    cmd.brake,    480, 105, color=(0, 0, 220))

        # Emergency / pothole indicator
        if world.emergency_stop:
            cv2.putText(panel, "⚠ EMERGENCY STOP", (750, 70), font, 0.7, (0,0,255), 2)
        if world.pothole_ahead:
            cv2.putText(panel, "⚠ POTHOLE", (750, 100), font, 0.6, (0,160,255), 1)

        return panel

    def close(self):
        cv2.destroyAllWindows()
