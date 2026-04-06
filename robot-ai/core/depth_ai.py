import torch
import cv2
import numpy as np
from transformers import pipeline
import sys

class DepthAI:
    """
    Wraps DepthAnything v2 for real-time monocular depth estimation.
    Works on a single RGB camera — no stereo or LIDAR required.
    """

    def __init__(self, model_size="small"):
        """
        model_size: "small"  → fast, ~10 FPS on CPU, 60 FPS on GPU
                    "base"   → balanced
                    "large"  → most accurate, needs strong GPU
        """
        model_map = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base":  "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf",
        }
        print(f"🔍 Loading DepthAnything-v2-{model_size}...")
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_map[model_size],
            device=self.device,
        )
        print(f"✅ DepthAnything loaded on {'GPU' if self.device == 0 else 'CPU'}")

    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Returns normalized depth map (0=near, 1=far) same size as input"""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(rgb)

        result     = self.pipe(pil_img)
        depth_raw  = np.array(result["depth"])  # float32

        # Normalize to 0-1
        d_min, d_max = depth_raw.min(), depth_raw.max()
        depth_norm = (depth_raw - d_min) / (d_max - d_min + 1e-6)

        # Resize to match original frame
        depth_resized = cv2.resize(
            depth_norm, (bgr_frame.shape[1], bgr_frame.shape[0])
        )
        return depth_resized

    def get_obstacle_distance(self, depth_map: np.ndarray, zone="center") -> float:
        """
        Returns min depth value in a region (lower = closer = danger).
        zone: "center", "left", "right"
        """
        h, w = depth_map.shape
        zones = {
            "center": depth_map[h//3 : 2*h//3, w//3 : 2*w//3],
            "left"  : depth_map[h//3 : 2*h//3, :w//3],
            "right" : depth_map[h//3 : 2*h//3, 2*w//3:],
        }
        region = zones.get(zone, zones["center"])
        # Low depth value = obstacle is CLOSE
        return float(region.min())

    def visualize(self, bgr_frame: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Returns side-by-side: original + colorized depth"""
        depth_colored = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA
        )
        return np.hstack([bgr_frame, depth_colored])


if __name__ == "__main__":
    depth_ai = DepthAI(model_size="small")
    # For testing, we can use a sample image if camera is not available
    print("🚀 Depth AI initialized. Usage: import DepthAI and call estimate(frame).")
