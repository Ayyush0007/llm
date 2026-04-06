"""
extract_frames.py — Extract training frames from phone video footage.

Use this to create your own Jalgaon road dataset from videos
recorded on your phone. Extracted frames can be labeled in Roboflow.

Usage:
    python3 scripts/extract_frames.py --video /path/to/jalgaon_road.mp4
    python3 scripts/extract_frames.py --video /path/to/video.mp4 --every 3
"""

import cv2
import os
import argparse

def extract_frames(video_path: str, output_dir: str, every_n: int = 5):
    """
    Extract every N-th frame from a video.

    Args:
        video_path: Path to input video file (.mp4, .mov, etc.)
        output_dir: Directory to save extracted frames.
        every_n:    Save one frame every N frames (e.g. 5 = 1/5 of all frames).
    """
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    cap         = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS)

    print(f"🎬 Video: {video_path}")
    print(f"   Frames: {total_frames} @ {fps:.1f} fps → ~{total_frames/fps:.0f} seconds")
    print(f"   Saving every {every_n} frames → ~{total_frames // every_n} output images")

    frame_idx   = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"✅ Extracted {saved_count} frames to: {output_dir}")
    print(f"   Next step: Upload to Roboflow (free) to label images")
    print(f"   URL: https://roboflow.com")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract training frames from video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default="./datasets/jalgaon_frames",
                        help="Output directory for extracted frames")
    parser.add_argument("--every", type=int, default=5,
                        help="Save one frame every N frames (default: 5)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.every)
