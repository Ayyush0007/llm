#!/usr/bin/env python3
"""
run_robot.py — Single-command launcher for the full self-drive stack.

Usage:
    # PHASE 1 — Simulation (CARLA must be running)
    python3 run_robot.py

    # With custom options:
    python3 run_robot.py --mode sim --model models/vision/best.pt --depth small

    # PHASE 2 — Real hardware (coming later)
    python3 run_robot.py --mode hardware
"""

import argparse
import os
import sys
import subprocess
import time

# ─── ASCII Banner ─────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════╗
║   🤖  BENTO ROBOT AI — SELF-DRIVE SYSTEM v1.0           ║
║   Indian Road Autonomous Navigation · Jalgaon, MH       ║
╠══════════════════════════════════════════════════════════╣
║   PHASE 1: CARLA Simulation Training                     ║
║   Stack:   YOLOv8 + DepthAnything v2 + FSM              ║
╚══════════════════════════════════════════════════════════╝
"""

def check_dependencies():
    """Check all required packages are installed."""
    print("🔍 Checking dependencies...")
    missing = []
    checks = [
        ("ultralytics", "YOLOv8"),
        ("torch",        "PyTorch"),
        ("transformers", "HuggingFace (DepthAnything)"),
        ("cv2",          "OpenCV"),
        ("numpy",        "NumPy"),
    ]
    for pkg, name in checks:
        try:
            __import__(pkg)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} — not installed")
            missing.append(pkg)

    try:
        import carla
        print(f"   ✅ CARLA Python API")
    except ImportError:
        print(f"   ⚠️  CARLA Python API — not installed (simulation won't work)")
        print(f"      Install: pip install carla==0.9.15")

    if missing:
        print(f"\n❌ Missing packages: {missing}")
        print(f"   Run: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("✅ All dependencies OK\n")


def check_carla_running(host="localhost", port=2000):
    """Attempt to connect to CARLA — warn if not reachable."""
    try:
        import carla
        client = carla.Client(host, port)
        client.set_timeout(3.0)
        world = client.get_world()
        print(f"✅ CARLA server found — Map: {world.get_map().name}")
        return True
    except Exception:
        print("⚠️  CARLA server not reachable at localhost:2000")
        print("   Start it with: ./CarlaUE4.sh -quality-level=Low -fps=20")
        return False


def run_simulation_mode(args):
    """Launch the CARLA self-drive loop."""
    print("🚗 Starting SIMULATION mode...\n")

    carla_ok = check_carla_running(args.host, args.port)
    if not carla_ok and not args.force:
        print("\nAbort — CARLA not running.")
        print("To run anyway (demo/no-CARLA mode): add --force flag")
        return

    # Set environment variables used by self_drive.py
    os.environ["YOLO_MODEL"]  = args.model
    os.environ["DEPTH_SIZE"]  = args.depth
    os.environ["CARLA_HOST"]  = args.host
    os.environ["CARLA_PORT"]  = str(args.port)

    # Import and run
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core.self_drive import SelfDriveSystem
    system = SelfDriveSystem()
    try:
        system.start()
    except KeyboardInterrupt:
        system.stop()


def run_hardware_mode(args):
    """[PHASE 2] Launch on real Jetson hardware."""
    print("🤖 HARDWARE mode selected — launching motor controller + self-drive...\n")
    print("⚠️  Hardware mode requires:")
    print("   1. NVIDIA Jetson Orin NX running Ubuntu 22.04")
    print("   2. USB camera at /dev/video0")
    print("   3. L298N motor driver wired to GPIO pins 17,18,22,23,24,27")
    print("   4. ROS 2 Humble installed")
    print("\n   When ready, this will start:")
    print("   - motor_control.py (motor driver node)")
    print("   - self_drive.py --mode hardware (camera + AI loop)")
    print("\n⏳ Hardware mode coming in Phase 2. Run --mode sim for now.")


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description="Bento Robot Self-Drive Launcher")
    parser.add_argument(
        "--mode",  choices=["sim", "hardware"], default="sim",
        help="sim = CARLA simulation (default), hardware = real Jetson"
    )
    parser.add_argument("--model", default="models/vision/best.pt",
                        help="Path to YOLOv8 .pt model")
    parser.add_argument("--depth", choices=["small","base","large"], default="small",
                        help="DepthAnything v2 model size")
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000,  help="CARLA server port")
    parser.add_argument("--force", action="store_true",
                        help="Run even if CARLA is not reachable (demo mode)")
    args = parser.parse_args()

    check_dependencies()

    print(f"📡 Mode : {args.mode.upper()}")
    print(f"🧠 Model: {args.model}")
    print(f"📐 Depth: DepthAnything-v2-{args.depth}")
    print()

    if args.mode == "sim":
        run_simulation_mode(args)
    else:
        run_hardware_mode(args)


if __name__ == "__main__":
    main()
