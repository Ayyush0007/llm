"""
export_to_trt.py — Export YOLOv8 to TensorRT for 10x faster inference on Jetson.

Run this on the Jetson Orin itself (not on Mac), since TensorRT engine files
are hardware-specific and must be compiled on the target device.

Usage:
    python3 scripts/export_to_trt.py
"""

from ultralytics import YOLO
import os

MODEL_PATH = "/Users/yashmogare/robot-ai/llm/robot-ai/models/vision/best.pt"
OUTPUT_DIR = "/Users/yashmogare/robot-ai/llm/robot-ai/models/vision/"

def export_trt(model_path: str, half: bool = True):
    """
    Exports a YOLOv8 .pt model to TensorRT .engine format.
    half=True: FP16 — best for Jetson Orin (2x speed vs FP32).
    """
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Train your model first with: python3 scripts/train_vision.py")
        return

    model = YOLO(model_path)
    print(f"🚀 Exporting {model_path} to TensorRT...")

    model.export(
        format   = "engine",    # TensorRT .engine
        device   = 0,           # GPU (required for TRT export)
        half     = half,        # FP16 precision
        imgsz    = 640,
        simplify = True,
        workspace= 4,           # GB — adjust based on Jetson RAM
    )

    engine_path = model_path.replace(".pt", ".engine")
    print(f"✅ Exported to: {engine_path}")
    print(f"   Copy this to Jetson: scp {engine_path} jetson@<IP>:/home/jetson/robot/models/")


def export_onnx(model_path: str):
    """
    Fallback: Export to ONNX format for CPU/other runtimes.
    More portable than TensorRT but slower.
    """
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    model = YOLO(model_path)
    print(f"🚀 Exporting {model_path} to ONNX...")

    model.export(
        format   = "onnx",
        imgsz    = 640,
        simplify = True,
        opset    = 17,
    )
    onnx_path = model_path.replace(".pt", ".onnx")
    print(f"✅ Exported to: {onnx_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export YOLOv8 model for deployment")
    parser.add_argument("--format", choices=["trt", "onnx"], default="trt",
                        help="Export format: 'trt'=TensorRT (Jetson), 'onnx'=ONNX (portable)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .pt model")
    args = parser.parse_args()

    if args.format == "trt":
        export_trt(args.model)
    else:
        export_onnx(args.model)
