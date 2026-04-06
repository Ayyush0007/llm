from ultralytics import YOLO
import sys
import os

def infer(image_path):
    # Path to the best weights from the training run
    model_path = '/Users/yashmogare/robot-ai/llm/runs/detect/idd-yolov8-v13/weights/last.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Load the model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path)
    
    # Show results
    for i, result in enumerate(results):
        out = f"result_{i}.jpg"
        result.save(filename=out)
        print(f"  Saved → {out}")
    
    print(f"✅ Inference complete for {image_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 infer_idd.py <path_to_image>")
    else:
        infer(sys.argv[1])
