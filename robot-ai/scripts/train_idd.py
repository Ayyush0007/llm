from ultralytics import YOLO
import os

def train():
    # Absolute path to data.yaml
    data_path = '/Users/yashmogare/robot-ai/llm/robot-ai/idd-detection1-1/data.yaml'
    
    # RESUME from the latest checkpoint
    model_path = '/Users/yashmogare/robot-ai/llm/runs/detect/idd-yolov8-v13/weights/last.pt'
    
    # Initialize the model from checkpoint
    model = YOLO(model_path)
    
    # Resume training with validation DISABLED to prevent OOM/NMS timeouts on Mac
    results = model.train(
        resume=True,
        val=False,
        # Other parameters are inherited from the checkpoint (args.yaml)
    )
    
    print("✅ Training session initiated/completed!")

if __name__ == "__main__":
    train()
