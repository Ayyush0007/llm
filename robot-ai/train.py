from ultralytics import YOLO
import os

# Define paths
data_path = os.path.abspath('/Users/yashmogare/robot-ai/llm/robot-ai/idd-detection1-1/data.yaml')

# Load latest checkpoint
model = YOLO('/Users/yashmogare/robot-ai/llm/runs/detect/indian-roads-v12/weights/last.pt')

# Resume training with minimum batch size to avoid OOM
model.train(
    resume=True,
    batch=1,
    patience=10,
    device='mps'        # uses Mac GPU (Apple Silicon)
)

print("✅ Training complete!")
