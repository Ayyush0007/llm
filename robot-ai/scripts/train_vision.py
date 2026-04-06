from ultralytics import YOLO
import torch
import os

# ─── Config ───────────────────────────────────────────────
MODEL_BASE   = "yolov8m.pt"       # medium model — good balance
DATA_CONFIG  = "/Users/yashmogare/robot-ai/llm/robot-ai/configs/indian_roads.yaml"
EPOCHS       = 100
BATCH_SIZE   = 16
IMAGE_SIZE   = 640
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_NAME = "indian_road_robot"
# ──────────────────────────────────────────────────────────

def train():
    print(f"🚀 Starting training on {DEVICE}")
    
    # Load base YOLOv8 model (pre-trained on COCO)
    # Check if a last.pt exists to resume
    checkpoint_path = "/Users/yashmogare/robot-ai/llm/runs/detect/vision_ai_v1/weights/last.pt"
    if os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        print(f"🆕 Initializing new model from {MODEL_BASE}")
        model = YOLO(MODEL_BASE)
    
    # Fine-tune on Indian road data
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name="vision_ai_v1",
        resume=os.path.exists(checkpoint_path),
        
        # Augmentation — critical for Indian road variety
        hsv_h=0.015,      # hue shift (lighting changes)
        hsv_s=0.7,        # saturation (dust, rain)
        hsv_v=0.4,        # brightness (shadow, night)
        flipud=0.0,        # no vertical flip
        fliplr=0.5,        # horizontal flip OK
        mosaic=1.0,        # mosaic augmentation ON
        mixup=0.1,         # slight mixup
        
        # Save best model
        save=True,
    )
    
    print(f"✅ Training complete!")
    print(f"   Model saved to: {PROJECT_NAME}/vision_ai_v1/weights/best.pt")
    
    return results

def validate(model_path: str):
    """Run validation on trained model"""
    model = YOLO(model_path)
    metrics = model.val(data=DATA_CONFIG, device=DEVICE)
    print(f"📊 Validation mAP50: {metrics.box.map50:.4f}")
    return metrics

if __name__ == "__main__":
    train()
