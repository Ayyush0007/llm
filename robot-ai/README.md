# 🤖 Bento Robot AI — Indian Road Autonomous Navigation

An end-to-end autonomous driving AI stack built for Indian roads (Jalgaon, Maharashtra).

## 🧱 Stack

| Layer | Technology |
|-------|-----------|
| Vision AI | YOLOv8 fine-tuned on IDD + Indian road classes |
| Depth AI | DepthAnything v2 — monocular obstacle detection |
| Navigation AI | PPO (Stable-Baselines3) in CARLA simulator |
| ROS 2 Bridge | Custom decision node fusing all AI signals |
| Hardware | NVIDIA Jetson Orin NX + L298N motor driver |
| Inference Server | FastAPI on Hugging Face Spaces |

## 📁 Project Structure

```
robot-ai/
├── configs/           ← Dataset YAML configs
├── core/              ← Core AI modules
│   ├── carla_env.py   ← CARLA Gym environment (Indian weather + reward)
│   ├── depth_ai.py    ← DepthAnything v2 (3-zone obstacle detection)
│   ├── robot_node.py  ← ROS 2 decision brain
│   └── motor_control.py ← GPIO L298N driver for Jetson
├── datasets/          ← Training data (IDD + Jalgaon + Roboflow)
├── models/            ← Trained weights (vision + navigation)
├── scripts/           ← Training & utility scripts
│   ├── train_vision.py      ← YOLOv8 fine-tuning
│   ├── train_navigation.py  ← PPO RL training
│   ├── build_jalgaon_map.py ← CARLA Indian environment
│   ├── extract_frames.py    ← Phone video → training frames
│   ├── download_datasets.py ← Download Roboflow datasets
│   └── export_to_trt.py     ← TensorRT export for Jetson
├── server/            ← Hugging Face Spaces inference API
│   ├── main.py        ← FastAPI YOLOv8 inference server
│   └── Dockerfile     ← Docker config for HF deployment
└── third_party/       ← Cloned dependency repos
    ├── ros-bridge/    ← carla-simulator/ros-bridge
    ├── gym-carla/     ← cjy1992/gym-carla
    ├── Depth-Anything-V2/  ← Official DepthAnything repo
    └── supervision/   ← roboflow/supervision
```

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/Ayyush0007/llm.git
cd llm/robot-ai
pip install -r requirements.txt
```

### 2. Download datasets
```bash
python3 scripts/download_datasets.py --rf-key YOUR_ROBOFLOW_KEY
```

### 3. Train Vision AI
```bash
python3 scripts/train_vision.py
# Monitor: tensorboard --logdir indian_road_robot/
```

### 4. Build CARLA environment
```bash
# Terminal 1: ./CarlaUE4.sh -quality-level=Low
python3 scripts/build_jalgaon_map.py
```

### 5. Train Navigation AI
```bash
python3 scripts/train_navigation.py
```

### 6. Run full ROS 2 stack
```bash
python3 core/robot_node.py
```

## 🌐 Live API

The Vision AI server is live at:  
**https://sarasproject-bento.hf.space**

```bash
# Test inference
curl -X POST -F "file=@/path/to/image.jpg" \
  https://sarasproject-bento.hf.space/predict
```

## 🖥️ Hardware BOM (~₹33,500)

| Part | Price |
|------|-------|
| NVIDIA Jetson Orin NX 8GB | ₹25,000 |
| 4WD chassis + motors | ₹3,500 |
| USB wide-angle camera | ₹1,200 |
| L298N motor driver | ₹300 |
| 12V LiPo battery | ₹3,000 |
| Wiring + misc | ₹500 |

## 📊 Target Metrics

| Model | Target | Current |
|-------|--------|---------|
| Vision mAP50 | > 0.75 | ~0.25 (18/50 epochs) |
| Nav collision rate | < 20% | TBD (not trained yet) |
| Inference latency (Jetson TRT) | < 30ms | TBD |

## 🗺️ Roadmap

- [x] Vision AI server deployed on Hugging Face
- [x] CARLA environment with Indian weather randomization
- [x] DepthAnything v2 integration (3-zone detection)
- [x] ROS 2 decision node
- [x] Jetson GPIO motor controller
- [ ] Resume to 50 epoch Vision AI training
- [ ] 1M step Navigation AI PPO training
- [ ] Jetson hardware assembly
- [ ] Jalgaon street testing

---

*Built for Jalgaon, Maharashtra, India 🇮🇳*  
*Stack: CARLA · ROS 2 · YOLOv8 · DepthAnything · PPO · Jetson Orin*
