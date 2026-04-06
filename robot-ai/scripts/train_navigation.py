"""
train_navigation.py — Reinforcement Learning training script using PPO.

Uses Stable-Baselines3 to train an agent in the custom IndianCarlaEnv.
Logs to TensorBoard and saves checkpoints.

Usage:
  # Terminal 1: ./CarlaUE4.sh (Start Simulator)
  # Terminal 2: python3 scripts/train_navigation.py
  # Terminal 3: tensorboard --logdir=runs/ppo_navigation/
  
  # For syntax testing without CARLA:
  python3 scripts/train_navigation.py --dry-run
"""

import os
import sys
import argparse

# Add parent dir to path so we can import core.rl.indian_carla_env
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

from core.rl.indian_carla_env import IndianCarlaEnv

# We use MultiInputPolicy because our ObservationSpace is a Dict (Image + Telemetry Vector)
POLICY_NAME = "MultiInputPolicy"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a quick syntax check without connecting to CARLA.")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps to train.")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA Host ID")
    parser.add_argument("--port", type=int, default=2000, help="CARLA Port")
    args = parser.parse_args()

    print("====================================")
    print("🇮🇳  Bento Indian Road RL Trainer   🇮🇳")
    print("====================================")

    # 1. Init environment
    print(f"Connecting to environment on {args.host}:{args.port}...")
    env = IndianCarlaEnv(host=args.host, port=args.port, max_steps=500)

    # Verify environment follows Gym API
    print("Validating Gymnasium API...")
    check_env(env, warn=True)
    print("✅ Environment is valid.")

    if args.dry_run:
        print("✅ Dry run completed successfully. Syntax and spaces are correct.")
        sys.exit(0)

    # 2. Callbacks
    os.makedirs("models/rl", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/rl/",
        name_prefix="ppo_indian_driver"
    )

    # 3. Model Definition
    print(f"Using Device: {'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'}")
    
    model = PPO(
        POLICY_NAME,
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./runs/ppo_navigation/"
    )

    # 4. Training
    print(f"Starting training for {args.timesteps} timesteps...")
    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving early checkpoint...")
    finally:
        # Save final model
        model.save("models/rl/ppo_indian_driver_final")
        print("✅ Final model saved to models/rl/ppo_indian_driver_final.zip")
        env.close()

if __name__ == "__main__":
    main()
