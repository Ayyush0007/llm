import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import sys

# Add core to path to import CarlaIndianEnv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.carla_env import CarlaIndianEnv

# ─── Config ────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_000_000
SAVE_DIR        = "/Users/yashmogare/robot-ai/llm/robot-ai/models/navigation/"
LOG_DIR         = "/Users/yashmogare/robot-ai/llm/robot-ai/logs/navigation/"
CHECKPOINT_FREQ = 50_000
# ───────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

def make_env():
    env = CarlaIndianEnv(image_size=84)
    return env

# Vectorized environment
env      = DummyVecEnv([make_env])
env      = VecTransposeImage(env)

eval_env = DummyVecEnv([make_env])
eval_env = VecTransposeImage(eval_env)

# PPO with CNN policy
model = PPO(
    policy          = "CnnPolicy",
    env             = env,
    n_steps         = 2048,
    batch_size      = 64,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    learning_rate   = 3e-4,
    ent_coef        = 0.01,
    verbose         = 1,
    tensorboard_log = LOG_DIR,
    device          = "cuda",
)

# Callbacks
checkpoint_cb = CheckpointCallback(
    save_freq   = CHECKPOINT_FREQ,
    save_path   = SAVE_DIR,
    name_prefix = "nav_ppo",
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path = os.path.join(SAVE_DIR, "best"),
    log_path             = LOG_DIR,
    eval_freq            = 25_000,
    n_eval_episodes      = 5,
    deterministic        = True,
)

if __name__ == "__main__":
    print("🚀 Starting Navigation AI training...")
    try:
        model.learn(
            total_timesteps = TOTAL_TIMESTEPS,
            callback        = [checkpoint_cb, eval_cb],
            progress_bar    = True,
        )
        model.save(os.path.join(SAVE_DIR, "nav_final"))
        print(f"✅ Navigation AI saved to {SAVE_DIR}/nav_final.zip")
    except Exception as e:
        print(f"❌ Training failed: {e}")
