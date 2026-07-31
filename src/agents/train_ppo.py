"""
train_ppo.py

Train PPO Agent for Financial Trading
using Stable-Baselines3.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from src.env.trading_env import TradingEnv

# ==========================================================
# Configuration
# ==========================================================

MODEL_DIR = Path("results/models")
LOG_DIR = Path("results/logs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "ppo_trading_agent"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ==========================================================
# Environment Factory
# ==========================================================

def make_env(train=True):

    def _init():

        env = TradingEnv(train=train)

        env = Monitor(env)

        return env

    return _init




# ==========================================================
# PPO Hyperparameters
# ==========================================================

PPO_PARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
    "tensorboard_log": str(LOG_DIR),
}

TOTAL_TIMESTEPS = 1_000_000


# ==========================================================
# Create PPO Model
# ==========================================================

def build_model(env):

    logger.info("Building PPO model...")

    model = PPO(
        env=env,
        **PPO_PARAMS,
    )

    logger.info("PPO model created successfully.")

    return model


# ==========================================================
# Training
# ==========================================================

def train_model():

    logger.info("Creating training environment...")

    train_env = DummyVecEnv([make_env(train=True)])

    logger.info("Creating evaluation environment...")

    eval_env = DummyVecEnv([make_env(train=False)])

    logger.info("Building PPO model...")

    model = build_model(train_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=5000,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_checkpoint",
    )

    logger.info("Starting PPO training...")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    logger.info("Training completed.")

    model.save(MODEL_PATH)

    logger.info(f"Model saved to: {MODEL_PATH}")

    return model

    # ==========================================================
# Evaluate Trained Model
# ==========================================================
def evaluate_model(model):

    logger.info("Creating evaluation environment...")

    test_env = DummyVecEnv([make_env(train=False)])

    obs = test_env.reset()

    total_reward = 0.0
    final_info = None
                           
    while True:

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, rewards, dones, infos = test_env.step(action)

        total_reward += float(rewards[0])

        if dones[0]:

            final_info = infos[0]
            break

    logger.info("Evaluation completed.")
    logger.info(f"Total Reward : {total_reward:.6f}")

    print("\n========== Evaluation Summary ==========")
    print(f"Portfolio Value : {final_info['final_portfolio']:.2f}")
    print(f"Total Reward    : {final_info['final_reward']:.6f}")
    print(f"Max Drawdown    : {final_info['final_drawdown']:.4f}")
    print(f"Trades          : {final_info['final_trades']}")
    print(f"Winning Trades  : {final_info['winning_trades']}")
    print(f"Losing Trades   : {final_info['losing_trades']}")
    print("========================================")

    return total_reward


# ==========================================================
# Load Saved Model
# ==========================================================

def load_model():

    logger.info("Loading saved PPO model...")

    env = DummyVecEnv([make_env(train=False)])

    model = PPO.load(
        MODEL_PATH,
        env=env,
    )

    logger.info("Model loaded successfully.")

    return model


# ==========================================================
# Main
# ==========================================================

def main():

    logger.info("=" * 60)
    logger.info("PPO Trading Agent Training")
    logger.info("=" * 60)

    model = train_model()

    logger.info("Evaluating trained model...")

    evaluate_model(model)

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":

    main()