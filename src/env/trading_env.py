"""
trading_env.py

Research-grade Gymnasium Trading Environment
for PPO-based Financial Trading.
"""


from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ==========================================================
# Configuration
# ==========================================================

DATA_DIR = Path("data/processed")

TENSOR_FILE = DATA_DIR / "training_tensors.npz"

INITIAL_BALANCE = 100000.0
TRANSACTION_COST = 0.001          # 0.1%
MAX_POSITION = 1                  # Long / Short / Flat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ==========================================================
# Trading Environment
# ==========================================================


class TradingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        train: bool = True,
        initial_balance: float = INITIAL_BALANCE,
        transaction_cost: float = TRANSACTION_COST,
    ):

        super().__init__()

        logger.info("Initializing Trading Environment...")

        # ---------------------------------------
        # Load tensors
        # ---------------------------------------

        data = np.load(TENSOR_FILE)

        if train:

            self.X = data["X_train"]

            self.y = data["y_train"]

        else:

            self.X = data["X_test"]

            self.y = data["y_test"]

        logger.info(f"Samples : {len(self.X):,}")

        logger.info(f"Observation Shape : {self.X.shape[1:]}")

        self.initial_balance = initial_balance

        self.transaction_cost = transaction_cost

        # ---------------------------------------
        # Observation Space
        # Shape:
        # (lookback, features)
        # ---------------------------------------

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.X.shape[1:],
            dtype=np.float32,
        )

        # ---------------------------------------
        # Action Space
        #
        # 0 = Hold
        # 1 = Buy
        # 2 = Sell
        # ---------------------------------------

        self.action_space = spaces.Discrete(3)

        # ---------------------------------------
        # Internal Variables
        # ---------------------------------------

        self.current_step = 0

        self.balance = self.initial_balance

        self.portfolio_value = self.initial_balance

        self.max_portfolio = self.initial_balance

        self.position = 0
        # -1 = Short
        #  0 = Flat
        #  1 = Long

        self.entry_price = 0.0

        self.total_reward = 0.0

        self.total_trades = 0

        self.win_trades = 0

        self.loss_trades = 0

        self.max_drawdown = 0.0

        self.daily_returns = []

        self.trade_history = []

        logger.info("Environment Ready.")

    # ======================================================
    # Observation
    # ======================================================

    def _get_observation(self):

        return self.X[self.current_step].astype(
            np.float32
        )

            # ======================================================
    # Reset Environment
    # ======================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(seed=seed)

        self.current_step = 0

        self.balance = self.initial_balance

        self.portfolio_value = self.initial_balance

        self.max_portfolio = self.initial_balance

        self.position = 0

        self.entry_price = 0.0

        self.total_reward = 0.0

        self.total_trades = 0

        self.win_trades = 0

        self.loss_trades = 0

        self.max_drawdown = 0.0

        self.daily_returns = []

        self.trade_history = []

        return self._get_observation(), {}

    # ======================================================
    # Portfolio Update
    # ======================================================

    def _update_portfolio(
        self,
        market_return,
    ):

        reward = 0.0

        # -----------------------------
        # LONG Position
        # -----------------------------

        if self.position == 1:

            reward = market_return

        # -----------------------------
        # SHORT Position
        # -----------------------------

        elif self.position == -1:

            reward = -market_return

        else:

            reward = 0.0

        # -----------------------------
        # Update Portfolio
        # -----------------------------

        self.portfolio_value *= (1 + reward)

        return reward

    # ======================================================
    # Execute Trade
    # ======================================================

    def _execute_trade(
        self,
        action,
    ):

        trade_cost = 0.0

        # -----------------------------
        # BUY
        # -----------------------------

        if action == 1:

            if self.position != 1:

                self.position = 1

                trade_cost = (
                    self.transaction_cost
                )

                self.total_trades += 1

        # -----------------------------
        # SELL
        # -----------------------------

        elif action == 2:

            if self.position != -1:

                self.position = -1

                trade_cost = (
                    self.transaction_cost
                )

                self.total_trades += 1

        return trade_cost

    # ======================================================
    # Drawdown
    # ======================================================

    def _update_drawdown(self):

        if self.portfolio_value > self.max_portfolio:

            self.max_portfolio = self.portfolio_value

        drawdown = (

            self.max_portfolio
            - self.portfolio_value

        ) / self.max_portfolio

        self.max_drawdown = max(

            self.max_drawdown,

            drawdown,

        )

        return drawdown

    # ======================================================
    # Trade Statistics
    # ======================================================

    def _update_trade_statistics(
        self,
        reward,
    ):

        if reward > 0:

            self.win_trades += 1

        elif reward < 0:

            self.loss_trades += 1

        self.daily_returns.append(
            reward
        )

    # ======================================================
    # Info Dictionary
    # ======================================================

    def _build_info(self):

        return {

            "portfolio_value":
                self.portfolio_value,

            "balance":
                self.balance,

            "position":
                self.position,

            "step":
                self.current_step,

            "trades":
                self.total_trades,

            "wins":
                self.win_trades,

            "losses":
                self.loss_trades,

            "drawdown":
                self.max_drawdown,

        }

            # ======================================================
    # Reward Function
    # ======================================================

    def _calculate_reward(
        self,
        portfolio_return,
        trade_cost,
        drawdown,
    ):

        reward = portfolio_return

        # Penalize transaction cost
        reward -= trade_cost

        # Penalize excessive drawdown
        reward -= drawdown * 0.05

        return reward

    # ======================================================
    # Step
    # ======================================================

    def step(
        self,
        action,
    ):

        # Current market return
        market_return = self.y[self.current_step]

        # Execute trade
        trade_cost = self._execute_trade(action)

        # Update portfolio
        portfolio_return = self._update_portfolio(
            market_return
        )

        # Update drawdown
        drawdown = self._update_drawdown()

        # Final reward
        reward = self._calculate_reward(
            portfolio_return,
            trade_cost,
            drawdown,
        )

        self.total_reward += reward

        self._update_trade_statistics(reward)

        # Store trade history
        self.trade_history.append(
            {
                "step": self.current_step,
                "action": int(action),
                "position": self.position,
                "market_return": float(market_return),
                "portfolio_value": float(self.portfolio_value),
                "reward": float(reward),
            }
        )

        # Move to next timestep
        self.current_step += 1

        # Episode finished?
        terminated = (
            self.current_step >= len(self.X) - 1
        )

        truncated = False

        if terminated:

            observation = np.zeros(
                self.observation_space.shape,
                dtype=np.float32,
            )

        else:

            observation = self._get_observation()

        info = self._build_info()

        if terminated:

            info["final_portfolio"] = self.portfolio_value
            info["final_reward"] = self.total_reward
            info["final_drawdown"] = self.max_drawdown
            info["final_trades"] = self.total_trades
            info["winning_trades"] = self.win_trades
            info["losing_trades"] = self.loss_trades

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
)

    # ======================================================
    # Render
    # ======================================================

    # ======================================================
    # Render
    # ======================================================

    def render(self):

        print("\n==============================")
        print("Trading Environment")
        print("==============================")

        print(f"Step            : {self.current_step}")
        print(f"Position        : {self.position}")
        print(f"Portfolio Value : {self.portfolio_value:.2f}")
        print(f"Balance         : {self.balance:.2f}")
        print(f"Max Drawdown    : {self.max_drawdown:.4f}")
        print(f"Trades          : {self.total_trades}")
        print(f"Winning Trades  : {self.win_trades}")
        print(f"Losing Trades   : {self.loss_trades}")
        print(f"Total Reward    : {self.total_reward:.6f}")

        if self.total_trades > 0:

            win_rate = (
                self.win_trades / self.total_trades
            ) * 100

            print(f"Win Rate        : {win_rate:.2f}%")

        print("==============================")

    # ======================================================
    # Close
    # ======================================================

    def close(self):

        logger.info("Closing Trading Environment.")

    # ======================================================
    # Utility Functions
    # ======================================================

    def get_trade_history(self):
        return self.trade_history

    def get_portfolio_value(self):
        return self.portfolio_value

    def get_total_reward(self):
        return self.total_reward

    def get_max_drawdown(self):
        return self.max_drawdown

    def get_total_trades(self):
        return self.total_trades

    def get_win_trades(self):
        return self.win_trades

    def get_loss_trades(self):
        return self.loss_trades


# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    print("Starting Trading Environment...")

    env = TradingEnv(train=True)

    obs, info = env.reset()

    print("Observation Shape:", obs.shape)

    done = False
    total_reward = 0.0

    while not done:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        done = terminated or truncated

    env.render()

    print(f"\nTotal Reward : {total_reward:.6f}")
    print(f"Portfolio    : {env.get_portfolio_value():.2f}")
    print(f"Drawdown     : {env.get_max_drawdown():.4f}")
    print(f"Trades       : {env.get_total_trades()}")

    env.close()