"""
Reinforcement Learning pipeline for training and evaluating an agent in a custom environment.

Core reinforcement methodology is based on LSTM-DDPG (Long Short-Term Memory Deep Deterministic Policy Gradient).
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    A custom trading environment for reinforcement learning.
    The environment simulates trading a single stock with OHLCV data.
    There is a returns array and a features array.
    The agent can take actions to buy, sell, or hold the stock.
    """
    def __init__(self, returns, features, initial_balance=1_00_000, transaction_cost=0.002):
        super(TradingEnv, self).__init__()

        self.returns = returns
        self.features = features
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # Action space: discrete action space for 0&1 position
        self.action_space = spaces.Discrete(2)

        # Observation space: features+ position
        # Features are in LSTM format: (timesteps, features)
        seq_length, n_features = features.shape[1], features.shape[2]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(seq_length, n_features+1), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance
        self.lock_counter = 0

        return self._next_observation(), {}

    def _next_observation(self):
        obs_feat = self.features[self.current_step]

        pos_col = np.zeros((obs_feat.shape[0], 1))
        pos_col[-1, 0] = self.position

        obs = np.hstack((obs_feat, pos_col))

        return obs

    def step(self, action):

        action = float(action)

        # Enforce T+2 lock in period
        if self.lock_counter > 0:
            new_position = self.position
            self.lock_counter -= 1
        else:
            new_position = action
            if new_position != self.position:
                self.lock_counter = 2

        # Calculate turnover
        turnover = abs(new_position - self.position)

        ret = self.returns[self.current_step]

        self.portfolio_value *= (1 + self.position * ret)

        if turnover > 0:
            self.portfolio_value *= (1 - self.transaction_cost * turnover)

        self.position = new_position
        self.current_step += 1

        reward = np.log(max(self.portfolio_value, 1e-8)) - np.log(max(self.prev_portfolio_value, 1e-8))

        self.prev_portfolio_value = self.portfolio_value

        terminated = (self.current_step >= len(self.returns) - 1) or (self.portfolio_value <= 0)

        obs = self._next_observation()

        info = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "prev_portfolio_value": self.prev_portfolio_value,
            "position": self.position,
            "locked_days_left": self.lock_counter,
            "daily_return": self.portfolio_value/self.prev_portfolio_value - 1,
            "turnover": turnover
        }

        return obs, reward, terminated, False, info

