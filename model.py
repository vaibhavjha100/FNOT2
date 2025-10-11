"""
Reinforcement Learning pipeline for training and evaluating an agent in a custom environment.

Core reinforcement methodology is based on LSTM-PPO (Long Short-Term Memory Proximal Policy Optimization).
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from dotenv import load_dotenv
load_dotenv()
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
import json
from datetime import datetime
import torch


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
        self.position = 0
        self.portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance
        self.lock_counter = 0
        self.peak_value = self.initial_balance
        self.episode_rewards = []

        return self._next_observation(), {}

    def _next_observation(self):
        obs_feat = self.features[self.current_step]

        pos_col = np.zeros((obs_feat.shape[0], 1))
        pos_col[-1, 0] = self.position

        obs = np.hstack((obs_feat, pos_col))

        return obs

    def step(self, action):

        action = int(action)

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

        # Step reward: log return of portfolio value
        reward = np.log(max(self.portfolio_value, 1e-8)) - np.log(max(self.prev_portfolio_value, 1e-8))
        self.episode_rewards.append(reward)

        # Time decay penalty on holding cash position
        if self.position == 0:
            reward -= 0.0002

        daily_return = self.portfolio_value/self.prev_portfolio_value - 1

        self.prev_portfolio_value = self.portfolio_value

        self.peak_value = max(self.peak_value, self.portfolio_value)

        terminated = (self.current_step >= len(self.returns) - 1) or (self.portfolio_value <= 0)

        if terminated:
            final_return = self.portfolio_value / self.initial_balance - 1
            reward += final_return

            # Drawdown penalty
            drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            reward -= 0.5 * max(drawdown - 0.2, 0)

            # Downside volatility penalty
            downside_returns = [r for r in self.episode_rewards if r < 0]
            if len(downside_returns) > 1:
                reward -= 0.5 * np.std(downside_returns)

        obs = self._next_observation()

        info = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "prev_portfolio_value": self.prev_portfolio_value,
            "position": self.position,
            "locked_days_left": self.lock_counter,
            "daily_return": daily_return,
            "turnover": turnover,
        }

        return obs, reward, terminated, False, info

modeldir = "./models"
featuresdir = "./features"

class TradingMetricsCallback(BaseCallback):
    """
    Custom callback for logging trading-specific metrics during training.
    Tracks portfolio performance, Sharpe ratio, and trading statistics.
    """

    def __init__(self, verbose=0):
        super(TradingMetricsCallback, self).__init__(verbose)
        self.episode_returns = []
        self.episode_sharpes = []
        self.episode_portfolios = []
        self.episode_trades = []

    def _on_step(self) -> bool:
        # Log metrics at episode end
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]

            if "episode" in info:
                episode_return = info["episode"]["r"]
                episode_length = info["episode"]["l"]

                self.episode_returns.append(episode_return)

                # Get portfolio value from info
                if "portfolio_value" in info:
                    final_portfolio = info["portfolio_value"]
                    self.episode_portfolios.append(final_portfolio)

                # Calculate rolling Sharpe ratio (last 30 episodes)
                if len(self.episode_returns) >= 30:
                    recent_returns = self.episode_returns[-30:]
                    mean_ret = np.mean(recent_returns)
                    std_ret = np.std(recent_returns)
                    sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252)
                    self.episode_sharpes.append(sharpe)

                    if self.verbose > 0 and len(self.episode_returns) % 10 == 0:
                        avg_portfolio = np.mean(self.episode_portfolios[-30:]) if self.episode_portfolios else 0
                        print(f"\nEpisode {len(self.episode_returns)}:")
                        print(f"  Return: {episode_return:.4f}")
                        print(f"  Rolling Sharpe (30 ep): {sharpe:.2f}")
                        print(f"  Avg Portfolio Value: ₹{avg_portfolio:,.0f}")

        return True


def evaluate_agent(model, env, n_eval_episodes=20, deterministic=True):
    """
    Comprehensive evaluation of the trained agent.
    Returns detailed trading metrics and statistics.
    """
    episode_returns = []
    episode_sharpes = []
    episode_max_drawdowns = []
    episode_win_rates = []
    final_portfolios = []
    total_trades = []

    for episode in range(n_eval_episodes):
        obs = env.reset()
        # RecurrentPPO needs lstm_states and episode_start
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

        done = False
        episode_reward = 0
        portfolio_values = [env.get_attr('initial_balance')[0]]
        daily_returns = []
        trades = 0
        prev_position = 0

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic
            )
            #obs, reward, terminated, truncated, info = env.step(action)
            obs, reward, done, info = env.step(action)
            #done = terminated[0] or truncated[0]
            done = done[0]
            episode_start = np.array([False])

            episode_reward += reward[0]

            # Track metrics from info
            if isinstance(info, list):
                info = info[0]

            portfolio_values.append(info.get("portfolio_value", portfolio_values[-1]))

            if "daily_return" in info:
                daily_returns.append(info["daily_return"])

            # Count trades
            curr_position = info.get("position", 0)
            if curr_position != prev_position:
                trades += 1
            prev_position = curr_position

        episode_returns.append(episode_reward)
        final_portfolios.append(portfolio_values[-1])
        total_trades.append(trades)

        # Calculate Sharpe ratio for this episode
        if len(daily_returns) > 1:
            mean_ret = np.mean(daily_returns)
            std_ret = np.std(daily_returns)
            sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252)
            episode_sharpes.append(sharpe)

        # Calculate maximum drawdown
        peak = portfolio_values[0]
        max_dd = 0
        for pv in portfolio_values:
            if pv > peak:
                peak = pv
            dd = (peak - pv) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        episode_max_drawdowns.append(max_dd)

        # Win rate
        wins = sum(1 for r in daily_returns if r > 0)
        win_rate = wins / len(daily_returns) if daily_returns else 0
        episode_win_rates.append(win_rate)

    # Aggregate metrics
    metrics = {
        "mean_episode_return": float(np.mean(episode_returns)),
        "std_episode_return": float(np.std(episode_returns)),
        "mean_sharpe_ratio": float(np.mean(episode_sharpes)) if episode_sharpes else 0,
        "std_sharpe_ratio": float(np.std(episode_sharpes)) if episode_sharpes else 0,
        "mean_max_drawdown": float(np.mean(episode_max_drawdowns)),
        "max_drawdown": float(np.max(episode_max_drawdowns)),
        "mean_win_rate": float(np.mean(episode_win_rates)),
        "mean_final_portfolio": float(np.mean(final_portfolios)),
        "std_final_portfolio": float(np.std(final_portfolios)),
        "mean_total_trades": float(np.mean(total_trades)),
        "total_return_pct": float((np.mean(final_portfolios) / env.get_attr('initial_balance')[0] - 1) * 100)
    }

    return metrics


def model_pipeline():
    """
    Main pipeline to train and evaluate the RL agent in the custom trading environment.
    """

    # Load ticker from environment variable
    ticker = os.getenv('TICKER')
    if ticker is None:
        raise ValueError("TICKER environment variable not set.")

    # Load features and returns from featuresdir
    x_train = np.load(os.path.join(featuresdir, f"{ticker}_x_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(featuresdir, f"{ticker}_y_train.npy"), allow_pickle=True)
    x_test = np.load(os.path.join(featuresdir, f"{ticker}_x_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(featuresdir, f"{ticker}_y_test.npy"), allow_pickle=True)

    print(f"✓ Loaded features and returns for {ticker}")
    print(f"  x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"  x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # Create trading environment
    train_env = TradingEnv(y_train, x_train)
    test_env = TradingEnv(y_test, x_test)

    # Wrap environments
    train_env = Monitor(train_env)
    train_env = DummyVecEnv([lambda: train_env])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,  # Normalize observations
        norm_reward=True,  # Normalize rewards during training
        clip_obs=10.0,  # Clip normalized observations
        clip_reward=10.0  # Clip normalized rewards
    )

    test_env = Monitor(test_env)
    test_env = DummyVecEnv([lambda: test_env])
    test_env = VecNormalize(
        test_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards during testing
        clip_obs=10.0,
        training=False  # Don't update normalization stats during testing
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(modeldir, f"{ticker}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)

    print(f"✓ Model directory: {save_path}\n")

    # Check for gpu acceleration
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(
        f"GPU acceleration is {'enabled' if device == 'cuda' else 'disabled'}. Using device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    # Model Architecture

    policy_kwargs = dict(
        lstm_hidden_size=128,  # LSTM hidden layer size
        n_lstm_layers=2,  # Number of LSTM layers (1 or 2 recommended)
        shared_lstm=False,  # Separate LSTMs for actor and critic
        enable_critic_lstm=True,  # Use LSTM for critic (value function)
        net_arch=dict(
            pi=[64],  # Actor MLP after LSTM
            vf=[64]  # Critic MLP after LSTM
        ),
        )

    print("Initializing RecurrentPPO with LSTM policy...")
    print(f"  LSTM hidden size: {policy_kwargs['lstm_hidden_size']}")
    print(f"  LSTM layers: {policy_kwargs['n_lstm_layers']}")
    print(f"  Shared LSTM: {policy_kwargs['shared_lstm']}")

    model = RecurrentPPO(
        policy=MlpLstmPolicy,
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,  # Steps to collect before update
        batch_size=64,  # Minibatch size
        n_epochs=10,  # Optimization epochs per update
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda
        clip_range=0.2,  # PPO clip range
        ent_coef=0.01,  # Entropy coefficient for exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(save_path, "tensorboard"),
        device=device
    )

    print("✓ RecurrentPPO agent initialized\n")

    # Setup callbacks
    trading_callback = TradingMetricsCallback(verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(save_path, "checkpoints"),
        name_prefix="recurrent_ppo_trading",
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=os.path.join(save_path, "best_model"),
        log_path=os.path.join(save_path, "eval_logs"),
        eval_freq=5000,  # Evaluate every 5000 steps
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    print("✓ Callbacks configured:")
    print("  - TradingMetrics: Track episode returns and Sharpe")
    print("  - Checkpoint: Save model every 10k steps")
    print("  - Eval: Evaluate on test set every 5k steps\n")

    # Train the agent
    print("=" * 70)
    print("Starting Training...")
    print("=" * 70 + "\n")

    # Adjust timesteps based on your data size
    # Rule of thumb: 5-10x the number of training samples
    total_timesteps = max(500_000, len(y_train) * 10)

    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Training samples: {len(y_train):,}")
    print(f"Estimated episodes: ~{total_timesteps // len(y_train)}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[trading_callback, checkpoint_callback, eval_callback],
        progress_bar=True
    )

    print("\n" + "=" * 70)
    print("Training Completed!")
    print("=" * 70 + "\n")

    # Save final model and normalization stats
    model.save(os.path.join(save_path, "final_model"))
    train_env.save(os.path.join(save_path, "vec_normalize.pkl"))

    print(f"✓ Saved final model to {save_path}")
    print(f"✓ Saved normalization stats\n")

    # Comprehensive evaluation on test set
    print("=" * 70)
    print("Evaluating on Test Set...")
    print("=" * 70 + "\n")

    # Load normalization stats into test env
    test_env = VecNormalize.load(os.path.join(save_path, "vec_normalize.pkl"), test_env)
    test_env.training = False
    test_env.norm_reward = False

    # Evaluate
    test_metrics = evaluate_agent(model, test_env, n_eval_episodes=20)

    print("\n" + "=" * 70)
    print("Test Set Results:")
    print("=" * 70)
    print(
        f"Mean Episode Return:     {test_metrics['mean_episode_return']:.4f} ± {test_metrics['std_episode_return']:.4f}")
    print(f"Mean Sharpe Ratio:       {test_metrics['mean_sharpe_ratio']:.2f} ± {test_metrics['std_sharpe_ratio']:.2f}")
    print(f"Mean Max Drawdown:       {test_metrics['mean_max_drawdown']:.2%}")
    print(f"Worst Drawdown:          {test_metrics['max_drawdown']:.2%}")
    print(f"Mean Win Rate:           {test_metrics['mean_win_rate']:.2%}")
    print(
        f"Mean Final Portfolio:    ₹{test_metrics['mean_final_portfolio']:,.0f} ± ₹{test_metrics['std_final_portfolio']:,.0f}")
    print(f"Total Return:            {test_metrics['total_return_pct']:.2f}%")
    print(f"Mean Trades per Episode: {test_metrics['mean_total_trades']:.1f}")
    print("=" * 70 + "\n")

    # Save all metrics and configuration
    config_dict = {
        "ticker": ticker,
        "timestamp": timestamp,
        "model": "RecurrentPPO",
        "policy": "MlpLstmPolicy",
        "training_timesteps": total_timesteps,
        "data_shapes": {
            "x_train": list(x_train.shape),
            "y_train": list(y_train.shape),
            "x_test": list(x_test.shape),
            "y_test": list(y_test.shape)
        },
        "policy_kwargs": policy_kwargs,
        "test_metrics": test_metrics
    }

    with open(os.path.join(save_path, "config_and_metrics.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"✓ Saved configuration and metrics to config_and_metrics.json\n")

    print("=" * 70)
    print("Pipeline Completed Successfully!")
    print("=" * 70 + "\n")

    print("Files saved:")
    print(f"  📁 {save_path}/")
    print(f"     ├── final_model.zip")
    print(f"     ├── vec_normalize.pkl")
    print(f"     ├── config_and_metrics.json")
    print(f"     ├── best_model/")
    print(f"     ├── checkpoints/")
    print(f"     ├── eval_logs/")
    print(f"     └── tensorboard/\n")

    return model, train_env, test_env, test_metrics


if __name__ == "__main__":
    model, train_env, test_env, test_metrics = model_pipeline()
    print("Model pipeline executed successfully.")




