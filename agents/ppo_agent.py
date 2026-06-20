# agents/ppo_agent.py
import numpy as np
import pandas as pd
from typing import Dict, Any
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class TradingEnv(gym.Env):
    """
    Custom environment for PPO that simulates trading on historical 1-minute candles.
    Observation: 10 features (price, returns, balance, position, etc.)
    Action: 0=HOLD, 1=BUY, 2=SELL
    Reward: profit/loss from closed trade, else 0.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, candles: list = None, initial_balance: float = 10000.0, max_steps: int = 500):
        super().__init__()
        self.candles = candles if candles else []
        self.initial_balance = initial_balance
        self.max_steps = min(max_steps, len(self.candles) - 10)
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0          # 0=flat, 1=long, -1=short
        self.entry_price = 0.0

        # Action space: 0 (hold), 1 (buy), 2 (sell)
        self.action_space = spaces.Discrete(3)

        # Observation space: 10 float features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def _get_obs(self):
        if self.current_step >= len(self.candles):
            return np.zeros(10, dtype=np.float32)
        c = self.candles[self.current_step]
        price = c['close']
        # Features
        prev_price = self.candles[max(0, self.current_step - 1)]['close']
        returns = (price / prev_price - 1) * 100
        high_low = (c['high'] - c['low']) / price * 100
        obs = np.array([
            price / 2000.0,
            self.balance / 10000.0,
            self.position,
            returns,
            high_low,
            (price - self.candles[max(0, self.current_step - 5)]['close']) / price * 100,
            (price - self.candles[max(0, self.current_step - 10)]['close']) / price * 100,
            (price - self.candles[max(0, self.current_step - 20)]['close']) / price * 100,
            self.entry_price / 2000.0 if self.position != 0 else 0.0,
            0.0  # placeholder
        ], dtype=np.float32)
        return obs

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.candles) - 1 or self.balance <= 1000
        if self.current_step >= len(self.candles):
            return self._get_obs(), 0, True, False, {}

        price = self.candles[self.current_step]['close']
        reward = 0.0

        # Execute action
        if action == 1:   # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = price
            elif self.position == -1:  # close short and go long
                profit = (self.entry_price - price) / self.entry_price * self.balance * 0.01
                self.balance += profit
                reward = profit / 10.0
                self.position = 1
                self.entry_price = price
            # else do nothing

        elif action == 2:  # SELL
            if self.position == 0:
                self.position = -1
                self.entry_price = price
            elif self.position == 1:  # close long and go short
                profit = (price - self.entry_price) / self.entry_price * self.balance * 0.01
                self.balance += profit
                reward = profit / 10.0
                self.position = -1
                self.entry_price = price

        # If still in position and this is the last step, close it
        if done and self.position != 0:
            if self.position == 1:
                profit = (price - self.entry_price) / self.entry_price * self.balance * 0.01
            else:
                profit = (self.entry_price - price) / self.entry_price * self.balance * 0.01
            self.balance += profit
            reward += profit / 10.0
            self.position = 0

        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        return self._get_obs(), {}


class PPOAgent:
    """
    Deep Reinforcement Learning agent using PPO.
    Trains on historical 1-minute candles and produces BUY/SELL/HOLD signals.
    """
    def __init__(self, config=None):
        self.name = "PPO"
        self.model = None
        self.trained = False
        self.min_stop_distance = 1.10

    def train(self, candles: list, total_timesteps: int = 10000):
        """
        Train PPO on historical candle data.
        candles: list of dicts with 'open','high','low','close'
        """
        if len(candles) < 100:
            print("PPO: not enough data for training")
            return

        env = DummyVecEnv([lambda: TradingEnv(candles=candles)])
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=0.0003,
            n_steps=128,
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            policy_kwargs=dict(net_arch=[64, 64])
        )
        self.model.learn(total_timesteps=total_timesteps)
        self.trained = True
        print(f"✅ PPO trained on {len(candles)} candles")

    def get_signal(self, ohlc_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signal using PPO.
        ohlc_dict must contain 'highs', 'lows', 'closes' lists.
        Returns a dict like strategy_agent.
        """
        closes = pd.Series(ohlc_dict['closes'])
        highs  = pd.Series(ohlc_dict['highs'])
        lows   = pd.Series(ohlc_dict['lows'])
        current_price = closes.iloc[-1]

        signal = "HOLD"
        if self.trained and self.model is not None and len(closes) >= 20:
            # Build observation from recent candles
            prev_close = closes.iloc[-2] if len(closes) > 1 else current_price
            returns = (current_price / prev_close - 1) * 100
            high_low = (highs.iloc[-1] - lows.iloc[-1]) / current_price * 100
            obs = np.array([
                current_price / 2000.0,
                1.0,   # balance placeholder (not used for inference)
                0.0,   # position
                returns,
                high_low,
                (current_price - closes.iloc[-5] if len(closes) >=5 else current_price) / current_price * 100,
                (current_price - closes.iloc[-10] if len(closes) >=10 else current_price) / current_price * 100,
                (current_price - closes.iloc[-20] if len(closes) >=20 else current_price) / current_price * 100,
                0.0,
                0.0
            ], dtype=np.float32)
            action, _ = self.model.predict(obs, deterministic=True)
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            signal = action_map.get(action, "HOLD")
        else:
            # Fallback: simple EMA trend
            ema8 = closes.ewm(span=8).mean().iloc[-1]
            ema21 = closes.ewm(span=21).mean().iloc[-1]
            if ema8 > ema21 and current_price > ema21:
                signal = "BUY"
            elif ema8 < ema21 and current_price < ema21:
                signal = "SELL"

        # Calculate SL/TP using ATR
        atr = (highs.iloc[-14:].max() - lows.iloc[-14:].min()) / 2
        if atr < 0.5:
            atr = 1.0

        sl = tp = entry_zone = None
        if signal == "BUY":
            sl = current_price - max(1.5 * atr, self.min_stop_distance)
            tp = current_price + 2.0 * atr
            entry_zone = (current_price, current_price)
        elif signal == "SELL":
            sl = current_price + max(1.5 * atr, self.min_stop_distance)
            tp = current_price - 2.0 * atr
            entry_zone = (current_price, current_price)

        trend = "UPTREND" if current_price > closes.ewm(span=50).mean().iloc[-1] else "DOWNTREND"
        return {
            "signal": signal,
            "entry_zone": entry_zone,
            "sl": round(sl, 2) if sl else None,
            "tp": round(tp, 2) if tp else None,
            "trend": trend,
            "current_price": current_price
        }