# agents/ppo_agent.py
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    """Custom Trading Environment for PPO."""
    def __init__(self, ohlc_data=None, initial_balance=10000, max_steps=200):
        super().__init__()
        self.ohlc = ohlc_data if ohlc_data else []
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0: none, 1: long, -1: short
        self.entry_price = 0.0
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        # Observation space: [price, balance, position, returns...]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def _get_obs(self):
        if self.current_step >= len(self.ohlc):
            return np.zeros(10, dtype=np.float32)
        price = self.ohlc[self.current_step]["close"]
        # simple features
        returns = price / self.ohlc[max(0,self.current_step-1)]["close"] - 1
        obs = np.array([price/2000, self.balance/10000, self.position, returns,
                        price/2000, 0,0,0,0,0], dtype=np.float32)
        return obs

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.ohlc):
            return self._get_obs(), 0, True, False, {}
        price = self.ohlc[self.current_step]["close"]
        reward = 0
        if action == 1:  # buy
            if self.position == 0:
                self.position = 1
                self.entry_price = price
            elif self.position == -1:  # close short
                profit = (self.entry_price - price) / self.entry_price * self.balance
                self.balance += profit
                reward = profit / 100
                self.position = 0
        elif action == 2:  # sell
            if self.position == 0:
                self.position = -1
                self.entry_price = price
            elif self.position == 1:  # close long
                profit = (price - self.entry_price) / self.entry_price * self.balance
                self.balance += profit
                reward = profit / 100
                self.position = 0
        done = self.balance <= 5000 or self.current_step >= self.max_steps
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        return self._get_obs(), {}

class PPOAgent:
    """PPO-based trading agent."""
    def __init__(self, config=None):
        self.name = "PPO"
        self.model = None
        self.trained = False
        self.min_stop_distance = 1.10

    def train(self, candles):
        """Train PPO on historical candles."""
        if len(candles) < 100:
            return
        env = DummyVecEnv([lambda: TradingEnv(ohlc_data=candles)])
        self.model = PPO("MlpPolicy", env, verbose=0)
        self.model.learn(total_timesteps=5000)
        self.trained = True

    def get_signal(self, ohlc_dict):
        """Use PPO model to decide action. Fallback if not trained."""
        closes = pd.Series(ohlc_dict['closes'])
        current_price = closes.iloc[-1]
        if not self.trained or self.model is None:
            # fallback simple trend
            ema8 = closes.ewm(span=8).mean().iloc[-1]
            ema21 = closes.ewm(span=21).mean().iloc[-1]
            if ema8 > ema21: signal = "BUY"
            elif ema8 < ema21: signal = "SELL"
            else: signal = "HOLD"
        else:
            # Build observation from recent data
            obs = np.array([current_price/2000, 1.0, 0, 0, current_price/2000, 0,0,0,0,0], dtype=np.float32)
            action, _ = self.model.predict(obs, deterministic=True)
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            signal = action_map.get(action, "HOLD")

        # SL/TP calculation
        highs = pd.Series(ohlc_dict['highs'])
        lows  = pd.Series(ohlc_dict['lows'])
        atr = (highs.iloc[-14:].max() - lows.iloc[-14:].min()) / 2
        if atr < 0.5: atr = 1.0
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
            "sl": round(sl,2) if sl else None,
            "tp": round(tp,2) if tp else None,
            "trend": trend,
            "current_price": current_price
        }
