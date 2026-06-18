# agents/ai_agent.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

class RandomForestStrategy:
    """
    Self-learning strategy using Random Forest.
    Trains on historical OHLC patterns and weekly outcome labels.
    Lightweight – works perfectly on Render free tier.
    """
    def __init__(self, config=None):
        self.name = "RandomForest"
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.trained = False
        self.feature_window = 10
        self.balance = 10000.0

    def _make_features(self, ohlc_dict):
        """Build feature vector from last 10 candles."""
        closes = pd.Series(ohlc_dict['closes'])
        highs  = pd.Series(ohlc_dict['highs'])
        lows   = pd.Series(ohlc_dict['lows'])
        opens  = pd.Series(ohlc_dict.get('opens', closes.shift(1).fillna(closes.iloc[0])))
        if len(closes) < self.feature_window + 1:
            return None, 0

        features = []
        for i in range(1, self.feature_window + 1):
            features.extend([
                closes.iloc[-i] / 2000,
                highs.iloc[-i] / 2000,
                lows.iloc[-i] / 2000,
                opens.iloc[-i] / 2000,
                (closes.iloc[-i] - closes.iloc[-i-1]) / 10,
                (highs.iloc[-i] - lows.iloc[-i]) / 10,
            ])
        # Add SMA differences
        sma5 = closes.rolling(5).mean().iloc[-1] / 2000
        sma10 = closes.rolling(10).mean().iloc[-1] / 2000
        features.extend([sma5, sma10, closes.iloc[-1] / closes.rolling(20).mean().iloc[-1]])
        current_price = closes.iloc[-1]
        return np.array(features).reshape(1, -1), current_price

    def train_from_trades(self, trade_journal):
        """
        Learn from past trades: each trade -> features at entry, label = BUY/SELL/HOLD.
        This can be called weekly with all completed trades.
        """
        if len(trade_journal) < 10:
            return
        X, y = [], []
        for trade in trade_journal:
            entry = trade.get('entry', 0)
            sl = trade.get('sl', 0)
            tp = trade.get('tp', 0)
            if entry == 0:
                continue
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            features = [entry/2000, sl/2000, tp/2000, risk/10, reward/10]
            label = trade.get('signal', 'HOLD')
            if label == 'BUY':
                y.append(0)
            elif label == 'SELL':
                y.append(1)
            else:
                y.append(2)
            X.append(features)
        if len(set(y)) < 2:  # need at least two classes
            return
        self.model.fit(X, y)
        self.trained = True

    def get_signal(self, ohlc_dict):
        """
        Return signal dict like strategy_agent.
        If trained, use model; else fallback to trend.
        """
        closes = pd.Series(ohlc_dict['closes'])
        current_price = closes.iloc[-1]

        if not self.trained:
            # Fallback: simple trend following
            trend = "UPTREND" if closes.iloc[-1] > closes.iloc[-20] else "DOWNTREND"
            return {
                "signal": "HOLD",
                "entry_zone": None, "sl": None, "tp": None,
                "trend": trend, "current_price": current_price
            }

        features, price = self._make_features(ohlc_dict)
        if features is None:
            return {
                "signal": "HOLD",
                "entry_zone": None, "sl": None, "tp": None,
                "trend": "Unknown", "current_price": current_price
            }

        pred = self.model.predict(features)[0]  # 0=BUY, 1=SELL, 2=HOLD
        signal_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
        signal = signal_map.get(pred, "HOLD")

        # Calculate SL/TP based on ATR
        highs = pd.Series(ohlc_dict['highs'])
        lows  = pd.Series(ohlc_dict['lows'])
        atr = (highs.iloc[-14:].max() - lows.iloc[-14:].min()) / 2
        if atr < 0.5:
            atr = 1.0

        sl = tp = entry_zone = None
        if signal == "BUY":
            sl = current_price - 1.5 * atr
            tp = current_price + 2.0 * atr
            entry_zone = (current_price, current_price)
        elif signal == "SELL":
            sl = current_price + 1.5 * atr
            tp = current_price - 2.0 * atr
            entry_zone = (current_price, current_price)

        trend = "UPTREND" if closes.iloc[-1] > closes.iloc[-20] else "DOWNTREND"
        return {
            "signal": signal,
            "entry_zone": entry_zone,
            "sl": round(sl, 2) if sl else None,
            "tp": round(tp, 2) if tp else None,
            "trend": trend,
            "current_price": current_price
        }