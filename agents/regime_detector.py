# agents/regime_detector.py
import pandas as pd
import numpy as np

class RegimeDetector:
    def __init__(self, adx_trend_threshold=20, volatility_pct=0.005):
        self.adx_threshold = adx_trend_threshold
        self.volatility_threshold = volatility_pct

    def classify(self, candles):
        if not candles or len(candles) < 50: return "UNKNOWN"
        df = pd.DataFrame(candles)
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        current_price = close.iloc[-1]
        volatility_pct = atr / current_price if current_price > 0 else 0
        plus_dm = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(lower=0)
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(14).mean().iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]
        if len(close) >= 5:
            x = np.arange(5)
            y = close.iloc[-5:].values
            slope = np.polyfit(x, y, 1)[0] / current_price
        else: slope = 0
        if volatility_pct > self.volatility_threshold: return "HIGH_VOLATILITY"
        if adx > self.adx_threshold:
            if current_price > ema50 and slope > 0: return "UPTREND"
            elif current_price < ema50 and slope < 0: return "DOWNTREND"
            else: return "SIDEWAYS"
        else: return "SIDEWAYS"
