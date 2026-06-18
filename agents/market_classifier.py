# agents/market_classifier.py
import pandas as pd

class MarketClassifier:
    def __init__(self):
        pass

    def classify(self, candles_1h):
        """
        candles_1h: list of dicts with 'close' keys (1h data, at least 50)
        Returns: "UPTREND", "DOWNTREND", or "SIDEWAYS"
        """
        if not candles_1h or len(candles_1h) < 20:
            return "Unknown"
        closes = pd.Series([c['close'] for c in candles_1h])
        ema50 = closes.ewm(span=50).mean().iloc[-1] if len(closes) >= 50 else closes.mean()
        sma20 = closes.rolling(20).mean().iloc[-1]

        # ADX for trend strength (simplified)
        high = pd.Series([c['high'] for c in candles_1h])
        low  = pd.Series([c['low'] for c in candles_1h])
        tr1 = high - low
        tr2 = (high - closes.shift()).abs()
        tr3 = (low - closes.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        plus_dm = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(lower=0)
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(14).mean().iloc[-1]

        if adx > 25:
            if closes.iloc[-1] > ema50 and sma20 > ema50:
                return "UPTREND"
            elif closes.iloc[-1] < ema50 and sma20 < ema50:
                return "DOWNTREND"
        return "SIDEWAYS"
