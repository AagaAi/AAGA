# agents/strategy_agent.py
import numpy as np
import pandas as pd
from typing import Dict, Any

class EmaDmiStrategy:
    def __init__(self, config: dict = None):
        self.name = "EMA+DMI+BOS"
        self.ema_fast = 8
        self.ema_mid  = 21
        self.ema_slow = 50
        self.adx_threshold = 15           # lower than before (was 20)
        self.slope_threshold = 0.00005    # more sensitive
        self.lookback = 3
        self.min_stop_distance = 1.10
        self.atr_multiplier_sl = 1.5
        self.rr_ratio = 2.0

    def calculate_emas(self, closes: pd.Series):
        ema8  = closes.ewm(span=self.ema_fast).mean().iloc[-1]
        ema21 = closes.ewm(span=self.ema_mid).mean().iloc[-1]
        ema50 = closes.ewm(span=self.ema_slow).mean().iloc[-1]
        return ema8, ema21, ema50

    def calculate_dmi(self, highs, lows, closes, period=14):
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]

    def detect_bos(self, candles, lookback=3):
        if len(candles) < lookback + 2:
            return None
        recent = candles[-lookback-1:-1]
        curr = candles[-1]
        swing_high = max(c['high'] for c in recent)
        swing_low = min(c['low'] for c in recent)
        if curr['close'] > swing_high: return 'BULLISH'
        elif curr['close'] < swing_low: return 'BEARISH'
        return None

    def get_signal(self, ohlc: Dict[str, Any]) -> Dict[str, Any]:
        closes = pd.Series(ohlc['closes'])
        highs  = pd.Series(ohlc['highs'])
        lows   = pd.Series(ohlc['lows'])
        current_price = closes.iloc[-1]

        # 1. EMAs
        ema8, ema21, ema50 = self.calculate_emas(closes)

        # 2. DMI
        adx, plus_di, minus_di = self.calculate_dmi(highs, lows, closes)

        # 3. Slope (linear regression of last 5 bars)
        if len(closes) >= 5:
            x = np.arange(5)
            y = closes.iloc[-5:].values
            slope = np.polyfit(x, y, 1)[0] / current_price
        else:
            slope = 0

        # 4. Distance from EMA50 (%)
        distance = (current_price - ema50) / ema50 * 100

        # 5. BOS
        candles_list = [{'high': h, 'low': l, 'close': c} for h, l, c in zip(highs, lows, closes)]
        bos = self.detect_bos(candles_list)

        signal = "HOLD"
        buy_condition = (ema8 > ema21 > ema50 and adx > self.adx_threshold and
                         plus_di > minus_di and slope > self.slope_threshold and bos == 'BULLISH')
        sell_condition = (ema8 < ema21 < ema50 and adx > self.adx_threshold and
                          minus_di > plus_di and slope < -self.slope_threshold and bos == 'BEARISH')

        if buy_condition:
            signal = "BUY"
        elif sell_condition:
            signal = "SELL"

        # ---- FALLBACK: If no ICT signal, just follow the trend ----
        if signal == "HOLD":
            if current_price > ema50 and plus_di > minus_di:
                signal = "BUY"
            elif current_price < ema50 and minus_di > plus_di:
                signal = "SELL"

        sl_val = tp_val = entry_zone = None
        if signal != "HOLD":
            atr_val = (highs.iloc[-14:].max() - lows.iloc[-14:].min()) / 2  # simplified ATR
            if signal == "BUY":
                sl_val = current_price - self.atr_multiplier_sl * atr_val
                tp_val = current_price + self.rr_ratio * (current_price - sl_val)
                entry_zone = (current_price, current_price)
            else:
                sl_val = current_price + self.atr_multiplier_sl * atr_val
                tp_val = current_price - self.rr_ratio * (sl_val - current_price)
                entry_zone = (current_price, current_price)

        return {
            "signal": signal,
            "entry_zone": entry_zone,
            "sl": round(sl_val, 2) if sl_val else None,
            "tp": round(tp_val, 2) if tp_val else None,
            "trend": "UPTREND" if ema8 > ema50 else "DOWNTREND" if ema8 < ema50 else "SIDEWAYS",
            "adx": round(adx, 1),
            "bos": bos,
            "current_price": current_price
        }