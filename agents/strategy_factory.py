# agents/strategy_factory.py
import json, datetime
from agents.gemini_agent import GeminiAnalyst

class StrategyFactory:
    def __init__(self, gemini_analyst):
        self.gemini = gemini_analyst

    def generate_strategy_params(self, market_context, recent_trades):
        """
        Ask Gemini to suggest new strategy parameters for a generic rule-based strategy.
        Returns a dict with parameters, or None if failed.
        """
        prompt = f"""
You are an expert XAUUSD trading strategist. Based on the following market context and recent trades, suggest a new trading strategy.
Return ONLY a JSON object with these keys:
- "name": a short name for the strategy
- "description": one line
- "buy_conditions": list of simple conditions (e.g., ["ema8 > ema21", "rsi < 30"]) using indicators ema8, ema21, ema50, rsi, adx, price_slope, bos.
- "sell_conditions": similar list
- "sl_atr_multiplier": float (1.0-3.0)
- "tp_atr_multiplier": float (1.5-4.0)
- "min_stop_distance": float (0.5-2.0)

Market context: {json.dumps(market_context)}
Recent trades (last 10): {json.dumps(recent_trades, default=str)}

Only the JSON, no other text.
"""
        response = self.gemini._safe_generate(prompt)
        if not response:
            return None
        try:
            # Clean markdown fences
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            params = json.loads(response)
            return params
        except:
            return None

    def create_strategy_from_params(self, params):
        """
        Dynamically create a strategy instance based on Gemini's parameters.
        Uses a simple class that applies the given conditions.
        """
        if not params:
            return None

        class GeneratedStrategy:
            def __init__(self, cfg):
                self.name = cfg.get("name", "GeminiGen")
                self.buy_conditions = cfg.get("buy_conditions", [])
                self.sell_conditions = cfg.get("sell_conditions", [])
                self.sl_atr_multiplier = cfg.get("sl_atr_multiplier", 1.5)
                self.tp_atr_multiplier = cfg.get("tp_atr_multiplier", 2.0)
                self.min_stop_distance = cfg.get("min_stop_distance", 1.10)
                self.atr_multiplier_sl = self.sl_atr_multiplier
                self.rr_ratio = self.tp_atr_multiplier / self.sl_atr_multiplier

            def get_signal(self, ohlc_dict):
                import pandas as pd
                closes = pd.Series(ohlc_dict['closes'])
                highs  = pd.Series(ohlc_dict['highs'])
                lows   = pd.Series(ohlc_dict['lows'])
                current_price = closes.iloc[-1]

                # Calculate common indicators
                ema8 = closes.ewm(span=8).mean().iloc[-1]
                ema21 = closes.ewm(span=21).mean().iloc[-1]
                ema50 = closes.ewm(span=50).mean().iloc[-1]
                # RSI
                delta = closes.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(14).mean().iloc[-1]
                avg_loss = loss.rolling(14).mean().iloc[-1]
                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi = 100 - (100 / (1 + rs))
                # ADX (simplified)
                tr = pd.concat([highs - lows, (highs - closes.shift()).abs(), (lows - closes.shift()).abs()], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                plus_dm = highs.diff().clip(lower=0)
                minus_dm = -lows.diff().clip(lower=0)
                atr14 = tr.rolling(14).mean()
                plus_di = 100 * (plus_dm.rolling(14).mean() / atr14).iloc[-1]
                minus_di = 100 * (minus_dm.rolling(14).mean() / atr14).iloc[-1]
                dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
                adx = dx.rolling(14).mean().iloc[-1]
                # Slope
                import numpy as np
                if len(closes) >= 5:
                    slope = np.polyfit(range(5), closes.iloc[-5:], 1)[0] / current_price
                else:
                    slope = 0
                # BOS (simple)
                def detect_bos(candles, lookback=3):
                    if len(candles) < lookback + 2:
                        return None
                    recent = candles[-lookback-1:-1]
                    curr = candles[-1]
                    swing_high = max(c['high'] for c in recent)
                    swing_low = min(c['low'] for c in recent)
                    if curr['close'] > swing_high: return 'BULLISH'
                    elif curr['close'] < swing_low: return 'BEARISH'
                    return None
                candles_list = [{'high': h, 'low': l, 'close': c} for h, l, c in zip(highs, lows, closes)]
                bos = detect_bos(candles_list)

                # Evaluate conditions (simplified)
                local_vars = {
                    'ema8': ema8, 'ema21': ema21, 'ema50': ema50,
                    'rsi': rsi, 'adx': adx, 'price_slope': slope,
                    'bos': bos, 'current_price': current_price,
                    'plus_di': plus_di, 'minus_di': minus_di
                }
                signal = "HOLD"
                try:
                    if all(eval(cond, {}, local_vars) for cond in self.buy_conditions):
                        signal = "BUY"
                    elif all(eval(cond, {}, local_vars) for cond in self.sell_conditions):
                        signal = "SELL"
                except:
                    pass

                sl_val = tp_val = entry_zone = None
                if signal != "HOLD":
                    sl_val = current_price - self.atr_multiplier_sl * atr if signal == "BUY" else current_price + self.atr_multiplier_sl * atr
                    tp_val = current_price + self.tp_atr_multiplier * (current_price - sl_val) if signal == "BUY" else current_price - self.tp_atr_multiplier * (sl_val - current_price)
                    entry_zone = (current_price, current_price)

                return {
                    "signal": signal,
                    "entry_zone": entry_zone,
                    "sl": round(sl_val, 2) if sl_val else None,
                    "tp": round(tp_val, 2) if tp_val else None,
                    "trend": "UPTREND" if current_price > ema50 else "DOWNTREND",
                    "current_price": current_price
                }

        return GeneratedStrategy(params)