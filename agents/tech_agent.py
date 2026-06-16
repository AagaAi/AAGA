# agents/tech_agent.py
import yfinance as yf
import pandas as pd

def run_tech_analysis(pair):
    # Spot XAUUSD fix
    if pair == "XAUUSD":
        ticker = yf.Ticker("XAUUSD=X")
    else:
        ticker = yf.Ticker(pair + "=X")
    df = ticker.history(period="5d", interval="15m")
    if df.empty:
        return {
            "current_price": 0,
            "prob_buy": 0, "prob_sell": 0, "prob_hold": 1,
            "entry_zone": None, "stop_loss": None, "take_profit": None,
            "signal": None
        }

    # Convert to list of dicts
    candles = []
    for i in range(len(df)):
        candles.append({
            "high": float(df['High'].iloc[i]),
            "low": float(df['Low'].iloc[i]),
            "close": float(df['Close'].iloc[i])
        })
    current = candles[-1]["close"]

    # ----- 1. Swing Points (3-candle lookback) -----
    swing_highs = []
    swing_lows = []
    n = len(candles)
    for i in range(3, n - 3):
        if (candles[i]["high"] > max(c["high"] for c in candles[i-3:i]) and
            candles[i]["high"] > max(c["high"] for c in candles[i+1:i+4])):
            swing_highs.append(i)
        if (candles[i]["low"] < min(c["low"] for c in candles[i-3:i]) and
            candles[i]["low"] < min(c["low"] for c in candles[i+1:i+4])):
            swing_lows.append(i)

    # ----- 2. Liquidity Sweeps -----
    sweeps = []
    for i in range(1, n):
        c = candles[i]
        # Check for sell-side liquidity sweep (bearish sweep → BUY signal)
        for sl_idx in swing_lows:
            if sl_idx < i and c["low"] < candles[sl_idx]["low"] and c["close"] > candles[sl_idx]["low"]:
                sweeps.append({"type": "bearish_sweep", "idx": i, "level": candles[sl_idx]["low"]})
                break
        # Check for buy-side liquidity sweep (bullish sweep → SELL signal)
        for sh_idx in swing_highs:
            if sh_idx < i and c["high"] > candles[sh_idx]["high"] and c["close"] < candles[sh_idx]["high"]:
                sweeps.append({"type": "bullish_sweep", "idx": i, "level": candles[sh_idx]["high"]})
                break

    # ----- 3. Fair Value Gaps (FVGs) -----
    fvgs = []
    for i in range(1, n - 1):
        prev = candles[i-1]
        nxt = candles[i+1]
        # Bullish FVG: prev high < next low
        if prev["high"] < nxt["low"]:
            fvgs.append({"type": "bullish_fvg", "idx": i, "low": prev["high"], "high": nxt["low"]})
        # Bearish FVG: prev low > next high
        elif prev["low"] > nxt["high"]:
            fvgs.append({"type": "bearish_fvg", "idx": i, "low": nxt["high"], "high": prev["low"]})

    # ----- 4. Generate Signal (Sweep + Immediate FVG) -----
    signal = None
    entry_zone = None
    stop_loss = None
    take_profit = None

    # We look for a sweep immediately followed by an FVG (within 2 candles)
    for sw in sweeps:
        for fv in fvgs:
            if fv["idx"] >= sw["idx"] and fv["idx"] <= sw["idx"] + 2:
                if sw["type"] == "bearish_sweep" and fv["type"] == "bullish_fvg":
                    signal = "BUY"
                    entry_zone = (fv["low"], fv["high"])
                    stop_loss = sw["level"]             # below swept low
                    take_profit = fv["high"] + (fv["high"] - fv["low"])   # 1:1 from FVG high
                    break
                elif sw["type"] == "bullish_sweep" and fv["type"] == "bearish_fvg":
                    signal = "SELL"
                    entry_zone = (fv["low"], fv["high"])
                    stop_loss = sw["level"]             # above swept high
                    take_profit = fv["low"] - (fv["high"] - fv["low"])    # 1:1 from FVG low
                    break
        if signal:
            break

    # ----- 5. Probabilities -----
    if signal == "BUY":
        prob_buy, prob_sell, prob_hold = 0.80, 0.10, 0.10
    elif signal == "SELL":
        prob_buy, prob_sell, prob_hold = 0.10, 0.80, 0.10
    else:
        prob_buy, prob_sell, prob_hold = 0.30, 0.30, 0.40

    return {
        "current_price": current,
        "prob_buy": prob_buy, "prob_sell": prob_sell, "prob_hold": prob_hold,
        "entry_zone": entry_zone,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "signal": signal
    }