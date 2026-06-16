# agents/tech_agent.py
import yfinance as yf
import pandas as pd

def run_tech_analysis(pair):
    # Spot gold ticker fix
    if pair == "XAUUSD":
        ticker = yf.Ticker("XAUUSD=X")
    else:
        ticker = yf.Ticker(pair + "=X")
    df = ticker.history(period="5d", interval="15m")
    if df.empty:
        return {"current_price": 0, "prob_buy": 0, "prob_sell": 0, "prob_hold": 1,
                "entry_zone": None, "stop_loss": None, "take_profit": None}

    candles = [{"high": r.High, "low": r.Low, "close": r.Close} for r in df.itertuples()]
    current = candles[-1]["close"]

    # ---- 1. Swing Points ----
    swing_highs = []
    swing_lows = []
    for i in range(3, len(candles)-3):
        if candles[i]["high"] > max(c[i]["high"] for i in range(i-3,i)) and \
           candles[i]["high"] > max(c[i]["high"] for i in range(i+1,i+4)):
            swing_highs.append(i)
        if candles[i]["low"] < min(c[i]["low"] for i in range(i-3,i)) and \
           candles[i]["low"] < min(c[i]["low"] for i in range(i+1,i+4)):
            swing_lows.append(i)

    # ---- 2. Liquidity Sweeps ----
    sweeps = []
    for i in range(1, len(candles)):
        c = candles[i]
        for idx in swing_lows:
            if idx < i and c["low"] < candles[idx]["low"] and c["close"] > candles[idx]["low"]:
                sweeps.append({"type":"bearish_sweep","idx":i,"level":candles[idx]["low"]})
                break
        for idx in swing_highs:
            if idx < i and c["high"] > candles[idx]["high"] and c["close"] < candles[idx]["high"]:
                sweeps.append({"type":"bullish_sweep","idx":i,"level":candles[idx]["high"]})
                break

    # ---- 3. FVGs ----
    fvgs = []
    for i in range(1, len(candles)-1):
        prev = candles[i-1]
        nxt = candles[i+1]
        if prev["high"] < nxt["low"]:
            fvgs.append({"type":"bullish_fvg","idx":i,"low":prev["high"],"high":nxt["low"]})
        elif prev["low"] > nxt["high"]:
            fvgs.append({"type":"bearish_fvg","idx":i,"low":nxt["high"],"high":prev["low"]})

    # ---- 4. Signal Generation (Sweep → FVG) ----
    signal = None
    entry_zone = None
    stop_loss = None
    take_profit = None

    for sw in sweeps:
        for fv in fvgs:
            if fv["idx"] == sw["idx"] + 1:  # FVG right after sweep
                if sw["type"] == "bearish_sweep" and fv["type"] == "bullish_fvg":
                    entry_zone = (fv["low"], fv["high"])
                    stop_loss = sw["level"]  # below swept low
                    take_profit = fv["high"] + (fv["high"] - fv["low"])  # 1:1 target from FVG
                    signal = "BUY"
                elif sw["type"] == "bullish_sweep" and fv["type"] == "bearish_fvg":
                    entry_zone = (fv["low"], fv["high"])
                    stop_loss = sw["level"]  # above swept high
                    take_profit = fv["low"] - (fv["high"] - fv["low"])
                    signal = "SELL"
                break
        if signal:
            break

    # Probabilities
    if signal == "BUY":
        prob_buy, prob_sell, prob_hold = 0.8, 0.1, 0.1
    elif signal == "SELL":
        prob_buy, prob_sell, prob_hold = 0.1, 0.8, 0.1
    else:
        prob_buy, prob_sell, prob_hold = 0.3, 0.3, 0.4

    return {
        "current_price": current,
        "prob_buy": prob_buy, "prob_sell": prob_sell, "prob_hold": prob_hold,
        "entry_zone": entry_zone,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "signal": signal
    }