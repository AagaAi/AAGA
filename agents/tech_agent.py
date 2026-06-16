# agents/tech_agent.py
import yfinance as yf
import numpy as np

def run_tech_analysis(pair):
    # Spot XAUUSD fix
    if pair == "XAUUSD":
        ticker = yf.Ticker("XAUUSD=X")
    else:
        ticker = yf.Ticker(pair + "=X")

    # Fetch 15m data (5 days enough)
    df = ticker.history(period="5d", interval="15m")
    if df.empty:
        return no_signal_result()

    # ---- 1. ICT Sweep + FVG (Primary) ----
    signal, entry_zone, sl, tp = ict_sniper(df)

    # ---- 2. Fallback: Simple Trend (if no ICT signal) ----
    if not signal:
        signal, entry_zone, sl, tp = simple_trend_signal(df)

    # ---- 3. Build probabilities ----
    current = float(df['Close'].iloc[-1])
    if signal == "BUY":
        prob_buy, prob_sell, prob_hold = 0.80, 0.10, 0.10
    elif signal == "SELL":
        prob_buy, prob_sell, prob_hold = 0.10, 0.80, 0.10
    else:
        prob_buy, prob_sell, prob_hold = 0.30, 0.30, 0.40

    return {
        "current_price": current,
        "prob_buy": prob_buy,
        "prob_sell": prob_sell,
        "prob_hold": prob_hold,
        "entry_zone": entry_zone,
        "stop_loss": sl,
        "take_profit": tp,
        "signal": signal
    }

def no_signal_result():
    return {
        "current_price": 0,
        "prob_buy": 0, "prob_sell": 0, "prob_hold": 1,
        "entry_zone": None, "stop_loss": None, "take_profit": None,
        "signal": None
    }

def ict_sniper(df):
    # Convert to candles
    candles = [{"high": float(df['High'].iloc[i]),
                "low": float(df['Low'].iloc[i]),
                "close": float(df['Close'].iloc[i])}
               for i in range(len(df))]
    n = len(candles)

    # Swing points
    swing_highs, swing_lows = [], []
    for i in range(3, n-3):
        if (candles[i]["high"] > max(c["high"] for c in candles[i-3:i]) and
            candles[i]["high"] > max(c["high"] for c in candles[i+1:i+4])):
            swing_highs.append(i)
        if (candles[i]["low"] < min(c["low"] for c in candles[i-3:i]) and
            candles[i]["low"] < min(c["low"] for c in candles[i+1:i+4])):
            swing_lows.append(i)

    # Sweeps
    sweeps = []
    for i in range(1, n):
        c = candles[i]
        for sl_idx in swing_lows:
            if sl_idx < i and c["low"] < candles[sl_idx]["low"] and c["close"] > candles[sl_idx]["low"]:
                sweeps.append({"type":"bearish","idx":i,"level":candles[sl_idx]["low"]})
                break
        for sh_idx in swing_highs:
            if sh_idx < i and c["high"] > candles[sh_idx]["high"] and c["close"] < candles[sh_idx]["high"]:
                sweeps.append({"type":"bullish","idx":i,"level":candles[sh_idx]["high"]})
                break

    # FVGs
    fvgs = []
    for i in range(1, n-1):
        prev = candles[i-1]
        nxt = candles[i+1]
        if prev["high"] < nxt["low"]:
            fvgs.append({"type":"bullish","idx":i,"low":prev["high"],"high":nxt["low"]})
        elif prev["low"] > nxt["high"]:
            fvgs.append({"type":"bearish","idx":i,"low":nxt["high"],"high":prev["low"]})

    # Match sweep + FVG within 2 candles
    for sw in sweeps:
        for fv in fvgs:
            if fv["idx"] >= sw["idx"] and fv["idx"] <= sw["idx"] + 2:
                if sw["type"] == "bearish" and fv["type"] == "bullish":
                    entry = (fv["low"], fv["high"])
                    sl = sw["level"]
                    tp = fv["high"] + (fv["high"] - fv["low"])
                    return "BUY", entry, sl, tp
                elif sw["type"] == "bullish" and fv["type"] == "bearish":
                    entry = (fv["low"], fv["high"])
                    sl = sw["level"]
                    tp = fv["low"] - (fv["high"] - fv["low"])
                    return "SELL", entry, sl, tp
    return None, None, None, None

def simple_trend_signal(df):
    # 15-minute 50-period SMA slope
    if len(df) < 50:
        return None, None, None, None
    sma = df['Close'].rolling(50).mean()
    slope = sma.iloc[-1] - sma.iloc[-2]
    current = float(df['Close'].iloc[-1])
    atr = float(df['High'].iloc[-14:].max() - df['Low'].iloc[-14:].min())  # simplified ATR

    if slope > 0:
        # Uptrend → BUY with SL below recent low, TP 2x ATR
        sl = current - atr * 1.5
        tp = current + atr * 2
        entry_zone = (current, current)  # market entry
        return "BUY", entry_zone, max(0, sl), tp
    elif slope < 0:
        # Downtrend → SELL
        sl = current + atr * 1.5
        tp = current - atr * 2
        entry_zone = (current, current)
        return "SELL", entry_zone, sl, tp
    else:
        return None, None, None, None