# agents/tech_agent.py
import yfinance as yf

def run_tech_analysis(pair):
    if pair == "XAUUSD":
        ticker = yf.Ticker("XAUUSD=X")
    else:
        ticker = yf.Ticker(pair + "=X")
    df = ticker.history(period="5d", interval="15m")
    if df.empty:
        return no_signal()

    candles = [{"high": float(df['High'].iloc[i]), "low": float(df['Low'].iloc[i]), "close": float(df['Close'].iloc[i])} for i in range(len(df))]
    n = len(candles)
    current = candles[-1]["close"]

    # Swing points (3-candle lookback)
    swing_highs, swing_lows = [], []
    for i in range(3, n-3):
        if (candles[i]["high"] > max(c["high"] for c in candles[i-3:i]) and
            candles[i]["high"] > max(c["high"] for c in candles[i+1:i+4])):
            swing_highs.append(i)
        if (candles[i]["low"] < min(c["low"] for c in candles[i-3:i]) and
            candles[i]["low"] < min(c["low"] for c in candles[i+1:i+4])):
            swing_lows.append(i)

    # Liquidity sweeps
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
                    return signal_result("BUY", entry, sl, tp, current)
                elif sw["type"] == "bullish" and fv["type"] == "bearish":
                    entry = (fv["low"], fv["high"])
                    sl = sw["level"]
                    tp = fv["low"] - (fv["high"] - fv["low"])
                    return signal_result("SELL", entry, sl, tp, current)

    # Fallback: simple trend
    sma = df['Close'].rolling(50).mean()
    if len(sma) < 2:
        return no_signal()
    slope = sma.iloc[-1] - sma.iloc[-2]
    atr = float(df['High'].iloc[-14:].max() - df['Low'].iloc[-14:].min())
    if slope > 0:
        sl = current - atr * 1.5
        tp = current + atr * 2
        return signal_result("BUY", (current, current), sl, tp, current)
    elif slope < 0:
        sl = current + atr * 1.5
        tp = current - atr * 2
        return signal_result("SELL", (current, current), sl, tp, current)
    else:
        return no_signal()

def signal_result(signal, entry_zone, sl, tp, price):
    return {
        "current_price": price,
        "prob_buy": 0.8 if signal=="BUY" else 0.1,
        "prob_sell": 0.8 if signal=="SELL" else 0.1,
        "prob_hold": 0.1,
        "entry_zone": entry_zone,
        "stop_loss": round(sl,2),
        "take_profit": round(tp,2),
        "signal": signal
    }

def no_signal():
    return {
        "current_price": 0,
        "prob_buy": 0.3, "prob_sell": 0.3, "prob_hold": 0.4,
        "entry_zone": None, "stop_loss": None, "take_profit": None,
        "signal": None
    }