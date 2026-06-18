# agents/sniper_agent.py
import yfinance as yf

def run_sniper_analysis(pair="XAUUSD"):
    # Symbols
    if pair == "XAUUSD":
        sym15 = sym1 = "XAUUSD=X"
    else:
        sym15 = sym1 = pair + "=X"

    try:
        df15 = yf.Ticker(sym15).history(period="5d", interval="15m")
        df1  = yf.Ticker(sym1).history(period="5d", interval="1m")
    except Exception as e:
        return error_result(f"Data fetch failed: {e}")

    if df15.empty or len(df15) < 50:
        return error_result("Not enough 15m data")
    if df1.empty or len(df1) < 10:
        return error_result("Not enough 1m data")

    # Convert to list of dicts
    try:
        candles15 = [{"high": float(r.High), "low": float(r.Low), "close": float(r.Close)} for r in df15.itertuples()]
        candles1  = [{"high": float(r.High), "low": float(r.Low), "close": float(r.Close)} for r in df1.itertuples()]
    except Exception as e:
        return error_result(f"Data conversion error: {e}")

    current_price = candles1[-1]["close"]

    # 1. Market Trend (15m)
    try:
        sma = df15["Close"].rolling(50).mean()
        if len(sma) < 2:
            trend = "Neutral"
        else:
            slope = sma.iloc[-1] - sma.iloc[-2]
            trend = "UPTREND" if slope > 0 else "DOWNTREND" if slope < 0 else "SIDEWAYS"
    except:
        trend = "Unknown"

    # 2. 15m Swing points & Sweeps
    try:
        sh15, sl15 = detect_swing_points(candles15, lookback=3)
        sweeps15 = detect_liquidity_sweep(candles15, sh15, sl15)
        latest_sweep = sweeps15[-1] if sweeps15 else None
    except Exception as e:
        return error_result(f"Sweep detection error: {e}")

    if not latest_sweep:
        return no_signal_result(current_price, trend)

    # 3. 1m MSS & FVG
    try:
        mss = detect_mss(candles1, lookback=3)
        fvg = detect_fvg(candles1)
    except Exception as e:
        return error_result(f"MSS/FVG error: {e}")

    # 4. Trade signal decision
    signal = None
    entry_zone = None
    sl = None
    tp = None

    if latest_sweep["type"] == "sell_side" and mss == "BULLISH" and fvg and fvg["type"] == "bullish":
        signal = "BUY"
        entry_zone = (fvg["zone_low"], fvg["zone_high"])
        sl = latest_sweep["level"]
        tp = fvg["zone_high"] + (fvg["zone_high"] - fvg["zone_low"]) * 2
    elif latest_sweep["type"] == "buy_side" and mss == "BEARISH" and fvg and fvg["type"] == "bearish":
        signal = "SELL"
        entry_zone = (fvg["zone_low"], fvg["zone_high"])
        sl = latest_sweep["level"]
        tp = fvg["zone_low"] - (fvg["zone_high"] - fvg["zone_low"]) * 2
    else:
        # Sweep exists but no confirmation → HOLD
        return {
            "signal": "HOLD",
            "entry_zone": None,
            "sl": None,
            "tp": None,
            "trend": trend,
            "sweep_detected": True,
            "sweep_type": latest_sweep["type"],
            "sweep_level": latest_sweep["level"],
            "mss": mss,
            "fvg": fvg,
            "current_price": current_price
        }

    return {
        "signal": signal,
        "entry_zone": entry_zone,
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "trend": trend,
        "sweep_detected": True,
        "sweep_type": latest_sweep["type"],
        "sweep_level": latest_sweep["level"],
        "mss": mss,
        "fvg": fvg,
        "current_price": current_price
    }


def detect_swing_points(candles, lookback=3):
    highs, lows = [], []
    for i in range(lookback, len(candles) - lookback):
        if candles[i]["high"] > max(c["high"] for c in candles[i-lookback:i]) and \
           candles[i]["high"] > max(c["high"] for c in candles[i+1:i+lookback+1]):
            highs.append(i)
        if candles[i]["low"] < min(c["low"] for c in candles[i-lookback:i]) and \
           candles[i]["low"] < min(c["low"] for c in candles[i+1:i+lookback+1]):
            lows.append(i)
    return highs, lows

def detect_liquidity_sweep(candles, swing_highs, swing_lows):
    sweeps = []
    for i in range(1, len(candles)):
        c = candles[i]
        for sl_idx in swing_lows:
            if sl_idx < i and c["low"] < candles[sl_idx]["low"] and c["close"] > candles[sl_idx]["low"]:
                sweeps.append({"type": "sell_side", "index": i, "level": candles[sl_idx]["low"]})
                break
        for sh_idx in swing_highs:
            if sh_idx < i and c["high"] > candles[sh_idx]["high"] and c["close"] < candles[sh_idx]["high"]:
                sweeps.append({"type": "buy_side", "index": i, "level": candles[sh_idx]["high"]})
                break
    return sweeps

def detect_mss(candles, lookback=3):
    if len(candles) < lookback + 2:
        return None
    recent = candles[-lookback-1:-1]
    curr = candles[-1]
    swing_high = max(c["high"] for c in recent)
    swing_low = min(c["low"] for c in recent)
    if curr["close"] > swing_high:
        return "BULLISH"
    elif curr["close"] < swing_low:
        return "BEARISH"
    return None

def detect_fvg(candles):
    if len(candles) < 3:
        return None
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if c1["high"] < c3["low"]:
        return {"type": "bullish", "zone_low": c1["high"], "zone_high": c3["low"]}
    elif c1["low"] > c3["high"]:
        return {"type": "bearish", "zone_low": c3["high"], "zone_high": c1["low"]}
    return None

def no_signal_result(current_price, trend):
    return {
        "signal": "HOLD",
        "entry_zone": None,
        "sl": None,
        "tp": None,
        "trend": trend,
        "sweep_detected": False,
        "sweep_type": None,
        "sweep_level": None,
        "mss": None,
        "fvg": None,
        "current_price": current_price
    }

def error_result(msg):
    return {
        "signal": "HOLD",
        "entry_zone": None,
        "sl": None,
        "tp": None,
        "trend": "Error",
        "sweep_detected": False,
        "sweep_type": None,
        "sweep_level": None,
        "mss": None,
        "fvg": None,
        "current_price": 0,
        "error": msg
    }
