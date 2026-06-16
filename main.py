# main.py – Tradevil AGI OS (Twelve Data XAUUSD Sniper + Paper Trading)
import os, json, sqlite3, datetime, threading, time as _time
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import pandas as pd

# ---------- Config ----------
CONFIG_PATH = "config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")

app = FastAPI(title="Tradevil AGI OS")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ========== Helper: Fetch candles via Twelve Data ==========
def fetch_twelvedata(symbol, interval, outputsize=500):
    """Fetch historical candles from Twelve Data. Returns DataFrame or None."""
    if not TWELVE_DATA_API_KEY:
        return None
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if "values" not in data:
            return None
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
        df.set_index("datetime", inplace=True)
        df = df.sort_index()
        return df
    except Exception:
        return None

# ========== Sniper Logic (using Twelve Data) ==========
def detect_swing_points(candles, lookback=3):
    highs, lows = [], []
    for i in range(lookback, len(candles) - lookback):
        if (candles[i]["high"] > max(c["high"] for c in candles[i-lookback:i]) and
            candles[i]["high"] > max(c["high"] for c in candles[i+1:i+lookback+1])):
            highs.append(i)
        if (candles[i]["low"] < min(c["low"] for c in candles[i-lookback:i]) and
            candles[i]["low"] < min(c["low"] for c in candles[i+1:i+lookback+1])):
            lows.append(i)
    return highs, lows

def detect_liquidity_sweep(candles, swing_highs, swing_lows):
    sweeps = []
    for i in range(1, len(candles)):
        c = candles[i]
        for sl_idx in swing_lows:
            if sl_idx < i and c["low"] < candles[sl_idx]["low"] and c["close"] > candles[sl_idx]["low"]:
                sweeps.append({"type":"sell_side","index":i,"level":round(candles[sl_idx]["low"],2)})
                break
        for sh_idx in swing_highs:
            if sh_idx < i and c["high"] > candles[sh_idx]["high"] and c["close"] < candles[sh_idx]["high"]:
                sweeps.append({"type":"buy_side","index":i,"level":round(candles[sh_idx]["high"],2)})
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
        return {"type":"bullish","zone_low":round(c1["high"],2),"zone_high":round(c3["low"],2)}
    elif c1["low"] > c3["high"]:
        return {"type":"bearish","zone_low":round(c3["high"],2),"zone_high":round(c1["low"],2)}
    return None

def run_sniper_analysis():
    # Fetch 15m and 1m data for XAU/USD from Twelve Data
    df15 = fetch_twelvedata("XAU/USD", "15min", 500)
    df1  = fetch_twelvedata("XAU/USD", "1min", 500)

    if df15 is None or df1 is None or df15.empty or df1.empty or len(df15) < 50 or len(df1) < 10:
        return {
            "signal": "HOLD",
            "entry_zone": None, "sl": None, "tp": None,
            "trend": "Data Unavailable",
            "sweep_detected": False, "sweep_type": None, "sweep_level": None,
            "mss": None, "fvg": None,
            "current_price": 0,
            "source": "twelvedata" if TWELVE_DATA_API_KEY else "no_key"
        }

    # Convert DataFrames to list of dicts
    candles15 = [{"high": float(r.High), "low": float(r.Low), "close": float(r.Close)} for r in df15.itertuples()]
    candles1  = [{"high": float(r.High), "low": float(r.Low), "close": float(r.Close)} for r in df1.itertuples()]
    current_price = candles1[-1]["close"]

    # Trend from 15m SMA
    sma15 = df15["Close"].rolling(50).mean()
    if len(sma15) >= 2:
        slope = sma15.iloc[-1] - sma15.iloc[-2]
        trend = "UPTREND" if slope > 0 else "DOWNTREND" if slope < 0 else "SIDEWAYS"
    else:
        trend = "Unknown"

    # 15m Sweep
    sh, sl = detect_swing_points(candles15, 3)
    sweeps = detect_liquidity_sweep(candles15, sh, sl)
    latest_sweep = sweeps[-1] if sweeps else None

    if not latest_sweep:
        return {
            "signal": "HOLD",
            "entry_zone": None, "sl": None, "tp": None,
            "trend": trend, "sweep_detected": False,
            "sweep_type": None, "sweep_level": None,
            "mss": None, "fvg": None,
            "current_price": current_price,
            "source": "twelvedata"
        }

    # 1m confirmation
    mss = detect_mss(candles1, 3)
    fvg = detect_fvg(candles1)

    signal = "HOLD"
    entry_zone = None
    sl_val = None
    tp_val = None

    if latest_sweep["type"] == "sell_side" and mss == "BULLISH" and fvg and fvg["type"] == "bullish":
        signal = "BUY"
        entry_zone = (fvg["zone_low"], fvg["zone_high"])
        sl_val = latest_sweep["level"]
        tp_val = round(fvg["zone_high"] + (fvg["zone_high"] - fvg["zone_low"]) * 2, 2)
    elif latest_sweep["type"] == "buy_side" and mss == "BEARISH" and fvg and fvg["type"] == "bearish":
        signal = "SELL"
        entry_zone = (fvg["zone_low"], fvg["zone_high"])
        sl_val = latest_sweep["level"]
        tp_val = round(fvg["zone_low"] - (fvg["zone_high"] - fvg["zone_low"]) * 2, 2)

    return {
        "signal": signal,
        "entry_zone": entry_zone,
        "sl": sl_val,
        "tp": tp_val,
        "trend": trend,
        "sweep_detected": True,
        "sweep_type": latest_sweep["type"],
        "sweep_level": latest_sweep["level"],
        "mss": mss,
        "fvg": fvg,
        "current_price": current_price,
        "source": "twelvedata"
    }

# ========== Paper Trading Engine ==========
def check_open_trades():
    while True:
        try:
            con = sqlite3.connect(DB_PATH)
            trades = con.execute(
                "SELECT id, pair, signal, entry, sl, tp FROM trade_journal WHERE outcome IS NULL"
            ).fetchall()
            for tid, pair, sig, entry, sl, tp in trades:
                if entry is None:
                    continue
                # Get live price from Twelve Data (or fallback)
                try:
                    url = "https://api.twelvedata.com/price?symbol=XAU/USD&apikey=" + TWELVE_DATA_API_KEY
                    resp = requests.get(url, timeout=5).json()
                    current = float(resp.get("price", 0))
                except:
                    continue

                outcome = None
                pnl = None
                if sig == "BUY":
                    if current <= sl:
                        outcome = "LOSS"
                        pnl = round((sl - entry) * 100, 2)
                    elif current >= tp:
                        outcome = "WIN"
                        pnl = round((tp - entry) * 100, 2)
                else:  # SELL
                    if current >= sl:
                        outcome = "LOSS"
                        pnl = round((entry - sl) * 100, 2)
                    elif current <= tp:
                        outcome = "WIN"
                        pnl = round((entry - tp) * 100, 2)

                if outcome:
                    con.execute("UPDATE trade_journal SET outcome=?, pnl=? WHERE id=?",
                                (outcome, pnl, tid))
                    con.commit()
            con.close()
        except Exception as e:
            print(f"Paper trade checker: {e}")
        _time.sleep(15)

# ---------- Agents ----------
def run_news_analysis():
    return {"prob_buy":0.5, "prob_sell":0.5, "prob_hold":0.0}

class RiskManager:
    def evaluate(self):
        return {"prob_buy":0.33, "prob_sell":0.33, "prob_hold":0.34}

class StrategySelector:
    def __init__(self, config): pass
    def select(self, pair, mtf):
        return {"name":"default", "prob_buy":0.5, "prob_sell":0.5, "prob_hold":0.0}

class GeminiAnalyst:
    def __init__(self, key): pass
    def analyze(self, *args):
        return {"summary": "Gemini throttled"}

# ---------- Initialize ----------
daily_tracker = type('',(object,),{"check_reset":lambda self:None,"stats":lambda self:{"daily_pnl":0}})()
risk_manager = RiskManager()
strategy_selector = StrategySelector(config)
gemini_analyst = GeminiAnalyst(GEMINI_API_KEY)

DB_PATH = "journal.db"
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, pair TEXT, signal TEXT, entry REAL, sl REAL, tp REAL,
            outcome TEXT, pnl REAL, strategy_used TEXT, gemini_insight TEXT
        );
        CREATE TABLE IF NOT EXISTS agent_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour TEXT, agent TEXT, action TEXT, prob REAL, details TEXT
        );
    """)
    con.commit()
    con.close()
init_db()

threading.Thread(target=check_open_trades, daemon=True).start()

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return FileResponse("dashboard.html")

@app.get("/master-signal")
def master_signal():
    sniper = run_sniper_analysis()
    news = run_news_analysis()
    mtf = {"htf_bias":"Bullish","confluence_score":6}
    risk_eval = risk_manager.evaluate()
    strategy = strategy_selector.select("XAUUSD", mtf)

    now = datetime.datetime.utcnow()
    gemini_advice = {"summary":"Gemini throttled"}

    if sniper["signal"] == "BUY":
        tech_probs = {"BUY":0.9, "SELL":0.0, "HOLD":0.1}
    elif sniper["signal"] == "SELL":
        tech_probs = {"BUY":0.0, "SELL":0.9, "HOLD":0.1}
    else:
        tech_probs = {"BUY":0.3, "SELL":0.3, "HOLD":0.4}

    news_probs = {"BUY":0.5, "SELL":0.5, "HOLD":0.0}
    risk_probs = {"BUY":0.33, "SELL":0.33, "HOLD":0.34}
    strat_probs = {"BUY":0.5, "SELL":0.5, "HOLD":0.0}

    w = {"tech":0.4,"news":0.2,"risk":0.2,"strategy":0.2}
    final_probs = {
        "BUY": tech_probs["BUY"]*w["tech"] + news_probs["BUY"]*w["news"] + risk_probs["BUY"]*w["risk"] + strat_probs["BUY"]*w["strategy"],
        "SELL": tech_probs["SELL"]*w["tech"] + news_probs["SELL"]*w["news"] + risk_probs["SELL"]*w["risk"] + strat_probs["SELL"]*w["strategy"],
        "HOLD": tech_probs["HOLD"]*w["tech"] + news_probs["HOLD"]*w["news"] + risk_probs["HOLD"]*w["risk"] + strat_probs["HOLD"]*w["strategy"]
    }
    decision = max(final_probs, key=final_probs.get)

    risk_brief = None
    if decision in ("BUY","SELL") and sniper["entry_zone"] and sniper["sl"] and sniper["tp"]:
        entry = sniper["entry_zone"][0] if decision == "BUY" else sniper["entry_zone"][1]
        risk_brief = {"entry": round(entry,2), "sl": sniper["sl"], "tp": sniper["tp"]}

        con = sqlite3.connect(DB_PATH)
        con.execute("INSERT INTO trade_journal (timestamp, pair, signal, entry, sl, tp, strategy_used, gemini_insight) VALUES (?,?,?,?,?,?,?,?)",
                    (now.isoformat(), "XAUUSD", decision, risk_brief["entry"], risk_brief["sl"], risk_brief["tp"], strategy["name"], gemini_advice.get("summary","")))
        con.commit()
        con.close()

    hour = now.strftime("%Y-%m-%d %H:00")
    con2 = sqlite3.connect(DB_PATH)
    for ag, probs in [("Tech", tech_probs), ("News", news_probs), ("Risk", risk_probs), ("Strategy", strat_probs)]:
        act = max(probs, key=probs.get)
        con2.execute("INSERT INTO agent_log (hour, agent, action, prob, details) VALUES (?,?,?,?,?)",
                     (hour, ag, act, probs[act], json.dumps(probs)))
    con2.commit()
    con2.close()

    return {
        "decision": decision,
        "probabilities": final_probs,
        "risk_brief": risk_brief,
        "strategy_used": strategy["name"],
        "gemini_advice": gemini_advice,
        "market_trend": sniper["trend"],
        "sweep_detected": sniper["sweep_detected"],
        "sweep_type": sniper["sweep_type"],
        "mss": sniper["mss"],
        "fvg": sniper["fvg"],
        "current_price": sniper["current_price"],
        "source": sniper.get("source","twelvedata")
    }

@app.get("/agent-log")
def agent_log():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT hour, agent, action, prob, details FROM agent_log WHERE hour >= datetime('now','-1 hour') ORDER BY id DESC LIMIT 20").fetchall()
    con.close()
    return [{"hour":r[0],"agent":r[1],"action":r[2],"prob":r[3],"details":r[4]} for r in rows]

@app.get("/today-trades")
def today_trades():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT timestamp, pair, signal, entry, sl, tp, outcome, pnl FROM trade_journal WHERE date(timestamp) = date('now') ORDER BY timestamp DESC").fetchall()
    con.close()
    return [{"timestamp":r[0],"pair":r[1],"signal":r[2],"entry":r[3],"sl":r[4],"tp":r[5],"outcome":r[6],"pnl":r[7]} for r in rows]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))