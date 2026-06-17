# main.py – A.A.G.A AI Trading OS (Gemini Optimized, No Throttling)
import os, json, sqlite3, datetime, time as _time, asyncio, aiohttp
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
from metaapi_cloud_sdk import MetaApi
from agents.gemini_agent import GeminiAnalyst

# ---------- Config ----------
CONFIG_PATH = "config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)

GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
METAAPI_TOKEN      = os.environ.get("METAAPI_TOKEN", "")
METAAPI_ACCOUNT_ID = os.environ.get("METAAPI_ACCOUNT_ID", "")

DB_PATH = "journal.db"

app = FastAPI(title="A.A.G.A AI OS")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

gemini_analyst = GeminiAnalyst(GEMINI_API_KEY)

# ============================================================
# SINGLE shared MetaApi instance + account (async)
# ============================================================
_api_instance   = None
_account_cache  = None

async def get_account():
    global _api_instance, _account_cache
    if _account_cache is not None:
        return _account_cache
    _api_instance  = MetaApi(METAAPI_TOKEN)
    account        = await _api_instance.metatrader_account_api.get_account(METAAPI_ACCOUNT_ID)
    if account.state not in ('DEPLOYING', 'DEPLOYED'):
        await account.deploy()
    await account.wait_connected()
    _account_cache = account
    return account

# ============================================================
# Async MetaApi helpers (unchanged)
# ============================================================
async def sdk_get_candles_async(symbol: str, timeframe: str, limit: int = 500):
    if not METAAPI_TOKEN or not METAAPI_ACCOUNT_ID: return None
    account = await get_account()
    start_time = datetime.datetime.utcnow() + datetime.timedelta(days=1)
    try:
        candles_raw = await account.get_historical_candles(
            symbol=symbol, timeframe=timeframe, start_time=start_time, limit=limit)
        if not candles_raw: return None
        return [{"time": str(c.get("time", "")), "open": float(c.get("open", 0)), "high": float(c.get("high", 0)),
                 "low": float(c.get("low", 0)), "close": float(c.get("close", 0)),
                 "volume": float(c.get("tickVolume", c.get("volume", 0)))} for c in candles_raw]
    except Exception as e:
        print(f"SDK fetch error: {e}"); return None

async def sdk_get_price_async(symbol: str):
    if not METAAPI_TOKEN or not METAAPI_ACCOUNT_ID: return None
    account = await get_account()
    connection = account.get_rpc_connection()
    await connection.connect(); await connection.wait_synchronized()
    try:
        price = await connection.get_symbol_price(symbol=symbol)
        if price: return (float(price.get("bid", 0)) + float(price.get("ask", 0))) / 2
    finally: await connection.close()
    return None

async def execute_trade_async(signal, sl, tp, lot=0.01):
    if not METAAPI_TOKEN or not METAAPI_ACCOUNT_ID: return None
    account = await get_account()
    connection = account.get_rpc_connection()
    await connection.connect(); await connection.wait_synchronized()
    try:
        if signal == "BUY":
            return await connection.create_market_buy_order(symbol="XAUUSD", volume=lot, stop_loss=sl, take_profit=tp)
        else:
            return await connection.create_market_sell_order(symbol="XAUUSD", volume=lot, stop_loss=sl, take_profit=tp)
    finally: await connection.close()

# ============================================================
# Sniper Logic (minimum distance 2.20 – stable)
# ============================================================
MIN_STOP_DISTANCE = 2.20

def detect_swing_points(candles, lookback=3):
    highs, lows = [], []
    for i in range(lookback, len(candles) - lookback):
        if (candles[i]["high"] > max(c["high"] for c in candles[i-lookback:i]) and
                candles[i]["high"] > max(c["high"] for c in candles[i+1:i+lookback+1])): highs.append(i)
        if (candles[i]["low"] < min(c["low"] for c in candles[i-lookback:i]) and
                candles[i]["low"] < min(c["low"] for c in candles[i+1:i+lookback+1])): lows.append(i)
    return highs, lows

def detect_liquidity_sweep(candles, swing_highs, swing_lows):
    sweeps = []
    for i in range(1, len(candles)):
        c = candles[i]
        for sl_idx in swing_lows:
            level = candles[sl_idx]["low"]
            if sl_idx < i and c["low"] < level and c["close"] > level:
                sweeps.append({"type": "sell_side", "index": i, "level": round(level, 2) if level is not None else None}); break
        for sh_idx in swing_highs:
            level = candles[sh_idx]["high"]
            if sh_idx < i and c["high"] > level and c["close"] < level:
                sweeps.append({"type": "buy_side", "index": i, "level": round(level, 2) if level is not None else None}); break
    return sweeps

def detect_mss(candles, lookback=3):
    if len(candles) < lookback + 2: return None
    recent = candles[-lookback - 1:-1]
    curr = candles[-1]
    swing_high = max(c["high"] for c in recent); swing_low = min(c["low"] for c in recent)
    if curr["close"] > swing_high: return "BULLISH"
    elif curr["close"] < swing_low: return "BEARISH"
    return None

def detect_fvg(candles):
    if len(candles) < 3: return None
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if c1["high"] < c3["low"]: return {"type": "bullish", "zone_low": round(c1["high"], 2), "zone_high": round(c3["low"], 2)}
    elif c1["low"] > c3["high"]: return {"type": "bearish", "zone_low": round(c3["high"], 2), "zone_high": round(c1["low"], 2)}
    return None

def validate_stops(signal, entry, sl, tp):
    if sl is None or tp is None or entry is None: return False
    if signal == "BUY":
        if sl > entry - MIN_STOP_DISTANCE or tp < entry + MIN_STOP_DISTANCE: return False
    else:
        if sl < entry + MIN_STOP_DISTANCE or tp > entry - MIN_STOP_DISTANCE: return False
    return True

def adjust_stops_to_minimum(signal, entry, sl, tp):
    if signal == "BUY":
        sl = min(sl, entry - MIN_STOP_DISTANCE)
        tp = max(tp, entry + MIN_STOP_DISTANCE)
    else:
        sl = max(sl, entry + MIN_STOP_DISTANCE)
        tp = min(tp, entry - MIN_STOP_DISTANCE)
    return sl, tp

async def run_sniper_analysis():
    raw15 = await sdk_get_candles_async("XAUUSD", "15m", 500)
    raw1  = await sdk_get_candles_async("XAUUSD", "1m", 200)
    if not raw15 or not raw1 or len(raw15) < 50 or len(raw1) < 10:
        price = await sdk_get_price_async("XAUUSD") or 0
        return {"signal":"HOLD","entry_zone":None,"sl":None,"tp":None,"trend":"Data Unavailable",
                "sweep_detected":False,"sweep_type":None,"sweep_level":None,"mss":None,"fvg":None,"current_price":price}
    candles15 = [{"high": c["high"], "low": c["low"], "close": c["close"]} for c in raw15]
    candles1  = [{"high": c["high"], "low": c["low"], "close": c["close"]} for c in raw1]
    current_price = candles1[-1]["close"]
    closes15 = pd.Series([c["close"] for c in candles15])
    sma15    = closes15.rolling(50).mean()
    trend    = "Unknown"
    if len(sma15) >= 2 and not pd.isna(sma15.iloc[-1]):
        slope = sma15.iloc[-1] - sma15.iloc[-2]
        trend = "UPTREND" if slope > 0 else ("DOWNTREND" if slope < 0 else "SIDEWAYS")
    sh, sl = detect_swing_points(candles15, 3)
    sweeps = detect_liquidity_sweep(candles15, sh, sl)
    latest_sweep = sweeps[-1] if sweeps else None
    if not latest_sweep:
        return {"signal":"HOLD","entry_zone":None,"sl":None,"tp":None,"trend":trend,"sweep_detected":False,
                "sweep_type":None,"sweep_level":None,"mss":None,"fvg":None,"current_price":current_price}
    mss = detect_mss(candles1, 3)
    fvg = detect_fvg(candles1)
    signal = "HOLD"; entry_zone = sl_val = tp_val = None
    if latest_sweep["type"]=="sell_side" and mss=="BULLISH" and fvg and fvg["type"]=="bullish":
        signal="BUY"; entry_zone = (fvg["zone_low"], fvg["zone_high"]); sl_val = latest_sweep["level"]
        tp_val = round(fvg["zone_high"] + (fvg["zone_high"]-fvg["zone_low"])*2, 2)
    elif latest_sweep["type"]=="buy_side" and mss=="BEARISH" and fvg and fvg["type"]=="bearish":
        signal="SELL"; entry_zone = (fvg["zone_low"], fvg["zone_high"]); sl_val = latest_sweep["level"]
        tp_val = round(fvg["zone_low"] - (fvg["zone_high"]-fvg["zone_low"])*2, 2)
    else:
        sweep_level = latest_sweep["level"]
        if sweep_level and not (isinstance(sweep_level, float) and pd.isna(sweep_level)) and sweep_level > 0:
            risk_dist = abs(current_price - sweep_level)
            if latest_sweep["type"] == "sell_side":
                signal="BUY"; sl_val = sweep_level
                tp_val = round(current_price + max(2 * risk_dist, MIN_STOP_DISTANCE*2), 2)
                entry_zone = (current_price, current_price)
            elif latest_sweep["type"] == "buy_side":
                signal="SELL"; sl_val = sweep_level
                tp_val = round(current_price - max(2 * risk_dist, MIN_STOP_DISTANCE*2), 2)
                entry_zone = (current_price, current_price)
    if signal in ("BUY","SELL") and entry_zone:
        entry = entry_zone[0] if signal=="BUY" else entry_zone[1]
        sl_val, tp_val = adjust_stops_to_minimum(signal, entry, sl_val, tp_val)
        if not validate_stops(signal, entry, sl_val, tp_val):
            signal = "HOLD"; entry_zone = None; sl_val = None; tp_val = None
    return {"signal":signal,"entry_zone":entry_zone,"sl":sl_val,"tp":tp_val,"trend":trend,
            "sweep_detected":True,"sweep_type":latest_sweep["type"],"sweep_level":latest_sweep["level"],
            "mss":mss,"fvg":fvg,"current_price":current_price}

# ============================================================
# Outcome checker
# ============================================================
async def update_pending_trades():
    price = await sdk_get_price_async("XAUUSD")
    if price is None: return
    con = sqlite3.connect(DB_PATH)
    trades = con.execute("SELECT id, pair, signal, entry, sl, tp FROM trade_journal WHERE outcome IS NULL").fetchall()
    for tid, pair, sig, entry, sl, tp in trades:
        if entry is None: continue
        outcome = pnl = None
        if sig == "BUY":
            if price <= sl: outcome, pnl = "LOSS", round((sl - entry) * 100, 2)
            elif price >= tp: outcome, pnl = "WIN",  round((tp - entry) * 100, 2)
        else:
            if price >= sl: outcome, pnl = "LOSS", round((entry - sl) * 100, 2)
            elif price <= tp: outcome, pnl = "WIN",  round((entry - tp) * 100, 2)
        if outcome: con.execute("UPDATE trade_journal SET outcome=?, pnl=? WHERE id=?", (outcome, pnl, tid))
    con.commit(); con.close()

# ============================================================
# Nightly Gemini Analysis – ONE call per day (quota friendly)
# ============================================================
async def nightly_gemini_analysis():
    while True:
        now = datetime.datetime.utcnow()
        next_midnight = (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
        wait_seconds = (next_midnight - now).total_seconds()
        await asyncio.sleep(wait_seconds)

        # Fetch all completed trades from yesterday
        con = sqlite3.connect(DB_PATH)
        trades = con.execute("""
            SELECT id, signal, entry, sl, tp, outcome, pnl, timestamp
            FROM trade_journal
            WHERE date(timestamp) = date('now', '-1 day') AND outcome IS NOT NULL
            ORDER BY timestamp
        """).fetchall()
        con.close()

        if trades:
            trade_list = [{
                "id": t[0], "signal": t[1], "entry": t[2], "sl": t[3], "tp": t[4],
                "outcome": t[5], "pnl": t[6], "timestamp": t[7]
            } for t in trades]

            # One Gemini call for daily summary
            daily_insight = gemini_analyst.analyze_daily_trades(trade_list)
            print(f"📊 A.A.G.A Nightly Analysis: {daily_insight.get('summary', 'No insight')}")

            # Store the daily insight in agent_log
            hour = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00")
            con = sqlite3.connect(DB_PATH)
            con.execute("INSERT INTO agent_log (hour, agent, action, prob, details) VALUES (?,?,?,?,?)",
                        (hour, "A.A.G.A Gemini", "DAILY_REVIEW", 1.0, json.dumps(daily_insight)))
            con.commit()
            con.close()

            # Apply parameter suggestions if any
            suggestions = daily_insight.get("parameter_suggestions", {})
            if suggestions:
                with open(CONFIG_PATH, "r+") as f:
                    cfg = json.load(f)
                    params = cfg.get("parameters", {})
                    for key, val in suggestions.items():
                        if key in params:
                            params[key] = val
                    cfg["parameters"] = params
                    f.seek(0)
                    json.dump(cfg, f, indent=2)
                    f.truncate()
                print("✅ A.A.G.A self‑optimized parameters:", suggestions)
        else:
            print("ℹ️ No completed trades yesterday for A.A.G.A analysis.")

# ============================================================
# Autonomous Trading Loop + Keep‑alive
# ============================================================
AUTONOMOUS_INTERVAL_SEC = 60
KEEPALIVE_INTERVAL_SEC  = 180

async def keep_alive_task():
    await asyncio.sleep(10)
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://aaga.onrender.com/health") as resp:
                    if resp.status == 200: print("✅ Keep‑alive ping OK")
                    else: print("⚠️ Keep‑alive ping failed with status", resp.status)
        except Exception as e: print("❌ Keep‑alive ping error:", e)
        await asyncio.sleep(KEEPALIVE_INTERVAL_SEC)

async def autonomous_trading_loop():
    while True:
        try:
            await update_pending_trades()
            sniper = await run_sniper_analysis()
            if sniper["signal"] in ("BUY", "SELL") and sniper["entry_zone"] and sniper["sl"] and sniper["tp"]:
                entry = sniper["entry_zone"][0] if sniper["signal"] == "BUY" else sniper["entry_zone"][1]
                if not validate_stops(sniper["signal"], entry, sniper["sl"], sniper["tp"]): continue
                con = sqlite3.connect(DB_PATH)
                existing = con.execute("SELECT COUNT(*) FROM trade_journal WHERE outcome IS NULL AND signal = ?", (sniper["signal"],)).fetchone()[0]
                con.close()
                if existing == 0:
                    risk_brief = {"entry": round(entry,2), "sl": sniper["sl"], "tp": sniper["tp"]}
                    con = sqlite3.connect(DB_PATH)
                    con.execute("INSERT INTO trade_journal (timestamp,pair,signal,entry,sl,tp,strategy_used,gemini_insight) VALUES (?,?,?,?,?,?,?,?)",
                                (datetime.datetime.utcnow().isoformat(), "XAUUSD", sniper["signal"],
                                 risk_brief["entry"], risk_brief["sl"], risk_brief["tp"], "default", ""))
                    con.commit(); con.close()
                    order = await execute_trade_async(sniper["signal"], sniper["sl"], sniper["tp"], lot=0.01)
                    if order: print(f"✅ A.A.G.A Auto trade: {sniper['signal']}")
                    else: print("❌ A.A.G.A Auto trade failed")
        except Exception as e: print(f"Autonomous loop error: {e}")
        await asyncio.sleep(AUTONOMOUS_INTERVAL_SEC)

@app.on_event("startup")
async def startup():
    asyncio.create_task(keep_alive_task())
    asyncio.create_task(autonomous_trading_loop())
    asyncio.create_task(nightly_gemini_analysis())

# ============================================================
# Database init
# ============================================================
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS trade_journal (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT,
            pair          TEXT,
            signal        TEXT,
            entry         REAL,
            sl            REAL,
            tp            REAL,
            outcome       TEXT,
            pnl           REAL,
            strategy_used TEXT,
            gemini_insight TEXT
        );
        CREATE TABLE IF NOT EXISTS agent_log (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            hour    TEXT,
            agent   TEXT,
            action  TEXT,
            prob    REAL,
            details TEXT
        );
    """)
    con.commit()
    con.close()

init_db()

# ============================================================
# Routes
# ============================================================
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return FileResponse("dashboard.html")

@app.get("/health")
async def health():
    return {"status": "ok", "autonomous_loop": "active", "gemini": gemini_analyst.available}

@app.get("/master-signal")
async def master_signal():
    sniper = await run_sniper_analysis()
    mtf = {"htf_bias":"Bullish","confluence_score":6}
    now = datetime.datetime.utcnow()
    # Gemini not called here to save quota; use nightly for insights
    gemini_advice = {"summary": "Available overnight"}

    if sniper["signal"] == "BUY": tech_probs = {"BUY":0.9,"SELL":0.0,"HOLD":0.1}
    elif sniper["signal"] == "SELL": tech_probs = {"BUY":0.0,"SELL":0.9,"HOLD":0.1}
    else: tech_probs = {"BUY":0.3,"SELL":0.3,"HOLD":0.4}

    news_probs  = {"BUY":0.5,"SELL":0.5,"HOLD":0.0}
    risk_probs  = {"BUY":0.33,"SELL":0.33,"HOLD":0.34}
    strat_probs = {"BUY":0.5,"SELL":0.5,"HOLD":0.0}
    w = {"tech":0.4,"news":0.2,"risk":0.2,"strategy":0.2}

    final_probs = {
        "BUY":  tech_probs["BUY"]*w["tech"] + news_probs["BUY"]*w["news"] + risk_probs["BUY"]*w["risk"] + strat_probs["BUY"]*w["strategy"],
        "SELL": tech_probs["SELL"]*w["tech"] + news_probs["SELL"]*w["news"] + risk_probs["SELL"]*w["risk"] + strat_probs["SELL"]*w["strategy"],
        "HOLD": tech_probs["HOLD"]*w["tech"] + news_probs["HOLD"]*w["news"] + risk_probs["HOLD"]*w["risk"] + strat_probs["HOLD"]*w["strategy"],
    }
    decision = max(final_probs, key=final_probs.get)

    risk_brief = None
    if decision in ("BUY","SELL") and sniper["entry_zone"] and sniper["sl"] and sniper["tp"]:
        entry = sniper["entry_zone"][0] if decision=="BUY" else sniper["entry_zone"][1]
        risk_brief = {"entry":round(entry,2),"sl":sniper["sl"],"tp":sniper["tp"]}

    hour = now.strftime("%Y-%m-%d %H:00")
    con2 = sqlite3.connect(DB_PATH)
    for ag, probs in [("A.A.G.A Tech",tech_probs),("A.A.G.A News",news_probs),("A.A.G.A Risk",risk_probs),("A.A.G.A Strategy",strat_probs)]:
        act = max(probs, key=probs.get)
        con2.execute("INSERT INTO agent_log (hour,agent,action,prob,details) VALUES (?,?,?,?,?)",
                     (hour,ag,act,probs[act],json.dumps(probs)))
    con2.commit(); con2.close()

    return {
        "decision":decision,"probabilities":final_probs,"risk_brief":risk_brief,"strategy_used":"default",
        "gemini_advice":gemini_advice,"market_trend":sniper["trend"],"sweep_detected":sniper["sweep_detected"],
        "sweep_type":sniper["sweep_type"],"mss":sniper["mss"],"fvg":sniper["fvg"],"current_price":sniper["current_price"]
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

@app.get("/trade-reviews")
def trade_reviews():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT id, timestamp, signal, entry, sl, tp, outcome, pnl, gemini_insight FROM trade_journal WHERE outcome IS NOT NULL ORDER BY timestamp DESC LIMIT 50").fetchall()
    con.close()
    return [{"id": r[0], "timestamp": r[1], "signal": r[2], "entry": r[3], "sl": r[4], "tp": r[5],
             "outcome": r[6], "pnl": r[7], "gemini_insight": r[8]} for r in rows]

# ============================================================
# Stub Agents
# ============================================================
def run_news_analysis(): return {"prob_buy":0.5,"prob_sell":0.5,"prob_hold":0.0}
class RiskManager: pass
class StrategySelector: pass
class GeminiAnalyst: pass

daily_tracker = type('',(object,),{"check_reset":lambda s:None,"stats":lambda s:{"daily_pnl":0}})()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))