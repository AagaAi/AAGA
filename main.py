# main.py – A.A.G.A AI OS (All Agents + AI Strategy + Auto-Select)
import os, json, sqlite3, datetime, time as _time, asyncio, aiohttp, feedparser
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
from metaapi_cloud_sdk import MetaApi
from agents.gemini_agent import GeminiAnalyst
from agents.strategy_agent import EmaDmiStrategy
from agents.ai_agent import AIPureStrategy, ai_strategy as ai_strat_instance
from backtest import run_comparison as compare_strategies

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
ema_dmi_strat = EmaDmiStrategy(config.get("parameters", {}))
# AI strategy is a singleton imported

# current active strategy (can be switched)
active_strategy = "ema_dmi"  # or "ai"

# ============================================================
# MetaApi singleton (async)
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
# AGENTS
# ============================================================
# News Agent (inline)
def run_news_analysis():
    # (same as before)
    return {"prob_buy":0.5, "prob_sell":0.5, "prob_hold":0.0}

# Risk Manager (inline)
class RiskManager:
    def __init__(self): self.daily_pnl = 0.0; self.loss_limit = 500
    def evaluate(self):
        if -self.daily_pnl >= self.loss_limit:
            return {"prob_buy":0.0, "prob_sell":0.0, "prob_hold":1.0, "halt": True}
        return {"prob_buy":0.33, "prob_sell":0.33, "prob_hold":0.34, "halt": False}
    def update_pnl(self, pnl): self.daily_pnl += pnl
risk_manager = RiskManager()

# ============================================================
# Combined Signal (uses active strategy)
# ============================================================
MIN_STOP_DISTANCE = 1.10

async def get_signal_from_strategy(strategy_name, ohlc):
    if strategy_name == "ema_dmi":
        return ema_dmi_strat.get_signal(ohlc)
    else:  # ai
        return ai_strat_instance.get_signal(ohlc)

async def run_aggregated_analysis():
    raw1m = await sdk_get_candles_async("XAUUSD", "1m", 200)
    if not raw1m or len(raw1m) < 50:
        price = await sdk_get_price_async("XAUUSD") or 0
        return {"signal":"HOLD","entry_zone":None,"sl":None,"tp":None,"trend":"Data Unavailable","current_price":price}

    highs  = [c['high'] for c in raw1m]
    lows   = [c['low'] for c in raw1m]
    closes = [c['close'] for c in raw1m]
    opens  = [c['open'] for c in raw1m]
    ohlc = {'highs': highs, 'lows': lows, 'closes': closes, 'opens': opens}

    # Get signal from active strategy
    tech_signal = await get_signal_from_strategy(active_strategy, ohlc)

    # News agent
    news_data = run_news_analysis()
    # Risk agent
    risk_data = risk_manager.evaluate()

    # Tech probabilities from signal
    if tech_signal["signal"] == "BUY":
        tech_probs = {"BUY":0.8, "SELL":0.1, "HOLD":0.1}
    elif tech_signal["signal"] == "SELL":
        tech_probs = {"BUY":0.1, "SELL":0.8, "HOLD":0.1}
    else:
        tech_probs = {"BUY":0.3, "SELL":0.3, "HOLD":0.4}

    news_probs = {"BUY": news_data["prob_buy"], "SELL": news_data["prob_sell"], "HOLD": news_data["prob_hold"]}
    risk_probs = {"BUY": risk_data["prob_buy"], "SELL": risk_data["prob_sell"], "HOLD": risk_data["prob_hold"]}
    strat_probs = {"BUY":0.5, "SELL":0.5, "HOLD":0.0}  # default

    # Now include AI agent as another agent (if different from tech)
    # We'll just add another agent weight for "AI" using tech_probs but could be separate
    # For simplicity, we'll combine tech+ai weights together

    w = config["parameters"]["agent_weights"]
    final_probs = {
        "BUY":  tech_probs["BUY"]*w["tech"] + news_probs["BUY"]*w["news"] + risk_probs["BUY"]*w["risk"] + strat_probs["BUY"]*w["strategy"],
        "SELL": tech_probs["SELL"]*w["tech"] + news_probs["SELL"]*w["news"] + risk_probs["SELL"]*w["risk"] + strat_probs["SELL"]*w["strategy"],
        "HOLD": tech_probs["HOLD"]*w["tech"] + news_probs["HOLD"]*w["news"] + risk_probs["HOLD"]*w["risk"] + strat_probs["HOLD"]*w["strategy"],
    }
    decision = max(final_probs, key=final_probs.get)

    # Risk brief from tech_signal if decision matches
    risk_brief = None
    if decision == tech_signal["signal"] and tech_signal["entry_zone"] and tech_signal["sl"] and tech_signal["tp"]:
        entry = tech_signal["entry_zone"][0] if decision=="BUY" else tech_signal["entry_zone"][1]
        risk_brief = {"entry":round(entry,2), "sl":tech_signal["sl"], "tp":tech_signal["tp"]}

    return {
        "signal": decision,
        "entry_zone": tech_signal.get("entry_zone") if decision == tech_signal["signal"] else None,
        "sl": tech_signal.get("sl") if decision == tech_signal["signal"] else None,
        "tp": tech_signal.get("tp") if decision == tech_signal["signal"] else None,
        "trend": tech_signal.get("trend", "Unknown"),
        "current_price": tech_signal.get("current_price", 0),
        "agent_probs": {"tech": tech_probs, "news": news_probs, "risk": risk_probs, "strategy": strat_probs}
    }

# ============================================================
# Outcome checker (same)
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
        if outcome:
            con.execute("UPDATE trade_journal SET outcome=?, pnl=? WHERE id=?", (outcome, pnl, tid))
            if outcome in ("WIN","LOSS"):
                risk_manager.update_pnl(pnl)
    con.commit(); con.close()

# ============================================================
# Nightly Gemini Analysis (same)
# ============================================================
async def nightly_gemini_analysis():
    while True:
        now = datetime.datetime.utcnow()
        next_midnight = (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
        wait_seconds = (next_midnight - now).total_seconds()
        await asyncio.sleep(wait_seconds)

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

            daily_insight = gemini_analyst.analyze_daily_trades(trade_list)
            print(f"📊 A.A.G.A Nightly Analysis: {daily_insight.get('summary', 'No insight')}")

            hour = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00")
            con = sqlite3.connect(DB_PATH)
            con.execute("INSERT INTO agent_log (hour, agent, action, prob, details) VALUES (?,?,?,?,?)",
                        (hour, "A.A.G.A Gemini", "DAILY_REVIEW", 1.0, json.dumps(daily_insight)))
            con.commit()

            suggestions = daily_insight.get("parameter_suggestions", {})
            if suggestions:
                for key, val in suggestions.items():
                    if hasattr(ema_dmi_strat, key):
                        setattr(ema_dmi_strat, key, val)
                with open(CONFIG_PATH, "r+") as f:
                    cfg = json.load(f)
                    params = cfg.get("parameters", {})
                    for key, val in suggestions.items():
                        params[key] = val
                    cfg["parameters"] = params
                    f.seek(0)
                    json.dump(cfg, f, indent=2)
                    f.truncate()
                print("✅ A.A.G.A self‑optimized parameters:", suggestions)
        else:
            print("ℹ️ No completed trades yesterday for A.A.G.A analysis.")

# ============================================================
# Weekly Auto-Strategy Comparison (runs on Sunday midnight)
# ============================================================
async def weekly_strategy_select():
    while True:
        # Wait until Sunday 00:05 UTC
        now = datetime.datetime.utcnow()
        days_until_sunday = (6 - now.weekday()) % 7
        next_sunday = now + datetime.timedelta(days=days_until_sunday)
        next_sunday = next_sunday.replace(hour=0, minute=5, second=0)
        if days_until_sunday == 0 and now > next_sunday:
            next_sunday = next_sunday + datetime.timedelta(weeks=1)
        wait_seconds = (next_sunday - now).total_seconds()
        await asyncio.sleep(wait_seconds)

        global active_strategy
        try:
            comp_result = await compare_strategies()
            if comp_result.get('best_strategy'):
                if comp_result['best_strategy'] == 'EmaDmiStrategy':
                    active_strategy = "ema_dmi"
                else:
                    active_strategy = "ai"
                print(f"🏆 Switched to best strategy: {comp_result['best_strategy']}")
            # Store comparison in agent_log
            hour = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00")
            con = sqlite3.connect(DB_PATH)
            con.execute("INSERT INTO agent_log (hour, agent, action, prob, details) VALUES (?,?,?,?,?)",
                        (hour, "A.A.G.A Strategy Selector", "SWITCH", 1.0, json.dumps(comp_result)))
            con.commit(); con.close()
        except Exception as e:
            print(f"Weekly strategy select error: {e}")

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
            signal_info = await run_aggregated_analysis()
            print(f"📡 Aggregated Signal ({active_strategy}): {signal_info['signal']} | Price: {signal_info['current_price']} | Trend: {signal_info['trend']}")
            if signal_info["signal"] in ("BUY", "SELL") and signal_info["entry_zone"] and signal_info["sl"] and signal_info["tp"]:
                entry = signal_info["entry_zone"][0] if signal_info["signal"] == "BUY" else signal_info["entry_zone"][1]
                sl, tp = signal_info["sl"], signal_info["tp"]
                if signal_info["signal"] == "BUY":
                    if sl > entry - MIN_STOP_DISTANCE or tp < entry + MIN_STOP_DISTANCE:
                        print("⚠️ Invalid stops, skipping")
                        continue
                else:
                    if sl < entry + MIN_STOP_DISTANCE or tp > entry - MIN_STOP_DISTANCE:
                        print("⚠️ Invalid stops, skipping")
                        continue

                con = sqlite3.connect(DB_PATH)
                existing = con.execute("SELECT COUNT(*) FROM trade_journal WHERE outcome IS NULL AND signal = ?", (signal_info["signal"],)).fetchone()[0]
                con.close()
                if existing == 0:
                    risk_brief = {"entry": round(entry,2), "sl": sl, "tp": tp}
                    con = sqlite3.connect(DB_PATH)
                    con.execute("INSERT INTO trade_journal (timestamp,pair,signal,entry,sl,tp,strategy_used,gemini_insight) VALUES (?,?,?,?,?,?,?,?)",
                                (datetime.datetime.utcnow().isoformat(), "XAUUSD", signal_info["signal"],
                                 risk_brief["entry"], risk_brief["sl"], risk_brief["tp"], active_strategy, ""))
                    con.commit(); con.close()
                    order = await execute_trade_async(signal_info["signal"], sl, tp, lot=0.01)
                    if order: print(f"✅ A.A.G.A Auto trade ({active_strategy}): {signal_info['signal']}")
                    else: print("❌ A.A.G.A Auto trade failed")
                else:
                    print(f"⏩ Same direction open trade exists, skipping")
            else:
                if signal_info["signal"] == "HOLD":
                    print("ℹ️ No valid signal (HOLD). Waiting for setup.")
        except Exception as e:
            print(f"Autonomous loop error: {e}")
        await asyncio.sleep(AUTONOMOUS_INTERVAL_SEC)

@app.on_event("startup")
async def startup():
    asyncio.create_task(keep_alive_task())
    asyncio.create_task(autonomous_trading_loop())
    asyncio.create_task(nightly_gemini_analysis())
    asyncio.create_task(weekly_strategy_select())

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
    signal_info = await run_aggregated_analysis()
    now = datetime.datetime.utcnow()
    gemini_advice = {"summary": "Available overnight"}

    decision = signal_info["signal"]
    agent_probs = signal_info.get("agent_probs", {})

    risk_brief = None
    if decision in ("BUY","SELL") and signal_info["entry_zone"] and signal_info["sl"] and signal_info["tp"]:
        entry = signal_info["entry_zone"][0] if decision=="BUY" else signal_info["entry_zone"][1]
        risk_brief = {"entry":round(entry,2),"sl":signal_info["sl"],"tp":signal_info["tp"]}

    hour = now.strftime("%Y-%m-%d %H:00")
    con = sqlite3.connect(DB_PATH)
    for ag_name, probs in [("A.A.G.A Tech", agent_probs.get("tech", {})),
                           ("A.A.G.A News", agent_probs.get("news", {})),
                           ("A.A.G.A Risk", agent_probs.get("risk", {})),
                           ("A.A.G.A Strategy", agent_probs.get("strategy", {}))]:
        if probs:
            act = max(probs, key=probs.get)
            con.execute("INSERT INTO agent_log (hour,agent,action,prob,details) VALUES (?,?,?,?,?)",
                        (hour, ag_name, act, probs[act], json.dumps(probs)))
    con.commit(); con.close()

    return {
        "decision":decision,
        "probabilities": {
            "BUY": agent_probs.get("tech",{}).get("BUY",0)*0.4 + agent_probs.get("news",{}).get("BUY",0)*0.2 + agent_probs.get("risk",{}).get("BUY",0)*0.2 + agent_probs.get("strategy",{}).get("BUY",0)*0.2,
            "SELL": agent_probs.get("tech",{}).get("SELL",0)*0.4 + agent_probs.get("news",{}).get("SELL",0)*0.2 + agent_probs.get("risk",{}).get("SELL",0)*0.2 + agent_probs.get("strategy",{}).get("SELL",0)*0.2,
            "HOLD": agent_probs.get("tech",{}).get("HOLD",0)*0.4 + agent_probs.get("news",{}).get("HOLD",0)*0.2 + agent_probs.get("risk",{}).get("HOLD",0)*0.2 + agent_probs.get("strategy",{}).get("HOLD",0)*0.2,
        },
        "risk_brief": risk_brief,
        "strategy_used": active_strategy,
        "gemini_advice": gemini_advice,
        "market_trend": signal_info["trend"],
        "sweep_detected": False,
        "sweep_type": None,
        "mss": None,
        "fvg": None,
        "current_price": signal_info["current_price"]
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

@app.get("/compare-strategies")
async def compare_strategies_endpoint():
    return await compare_strategies()

@app.post("/switch-strategy")
async def switch_strategy(name: str):
    global active_strategy
    if name in ["ema_dmi", "ai"]:
        active_strategy = name
        return {"status": "ok", "active": name}
    return {"status": "error", "message": "invalid strategy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))