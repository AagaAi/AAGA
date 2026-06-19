# main.py – A.A.G.A AI (Stable, Hourly Intelligence every 3h, Gemini Fixed)
import os, json, sqlite3, datetime, time as _time, asyncio, aiohttp
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
from metaapi_cloud_sdk import MetaApi
from agents.gemini_agent import GeminiAnalyst
from agents.strategy_agent import EmaDmiStrategy
from agents.ai_agent import RandomForestStrategy
from agents.master_agent import MasterAgent
from agents.memory import TradeMemory
from agents.news_agent import NewsSentimentAgent
from agents.market_classifier import MarketClassifier
from agents.strategy_factory import StrategyFactory
from agents.weekly_pipeline import weekly_self_retraining
from backtest import run_comparison as compare_strategies

# ---------- Config ----------
CONFIG_PATH = "config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)

GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
METAAPI_TOKEN      = os.environ.get("METAAPI_TOKEN", "")
METAAPI_ACCOUNT_ID = os.environ.get("METAAPI_ACCOUNT_ID", "")

DB_PATH = "journal.db"

app = FastAPI(title="A.A.G.A AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

gemini_analyst = GeminiAnalyst(GEMINI_API_KEY)
ema_dmi_strat = EmaDmiStrategy(config.get("parameters", {}))
ai_strat_instance = RandomForestStrategy()
active_strategies = [ema_dmi_strat, ai_strat_instance]

news_agent = NewsSentimentAgent()
market_classifier = MarketClassifier()

# Strategy factory only if Gemini available
strategy_factory = StrategyFactory(gemini_analyst) if gemini_analyst.available else None

# Stores the latest hourly insight for dashboard
latest_hourly_insight = {"summary": "Waiting for first analysis...", "timestamp": None}

def run_news_analysis():
    result = news_agent.analyze()
    return {"prob_hold": result.get("prob_hold", 0.0), "full": result}

class RiskManager:
    def __init__(self, balance=10000, risk_per_trade=0.02):
        self.balance = balance
        self.risk_per_trade = risk_per_trade
        self.daily_pnl = 0.0
        self.loss_limit = balance * 0.05

    def evaluate(self):
        if -self.daily_pnl >= self.loss_limit:
            return {"prob_buy":0.0, "prob_sell":0.0, "prob_hold":1.0, "halt": True}
        return {"prob_buy":0.33, "prob_sell":0.33, "prob_hold":0.34, "halt": False}

    def update_pnl(self, pnl):
        self.daily_pnl += pnl
        self.balance += pnl

    def calculate_lot(self, entry, sl):
        if sl is None or entry is None: return 0.01
        risk_amount = self.balance * self.risk_per_trade
        sl_distance = abs(entry - sl)
        if sl_distance < 0.5: sl_distance = 0.5
        lot = risk_amount / (sl_distance * 100)
        lot = max(0.01, round(lot, 2))
        return lot

risk_manager = RiskManager()
master_agent = MasterAgent(active_strategies, risk_manager.evaluate, run_news_analysis)
trade_memory = TradeMemory(DB_PATH)

# MetaApi singleton with health flag
_api_instance   = None
_account_cache  = None
_metaapi_healthy = True

async def get_account():
    global _api_instance, _account_cache, _metaapi_healthy
    if _account_cache is not None:
        return _account_cache
    if not _metaapi_healthy:
        raise Exception("MetaApi currently unavailable")
    _api_instance  = MetaApi(METAAPI_TOKEN)
    account        = await _api_instance.metatrader_account_api.get_account(METAAPI_ACCOUNT_ID)
    if account.state not in ('DEPLOYING', 'DEPLOYED'):
        await account.deploy()
    await account.wait_connected()
    _account_cache = account
    _metaapi_healthy = True
    return account

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
    except Exception as e: print(f"SDK fetch error: {e}"); return None

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

async def execute_trade_async(signal, sl, tp, lot):
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

MIN_STOP_DISTANCE = 1.10

async def get_strategy_signal(strat, ohlc):
    return strat.get_signal(ohlc)

async def run_all_strategies():
    raw1m = await sdk_get_candles_async("XAUUSD", "1m", 200)
    if not raw1m or len(raw1m) < 50: return None, None
    highs  = [c['high'] for c in raw1m]
    lows   = [c['low'] for c in raw1m]
    closes = [c['close'] for c in raw1m]
    opens  = [c['open'] for c in raw1m]
    ohlc = {'highs': highs, 'lows': lows, 'closes': closes, 'opens': opens}
    signals = []
    for strat in active_strategies:
        sig = await get_strategy_signal(strat, ohlc)
        signals.append((strat, sig))
    return ohlc, signals

# Database helper with retry for locked DB
def db_execute(statement, params=(), commit=False, fetch=False):
    for _ in range(5):
        try:
            con = sqlite3.connect(DB_PATH)
            if commit:
                con.execute(statement, params)
                con.commit()
                result = None
            elif fetch:
                rows = con.execute(statement, params).fetchall()
                result = rows
            con.close()
            return result
        except sqlite3.OperationalError as e:
            if "locked" in str(e): _time.sleep(0.2)
            else: raise
    return None

async def update_pending_trades():
    price = await sdk_get_price_async("XAUUSD")
    if price is None: return
    candles_1h = await sdk_get_candles_async("XAUUSD", "1h", 100)
    market_condition = market_classifier.classify(candles_1h) if candles_1h else "Unknown"
    trades = db_execute("SELECT id, pair, signal, entry, sl, tp, outcome, pnl, strategy_used FROM trade_journal WHERE outcome IS NULL", fetch=True)
    if not trades: return
    for tid, pair, sig, entry, sl, tp, _, _, strat_name in trades:
        if entry is None: continue
        outcome = pnl = None
        if sig == "BUY":
            if price <= sl: outcome, pnl = "LOSS", round((sl - entry) * 100, 2)
            elif price >= tp: outcome, pnl = "WIN",  round((tp - entry) * 100, 2)
        else:
            if price >= sl: outcome, pnl = "LOSS", round((entry - sl) * 100, 2)
            elif price <= tp: outcome, pnl = "WIN",  round((entry - tp) * 100, 2)
        if outcome:
            db_execute("UPDATE trade_journal SET outcome=?, pnl=?, market_condition=? WHERE id=?", 
                       (outcome, pnl, market_condition, tid), commit=True)
            if outcome in ("WIN","LOSS"):
                risk_manager.update_pnl(pnl)
                action_map = {"BUY": 0, "SELL": 1, "HOLD": 2}
                features = [entry/2000, sl/2000, tp/2000, abs(entry-sl)/10, abs(tp-entry)/10]
                trade_memory.add_experience({"features": features, "action": action_map.get(sig, 2), "reward": pnl / 100})

# ============================================================
# Hourly Gemini Intelligence (every 3 hours)
# ============================================================
def _json_safe_params(strat):
    """Return a dict of only JSON‑serializable attributes of a strategy."""
    safe = {}
    for attr in dir(strat):
        if attr.startswith('_') or callable(getattr(strat, attr)):
            continue
        val = getattr(strat, attr)
        if isinstance(val, (int, float, str, bool, list, dict, tuple)):
            safe[attr] = val
    return safe

async def hourly_gemini_intelligence():
    global latest_hourly_insight
    await asyncio.sleep(5)
    while True:
        try:
            if not gemini_analyst.available or gemini_analyst.disabled_until > _time.time():
                print("⏳ Gemini unavailable for intelligence cycle")
                await asyncio.sleep(10800)
                continue

            trades_24h = db_execute("""
                SELECT signal, entry, sl, tp, outcome, pnl, market_condition, timestamp 
                FROM trade_journal 
                WHERE timestamp >= datetime('now', '-1 day') AND outcome IS NOT NULL
                ORDER BY timestamp DESC LIMIT 100
            """, fetch=True) or []
            
            news_result = news_agent.analyze()
            candles_1h = await sdk_get_candles_async("XAUUSD", "1h", 100)
            market_state = "Unknown"
            if candles_1h and len(candles_1h) >= 20:
                closes_1h = [c['close'] for c in candles_1h]
                ema50 = pd.Series(closes_1h).ewm(span=50).mean().iloc[-1] if len(closes_1h) >= 50 else sum(closes_1h)/len(closes_1h)
                market_state = f"Price: {closes_1h[-1]:.2f}, Trend: {'UPTREND' if closes_1h[-1] > ema50 else 'DOWNTREND'}"
            
            # Build JSON‑safe strategy parameters
            strategy_params = {}
            for strat in active_strategies:
                strategy_params[strat.name] = _json_safe_params(strat)

            prompt = f"""
You are A.A.G.A AI's central intelligence. Analyze the following comprehensive data and provide a single JSON response with these keys:
- "performance_summary": brief assessment of recent trades (win rate, P&L, market condition).
- "parameter_adjustments": dict of strategy name -> dict of parameters to change. Only include if needed.
- "suggested_trade": {{"action": "BUY"/"SELL"/"HOLD", "entry": price, "sl": price, "tp": price, "confidence": 0.0-1.0}}.
- "new_strategy_params": optional, provide a complete strategy definition like the factory format (name, buy_conditions, sell_conditions, sl_atr_multiplier, tp_atr_multiplier, min_stop_distance).
- "market_outlook": short sentence about next few hours.

Data:
Recent Trades: {json.dumps(trades_24h[:30], default=str)}
Current Market: {market_state}
News: {json.dumps(news_result)}
Strategy Parameters: {json.dumps(strategy_params)}
Active Strategies: {[s.name for s in active_strategies]}

Respond ONLY with the JSON.
"""
            response = gemini_analyst._safe_generate(prompt)
            if not response:
                print("❌ Gemini call failed")
                await asyncio.sleep(10800)
                continue

            try:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                insight = json.loads(response)
            except Exception as e:
                print(f"❌ Gemini response parse error: {e}")
                await asyncio.sleep(10800)
                continue

            print(f"🧠 Gemini Insight: {insight.get('performance_summary', 'No summary')}")
            
            param_adjustments = insight.get("parameter_adjustments", {})
            for strat_name, params in param_adjustments.items():
                for strat in active_strategies:
                    if strat.name == strat_name:
                        for key, val in params.items():
                            if hasattr(strat, key):
                                setattr(strat, key, val)
                                print(f"  🔧 {strat_name}: {key} = {val}")
            
            new_strat = insight.get("new_strategy_params")
            if new_strat and strategy_factory:
                try:
                    new_strategy = strategy_factory.create_strategy_from_params(new_strat)
                    if new_strategy and not any(s.name == new_strategy.name for s in active_strategies):
                        active_strategies.append(new_strategy)
                        master_agent.strategies = active_strategies
                        print(f"  🌟 New strategy added: {new_strategy.name}")
                except Exception as e:
                    print(f"  ⚠️ Failed to create new strategy: {e}")
            
            insight["timestamp"] = datetime.datetime.utcnow().isoformat()
            latest_hourly_insight = insight
            
            hour = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00")
            db_execute("INSERT INTO agent_log (hour, agent, action, prob, details) VALUES (?,?,?,?,?)",
                       (hour, "A.A.G.A Gemini", "INTELLIGENCE", 1.0, json.dumps(insight)), commit=True)

        except Exception as e:
            print(f"Intelligence cycle error: {e}")
        
        await asyncio.sleep(10800)   # every 3 hours

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
    global _metaapi_healthy, _account_cache
    while True:
        try:
            if not _metaapi_healthy:
                _account_cache = None
                try:
                    await get_account()
                    _metaapi_healthy = True
                except:
                    await asyncio.sleep(30)
                    continue

            await update_pending_trades()
            ohlc, strategy_signals = await run_all_strategies()
            if ohlc is None:
                await asyncio.sleep(AUTONOMOUS_INTERVAL_SEC)
                continue

            for strat, sig in strategy_signals:
                print(f"  📡 {strat.name}: {sig.get('signal')} | Price: {sig.get('current_price')}")

            signal_dicts = [{"strategy": strat.name, "signal": sig.get("signal"),
                             "entry_zone": sig.get("entry_zone"), "sl": sig.get("sl"),
                             "tp": sig.get("tp"), "current_price": sig.get("current_price")}
                            for strat, sig in strategy_signals]

            final_decision = master_agent.decide(signal_dicts)
            decision_signal = final_decision["signal"]
            if decision_signal not in ("BUY", "SELL"):
                await asyncio.sleep(AUTONOMOUS_INTERVAL_SEC)
                continue

            entry_zone = final_decision.get("entry_zone")
            sl = final_decision.get("sl")
            tp = final_decision.get("tp")
            if not entry_zone or not sl or not tp: continue

            entry = entry_zone[0] if decision_signal == "BUY" else entry_zone[1]
            if decision_signal == "BUY":
                if sl > entry - MIN_STOP_DISTANCE or tp < entry + MIN_STOP_DISTANCE: continue
            else:
                if sl < entry + MIN_STOP_DISTANCE or tp > entry - MIN_STOP_DISTANCE: continue

            existing = db_execute("SELECT COUNT(*) FROM trade_journal WHERE outcome IS NULL AND signal = ?", (decision_signal,), fetch=True)
            if existing and existing[0][0] > 0:
                print("⏩ Same direction open trade exists, skipping")
                continue

            lot = risk_manager.calculate_lot(entry, sl)
            risk_brief = {"entry": round(entry,2), "sl": sl, "tp": tp}
            strat_used = final_decision.get("strategy_used", "Master")
            candles_1h = await sdk_get_candles_async("XAUUSD", "1h", 100)
            mkt_condition = market_classifier.classify(candles_1h) if candles_1h else "Unknown"
            db_execute("INSERT INTO trade_journal (timestamp,pair,signal,entry,sl,tp,strategy_used,gemini_insight,market_condition) VALUES (?,?,?,?,?,?,?,?,?)",
                       (datetime.datetime.utcnow().isoformat(), "XAUUSD", decision_signal,
                        risk_brief["entry"], risk_brief["sl"], risk_brief["tp"],
                        strat_used, "", mkt_condition), commit=True)
            order = await execute_trade_async(decision_signal, sl, tp, lot)
            if order: print(f"✅ Master Auto trade ({strat_used}): {decision_signal} | Lot: {lot}")
            else: print("❌ Master Auto trade failed")
        except Exception as e:
            print(f"Autonomous loop error: {e}")
            _metaapi_healthy = False
            _account_cache = None
            await asyncio.sleep(30)
        await asyncio.sleep(AUTONOMOUS_INTERVAL_SEC)

# ============================================================
# Nightly tasks (retrain + weight update)
# ============================================================
async def nightly_tasks():
    while True:
        now = datetime.datetime.utcnow()
        next_midnight = (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
        await asyncio.sleep((next_midnight - now).total_seconds())

        if trade_memory.retrain_model(ai_strat_instance.model):
            ai_strat_instance.trained = True
            print("✅ RandomForest retrained (nightly)")

        rows = db_execute("""
            SELECT strategy_used, COUNT(*) as total,
                   SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins
            FROM trade_journal
            WHERE date(timestamp) = date('now', '-1 day') AND outcome IS NOT NULL
            GROUP BY strategy_used
        """, fetch=True)
        if rows:
            for r in rows:
                name, total, wins = r[0], r[1], r[2] or 0
                win_rate = wins / total if total > 0 else 0.5
                master_agent.update_performance(name, win_rate)

# ============================================================
# Weekly self‑retraining
# ============================================================
async def weekly_scheduler():
    while True:
        now = datetime.datetime.utcnow()
        days_until_sunday = (6 - now.weekday()) % 7
        next_sunday = now + datetime.timedelta(days=days_until_sunday)
        next_sunday = next_sunday.replace(hour=0, minute=5, second=0)
        if days_until_sunday == 0 and now > next_sunday:
            next_sunday = next_sunday + datetime.timedelta(weeks=1)
        await asyncio.sleep((next_sunday - now).total_seconds())
        await weekly_self_retraining(DB_PATH, active_strategies, master_agent, trade_memory, gemini_analyst, strategy_factory)

@app.on_event("startup")
async def startup():
    asyncio.create_task(keep_alive_task())
    asyncio.create_task(autonomous_trading_loop())
    asyncio.create_task(nightly_tasks())
    asyncio.create_task(weekly_scheduler())
    asyncio.create_task(hourly_gemini_intelligence())

def init_db():
    con = sqlite3.connect(DB_PATH)
    try: con.execute("ALTER TABLE trade_journal ADD COLUMN market_condition TEXT DEFAULT 'Unknown'")
    except: pass
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
            gemini_insight TEXT,
            market_condition TEXT DEFAULT 'Unknown'
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

# Dashboard routes
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return FileResponse("dashboard.html")

@app.get("/health")
async def health():
    return {"status": "ok", "autonomous_loop": "active", "metaapi_healthy": _metaapi_healthy, "strategies": [s.name for s in active_strategies]}

@app.get("/master-signal")
async def master_signal():
    ohlc, strategy_signals = await run_all_strategies()
    if ohlc is None or strategy_signals is None:
        return {"decision": "HOLD", "probabilities": {"BUY": 0.33, "SELL": 0.33, "HOLD": 0.34},
                "risk_brief": None, "strategy_used": "MetaApi offline",
                "gemini_advice": {"summary": "Market data unavailable"},
                "market_trend": "Unknown", "current_price": 0, "strategy_signals": []}

    strategy_signals_list = [{"name": strat.name, "signal": sig.get("signal"),
                              "entry_zone": sig.get("entry_zone"), "sl": sig.get("sl"), "tp": sig.get("tp"),
                              "trend": sig.get("trend"), "current_price": sig.get("current_price")}
                             for strat, sig in strategy_signals]
    signal_dicts = [{"strategy": s["name"], "signal": s["signal"], "entry_zone": s["entry_zone"],
                     "sl": s["sl"], "tp": s["tp"], "current_price": s["current_price"]} for s in strategy_signals_list]
    master_decision = master_agent.decide(signal_dicts)

    conf = master_decision.get("confidence", 0.5)
    probs = {"BUY": conf if master_decision["signal"]=="BUY" else 0,
             "SELL": conf if master_decision["signal"]=="SELL" else 0,
             "HOLD": 1-conf if master_decision["signal"] in ("BUY","SELL") else 1.0}
    risk_brief = None
    if master_decision["signal"] in ("BUY","SELL") and master_decision.get("entry_zone"):
        entry = master_decision["entry_zone"][0] if master_decision["signal"]=="BUY" else master_decision["entry_zone"][1]
        risk_brief = {"entry": round(entry,2), "sl": master_decision["sl"], "tp": master_decision["tp"]}

    now = datetime.datetime.utcnow()
    db_execute("INSERT INTO agent_log (hour, agent, action, prob, details) VALUES (?,?,?,?,?)",
               (now.strftime("%Y-%m-%d %H:00"), "A.A.G.A Master", master_decision["signal"], conf, json.dumps(master_decision)), commit=True)

    return {
        "decision": master_decision["signal"],
        "probabilities": probs,
        "risk_brief": risk_brief,
        "strategy_used": master_decision.get("strategy_used", "Master"),
        "gemini_advice": latest_hourly_insight,
        "market_trend": strategy_signals_list[0].get("trend") if strategy_signals_list else "Unknown",
        "current_price": strategy_signals_list[0].get("current_price", 0) if strategy_signals_list else 0,
        "strategy_signals": strategy_signals_list
    }

@app.get("/agent-log")
def agent_log():
    rows = db_execute("SELECT hour, agent, action, prob, details FROM agent_log WHERE hour >= datetime('now','-1 hour') ORDER BY id DESC LIMIT 20", fetch=True)
    return [{"hour":r[0],"agent":r[1],"action":r[2],"prob":r[3],"details":r[4]} for r in rows] if rows else []

@app.get("/today-trades")
def today_trades():
    rows = db_execute("SELECT timestamp, pair, signal, entry, sl, tp, outcome, pnl, strategy_used, market_condition FROM trade_journal WHERE date(timestamp) = date('now') ORDER BY timestamp DESC", fetch=True)
    return [{"timestamp":r[0],"pair":r[1],"signal":r[2],"entry":r[3],"sl":r[4],"tp":r[5],"outcome":r[6],"pnl":r[7],"strategy":r[8],"market_condition":r[9]} for r in rows] if rows else []

@app.get("/trade-reviews")
def trade_reviews():
    rows = db_execute("SELECT id, timestamp, signal, entry, sl, tp, outcome, pnl, gemini_insight, strategy_used, market_condition FROM trade_journal WHERE outcome IS NOT NULL ORDER BY timestamp DESC LIMIT 50", fetch=True)
    return [{"id": r[0], "timestamp": r[1], "signal": r[2], "entry": r[3], "sl": r[4], "tp": r[5],
             "outcome": r[6], "pnl": r[7], "gemini_insight": r[8], "strategy": r[9], "market_condition": r[10]} for r in rows] if rows else []

@app.get("/strategy-performance")
def strategy_performance():
    rows = db_execute("""
        SELECT strategy_used, COUNT(*) as total,
               SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) as losses,
               SUM(pnl) as total_pnl
        FROM trade_journal WHERE date(timestamp) = date('now') AND outcome IS NOT NULL
        GROUP BY strategy_used
    """, fetch=True)
    result = {}
    if rows:
        for r in rows:
            name = r[0] or "unknown"; total = r[1]; wins = r[2] or 0; losses = r[3] or 0
            win_rate = wins / total * 100 if total > 0 else 0
            result[name] = {"total_trades": total, "wins": wins, "losses": losses, "win_rate": round(win_rate,1), "pnl": round(r[4] or 0,2)}
    return result

@app.get("/market-condition-performance")
def market_condition_performance():
    rows = db_execute("""
        SELECT market_condition, COUNT(*) as total,
               SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) as losses,
               SUM(pnl) as total_pnl
        FROM trade_journal WHERE date(timestamp) = date('now') AND outcome IS NOT NULL AND market_condition IS NOT NULL
        GROUP BY market_condition
    """, fetch=True)
    result = {}
    if rows:
        for r in rows:
            cond = r[0] or "Unknown"; total = r[1]; wins = r[2] or 0; losses = r[3] or 0
            win_rate = wins / total * 100 if total > 0 else 0
            result[cond] = {"total_trades": total, "wins": wins, "losses": losses, "win_rate": round(win_rate,1), "pnl": round(r[4] or 0,2)}
    return result

@app.get("/ai-insights")
async def ai_insights():
    return latest_hourly_insight

@app.get("/compare-strategies")
async def compare_strategies_endpoint():
    return await compare_strategies()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))