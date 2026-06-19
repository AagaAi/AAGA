# main.py – A.A.G.A AI (More Trades + Logging)
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
from agents.weekly_pipeline import weekly_self_retraining
from backtest import run_comparison as compare_strategies

strategy_factory = None

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

if gemini_analyst.available:
    try:
        from agents.strategy_factory import StrategyFactory
        strategy_factory = StrategyFactory(gemini_analyst)
        print("✅ Strategy Factory enabled")
    except Exception as e:
        print(f"Strategy Factory init error: {e}")
        strategy_factory = None
else:
    print("⚠️ Gemini unavailable – Strategy Factory disabled")

# MetaApi singleton (async) – unchanged
_api_instance   = None
_account_cache  = None

async def get_account():
    global _api_instance, _account_cache
    if _account_cache is not None: return _account_cache
    _api_instance  = MetaApi(METAAPI_TOKEN)
    account        = await _api_instance.metatrader_account_api.get_account(METAAPI_ACCOUNT_ID)
    if account.state not in ('DEPLOYING', 'DEPLOYED'): await account.deploy()
    await account.wait_connected()
    _account_cache = account
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

async def update_pending_trades():
    price = await sdk_get_price_async("XAUUSD")
    if price is None: return
    candles_1h = await sdk_get_candles_async("XAUUSD", "1h", 100)
    market_condition = market_classifier.classify(candles_1h) if candles_1h else "Unknown"

    con = sqlite3.connect(DB_PATH)
    trades = con.execute("SELECT id, pair, signal, entry, sl, tp, outcome, pnl, strategy_used FROM trade_journal WHERE outcome IS NULL").fetchall()
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
            con.execute("UPDATE trade_journal SET outcome=?, pnl=?, market_condition=? WHERE id=?", 
                        (outcome, pnl, market_condition, tid))
            if outcome in ("WIN","LOSS"):
                risk_manager.update_pnl(pnl)
                action_map = {"BUY": 0, "SELL": 1, "HOLD": 2}
                features = [entry/2000, sl/2000, tp/2000, abs(entry-sl)/10, abs(tp-entry)/10]
                trade_memory.add_experience({
                    "features": features,
                    "action": action_map.get(sig, 2),
                    "reward": pnl / 100
                })
    con.commit(); con.close()

# Nightly tasks unchanged
async def nightly_tasks():
    while True:
        now = datetime.datetime.utcnow()
        next_midnight = (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
        wait_seconds = (next_midnight - now).total_seconds()
        await asyncio.sleep(wait_seconds)

        if trade_memory.retrain_model(ai_strat_instance.model):
            ai_strat_instance.trained = True
            print("✅ RandomForest retrained (nightly)")

        con = sqlite3.connect(DB_PATH)
        rows = con.execute("""
            SELECT strategy_used, COUNT(*) as total,
                   SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins
            FROM trade_journal
            WHERE date(timestamp) = date('now', '-1 day') AND outcome IS NOT NULL
            GROUP BY strategy_used
        """).fetchall()
        con.close()
        for r in rows:
            name, total, wins = r[0], r[1], r[2] if r[2] else 0
            win_rate = wins / total if total > 0 else 0.5
            master_agent.update_performance(name, win_rate)

        if gemini_analyst.available and gemini_analyst.disabled_until <= _time.time():
            con = sqlite3.connect(DB_PATH)
            rows = con.execute("""
                SELECT strategy_used, id, signal, entry, sl, tp, outcome, pnl, timestamp
                FROM trade_journal
                WHERE date(timestamp) = date('now', '-1 day') AND outcome IS NOT NULL
                ORDER BY strategy_used, timestamp
            """).fetchall()
            con.close()
            if rows:
                from collections import defaultdict
                grouped = defaultdict(list)
                for r in rows:
                    strat_name, tid, sig, entry, sl, tp, outc, pnl, ts = r
                    grouped[strat_name].append({
                        "id": tid, "signal": sig, "entry": entry, "sl": sl, "tp": tp,
                        "outcome": outc, "pnl": pnl, "timestamp": ts
                    })
                for strat_name, trades in grouped.items():
                    daily_insight = gemini_analyst.analyze_daily_trades(trades)
                    print(f"📊 {strat_name} Nightly Analysis: {daily_insight.get('summary', 'No insight')}")
                    hour = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00")
                    con = sqlite3.connect(DB_PATH)
                    con.execute("INSERT INTO agent_log (hour, agent, action, prob, details) VALUES (?,?,?,?,?)",
                                (hour, f"A.A.G.A Gemini ({strat_name})", "DAILY_REVIEW", 1.0, json.dumps(daily_insight)))
                    con.commit(); con.close()
                    suggestions = daily_insight.get("parameter_suggestions", {})
                    if suggestions:
                        for strat in active_strategies:
                            if strat.name == strat_name:
                                for key, val in suggestions.items():
                                    if hasattr(strat, key): setattr(strat, key, val)
                                print(f"✅ {strat_name} self‑optimized: {suggestions}")
        else:
            print("ℹ️ Gemini unavailable – skipping nightly analysis")

        if strategy_factory and gemini_analyst.available and gemini_analyst.disabled_until <= _time.time():
            try:
                candles_1h = await sdk_get_candles_async("XAUUSD", "1h", 100)
                if candles_1h and len(candles_1h) >= 50:
                    closes_1h = [c['close'] for c in candles_1h]
                    ema50_1h = pd.Series(closes_1h).ewm(span=50).mean().iloc[-1]
                    market_context = {
                        "current_price": closes_1h[-1],
                        "trend": "UPTREND" if closes_1h[-1] > ema50_1h else "DOWNTREND",
                        "volatility": round(pd.Series([c['high']-c['low'] for c in candles_1h[-14:]]).mean(), 2)
                    }
                else:
                    market_context = {"current_price": 0, "trend": "Unknown", "volatility": 0}

                con = sqlite3.connect(DB_PATH)
                recent = con.execute("SELECT signal, entry, sl, tp, outcome, pnl FROM trade_journal ORDER BY id DESC LIMIT 10").fetchall()
                con.close()
                recent_trades = [{"signal": r[0], "entry": r[1], "sl": r[2], "tp": r[3], "outcome": r[4], "pnl": r[5]} for r in recent]

                new_params = strategy_factory.generate_strategy_params(market_context, recent_trades)
                if new_params:
                    new_strategy = strategy_factory.create_strategy_from_params(new_params)
                    if new_strategy:
                        if not any(s.name == new_strategy.name for s in active_strategies):
                            active_strategies.append(new_strategy)
                            master_agent.strategies = active_strategies
                            print(f"🌟 New strategy '{new_strategy.name}' added!")
                            hour = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00")
                            con = sqlite3.connect(DB_PATH)
                            con.execute("INSERT INTO agent_log (hour, agent, action, prob, details) VALUES (?,?,?,?,?)",
                                        (hour, "A.A.G.A Factory", "NEW_STRATEGY", 1.0, json.dumps(new_params)))
                            con.commit(); con.close()
            except Exception as e:
                print(f"Strategy generation error: {e}")

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
            ohlc, strategy_signals = await run_all_strategies()
            if ohlc is None:
                await asyncio.sleep(AUTONOMOUS_INTERVAL_SEC)
                continue

            # ---- DETAILED LOGGING ----
            for strat, sig in strategy_signals:
                print(f"  📡 {strat.name}: {sig.get('signal')} | Price: {sig.get('current_price')} | ADX: {sig.get('adx')} | BOS: {sig.get('bos')}")

            signal_dicts = []
            for strat, sig in strategy_signals:
                signal_dicts.append({
                    "strategy": strat.name,
                    "signal": sig.get("signal"),
                    "entry_zone": sig.get("entry_zone"),
                    "sl": sig.get("sl"),
                    "tp": sig.get("tp"),
                    "current_price": sig.get("current_price")
                })

            final_decision = master_agent.decide(signal_dicts)
            decision_signal = final_decision["signal"]
            if decision_signal not in ("BUY", "SELL"):
                print(f"ℹ️ Master decision: {decision_signal} – reason: {final_decision.get('reason')}")
                await asyncio.sleep(AUTONOMOUS_INTERVAL_SEC)
                continue

            entry_zone = final_decision.get("entry_zone")
            sl = final_decision.get("sl")
            tp = final_decision.get("tp")
            if not entry_zone or not sl or not tp:
                print("⚠️ Master decision missing entry/SL/TP")
                continue

            entry = entry_zone[0] if decision_signal == "BUY" else entry_zone[1]
            if decision_signal == "BUY":
                if sl > entry - MIN_STOP_DISTANCE or tp < entry + MIN_STOP_DISTANCE:
                    print("⚠️ Invalid stops, skipping")
                    continue
            else:
                if sl < entry + MIN_STOP_DISTANCE or tp > entry - MIN_STOP_DISTANCE:
                    print("⚠️ Invalid stops, skipping")
                    continue

            con = sqlite3.connect(DB_PATH)
            existing = con.execute("SELECT COUNT(*) FROM trade_journal WHERE outcome IS NULL AND signal = ?", (decision_signal,)).fetchone()[0]
            con.close()
            if existing == 0:
                lot = risk_manager.calculate_lot(entry, sl)
                risk_brief = {"entry": round(entry,2), "sl": sl, "tp": tp}
                strat_used = final_decision.get("strategy_used", "Master")
                candles_1h = await sdk_get_candles_async("XAUUSD", "1h", 100)
                mkt_condition = market_classifier.classify(candles_1h) if candles_1h else "Unknown"
                con = sqlite3.connect(DB_PATH)
                con.execute("INSERT INTO trade_journal (timestamp,pair,signal,entry,sl,tp,strategy_used,gemini_insight,market_condition) VALUES (?,?,?,?,?,?,?,?,?)",
                            (datetime.datetime.utcnow().isoformat(), "XAUUSD", decision_signal,
                             risk_brief["entry"], risk_brief["sl"], risk_brief["tp"],
                             strat_used, "", mkt_condition))
                con.commit(); con.close()
                order = await execute_trade_async(decision_signal, sl, tp, lot)
                if order: print(f"✅ Master Auto trade ({strat_used}): {decision_signal} | Lot: {lot}")
                else: print("❌ Master Auto trade failed")
            else:
                print("⏩ Same direction open trade exists, skipping")
        except Exception as e:
            print(f"Autonomous loop error: {e}")
        await asyncio.sleep(AUTONOMOUS_INTERVAL_SEC)

@app.on_event("startup")
async def startup():
    asyncio.create_task(keep_alive_task())
    asyncio.create_task(autonomous_trading_loop())
    asyncio.create_task(nightly_tasks())
    asyncio.create_task(weekly_scheduler())

def init_db():
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("ALTER TABLE trade_journal ADD COLUMN market_condition TEXT DEFAULT 'Unknown'")
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

async def weekly_scheduler():
    while True:
        now = datetime.datetime.utcnow()
        days_until_sunday = (6 - now.weekday()) % 7
        next_sunday = now + datetime.timedelta(days=days_until_sunday)
        next_sunday = next_sunday.replace(hour=0, minute=5, second=0)
        if days_until_sunday == 0 and now > next_sunday:
            next_sunday = next_sunday + datetime.timedelta(weeks=1)
        wait_seconds = (next_sunday - now).total_seconds()
        await asyncio.sleep(wait_seconds)
        await weekly_self_retraining(DB_PATH, active_strategies, master_agent, trade_memory, gemini_analyst, strategy_factory)

# Routes unchanged (same as previous)
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return FileResponse("dashboard.html")

@app.get("/health")
async def health():
    return {"status": "ok", "autonomous_loop": "active", "strategies": [s.name for s in active_strategies], "gemini_available": gemini_analyst.available and gemini_analyst.disabled_until <= _time.time()}

# (other routes omitted for brevity – they are same as before)