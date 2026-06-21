# main.py – A.A.G.A AI (Portfolio Stats Initialized)
import os, json, sqlite3, datetime, time as _time, asyncio, aiohttp, feedparser
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
from metaapi_cloud_sdk import MetaApi
import yfinance as yf
from agents.gemini_agent import GeminiAnalyst
from agents.strategy_agent import EmaDmiStrategy
from agents.ai_agent import RandomForestStrategy
try:
    from agents.ppo_agent import PPOAgent
except:
    class PPOAgent:
        name = "PPO"
        def get_signal(self, ohlc):
            return {"signal":"HOLD","entry_zone":None,"sl":None,"tp":None,"trend":"Unknown","current_price":0}
        def train(self, *args, **kwargs): pass
from agents.master_agent import MasterAgent
from agents.memory import TradeMemory
from agents.news_agent import NewsSentimentAgent
from agents.market_classifier import MarketClassifier
from agents.strategy_factory import StrategyFactory
from agents.weekly_pipeline import weekly_self_retraining
from agents.economic_calendar import is_high_impact_now, get_upcoming_events
from agents.telegram_notifier import TelegramNotifier
from agents.regime_detector import RegimeDetector
from agents.genetic_optimizer import GeneticOptimizer
from agents.gan_augmentor import LightweightGAN
from agents.portfolio_manager import PortfolioManager
from backtest import run_comparison as compare_strategies

# ---------- Config ----------
CONFIG_PATH = "config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)

GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
METAAPI_TOKEN      = os.environ.get("METAAPI_TOKEN", "")
METAAPI_ACCOUNT_ID = os.environ.get("METAAPI_ACCOUNT_ID", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

DB_PATH = "journal.db"

app = FastAPI(title="A.A.G.A AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

gemini_analyst = GeminiAnalyst(GEMINI_API_KEY)
telegram = TelegramNotifier()
news_agent = NewsSentimentAgent()
try:
    news_agent.analyze()
    print("📰 NLP news agent loaded")
except: pass

market_classifier = MarketClassifier()
regime_detector = RegimeDetector()
genetic_optimizer = GeneticOptimizer()
gan_model = LightweightGAN()

# ---------- Risk Manager ----------
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

# ---------- Multi‑pair (3 pairs) ----------
pairs = ["XAUUSD", "EURUSD", "BTCUSD"]
YAHOO_SYMBOLS = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X", "BTCUSD": "BTC-USD"}

active_strategies = {}
for pair in pairs:
    active_strategies[pair] = [
        EmaDmiStrategy(config.get("parameters", {})),
        RandomForestStrategy(),
        PPOAgent()
    ]

risk_managers = {pair: RiskManager() for pair in pairs}
master_agents = {}
for pair in pairs:
    master_agents[pair] = MasterAgent(
        active_strategies[pair],
        risk_managers[pair].evaluate,
        lambda: {"prob_hold": news_agent.analyze().get("prob_hold", 0.0)}
    )

portfolio_mgr = PortfolioManager(pairs)

def run_news_analysis():
    result = news_agent.analyze()
    return {"prob_hold": result.get("prob_hold", 0.0), "full": result}

USE_FALLBACK = False

latest_signals_cache = {
    pair: {
        "decision": "HOLD",
        "probabilities": {"BUY": 0.33, "SELL": 0.33, "HOLD": 0.34},
        "risk_brief": None,
        "strategy_used": "Waiting for data",
        "market_trend": "Unknown",
        "current_price": 0,
        "strategy_signals": [],
        "regime": "UNKNOWN"
    } for pair in pairs
}

trade_memory = TradeMemory(DB_PATH)
strategy_factory = StrategyFactory(gemini_analyst) if gemini_analyst.available else None
latest_intelligence = {"summary": "Waiting for first 15‑min analysis...", "timestamp": None}

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

async def sdk_get_candles_metaapi(pair, timeframe, limit=500):
    if not METAAPI_TOKEN or not METAAPI_ACCOUNT_ID: return None
    try:
        account = await get_account()
        start_time = datetime.datetime.utcnow() + datetime.timedelta(days=1)
        candles_raw = await account.get_historical_candles(
            symbol=pair, timeframe=timeframe, start_time=start_time, limit=limit)
        if not candles_raw: return None
        return [{"time": str(c.get("time", "")), "open": float(c.get("open", 0)), "high": float(c.get("high", 0)),
                 "low": float(c.get("low", 0)), "close": float(c.get("close", 0)),
                 "volume": float(c.get("tickVolume", c.get("volume", 0)))} for c in candles_raw]
    except: return None

def sdk_get_candles_yahoo(pair, interval="1m", period="5d"):
    try:
        symbol = YAHOO_SYMBOLS.get(pair, pair + "=X")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty: return None
        candles = [{"time": str(i), "open": float(r["Open"]), "high": float(r["High"]),
                    "low": float(r["Low"]), "close": float(r["Close"]), "volume": float(r["Volume"])}
                   for i, r in df.iterrows()]
        return candles
    except: return None

async def sdk_get_candles_async(pair, timeframe, limit=500):
    if USE_FALLBACK:
        return sdk_get_candles_yahoo(pair, timeframe if timeframe != "1m" else "1m", period="5d")
    candles = await sdk_get_candles_metaapi(pair, timeframe, limit)
    if candles is None or len(candles) < 10:
        return sdk_get_candles_yahoo(pair, timeframe if timeframe != "1m" else "1m", period="5d")
    return candles

async def sdk_get_price_async(pair):
    try:
        account = await get_account()
        connection = account.get_rpc_connection()
        await connection.connect(); await connection.wait_synchronized()
        price = await connection.get_symbol_price(symbol=pair)
        await connection.close()
        if price: return (float(price.get("bid", 0)) + float(price.get("ask", 0))) / 2
    except: pass
    try:
        ticker = yf.Ticker(YAHOO_SYMBOLS.get(pair, pair+"=X"))
        df = ticker.history(period="1d", interval="1m")
        if not df.empty: return df["Close"].iloc[-1]
    except: pass
    return None

async def execute_trade_async(pair, signal, sl, tp, lot):
    if not METAAPI_TOKEN or not METAAPI_ACCOUNT_ID: return None
    try:
        account = await get_account()
        connection = account.get_rpc_connection()
        await connection.connect(); await connection.wait_synchronized()
        if signal == "BUY":
            order = await connection.create_market_buy_order(symbol=pair, volume=lot, stop_loss=sl, take_profit=tp)
        else:
            order = await connection.create_market_sell_order(symbol=pair, volume=lot, stop_loss=sl, take_profit=tp)
        await connection.close()
        return order
    except: return None

def db_execute(statement, params=(), commit=False, fetch=False):
    for _ in range(5):
        try:
            con = sqlite3.connect(DB_PATH)
            if commit:
                con.execute(statement, params); con.commit()
            elif fetch:
                rows = con.execute(statement, params).fetchall()
                result = rows
            con.close()
            return result if fetch else None
        except sqlite3.OperationalError as e:
            if "locked" in str(e): _time.sleep(0.2)
            else: raise
    return None

async def update_pending_trades(pair):
    price = await sdk_get_price_async(pair)
    if price is None: return
    trades = db_execute("SELECT id, pair, signal, entry, sl, tp, outcome, pnl, strategy_used FROM trade_journal WHERE outcome IS NULL AND pair=?", (pair,), fetch=True)
    if not trades: return
    for tid, p, sig, entry, sl, tp, _, _, strat_name in trades:
        outcome = pnl = None
        if sig == "BUY":
            if price <= sl: outcome, pnl = "LOSS", round((sl - entry) * 100, 2)
            elif price >= tp: outcome, pnl = "WIN",  round((tp - entry) * 100, 2)
        else:
            if price >= sl: outcome, pnl = "LOSS", round((entry - sl) * 100, 2)
            elif price <= tp: outcome, pnl = "WIN",  round((entry - tp) * 100, 2)
        if outcome:
            db_execute("UPDATE trade_journal SET outcome=?, pnl=? WHERE id=?", (outcome, pnl, tid), commit=True)
            if outcome in ("WIN","LOSS"):
                risk_managers[pair].update_pnl(pnl)

async def run_all_strategies(pair):
    raw1m = await sdk_get_candles_async(pair, "1m", 200)
    if not raw1m or len(raw1m) < 50: return None, None
    highs = [c['high'] for c in raw1m]; lows = [c['low'] for c in raw1m]
    closes = [c['close'] for c in raw1m]; opens = [c['open'] for c in raw1m]
    ohlc = {'highs': highs, 'lows': lows, 'closes': closes, 'opens': opens}
    signals = [(strat, strat.get_signal(ohlc)) for strat in active_strategies[pair]]
    return ohlc, signals

MIN_STOP_DISTANCE = 1.10

# ---------- Gemini (15m) ----------
async def gemini_intelligence_cycle():
    global latest_intelligence
    await asyncio.sleep(10)
    while True:
        try:
            if gemini_analyst.available and gemini_analyst.disabled_until <= _time.time():
                trades = db_execute("SELECT signal, entry, sl, tp, outcome, pnl, market_condition, timestamp FROM trade_journal WHERE timestamp >= datetime('now', '-1 day') AND outcome IS NOT NULL ORDER BY timestamp DESC LIMIT 10", fetch=True) or []
                if trades:
                    trade_list = [{"signal": r[0], "entry": r[1], "sl": r[2], "tp": r[3], "outcome": r[4], "pnl": r[5], "market_condition": r[6], "timestamp": r[7]} for r in trades]
                    daily_insight = gemini_analyst.analyze_daily_trades(trade_list)
                    if daily_insight.get("summary"):
                        print(f"🧠 15‑min Gemini: {daily_insight.get('summary')}")
                        suggestions = daily_insight.get("parameter_suggestions", {})
                        for strat_name, params in suggestions.items():
                            for pair in pairs:
                                for strat in active_strategies[pair]:
                                    if strat.name == strat_name:
                                        for key, val in params.items():
                                            if hasattr(strat, key): setattr(strat, key, val)
                        daily_insight["timestamp"] = datetime.datetime.utcnow().isoformat()
                        latest_intelligence = daily_insight
                        db_execute("INSERT INTO agent_log (hour,agent,action,prob,details) VALUES (?,?,?,?,?)",
                                   (datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00"), "A.A.G.A Gemini (15m)", "INTELLIGENCE", 1.0, json.dumps(daily_insight)), commit=True)
        except Exception as e: print(f"15‑min Gemini error: {e}")
        await asyncio.sleep(900)

# ---------- Autonomous Loop ----------
async def keep_alive_task():
    await asyncio.sleep(10)
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://aaga.onrender.com/health") as resp:
                    if resp.status == 200: print("✅ Keep‑alive ping OK")
        except: pass
        await asyncio.sleep(180)

# Initialize portfolio stats with default values
latest_portfolio_stats = {
    "correlation": {},
    "volatilities": {},
    "allocation_weights": {},
    "allocated_capital": {p: 10000/3 for p in pairs}
}

async def autonomous_trading_loop():
    global _metaapi_healthy, _account_cache, latest_signals_cache, latest_portfolio_stats
    while True:
        # Update portfolio allocation (once per cycle)
        try:
            candles_1h = {}
            for pair in pairs:
                candles_1h[pair] = await sdk_get_candles_async(pair, "1h", 200)
            current_prices = {}
            for pair in pairs:
                price = await sdk_get_price_async(pair)
                if price: current_prices[pair] = price
            if candles_1h and current_prices:
                portfolio_mgr.update_prices(current_prices)
                stats = portfolio_mgr.get_portfolio_stats(candles_1h, current_prices)
                latest_portfolio_stats = stats
                # Update risk managers' balance based on allocation
                for pair in pairs:
                    alloc = stats['allocated_capital'].get(pair, 10000/3)
                    risk_managers[pair].balance = alloc
        except Exception as e: print(f"Portfolio update error: {e}")

        for pair in pairs:
            try:
                high_impact, event, _ = is_high_impact_now()
                if high_impact: continue
                await update_pending_trades(pair)
                ohlc, strategy_signals = await run_all_strategies(pair)
                if ohlc is None: continue
                candles_15m = await sdk_get_candles_async(pair, "15m", 100)
                regime = regime_detector.classify(candles_15m) if candles_15m else "UNKNOWN"
                strategy_signals_list = [{"name": s[0].name, "signal": s[1].get("signal"), "entry_zone": s[1].get("entry_zone"),
                                          "sl": s[1].get("sl"), "tp": s[1].get("tp"), "trend": s[1].get("trend"),
                                          "current_price": s[1].get("current_price")} for s in strategy_signals]
                signal_dicts = [{"strategy": s["name"], "signal": s["signal"], "entry_zone": s["entry_zone"],
                                 "sl": s["sl"], "tp": s["tp"], "current_price": s["current_price"]} for s in strategy_signals_list]
                final_decision = master_agents[pair].decide(signal_dicts, regime=regime)
                decision_signal = final_decision["signal"]
                conf = final_decision.get("confidence", 0.5)
                probs = {"BUY": conf if decision_signal=="BUY" else 0, "SELL": conf if decision_signal=="SELL" else 0, "HOLD": 1-conf if decision_signal in ("BUY","SELL") else 1.0}
                risk_brief = None
                if decision_signal in ("BUY","SELL") and final_decision.get("entry_zone"):
                    entry = final_decision["entry_zone"][0] if decision_signal=="BUY" else final_decision["entry_zone"][1]
                    risk_brief = {"entry": round(entry,2), "sl": final_decision["sl"], "tp": final_decision["tp"]}
                latest_signals_cache[pair] = {"decision": decision_signal, "probabilities": probs, "risk_brief": risk_brief,
                                             "strategy_used": final_decision.get("strategy_used","Master"),
                                             "market_trend": strategy_signals_list[0].get("trend","Unknown"),
                                             "current_price": strategy_signals_list[0].get("current_price",0),
                                             "strategy_signals": strategy_signals_list, "regime": regime}
                now = datetime.datetime.utcnow(); hour = now.strftime("%Y-%m-%d %H:00")
                for s in strategy_signals_list:
                    db_execute("INSERT INTO agent_log (hour,agent,action,prob,details) VALUES (?,?,?,?,?)",
                               (hour, f"Agent {s['name']}", s.get("signal","HOLD"), 0.8 if s.get("signal") in ("BUY","SELL") else 0.3, json.dumps(s)), commit=True)
                if decision_signal in ("BUY","SELL"):
                    entry_zone = final_decision.get("entry_zone"); sl = final_decision.get("sl"); tp = final_decision.get("tp")
                    if not entry_zone or not sl or not tp: continue
                    entry = entry_zone[0] if decision_signal=="BUY" else entry_zone[1]
                    if (decision_signal=="BUY" and (sl > entry - MIN_STOP_DISTANCE or tp < entry + MIN_STOP_DISTANCE)) or \
                       (decision_signal=="SELL" and (sl < entry + MIN_STOP_DISTANCE or tp > entry - MIN_STOP_DISTANCE)): continue
                    existing = db_execute("SELECT COUNT(*) FROM trade_journal WHERE outcome IS NULL AND pair=? AND signal=?", (pair, decision_signal), fetch=True)
                    if existing and existing[0][0] > 0: continue
                    lot = risk_managers[pair].calculate_lot(entry, sl)
                    db_execute("INSERT INTO trade_journal (timestamp,pair,signal,entry,sl,tp,strategy_used,gemini_insight,market_condition) VALUES (?,?,?,?,?,?,?,?,?)",
                               (now.isoformat(), pair, decision_signal, entry, sl, tp, final_decision.get("strategy_used","Master"), "", regime), commit=True)
                    order = await execute_trade_async(pair, decision_signal, sl, tp, lot)
                    if order: print(f"✅ {pair} Master Auto trade: {decision_signal}")
            except Exception as e: print(f"Loop error {pair}: {e}")
        await asyncio.sleep(60)

# ---------- Nightly / Weekly ----------
async def nightly_tasks():
    while True:
        now = datetime.datetime.utcnow(); next_midnight = (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=5)
        await asyncio.sleep((next_midnight - now).total_seconds())
        for pair in pairs:
            for strat in active_strategies[pair]:
                if strat.name == "RandomForest" and trade_memory.retrain_model(strat.model):
                    strat.trained = True; print(f"✅ {pair} RF retrained")
            candles_1m = await sdk_get_candles_async(pair, "1m", 2000)
            if candles_1m and len(candles_1m)>200:
                for strat in active_strategies[pair]:
                    if strat.name == "PPO": strat.train(candles_1m, total_timesteps=5000); break
        rows = db_execute("SELECT pair, strategy_used, COUNT(*) as total, SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins FROM trade_journal WHERE date(timestamp) = date('now','-1 day') AND outcome IS NOT NULL GROUP BY pair, strategy_used", fetch=True)
        if rows:
            for r in rows: master_agents[r[0]].update_performance(r[1], (r[3] or 0)/r[2] if r[2]>0 else 0.5)

async def weekly_scheduler():
    while True:
        now = datetime.datetime.utcnow(); days_until_sunday = (6 - now.weekday()) % 7
        next_sunday = now + datetime.timedelta(days=days_until_sunday); next_sunday = next_sunday.replace(hour=0, minute=5)
        if days_until_sunday==0 and now>next_sunday: next_sunday += datetime.timedelta(weeks=1)
        await asyncio.sleep((next_sunday - now).total_seconds())
        for pair in pairs:
            await weekly_self_retraining(DB_PATH, active_strategies[pair], master_agents[pair], trade_memory, gemini_analyst, strategy_factory)
        # Genetic optimizer
        try:
            print("🧬 Genetic optimizer running...")
            strat = active_strategies["XAUUSD"][0]
            base_cfg = {"adx_threshold": strat.adx_threshold, "slope_threshold": strat.slope_threshold,
                        "atr_multiplier_sl": strat.atr_multiplier_sl, "rr_ratio": strat.rr_ratio,
                        "min_stop_distance": strat.min_stop_distance}
            best = await genetic_optimizer.optimize(EmaDmiStrategy, base_cfg=base_cfg)
            if best:
                genetic_optimizer.apply_to_strategy(strat, best)
                genetic_optimizer.apply_to_strategy(active_strategies["EURUSD"][0], best)
                cfg = json.load(open(CONFIG_PATH)); cfg["parameters"] = best
                with open(CONFIG_PATH,"w") as f: json.dump(cfg, f, indent=2)
                print("✅ Genetic optimizer applied")
        except Exception as e: print(f"Genetic error: {e}")
        # GAN training
        try:
            print("🎨 Training GAN on historical candles...")
            candles = await sdk_get_candles_async("XAUUSD", "1m", 2000)
            if candles and len(candles)>500:
                data = np.array([[c['open'], c['high'], c['low'], c['close']] for c in candles])
                mn = data.min(axis=0); mx = data.max(axis=0)
                norm = (data - mn) / (mx - mn + 1e-8)
                gan_model.train(norm, epochs=500, batch_size=64)
                synthetic = gan_model.generate_samples(500)
                synthetic = synthetic * (mx - mn) + mn
                with open("synthetic_candles.json", "w") as f:
                    json.dump(synthetic.tolist(), f)
                print("✅ GAN training complete, synthetic candles saved.")
        except Exception as e: print(f"GAN error: {e}")

@app.on_event("startup")
async def startup():
    asyncio.create_task(keep_alive_task())
    asyncio.create_task(autonomous_trading_loop())
    asyncio.create_task(nightly_tasks())
    asyncio.create_task(weekly_scheduler())
    asyncio.create_task(gemini_intelligence_cycle())

def init_db():
    con = sqlite3.connect(DB_PATH)
    try: con.execute("ALTER TABLE trade_journal ADD COLUMN market_condition TEXT DEFAULT 'Unknown'")
    except: pass
    con.executescript("""CREATE TABLE IF NOT EXISTS trade_journal (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, pair TEXT, signal TEXT, entry REAL, sl REAL, tp REAL, outcome TEXT, pnl REAL, strategy_used TEXT, gemini_insight TEXT, market_condition TEXT DEFAULT 'Unknown');
                         CREATE TABLE IF NOT EXISTS agent_log (id INTEGER PRIMARY KEY AUTOINCREMENT, hour TEXT, agent TEXT, action TEXT, prob REAL, details TEXT);""")
    con.commit(); con.close()
init_db()

# ---------- All Dashboard Routes ----------
@app.get("/", response_class=HTMLResponse)
def dashboard(): return FileResponse("dashboard.html")

@app.get("/health")
async def health(): return {"status":"ok","autonomous_loop":"active","metaapi_healthy":_metaapi_healthy}

@app.get("/debug")
async def debug():
    status = {}
    for pair in pairs:
        status[pair] = {"signal": latest_signals_cache[pair]["decision"], "price": latest_signals_cache[pair]["current_price"],
                        "regime": latest_signals_cache[pair]["regime"], "strategies": [s.name for s in active_strategies[pair]],
                        "risk_balance": risk_managers[pair].balance}
    return {"metaapi_healthy": _metaapi_healthy, "pairs": status, "news": news_agent.analyze().get("summary")}

@app.get("/master-signal")
async def master_signal(pair: str = Query("XAUUSD")):
    if pair not in master_agents: return {"error":"Invalid pair"}
    try:
        ohlc, strategy_signals = await run_all_strategies(pair)
        if ohlc is not None and strategy_signals is not None:
            strategy_signals_list = [{"name": s[0].name, "signal": s[1].get("signal"), "entry_zone": s[1].get("entry_zone"),
                                      "sl": s[1].get("sl"), "tp": s[1].get("tp"), "trend": s[1].get("trend"),
                                      "current_price": s[1].get("current_price")} for s in strategy_signals]
            signal_dicts = [{"strategy": s["name"], "signal": s["signal"], "entry_zone": s["entry_zone"],
                             "sl": s["sl"], "tp": s["tp"], "current_price": s["current_price"]} for s in strategy_signals_list]
            candles_15m = await sdk_get_candles_async(pair, "15m", 100)
            regime = regime_detector.classify(candles_15m) if candles_15m else "UNKNOWN"
            master_decision = master_agents[pair].decide(signal_dicts, regime=regime)
            conf = master_decision.get("confidence",0.5)
            decision_signal = master_decision["signal"]
            probs = {"BUY": conf if decision_signal=="BUY" else 0, "SELL": conf if decision_signal=="SELL" else 0, "HOLD": 1-conf if decision_signal in ("BUY","SELL") else 1.0}
            risk_brief = None
            if decision_signal in ("BUY","SELL") and master_decision.get("entry_zone"):
                entry = master_decision["entry_zone"][0] if decision_signal=="BUY" else master_decision["entry_zone"][1]
                risk_brief = {"entry": round(entry,2), "sl": master_decision["sl"], "tp": master_decision["tp"]}
            latest_signals_cache[pair] = {"decision": decision_signal, "probabilities": probs, "risk_brief": risk_brief,
                                         "strategy_used": master_decision.get("strategy_used","Master"),
                                         "market_trend": strategy_signals_list[0].get("trend","Unknown"),
                                         "current_price": strategy_signals_list[0].get("current_price",0),
                                         "strategy_signals": strategy_signals_list, "regime": regime}
    except Exception as e: print(f"Master signal error: {e}")
    cached = latest_signals_cache[pair]
    return {**cached, "gemini_advice": latest_intelligence, "news_sentiment": news_agent.analyze()}

@app.get("/agent-log")
def agent_log():
    rows = db_execute("SELECT hour, agent, action, prob, details FROM agent_log WHERE hour >= datetime('now', '-1 day') ORDER BY id DESC LIMIT 20", fetch=True)
    return [{"hour":r[0],"agent":r[1],"action":r[2],"prob":r[3],"details":r[4]} for r in rows] if rows else []

@app.get("/today-trades")
def today_trades(pair: str = Query("XAUUSD")):
    rows = db_execute("SELECT timestamp, pair, signal, entry, sl, tp, outcome, pnl, strategy_used, market_condition FROM trade_journal WHERE date(timestamp) = date('now') AND pair=? ORDER BY timestamp DESC", (pair,), fetch=True)
    return [{"timestamp":r[0],"pair":r[1],"signal":r[2],"entry":r[3],"sl":r[4],"tp":r[5],"outcome":r[6],"pnl":r[7],"strategy":r[8],"market_condition":r[9]} for r in rows] if rows else []

@app.get("/trade-reviews")
def trade_reviews(pair: str = Query("XAUUSD")):
    rows = db_execute("SELECT id, timestamp, signal, entry, sl, tp, outcome, pnl, gemini_insight, strategy_used, market_condition FROM trade_journal WHERE outcome IS NOT NULL AND pair=? ORDER BY timestamp DESC LIMIT 50", (pair,), fetch=True)
    return [{"id": r[0], "timestamp": r[1], "signal": r[2], "entry": r[3], "sl": r[4], "tp": r[5],
             "outcome": r[6], "pnl": r[7], "gemini_insight": r[8], "strategy": r[9], "market_condition": r[10]} for r in rows] if rows else []

@app.get("/strategy-performance")
def strategy_performance(pair: str = Query("XAUUSD")):
    rows = db_execute("""SELECT strategy_used, COUNT(*) as total, SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins, SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) as losses, SUM(pnl) as total_pnl FROM trade_journal WHERE date(timestamp) = date('now') AND pair=? AND outcome IS NOT NULL GROUP BY strategy_used""", (pair,), fetch=True)
    result = {}
    if rows:
        for r in rows:
            name = r[0] or "unknown"; total = r[1]; wins = r[2] or 0; losses = r[3] or 0
            win_rate = wins / total * 100 if total > 0 else 0
            result[name] = {"total_trades": total, "wins": wins, "losses": losses, "win_rate": round(win_rate,1), "pnl": round(r[4] or 0,2)}
    return result

@app.get("/market-condition-performance")
def market_condition_performance(pair: str = Query("XAUUSD")):
    rows = db_execute("""SELECT market_condition, COUNT(*) as total, SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins, SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) as losses, SUM(pnl) as total_pnl FROM trade_journal WHERE date(timestamp) = date('now') AND pair=? AND outcome IS NOT NULL AND market_condition IS NOT NULL GROUP BY market_condition""", (pair,), fetch=True)
    result = {}
    if rows:
        for r in rows:
            cond = r[0] or "Unknown"; total = r[1]; wins = r[2] or 0; losses = r[3] or 0
            win_rate = wins / total * 100 if total > 0 else 0
            result[cond] = {"total_trades": total, "wins": wins, "losses": losses, "win_rate": round(win_rate,1), "pnl": round(r[4] or 0,2)}
    return result

@app.get("/weekly-backtest")
async def weekly_backtest(pair: str = Query("XAUUSD")):
    try: return await compare_strategies()
    except Exception as e: return {"error": f"Backtest failed: {str(e)}"}

@app.get("/economic-calendar")
async def economic_calendar():
    return {"events": get_upcoming_events()}

@app.get("/explain/{pair}")
async def explain_decision(pair: str):
    if pair not in master_agents: return {"error":"Invalid pair"}
    ohlc, strategy_signals = await run_all_strategies(pair)
    if ohlc is None: return {"error":"No data"}
    strategy_signals_list = [{"name": s[0].name, "signal": s[1].get("signal"), "entry_zone": s[1].get("entry_zone"),
                              "sl": s[1].get("sl"), "tp": s[1].get("tp"), "trend": s[1].get("trend")} for s in strategy_signals]
    signal_dicts = [{"strategy": s["name"], "signal": s["signal"], "entry_zone": s["entry_zone"], "sl": s["sl"], "tp": s["tp"]} for s in strategy_signals_list]
    candles_15m = await sdk_get_candles_async(pair, "15m", 100)
    regime = regime_detector.classify(candles_15m) if candles_15m else "UNKNOWN"
    master_decision = master_agents[pair].decide(signal_dicts, regime=regime)
    rf_importance = None
    for strat in active_strategies[pair]:
        if strat.name == "RandomForest" and strat.trained:
            try: rf_importance = list(strat.model.feature_importances_)
            except: pass
    gemini_reasoning = latest_intelligence.get("summary", "Not available") if latest_intelligence else "Not available"
    return {"pair":pair, "regime":regime, "signals":strategy_signals_list, "master_decision":master_decision,
            "voting_weights":master_agents[pair].performance_memory, "random_forest_importance":rf_importance, "gemini_reasoning":gemini_reasoning}

@app.get("/ai-insights")
async def ai_insights(): return latest_intelligence

@app.get("/news-sentiment")
async def news_sentiment(): return news_agent.analyze()

@app.get("/portfolio-stats")
async def portfolio_stats():
    return latest_portfolio_stats

@app.get("/compare-strategies")
async def compare_strategies_endpoint(): return await compare_strategies()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))
