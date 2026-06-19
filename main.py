# main.py – A.A.G.A AI (Ultimate Upgrade: All 6 Features)
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
from agents.ppo_agent import PPOAgent
from agents.master_agent import MasterAgent
from agents.memory import TradeMemory
from agents.news_agent import NewsSentimentAgent
from agents.market_classifier import MarketClassifier
from agents.strategy_factory import StrategyFactory
from agents.weekly_pipeline import weekly_self_retraining
from agents.economic_calendar import is_high_impact_now, get_upcoming_events
from agents.telegram_notifier import TelegramNotifier
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

# Strategies per pair – now multi-pair
pairs = ["XAUUSD", "EURUSD", "BTCUSD"]  # can be extended
active_strategies = {}  # pair -> list of strategies
for pair in pairs:
    active_strategies[pair] = [
        EmaDmiStrategy(config.get("parameters", {})),
        RandomForestStrategy(),
        PPOAgent()
    ]

# Fallback data flag
USE_FALLBACK = False

# Cache latest signals per pair
latest_signals_cache = {}

# Risk managers per pair
risk_managers = {pair: RiskManager() for pair in pairs}

# Master agents per pair
master_agents = {}
for pair in pairs:
    master_agents[pair] = MasterAgent(active_strategies[pair], risk_managers[pair].evaluate, run_news_analysis)

# Trade memory
trade_memory = TradeMemory(DB_PATH)

# Strategy factory
strategy_factory = StrategyFactory(gemini_analyst) if gemini_analyst.available else None

# Latest intelligence
latest_intelligence = {"summary": "Waiting for first analysis...", "timestamp": None}

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

# MetaApi singleton (per pair)
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

async def sdk_get_candles_metaapi(pair: str, timeframe: str, limit: int = 500):
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
    except:
        return None

def sdk_get_candles_yahoo(pair: str, interval: str = "1m", period: str = "1d"):
    """Fallback data source (Yahoo Finance)."""
    try:
        ticker = yf.Ticker(pair + "=X")
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        candles = []
        for idx, row in df.iterrows():
            candles.append({
                "time": str(idx),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"])
            })
        return candles
    except:
        return None

async def sdk_get_candles_async(pair: str, timeframe: str, limit: int = 500):
    if USE_FALLBACK:
        return sdk_get_candles_yahoo(pair, timeframe if timeframe != "1m" else "1m", period="5d")
    candles = await sdk_get_candles_metaapi(pair, timeframe, limit)
    if candles is None or len(candles) < 10:
        # Fallback
        print(f"⚠️ MetaApi failed for {pair}, switching to Yahoo Finance")
        return sdk_get_candles_yahoo(pair, timeframe if timeframe != "1m" else "1m", period="5d")
    return candles

async def sdk_get_price_metaapi(pair: str):
    try:
        account = await get_account()
        connection = account.get_rpc_connection()
        await connection.connect(); await connection.wait_synchronized()
        price = await connection.get_symbol_price(symbol=pair)
        await connection.close()
        if price: return (float(price.get("bid", 0)) + float(price.get("ask", 0))) / 2
    except: pass
    return None

def sdk_get_price_yahoo(pair: str):
    try:
        ticker = yf.Ticker(pair + "=X")
        df = ticker.history(period="1d", interval="1m")
        if not df.empty:
            return df["Close"].iloc[-1]
    except: pass
    return None

async def sdk_get_price_async(pair: str):
    if USE_FALLBACK:
        return sdk_get_price_yahoo(pair)
    price = await sdk_get_price_metaapi(pair)
    if price is None:
        return sdk_get_price_yahoo(pair)
    return price

async def execute_trade_async(pair: str, signal, sl, tp, lot):
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

# Database helper
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

async def update_pending_trades(pair: str):
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

async def run_all_strategies(pair: str):
    raw1m = await sdk_get_candles_async(pair, "1m", 200)
    if not raw1m or len(raw1m) < 50: return None, None
    highs  = [c['high'] for c in raw1m]
    lows   = [c['low'] for c in raw1m]
    closes = [c['close'] for c in raw1m]
    opens  = [c['open'] for c in raw1m]
    ohlc = {'highs': highs, 'lows': lows, 'closes': closes, 'opens': opens}
    signals = []
    for strat in active_strategies[pair]:
        sig = await get_strategy_signal(strat, ohlc)
        signals.append((strat, sig))
    return ohlc, signals

async def get_strategy_signal(strat, ohlc):
    return strat.get_signal(ohlc)

MIN_STOP_DISTANCE = 1.10

# ============================================================
# Autonomous Trading Loop (multi-pair)
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
    global _metaapi_healthy, _account_cache, latest_signals_cache
    while True:
        for pair in pairs:
            try:
                # Check economic calendar
                high_impact, event, event_time = is_high_impact_now()
                if high_impact:
                    print(f"⛔ High‑impact event {event} in 15‑min window – HOLD all pairs")
                    continue  # skip trading for this cycle

                await update_pending_trades(pair)
                ohlc, strategy_signals = await run_all_strategies(pair)
                if ohlc is None:
                    continue

                strategy_signals_list = [{"name": strat.name, "signal": sig.get("signal"),
                                          "entry_zone": sig.get("entry_zone"), "sl": sig.get("sl"),
                                          "tp": sig.get("tp"), "trend": sig.get("trend"),
                                          "current_price": sig.get("current_price")}
                                         for strat, sig in strategy_signals]
                signal_dicts = [{"strategy": s["name"], "signal": s["signal"],
                                 "entry_zone": s["entry_zone"], "sl": s["sl"],
                                 "tp": s["tp"], "current_price": s["current_price"]}
                                for s in strategy_signals_list]
                final_decision = master_agents[pair].decide(signal_dicts)
                decision_signal = final_decision["signal"]

                # Update cache
                conf = final_decision.get("confidence", 0.5)
                probs = {"BUY": conf if decision_signal=="BUY" else 0,
                         "SELL": conf if decision_signal=="SELL" else 0,
                         "HOLD": 1-conf if decision_signal in ("BUY","SELL") else 1.0}
                risk_brief = None
                if decision_signal in ("BUY","SELL") and final_decision.get("entry_zone"):
                    entry = final_decision["entry_zone"][0] if decision_signal=="BUY" else final_decision["entry_zone"][1]
                    risk_brief = {"entry": round(entry,2), "sl": final_decision["sl"], "tp": final_decision["tp"]}
                latest_signals_cache[pair] = {
                    "decision": decision_signal,
                    "probabilities": probs,
                    "risk_brief": risk_brief,
                    "strategy_used": final_decision.get("strategy_used", "Master"),
                    "market_trend": strategy_signals_list[0].get("trend") if strategy_signals_list else "Unknown",
                    "current_price": strategy_signals_list[0].get("current_price", 0) if strategy_signals_list else 0,
                    "strategy_signals": strategy_signals_list,
                }

                # Trade execution
                if decision_signal in ("BUY", "SELL"):
                    entry_zone = final_decision.get("entry_zone")
                    sl = final_decision.get("sl")
                    tp = final_decision.get("tp")
                    if not entry_zone or not sl or not tp: continue
                    entry = entry_zone[0] if decision_signal == "BUY" else entry_zone[1]
                    # stop validation
                    if decision_signal == "BUY":
                        if sl > entry - MIN_STOP_DISTANCE or tp < entry + MIN_STOP_DISTANCE: continue
                    else:
                        if sl < entry + MIN_STOP_DISTANCE or tp > entry - MIN_STOP_DISTANCE: continue
                    existing = db_execute("SELECT COUNT(*) FROM trade_journal WHERE outcome IS NULL AND pair=? AND signal=?", (pair, decision_signal), fetch=True)
                    if existing and existing[0][0] > 0:
                        print(f"⏩ {pair}: Same direction open trade exists")
                    else:
                        lot = risk_managers[pair].calculate_lot(entry, sl)
                        risk_brief = {"entry": round(entry,2), "sl": sl, "tp": tp}
                        strat_used = final_decision.get("strategy_used", "Master")
                        now = datetime.datetime.utcnow()
                        db_execute("INSERT INTO trade_journal (timestamp,pair,signal,entry,sl,tp,strategy_used,gemini_insight,market_condition) VALUES (?,?,?,?,?,?,?,?,?)",
                                   (now.isoformat(), pair, decision_signal,
                                    risk_brief["entry"], risk_brief["sl"], risk_brief["tp"],
                                    strat_used, "", "Unknown"), commit=True)
                        order = await execute_trade_async(pair, decision_signal, sl, tp, lot)
                        if order:
                            print(f"✅ {pair} Master Auto trade ({strat_used}): {decision_signal} | Lot: {lot}")
                            telegram.send_message(f"A.A.G.A AI: {pair} {decision_signal} @ {entry}, SL {sl}, TP {tp}")
                        else:
                            print("❌ Trade failed")
            except Exception as e:
                print(f"Autonomous loop error for {pair}: {e}")
        await asyncio.sleep(AUTONOMOUS_INTERVAL_SEC)

# ============================================================
# Explainable AI Endpoint
# ============================================================
@app.get("/explain/{pair}")
async def explain_decision(pair: str):
    if pair not in master_agents:
        return {"error": "Invalid pair"}
    # Get latest signals
    ohlc, strategy_signals = await run_all_strategies(pair)
    if ohlc is None:
        return {"error": "No data"}
    strategy_signals_list = [{"name": strat.name, "signal": sig.get("signal"),
                              "entry_zone": sig.get("entry_zone"), "sl": sig.get("sl"),
                              "tp": sig.get("tp"), "trend": sig.get("trend")}
                             for strat, sig in strategy_signals]
    signal_dicts = [{"strategy": s["name"], "signal": s["signal"],
                     "entry_zone": s["entry_zone"], "sl": s["sl"],
                     "tp": s["tp"]} for s in strategy_signals_list]
    master_decision = master_agents[pair].decide(signal_dicts)
    # Get feature importance for RandomForest (if available)
    rf_importance = None
    for strat in active_strategies[pair]:
        if strat.name == "RandomForest" and strat.trained:
            try:
                importances = strat.model.feature_importances_
                rf_importance = list(importances)
            except: pass
    # Get Gemini reasoning (if available)
    gemini_reasoning = None
    if gemini_analyst.available:
        # quick gemini call (but we'll skip to avoid quota; reuse latest_intelligence)
        gemini_reasoning = latest_intelligence.get("summary", "Not available")
    return {
        "pair": pair,
        "signals": strategy_signals_list,
        "master_decision": master_decision,
        "voting_weights": master_agent.performance_memory,  # current weights
        "random_forest_importance": rf_importance,
        "gemini_reasoning": gemini_reasoning
    }

# Other routes (dashboard, agent-log, etc.) remain mostly the same, adding pair query param.

# ... (keep all existing routes and startup events)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))
