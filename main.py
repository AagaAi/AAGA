# main.py - A.A.G.A AI Trading System with FastAPI Web Service
import os
import sys
import time
import json
import sqlite3
import datetime
import asyncio
import threading
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# ---------- Environment Variables ----------
os.environ.setdefault("DERIV_API_TOKEN", "")

# ---------- Agents ----------
from agents.strategy_agent import EmaDmiStrategy
from agents.ai_agent import RandomForestStrategy
from agents.ppo_agent import PPOAgent
from agents.master_agent import MasterAgent
from agents.news_agent import NewsSentimentAgent
from agents.regime_detector import RegimeDetector
from agents.market_classifier import MarketClassifier
from agents.memory import TradeMemory
from agents.gemini_agent import GeminiAnalyst
from agents.strategy_factory import StrategyFactory
from agents.llm_router import MultiLLMRouter
from agents.self_optimizer import SelfOptimizer
from agents.genetic_optimizer import GeneticOptimizer
from agents.pdf_report import generate_daily_report
from agents.telegram_notifier import TelegramNotifier
from agents.broker_deriv import DerivBroker

# ---------- Global State ----------
latest_signals_cache = {}
_metaapi_healthy = True   # Kept for compatibility
autonomous_trading_loop_paused = False
last_regime_update = 0
signal_history = []
MAX_SIGNAL_HISTORY = 100

# ---------- Initialize Components ----------
deriv_broker = DerivBroker()
print("✅ Deriv Broker ready")

ema_strategy = EmaDmiStrategy()
rf_strategy = RandomForestStrategy()
ppo_strategy = PPOAgent()
active_strategies = [ema_strategy, rf_strategy, ppo_strategy]
print(f"✅ {len(active_strategies)} strategies loaded")

news_agent = NewsSentimentAgent()
regime_detector = RegimeDetector()
market_classifier = MarketClassifier()
trade_memory = TradeMemory("journal.db")
telegram = TelegramNotifier()

llm_router = MultiLLMRouter()
gemini_analyst = GeminiAnalyst()
strategy_factory = StrategyFactory(gemini_analyst)

self_optimizer = SelfOptimizer("journal.db", llm_router)
genetic_optimizer = GeneticOptimizer()

def risk_evaluator():
    con = sqlite3.connect("journal.db")
    today = datetime.date.today().isoformat()
    rows = con.execute(
        "SELECT SUM(pnl) FROM trade_journal WHERE date(timestamp) = ? AND outcome='LOSS'",
        (today,)
    ).fetchone()
    con.close()
    daily_loss = abs(rows[0] or 0)
    if daily_loss > 200:
        return {"halt": True, "reason": f"Daily loss ${daily_loss} > $200 limit"}
    return {"halt": False}

master_agent = MasterAgent(
    strategies=active_strategies,
    risk_evaluator=risk_evaluator,
    news_analyzer=news_agent.analyze
)
print("✅ Master Agent ready")

# ---------- Signal Aggregation ----------
def get_all_signals(ohlc_dict: Dict[str, Any], pair: str = "XAUUSD") -> list:
    signals = []
    for strat in active_strategies:
        try:
            sig = strat.get_signal(ohlc_dict)
            sig["strategy"] = strat.name
            signals.append(sig)
        except Exception as e:
            print(f"⚠️ {strat.name} error: {e}")
    return signals

# ---------- Trade Execution ----------
async def execute_decision(decision: dict, pair: str = "XAUUSD"):
    signal = decision.get("signal", "HOLD")
    if signal not in ("BUY", "SELL"):
        print(f"ℹ️ No trade: {signal}")
        return None
    current_price = await deriv_broker.get_tick("frxXAUUSD")
    if not current_price:
        print("❌ Cannot get price from Deriv")
        return None
    amount = 10.0
    multiplier = 100
    result = await deriv_broker.place_order(
        signal=signal,
        instrument="frxXAUUSD",
        amount=amount,
        multiplier=multiplier,
        duration=60
    )
    if result:
        print(f"✅ Trade executed: {signal} at ${current_price:.2f}")
        con = sqlite3.connect("journal.db")
        con.execute(
            """INSERT INTO trade_journal 
               (timestamp, pair, signal, entry, sl, tp, outcome, strategy_used, market_condition)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.datetime.utcnow().isoformat(),
                pair,
                signal,
                current_price,
                decision.get("sl"),
                decision.get("tp"),
                None,
                decision.get("strategy_used", "Master"),
                decision.get("regime", "UNKNOWN")
            )
        )
        con.commit()
        con.close()
        return result
    else:
        print(f"❌ Trade failed for {signal}")
        return None

# ---------- Trading Loop ----------
async def trading_loop():
    global latest_signals_cache, autonomous_trading_loop_paused
    print("🚀 A.A.G.A AI Trading Loop Started (Deriv)")
    while True:
        try:
            if autonomous_trading_loop_paused:
                await asyncio.sleep(5)
                continue
            current_price = await deriv_broker.get_tick("frxXAUUSD")
            if not current_price:
                await asyncio.sleep(10)
                continue
            # Simulate OHLC (simplified)
            ohlc_dict = {
                "closes": [current_price] * 60,
                "highs": [current_price * 1.001] * 60,
                "lows": [current_price * 0.999] * 60,
                "opens": [current_price * 0.999] * 60
            }
            # Regime detection (simplified)
            if current_price > 2800:
                regime = "UPTREND"
            elif current_price < 2600:
                regime = "DOWNTREND"
            else:
                regime = "SIDEWAYS"
            signals = get_all_signals(ohlc_dict)
            decision = master_agent.decide(signals, regime)
            decision["regime"] = regime
            decision["current_price"] = current_price
            decision["pair"] = "XAUUSD"
            latest_signals_cache["XAUUSD"] = decision
            if decision["signal"] in ("BUY", "SELL"):
                print(f"📊 Decision: {decision['signal']} @ {current_price:.2f}")
                await execute_decision(decision, "XAUUSD")
            # Self-Optimizer tasks (simplified intervals)
            if int(time.time()) % 1800 < 10:
                self_optimizer.auto_adjust_parameters({"XAUUSD": active_strategies})
                self_optimizer.update_regime_weights({"XAUUSD": master_agent})
            if int(time.time()) % 3600 < 10:
                suggestions = self_optimizer.llm_suggest_upgrade({"XAUUSD": active_strategies})
                if suggestions:
                    self_optimizer.apply_llm_suggestions(
                        suggestions,
                        {"XAUUSD": active_strategies},
                        {"XAUUSD": master_agent}
                    )
            await asyncio.sleep(60)
        except Exception as e:
            print(f"❌ Loop error: {e}")
            await asyncio.sleep(30)

# ---------- Lifespan Manager ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect to Deriv and start trading loop
    await deriv_broker.connect()
    task = asyncio.create_task(trading_loop())
    yield
    # Shutdown: clean up
    task.cancel()
    await deriv_broker.disconnect()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "A.A.G.A AI Trading System", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "deriv_connected": deriv_broker.connected}

@app.get("/status")
async def status():
    return {
        "broker": "Deriv",
        "connected": deriv_broker.connected,
        "strategies": [s.name for s in active_strategies],
        "latest_signal": latest_signals_cache.get("XAUUSD"),
        "trading_paused": autonomous_trading_loop_paused
    }

@app.post("/pause")
async def pause_trading():
    global autonomous_trading_loop_paused
    autonomous_trading_loop_paused = True
    return {"message": "Trading paused"}

@app.post("/resume")
async def resume_trading():
    global autonomous_trading_loop_paused
    autonomous_trading_loop_paused = False
    return {"message": "Trading resumed"}

@app.get("/report")
async def daily_report():
    report = generate_daily_report()
    return JSONResponse(content={"report": report})

# ---------- Main (for local testing) ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
