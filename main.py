# main.py - A.A.G.A AI Trading System with Deriv Integration
import os
import sys
import time
import json
import sqlite3
import datetime
import asyncio
import threading
from typing import Dict, Any, Optional

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
from agents.broker_deriv import DerivBroker          # NEW

# ---------- Initialize Components ----------

# 1. Broker (Deriv)
deriv_broker = DerivBroker()
print("✅ Deriv Broker ready")

# 2. Strategies
ema_strategy = EmaDmiStrategy()
rf_strategy = RandomForestStrategy()
ppo_strategy = PPOAgent()

active_strategies = [ema_strategy, rf_strategy, ppo_strategy]
print(f"✅ {len(active_strategies)} strategies loaded")

# 3. Other Agents
news_agent = NewsSentimentAgent()
regime_detector = RegimeDetector()
market_classifier = MarketClassifier()
trade_memory = TradeMemory("journal.db")
telegram = TelegramNotifier()

# 4. LLM & Strategy Factory
llm_router = MultiLLMRouter()
gemini_analyst = GeminiAnalyst()
strategy_factory = StrategyFactory(gemini_analyst)

# 5. Self-Optimizer
self_optimizer = SelfOptimizer("journal.db", llm_router)

# 6. Genetic Optimizer
genetic_optimizer = GeneticOptimizer()

# 7. Master Agent (Risk evaluator is a lambda/function)
def risk_evaluator():
    """Simple risk evaluator - checks daily loss limit"""
    con = sqlite3.connect("journal.db")
    today = datetime.date.today().isoformat()
    rows = con.execute(
        "SELECT SUM(pnl) FROM trade_journal WHERE date(timestamp) = ? AND outcome='LOSS'",
        (today,)
    ).fetchone()
    con.close()
    daily_loss = abs(rows[0] or 0)
    # If daily loss > 200, halt trading
    if daily_loss > 200:
        return {"halt": True, "reason": f"Daily loss ${daily_loss} > $200 limit"}
    return {"halt": False}

master_agent = MasterAgent(
    strategies=active_strategies,
    risk_evaluator=risk_evaluator,
    news_analyzer=news_agent.analyze
)
print("✅ Master Agent ready")

# ---------- Global State ----------
latest_signals_cache = {}
_metaapi_healthy = True   # Kept for compatibility, but not used
autonomous_trading_loop_paused = False
last_regime_update = 0
signal_history = []
MAX_SIGNAL_HISTORY = 100

# ---------- Signal Aggregation ----------
def get_all_signals(ohlc_dict: Dict[str, Any], pair: str = "XAUUSD") -> list:
    """Get signals from all active strategies"""
    signals = []
    for strat in active_strategies:
        try:
            sig = strat.get_signal(ohlc_dict)
            sig["strategy"] = strat.name
            signals.append(sig)
        except Exception as e:
            print(f"⚠️ {strat.name} error: {e}")
    return signals

# ---------- Decision Execution ----------
async def execute_decision(decision: dict, pair: str = "XAUUSD"):
    """Execute trade decision via Deriv Broker"""
    signal = decision.get("signal", "HOLD")
    if signal not in ("BUY", "SELL"):
        print(f"ℹ️ No trade: {signal}")
        return None
    
    # Get price from Deriv
    current_price = await deriv_broker.get_tick("frxXAUUSD")
    if not current_price:
        print("❌ Cannot get price from Deriv")
        return None
    
    # Use allocated capital from Portfolio Manager (if available)
    # Default: $10 with 100x leverage
    amount = 10.0
    multiplier = 100
    
    # Execute trade
    result = await deriv_broker.place_order(
        signal=signal,
        instrument="frxXAUUSD",
        amount=amount,
        multiplier=multiplier,
        duration=60  # 60 minutes
    )
    
    if result:
        print(f"✅ Trade executed: {signal} at ${current_price:.2f}")
        # Log to database
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
                None,  # outcome unknown
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

# ---------- Main Trading Loop ----------
async def trading_loop():
    """Main autonomous trading loop"""
    global latest_signals_cache, last_regime_update, autonomous_trading_loop_paused
    
    print("🚀 A.A.G.A AI Trading Loop Started (Deriv)")
    print("=" * 50)
    
    while True:
        try:
            # Check pause state
            if autonomous_trading_loop_paused:
                await asyncio.sleep(5)
                continue
            
            # 1. Get current price from Deriv
            current_price = await deriv_broker.get_tick("frxXAUUSD")
            if not current_price:
                print("⚠️ Waiting for price data...")
                await asyncio.sleep(10)
                continue
            
            # 2. Build OHLC dict (simplified - using tick data)
            # For real OHLC, you'd need historical candles from Deriv
            # Here we create a simple structure from tick
            ohlc_dict = {
                "closes": [current_price] * 60,  # Simulate 60 bars
                "highs": [current_price * 1.001] * 60,
                "lows": [current_price * 0.999] * 60,
                "opens": [current_price * 0.999] * 60
            }
            
            # 3. Classify regime (using cached data or simplified)
            regime = "UNKNOWN"
            try:
                # Simplified regime detection
                if current_price > 2800:
                    regime = "UPTREND"
                elif current_price < 2600:
                    regime = "DOWNTREND"
                else:
                    regime = "SIDEWAYS"
            except:
                regime = "UNKNOWN"
            
            # 4. Get signals from all strategies
            signals = get_all_signals(ohlc_dict)
            
            # 5. Master Agent decides
            decision = master_agent.decide(signals, regime)
            decision["regime"] = regime
            decision["current_price"] = current_price
            decision["pair"] = "XAUUSD"
            
            # 6. Cache for dashboard
            latest_signals_cache["XAUUSD"] = decision
            
            # 7. Execute if BUY/SELL
            if decision["signal"] in ("BUY", "SELL"):
                print(f"📊 Decision: {decision['signal']} @ {current_price:.2f} | "
                      f"Confidence: {decision.get('confidence', 0):.2f} | "
                      f"Reason: {decision.get('reason', '')}")
                
                # Execute trade
                await execute_decision(decision, "XAUUSD")
                
                # Save to signal history
                signal_history.append({
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "signal": decision["signal"],
                    "price": current_price,
                    "confidence": decision.get("confidence", 0),
                    "reason": decision.get("reason", "")
                })
                if len(signal_history) > MAX_SIGNAL_HISTORY:
                    signal_history.pop(0)
            
            # 8. Self-Optimizer (every 30 min)
            if int(time.time()) % 1800 < 10:
                print("🧬 Running Self-Optimizer...")
                # Rule-based adjustment (free)
                self_optimizer.auto_adjust_parameters({"XAUUSD": active_strategies})
                
                # Update regime weights
                self_optimizer.update_regime_weights({"XAUUSD": master_agent})
            
            # 9. LLM Deep Analysis (every hour)
            if int(time.time()) % 3600 < 10:
                suggestions = self_optimizer.llm_suggest_upgrade({"XAUUSD": active_strategies})
                if suggestions:
                    self_optimizer.apply_llm_suggestions(
                        suggestions,
                        {"XAUUSD": active_strategies},
                        {"XAUUSD": master_agent}
                    )
            
            # 10. Sleep before next iteration
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            print(f"❌ Loop error: {e}")
            await asyncio.sleep(30)

# ---------- Daily Report Generator ----------
async def daily_report_loop():
    """Generate daily report at midnight"""
    while True:
        now = datetime.datetime.utcnow()
        # Run at 00:05 UTC
        if now.hour == 0 and now.minute == 5:
            report = generate_daily_report()
            print(report)
            # Send to Telegram if configured
            telegram.send_message(report)
            await asyncio.sleep(60)  # Avoid double-run
        await asyncio.sleep(30)

# ---------- Weekly Pipeline ----------
async def weekly_pipeline_loop():
    """Run weekly self-retraining pipeline"""
    while True:
        now = datetime.datetime.utcnow()
        # Run every Sunday at 01:00 UTC
        if now.weekday() == 6 and now.hour == 1 and now.minute == 0:
            print("🚀 Weekly Self-Retraining Pipeline Started...")
            
            # Retrain RandomForest
            if trade_memory.retrain_model(rf_strategy.model):
                rf_strategy.trained = True
                print("✅ RandomForest retrained")
            
            # Run genetic optimization
            await genetic_optimizer.optimize(EmaDmiStrategy, {
                "adx_threshold": 15,
                "slope_threshold": 0.00005,
                "atr_multiplier_sl": 1.5,
                "rr_ratio": 2.0,
                "min_stop_distance": 1.10
            })
            
            # Generate new strategy via LLM
            market_context = {
                "current_price": await deriv_broker.get_tick("frxXAUUSD") or 2700,
                "trend": "UPTREND",
                "volatility": 15.0
            }
            con = sqlite3.connect("journal.db")
            recent = con.execute(
                "SELECT signal, entry, sl, tp, outcome, pnl FROM trade_journal ORDER BY id DESC LIMIT 10"
            ).fetchall()
            con.close()
            
            new_params = self_optimizer.llm_generate_new_strategy(
                market_context,
                [{"signal": r[0], "entry": r[1], "sl": r[2], "tp": r[3], "outcome": r[4], "pnl": r[5]} for r in recent]
            )
            if new_params:
                print(f"🌟 New strategy generated: {new_params.get('name', 'Unknown')}")
            
            await asyncio.sleep(60)
        await asyncio.sleep(60)

# ---------- Health Check ----------
async def health_check_loop():
    """Monitor Deriv connection health"""
    while True:
        try:
            # Check Deriv connection
            price = await deriv_broker.get_tick("frxXAUUSD")
            if price:
                print(f"💚 Deriv Online | Price: ${price:.2f}")
            else:
                print("💛 Deriv connection issue - attempting reconnect...")
                await deriv_broker.connect()
        except Exception as e:
            print(f"❌ Health check error: {e}")
        await asyncio.sleep(300)  # Every 5 minutes

# ---------- Main Entry ----------
async def main():
    """Main async entry point"""
    print("=" * 60)
    print("🏦 A.A.G.A AI Trading System (Deriv Version)")
    print("=" * 60)
    print(f"📅 Started at: {datetime.datetime.utcnow().isoformat()}")
    print(f"🔗 Broker: Deriv (ws.binaryws.com)")
    print(f"📊 Strategies: {', '.join(s.name for s in active_strategies)}")
    print("=" * 60)
    
    # Connect to Deriv
    await deriv_broker.connect()
    
    # Start all loops concurrently
    await asyncio.gather(
        trading_loop(),
        daily_report_loop(),
        weekly_pipeline_loop(),
        health_check_loop(),
        return_exceptions=True
    )

# ---------- Run ----------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down gracefully...")
        sys.exit(0)
