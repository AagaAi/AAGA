# agents/weekly_pipeline.py
import asyncio, datetime, json, sqlite3, time
import pandas as pd
from agents.strategy_agent import EmaDmiStrategy
from agents.ai_agent import RandomForestStrategy
from agents.master_agent import MasterAgent
from agents.memory import TradeMemory
from agents.gemini_agent import GeminiAnalyst
from agents.strategy_factory import StrategyFactory
from backtest import backtest_strategy, fetch_last_month_candles

async def weekly_self_retraining(db_path, active_strategies, master_agent, trade_memory, gemini_analyst, strategy_factory):
    print("🚀 Weekly Self‑Retraining Pipeline Started...")

    # 1. Retrain RandomForest
    if len(active_strategies) > 1 and trade_memory.retrain_model(active_strategies[1].model):
        active_strategies[1].trained = True
        print("✅ RandomForest retrained on all experiences")

    # 2. Backtest all active strategies on last 30 days
    candles = await fetch_last_month_candles()
    if candles and len(candles) >= 200:
        backtest_results = []
        for strat in active_strategies:
            result = backtest_strategy(strat, candles)
            result["name"] = strat.name
            backtest_results.append(result)
            print(f"📊 {strat.name} 30‑day backtest: Win Rate {result['win_rate']}%, ROI {result['roi']}%")

        # 3. Update master agent weights (top 3 get higher weights)
        sorted_results = sorted(backtest_results, key=lambda x: x['roi'], reverse=True)
        for i, res in enumerate(sorted_results[:3]):
            weight = 0.8 - i * 0.2
            master_agent.performance_memory[res['name']] = weight
            print(f"🔧 {res['name']} weight set to {weight}")
    else:
        print("⚠️ Not enough historical data for weekly backtest")

    # 4. Gemini strategy generation (if available)
    if strategy_factory and gemini_analyst.available and gemini_analyst.disabled_until <= time.time():
        try:
            # Market context
            candles_1h = await fetch_last_month_candles()  # reuse same data for simplicity
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

            con = sqlite3.connect(db_path)
            recent = con.execute("SELECT signal, entry, sl, tp, outcome, pnl FROM trade_journal ORDER BY id DESC LIMIT 10").fetchall()
            con.close()
            recent_trades = [{"signal": r[0], "entry": r[1], "sl": r[2], "tp": r[3], "outcome": r[4], "pnl": r[5]} for r in recent]

            new_params = strategy_factory.generate_strategy_params(market_context, recent_trades)
            if new_params:
                new_strategy = strategy_factory.create_strategy_from_params(new_params)
                if new_strategy:
                    # Save for evaluation – you can later add paper trading logic
                    with open("pending_strategy.json", "w") as f:
                        json.dump(new_params, f)
                    print(f"🌟 New strategy generated: {new_strategy.name}. Saved for evaluation.")
        except Exception as e:
            print(f"Strategy generation error: {e}")
    else:
        print("ℹ️ Gemini unavailable – skipping strategy generation")

    print("✅ Weekly Self‑Retraining Pipeline Completed")