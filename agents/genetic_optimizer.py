# agents/genetic_optimizer.py
import random, copy, asyncio, pandas as pd
from agents.strategy_agent import EmaDmiStrategy
from agents.ai_agent import RandomForestStrategy
from backtest import fetch_candles_sync, backtest_strategy  # Changed import

class GeneticOptimizer:
    # ... rest of the class stays the same ...
    
    async def optimize(self, strategy_class, base_cfg=None):
        # Use sync wrapper instead of async fetch
        candles = fetch_candles_sync()  # Changed from await fetch_last_month_candles()
        if not candles or len(candles) < 200:
            print("🧬 Not enough data for genetic optimization")
            return None
        # ... rest of the method stays the same ...
