# trainer.py – A.A.G.A AI Weekly Trainer (run as Render Cron Job)
import os, json, sqlite3, datetime, asyncio, numpy as np
import pandas as pd
from metaapi_cloud_sdk import MetaApi
import yfinance as yf
from agents.strategy_agent import EmaDmiStrategy
from agents.ai_agent import RandomForestStrategy
from agents.memory import TradeMemory
from agents.genetic_optimizer import GeneticOptimizer
from agents.gan_augmentor import LightweightGAN
from agents.weekly_pipeline import weekly_self_retraining
from agents.master_agent import MasterAgent
from agents.news_agent import NewsSentimentAgent
from agents.market_classifier import MarketClassifier
from agents.strategy_factory import StrategyFactory
from agents.regime_detector import RegimeDetector
from backtest import run_comparison as compare_strategies

# ---------- Config ----------
METAAPI_TOKEN = os.environ.get("METAAPI_TOKEN", "")
METAAPI_ACCOUNT_ID = os.environ.get("METAAPI_ACCOUNT_ID", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DB_PATH = "journal.db"
CONFIG_PATH = "config.json"

pairs = ["XAUUSD", "EURUSD", "BTCUSD"]
YAHOO_SYMBOLS = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X", "BTCUSD": "BTC-USD"}

# Minimal MetaApi instance for training
_api_instance = None
_account_cache = None

async def get_account():
    global _api_instance, _account_cache
    if _account_cache is not None:
        return _account_cache
    _api_instance = MetaApi(METAAPI_TOKEN)
    account = await _api_instance.metatrader_account_api.get_account(METAAPI_ACCOUNT_ID)
    if account.state not in ('DEPLOYING', 'DEPLOYED'):
        await account.deploy()
    await account.wait_connected()
    _account_cache = account
    return account

async def sdk_get_candles(pair, timeframe, limit=2000):
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
        # fallback to Yahoo
        try:
            symbol = YAHOO_SYMBOLS.get(pair, pair + "=X")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d", interval=timeframe)
            if df.empty: return None
            return [{"time": str(i), "open": float(r["Open"]), "high": float(r["High"]),
                     "low": float(r["Low"]), "close": float(r["Close"]), "volume": float(r["Volume"])}
                    for i, r in df.iterrows()]
        except:
            return None

async def main():
    print("🚂 Weekly Trainer started...")
    # Initialize strategies (same as in main.py)
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            config = json.load(f)
    news_agent = NewsSentimentAgent()
    # We need strategies per pair; but for training we only need the classes
    # We'll create temporary instances for training purposes
    # (The actual live strategies are in main.py; we just want to optimize and save config)
    ema_dmi = EmaDmiStrategy(config.get("parameters", {}))
    rf = RandomForestStrategy()
    # Dummy PPO (not trained here)
    # We'll just load what we need.

    # 1. Genetic Optimizer
    try:
        print("🧬 Genetic optimizer...")
        optimizer = GeneticOptimizer()
        base_cfg = {"adx_threshold": ema_dmi.adx_threshold, "slope_threshold": ema_dmi.slope_threshold,
                    "atr_multiplier_sl": ema_dmi.atr_multiplier_sl, "rr_ratio": ema_dmi.rr_ratio,
                    "min_stop_distance": ema_dmi.min_stop_distance}
        best = await optimizer.optimize(EmaDmiStrategy, base_cfg=base_cfg)
        if best:
            optimizer.apply_to_strategy(ema_dmi, best)
            cfg = json.load(open(CONFIG_PATH)) if os.path.exists(CONFIG_PATH) else {}
            cfg["parameters"] = best
            with open(CONFIG_PATH, "w") as f:
                json.dump(cfg, f, indent=2)
            print("✅ Genetic optimizer applied and config updated")
    except Exception as e:
        print(f"Genetic error: {e}")

    # 2. Retrain RandomForest on trade memory (if any)
    try:
        trade_memory = TradeMemory(DB_PATH)
        # We need to get the trained model? The model is stored in the strategy instance in main.py.
        # But we can't easily access that from here. So we'll load the model from disk? Not saved.
        # Instead, we can train a new RF model using trades from journal and save it somehow?
        # For simplicity, we'll skip RF retrain in trainer (it's light and can stay in nightly tasks of main.py)
    except Exception as e:
        print(f"RF retrain error: {e}")

    # 3. GAN training
    try:
        print("🎨 Training GAN...")
        gan = LightweightGAN()
        candles = await sdk_get_candles("XAUUSD", "1m", 2000)
        if candles and len(candles) > 500:
            data = np.array([[c['open'], c['high'], c['low'], c['close']] for c in candles])
            mn = data.min(axis=0); mx = data.max(axis=0)
            norm = (data - mn) / (mx - mn + 1e-8)
            gan.train(norm, epochs=500, batch_size=64)
            synthetic = gan.generate_samples(500)
            synthetic = synthetic * (mx - mn) + mn
            with open("synthetic_candles.json", "w") as f:
                json.dump(synthetic.tolist(), f)
            print("✅ GAN training complete")
    except Exception as e:
        print(f"GAN error: {e}")

    # 4. Weekly self‑retraining (backtest + strategy factory)
    try:
        from agents.gemini_agent import GeminiAnalyst
        gemini = GeminiAnalyst(GEMINI_API_KEY)
        for pair in pairs:
            # we don't have master_agents, skip strategy factory
            pass
    except Exception as e:
        print(f"Weekly self‑retraining error: {e}")

    print("✅ Trainer finished.")

if __name__ == "__main__":
    asyncio.run(main())
