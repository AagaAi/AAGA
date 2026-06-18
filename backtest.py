# backtest.py
import json, datetime, asyncio
import pandas as pd
import numpy as np
from metaapi_cloud_sdk import MetaApi
from agents.strategy_agent import EmaDmiStrategy
from agents.ai_agent import RandomForestStrategy
import os

METAAPI_TOKEN = os.environ.get("METAAPI_TOKEN", "")
METAAPI_ACCOUNT_ID = os.environ.get("METAAPI_ACCOUNT_ID", "")

async def fetch_last_month_candles(symbol="XAUUSD", timeframe="1m"):
    api = MetaApi(METAAPI_TOKEN)
    account = await api.metatrader_account_api.get_account(METAAPI_ACCOUNT_ID)
    await account.wait_connected()
    start_time = datetime.datetime.utcnow() - datetime.timedelta(days=30)
    candles = await account.get_historical_candles(
        symbol=symbol, timeframe=timeframe,
        start_time=start_time, limit=100000)
    return candles

def backtest_strategy(strategy, candles):
    trades = []
    balance = 10000.0
    position = 0
    entry_price = 0.0
    wins = losses = 0
    total_pnl = 0.0

    for i in range(50, len(candles)-1):
        slice_c = candles[:i+1]
        highs = [c['high'] for c in slice_c]
        lows  = [c['low'] for c in slice_c]
        closes = [c['close'] for c in slice_c]
        opens = [c['open'] for c in slice_c]
        ohlc = {'highs': highs, 'lows': lows, 'closes': closes, 'opens': opens}
        signal = strategy.get_signal(ohlc)
        current_price = closes[-1]

        if position != 0:
            if position == 1:  # long
                if current_price <= signal.get('sl', 0) or current_price >= signal.get('tp', 9999):
                    pnl = (current_price - entry_price) * 100
                    balance += pnl
                    total_pnl += pnl
                    if pnl > 0: wins += 1
                    else: losses += 1
                    position = 0
            else:  # short
                if current_price >= signal.get('sl', 9999) or current_price <= signal.get('tp', 0):
                    pnl = (entry_price - current_price) * 100
                    balance += pnl
                    total_pnl += pnl
                    if pnl > 0: wins += 1
                    else: losses += 1
                    position = 0

        if position == 0 and signal['signal'] in ('BUY','SELL'):
            if signal['entry_zone'] and signal['sl'] and signal['tp']:
                position = 1 if signal['signal'] == 'BUY' else -1
                entry_price = current_price

    if position != 0:
        final_price = closes[-1]
        if position == 1: pnl = (final_price - entry_price) * 100
        else: pnl = (entry_price - final_price) * 100
        balance += pnl
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1

    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    roi = (balance - 10000) / 10000 * 100
    return {
        "strategy": strategy.__class__.__name__,
        "trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "roi": round(roi, 2),
        "final_balance": round(balance, 2)
    }

async def run_comparison():
    candles = await fetch_last_month_candles()
    if not candles or len(candles) < 200:
        return {"error": "Not enough historical data"}

    ema_dmi = EmaDmiStrategy()
    rf_strat = RandomForestStrategy()
    # For Random Forest, we need to train it on some data – use a few random labels (placeholder)
    # In a real scenario, you'd train on past trades. Here we just initialize.
    # Since it's not trained, it'll fallback to trend, which is fine for comparison.
    # Better: train on a few fake samples to have it active
    # We'll quickly create dummy trade data to train RF
    dummy_trades = [{"signal": "BUY", "entry": 2000, "sl": 1998, "tp": 2005},
                    {"signal": "SELL", "entry": 2005, "sl": 2007, "tp": 2000}]
    rf_strat.train_from_trades(dummy_trades)

    results = []
    for strat in [ema_dmi, rf_strat]:
        result = backtest_strategy(strat, candles)
        results.append(result)

    best = max(results, key=lambda x: x['roi'])
    return {"results": results, "best_strategy": best['strategy']}