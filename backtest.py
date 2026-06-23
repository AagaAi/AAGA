# backtest.py - Using Deriv API instead of MetaApi
import asyncio
import json
import datetime
from typing import List, Dict, Any, Optional

# Import your Deriv broker
from agents.broker_deriv import DerivBroker


async def fetch_candles_from_deriv(
    symbol: str = "frxXAUUSD",
    granularity: int = 60,  # seconds: 60, 120, 180, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400
    count: int = 1000,
    end_time: Optional[int] = None
) -> List[Dict[str, float]]:
    """
    Fetch historical OHLC candles from Deriv using ticks_history endpoint.
    
    Returns list of candles: [{"open": x, "high": x, "low": x, "close": x, "time": epoch}, ...]
    """
    broker = DerivBroker()
    if not await broker.connect():
        print("❌ Failed to connect to Deriv")
        return []
    
    if end_time is None:
        end_time = int(datetime.datetime.utcnow().timestamp())
    
    # Build request for candles
    request = {
        "ticks_history": symbol,
        "end": str(end_time),
        "count": count,
        "granularity": granularity,
        "style": "candles"
    }
    
    try:
        await broker.websocket.send(json.dumps(request))
        response = json.loads(await broker.websocket.recv())
        
        await broker.disconnect()
        
        if "error" in response:
            print(f"❌ Deriv candles error: {response['error']['message']}")
            return []
        
        # Parse candles from response
        candles = []
        for candle in response.get("candles", []):
            candles.append({
                "time": candle.get("epoch", 0),
                "open": float(candle.get("open", 0)),
                "high": float(candle.get("high", 0)),
                "low": float(candle.get("low", 0)),
                "close": float(candle.get("close", 0))
            })
        
        print(f"✅ Fetched {len(candles)} candles from Deriv")
        return candles
        
    except Exception as e:
        print(f"❌ Error fetching candles: {e}")
        await broker.disconnect()
        return []


async def fetch_last_month_candles(symbol: str = "frxXAUUSD") -> List[Dict[str, float]]:
    """Fetch approximately 30 days of 1-hour candles."""
    # 30 days * 24 hours = 720 candles
    return await fetch_candles_from_deriv(
        symbol=symbol,
        granularity=3600,  # 1 hour
        count=720
    )


def backtest_strategy(strategy, candles: List[Dict[str, float]], initial_balance: float = 10000.0) -> Dict[str, Any]:
    """
    Simple backtest for a strategy on historical candle data.
    Returns performance metrics.
    """
    if not candles or len(candles) < 50:
        return {
            "win_rate": 0,
            "roi": 0,
            "final_balance": initial_balance,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "profit_factor": 1.0
        }
    
    balance = initial_balance
    trades = []
    position = None
    entry_price = 0.0
    sl = 0.0
    tp = 0.0
    
    for i in range(50, len(candles)):
        # Prepare OHLC dict for strategy
        ohlc_dict = {
            "closes": [c["close"] for c in candles[:i+1]],
            "highs": [c["high"] for c in candles[:i+1]],
            "lows": [c["low"] for c in candles[:i+1]],
            "opens": [c["open"] for c in candles[:i+1]]
        }
        
        # Get signal
        try:
            signal = strategy.get_signal(ohlc_dict)
        except Exception as e:
            print(f"⚠️ Strategy error in backtest: {e}")
            continue
        
        current_price = candles[i]["close"]
        
        # Check if we have an open position
        if position is not None:
            # Check SL/TP
            if position == "BUY":
                if current_price <= sl:
                    # Stop loss hit
                    pnl = (sl - entry_price) / entry_price * balance * 0.01
                    balance += pnl
                    trades.append({"outcome": "LOSS", "pnl": pnl})
                    position = None
                elif current_price >= tp:
                    # Take profit hit
                    pnl = (tp - entry_price) / entry_price * balance * 0.01
                    balance += pnl
                    trades.append({"outcome": "WIN", "pnl": pnl})
                    position = None
            elif position == "SELL":
                if current_price >= sl:
                    pnl = (entry_price - sl) / entry_price * balance * 0.01
                    balance += pnl
                    trades.append({"outcome": "LOSS", "pnl": pnl})
                    position = None
                elif current_price <= tp:
                    pnl = (entry_price - tp) / entry_price * balance * 0.01
                    balance += pnl
                    trades.append({"outcome": "WIN", "pnl": pnl})
                    position = None
        
        # Enter new position if signal and no position
        if position is None and signal.get("signal") in ("BUY", "SELL"):
            position = signal["signal"]
            entry_price = current_price
            sl = signal.get("sl", 0)
            tp = signal.get("tp", 0)
    
    # Calculate metrics
    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    losses = sum(1 for t in trades if t["outcome"] == "LOSS")
    total_trades = len(trades)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    roi = ((balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0
    
    # Profit factor
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0
    
    return {
        "win_rate": round(win_rate, 2),
        "roi": round(roi, 2),
        "final_balance": round(balance, 2),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "profit_factor": round(profit_factor, 2)
    }


# Sync wrapper for genetic_optimizer
def fetch_candles_sync(symbol: str = "frxXAUUSD") -> List[Dict[str, float]]:
    """Sync wrapper for async fetch."""
    return asyncio.run(fetch_last_month_candles(symbol))
