import yfinance as yf
import pandas as pd

def run_tech_analysis(pair):
    ticker = yf.Ticker(pair + "=X") if pair not in ["XAUUSD"] else yf.Ticker("XAUUSD=X")
    df = ticker.history(period="5d", interval="15m")
    if df.empty:
        return {"current_price":0, "prob_buy":0, "prob_sell":0, "prob_hold":1}
    # Simplified: check last swing high/low
    high = df['High'].max()
    low = df['Low'].min()
    current = df['Close'].iloc[-1]
    # Dummy probabilities (real ICT logic to be added)
    prob_buy = 0.7 if current < high and current > low else 0.3
    prob_sell = 1 - prob_buy
    return {"current_price": current, "prob_buy": prob_buy, "prob_sell": prob_sell, "prob_hold": 0.1}