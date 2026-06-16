# main.py – Tradevil AGI OS
import os, json, sqlite3, datetime, threading, time as _time
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from agents.tech_agent import run_tech_analysis
from agents.news_agent import run_news_analysis
from agents.risk_agent import RiskManager
from agents.strategy_agent import StrategySelector
from agents.gemini_agent import GeminiAnalyst
from agents.mtf_agent import analyze_mtf
from engines.backtest_engine import BacktestEngine
from engines.daily_risk import DailyRiskTracker

# ---------- Load Config ----------
CONFIG_PATH = "config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)
# ---------- Load Config ----------
CONFIG_PATH = "config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)

# 🔐 API Key from environment (safe)
import os
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
config["gemini_api_key"] = GEMINI_API_KEY
# ---------- Initialize ----------
app = FastAPI(title="Tradevil AGI OS")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

daily_tracker = DailyRiskTracker(config)
risk_manager = RiskManager(config)
strategy_selector = StrategySelector(config)
gemini_analyst = GeminiAnalyst(config.get("gemini_api_key", ""))
backtest_engine = BacktestEngine(config)

DB_PATH = "journal.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, pair TEXT, signal TEXT, entry REAL, sl REAL, tp REAL,
            outcome TEXT, pnl REAL, strategy_used TEXT, gemini_insight TEXT
        );
        CREATE TABLE IF NOT EXISTS agent_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour TEXT, agent TEXT, action TEXT, prob REAL, details TEXT
        );
    """)
    con.commit()
    con.close()

init_db()

# ---------- Current pair state ----------
current_pair = config["default_pair"]

# ---------- Dashboard ----------
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return FileResponse("dashboard.html")

# ---------- Pair Selector ----------
@app.post("/set-pair")
def set_pair(pair: str):
    global current_pair
    if pair in config["pairs"]:
        current_pair = pair
        return {"status":"ok", "current_pair": pair}
    raise HTTPException(400, "Invalid pair")

# ---------- Master Signal ----------
@app.get("/master-signal")
def master_signal():
    daily_tracker.check_reset()
    # 1. Market Data
    tech = run_tech_analysis(current_pair)
    news = run_news_analysis()
    mtf = analyze_mtf(current_pair)
    risk_eval = risk_manager.evaluate()

    # 2. Strategy Selection (AI picks best strategy)
    strategy = strategy_selector.select(current_pair, mtf)

    # 3. Gemini Learning (why previous trades lost, market pattern)
    gemini_advice = gemini_analyst.analyze(tech, news, mtf, daily_tracker.stats())

    # 4. Agent Probabilities (with strategy weight)
    probs = {
        "BUY": tech["prob_buy"]*0.4 + news["prob_buy"]*0.2 + risk_eval["prob_buy"]*0.2 + strategy["prob_buy"]*0.2,
        "SELL": tech["prob_sell"]*0.4 + news["prob_sell"]*0.2 + risk_eval["prob_sell"]*0.2 + strategy["prob_sell"]*0.2,
        "HOLD": tech["prob_hold"]*0.4 + news["prob_hold"]*0.2 + risk_eval["prob_hold"]*0.2 + strategy["prob_hold"]*0.2
    }
    decision = max(probs, key=probs.get)

    # 5. Risk & SL/TP from selected strategy
    if decision in ("BUY","SELL"):
        sl_tp = strategy["calculate_sltp"](decision, tech["current_price"])
    else:
        sl_tp = None

    # 6. Log
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO trade_journal (timestamp,pair,signal,entry,sl,tp,strategy_used,gemini_insight) VALUES (?,?,?,?,?,?,?,?)",
                (datetime.datetime.utcnow().isoformat(), current_pair, decision,
                 tech["current_price"], sl_tp["sl"] if sl_tp else None,
                 sl_tp["tp"] if sl_tp else None, strategy["name"], gemini_advice.get("summary")))
    con.commit()
    con.close()

    return {
        "decision": decision, "probabilities": probs,
        "risk_brief": sl_tp, "strategy_used": strategy["name"],
        "gemini_advice": gemini_advice
    }

# ---------- Backtest & Auto-Upgrade (triggered by scheduler) ----------
@app.post("/auto-upgrade")
def auto_upgrade():
    best_strategy = backtest_engine.run(current_pair, strategy_selector.get_all())
    strategy_selector.promote(best_strategy)
    return {"new_top_strategy": best_strategy["name"]}

# ---------- Agent logs for dashboard ----------
@app.get("/agent-log")
def agent_log():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT * FROM agent_log ORDER BY id DESC LIMIT 50").fetchall()
    con.close()
    return rows

# ---------- Scheduler for weekly optimization ----------
import schedule
def scheduler_thread():
    schedule.every().sunday.at("00:00").do(auto_upgrade)
    while True:
        schedule.run_pending()
        _time.sleep(3600)
threading.Thread(target=scheduler_thread, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))