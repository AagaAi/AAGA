# main.py – Tradevil AGI OS (Complete)
import os, json, sqlite3, datetime, threading, time as _time
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
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

# API Key from environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
config["gemini_api_key"] = GEMINI_API_KEY

# ---------- Initialize ----------
app = FastAPI(title="Tradevil AGI OS")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

daily_tracker = DailyRiskTracker(config)
risk_manager = RiskManager(config)
strategy_selector = StrategySelector(config)
gemini_analyst = GeminiAnalyst(GEMINI_API_KEY)
backtest_engine = BacktestEngine(config)

DB_PATH = "journal.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            pair TEXT,
            signal TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            outcome TEXT,
            pnl REAL,
            strategy_used TEXT,
            gemini_insight TEXT
        );
        CREATE TABLE IF NOT EXISTS agent_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour TEXT,
            agent TEXT,
            action TEXT,
            prob REAL,
            details TEXT
        );
    """)
    con.commit()
    con.close()

init_db()

# Current pair state
current_pair = config["default_pair"]

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return FileResponse("dashboard.html")

@app.post("/set-pair")
def set_pair(pair: str):
    global current_pair
    if pair in config["pairs"]:
        current_pair = pair
        return {"status": "ok", "current_pair": pair}
    raise HTTPException(400, "Invalid pair")

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/master-signal")
def master_signal():
    daily_tracker.check_reset()

    # 1. Agent analysis
    tech = run_tech_analysis(current_pair)
    news = run_news_analysis()
    mtf = analyze_mtf(current_pair)
    risk_eval = risk_manager.evaluate()
    strategy = strategy_selector.select(current_pair, mtf)
    gemini_advice = gemini_analyst.analyze(tech, news, mtf, daily_tracker.stats())

    # 2. Agent probabilities
    tech_probs = {"BUY": tech["prob_buy"], "SELL": tech["prob_sell"], "HOLD": tech["prob_hold"]}
    news_probs = {"BUY": news["prob_buy"], "SELL": news["prob_sell"], "HOLD": news["prob_hold"]}
    risk_probs = {"BUY": risk_eval["prob_buy"], "SELL": risk_eval["prob_sell"], "HOLD": risk_eval["prob_hold"]}
    strat_probs = {"BUY": strategy["prob_buy"], "SELL": strategy["prob_sell"], "HOLD": strategy["prob_hold"]}

    w = config["parameters"]["agent_weights"]
    final_probs = {
        "BUY": tech_probs["BUY"]*w["tech"] + news_probs["BUY"]*w["news"] + risk_probs["BUY"]*w["risk"] + strat_probs["BUY"]*w["strategy"],
        "SELL": tech_probs["SELL"]*w["tech"] + news_probs["SELL"]*w["news"] + risk_probs["SELL"]*w["risk"] + strat_probs["SELL"]*w["strategy"],
        "HOLD": tech_probs["HOLD"]*w["tech"] + news_probs["HOLD"]*w["news"] + risk_probs["HOLD"]*w["risk"] + strat_probs["HOLD"]*w["strategy"]
    }
    decision = max(final_probs, key=final_probs.get)

    # 3. Risk Brief from tech analysis (accurate entry, SL, TP)
    risk_brief = None
    if decision in ("BUY","SELL") and tech.get("entry_zone") and tech.get("stop_loss") and tech.get("take_profit"):
        if decision == "BUY":
            entry = tech["entry_zone"][0]   # buy at bottom of FVG
            sl = tech["stop_loss"]
            tp = tech["take_profit"]
        else:
            entry = tech["entry_zone"][1]   # sell at top of FVG
            sl = tech["stop_loss"]
            tp = tech["take_profit"]
        risk_brief = {"entry": round(entry,2), "sl": round(sl,2), "tp": round(tp,2)}

    # 4. Log to trade journal
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO trade_journal (timestamp, pair, signal, entry, sl, tp, strategy_used, gemini_insight)
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        datetime.datetime.utcnow().isoformat(),
        current_pair,
        decision,
        risk_brief["entry"] if risk_brief else None,
        risk_brief["sl"] if risk_brief else None,
        risk_brief["tp"] if risk_brief else None,
        strategy["name"],
        gemini_advice.get("summary", "")
    ))
    con.commit()
    con.close()

    # 5. Log agent activity
    hour = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00")
    con2 = sqlite3.connect(DB_PATH)
    for ag_name, probs in [("Tech", tech_probs), ("News", news_probs), ("Risk", risk_probs), ("Strategy", strat_probs)]:
        action = max(probs, key=probs.get)
        con2.execute("INSERT INTO agent_log (hour, agent, action, prob, details) VALUES (?,?,?,?,?)",
                     (hour, ag_name, action, probs[action], json.dumps(probs)))
    con2.commit()
    con2.close()

    return {
        "decision": decision,
        "probabilities": final_probs,
        "risk_brief": risk_brief,
        "strategy_used": strategy["name"],
        "gemini_advice": gemini_advice
    }

@app.get("/agent-log")
def agent_log():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT hour, agent, action, prob, details
        FROM agent_log
        WHERE hour >= datetime('now', '-1 hour')
        ORDER BY id DESC LIMIT 20
    """).fetchall()
    con.close()
    return [{"hour": r[0], "agent": r[1], "action": r[2], "prob": r[3], "details": r[4]} for r in rows]

@app.get("/today-trades")
def today_trades():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT timestamp, pair, signal, entry, sl, tp, outcome, pnl
        FROM trade_journal
        WHERE date(timestamp) = date('now')
        ORDER BY timestamp DESC
    """).fetchall()
    con.close()
    trades = []
    for r in rows:
        trades.append({
            "timestamp": r[0],
            "pair": r[1],
            "signal": r[2],
            "entry": r[3],
            "sl": r[4],
            "tp": r[5],
            "outcome": r[6],
            "pnl": r[7]
        })
    return trades

@app.post("/auto-upgrade")
def auto_upgrade():
    best_strategy = backtest_engine.run(current_pair, strategy_selector.get_all())
    strategy_selector.promote(best_strategy)
    return {"new_top_strategy": best_strategy["name"]}

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