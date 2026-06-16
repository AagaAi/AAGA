# main.py – Tradevil AGI OS (Sniper Upgrade)
import os, json, sqlite3, datetime, threading, time as _time
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agents.sniper_agent import run_sniper_analysis
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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
config["gemini_api_key"] = GEMINI_API_KEY

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

current_pair = config["default_pair"]
_last_gemini_call = None

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

@app.get("/test-sniper")
def test_sniper():
    return run_sniper_analysis(current_pair)

@app.get("/master-signal")
def master_signal():
    global _last_gemini_call
    daily_tracker.check_reset()

    # 1. Sniper Analysis (main trading engine)
    sniper = run_sniper_analysis(current_pair)

    # 2. Other agents (still provide probabilities)
    news = run_news_analysis()
    mtf = analyze_mtf(current_pair)
    risk_eval = risk_manager.evaluate()
    strategy = strategy_selector.select(current_pair, mtf)

    # 3. Gemini call (throttled)
    now = datetime.datetime.utcnow()
    if (sniper["signal"] in ("BUY", "SELL") and
        (_last_gemini_call is None or (now - _last_gemini_call).seconds > 3600)):
        gemini_advice = gemini_analyst.analyze(sniper, news, mtf, daily_tracker.stats())
        _last_gemini_call = now
    else:
        gemini_advice = {"summary": "Gemini throttled / quota exceeded", "recommended_action": sniper.get("signal", "HOLD"), "confidence": 0.5}

    # 4. Probability combination (weighted, but Sniper dominates)
    sniper_prob = {"BUY": 0.0, "SELL": 0.0, "HOLD": 1.0}
    if sniper["signal"] == "BUY":
        sniper_prob = {"BUY": 0.9, "SELL": 0.0, "HOLD": 0.1}
    elif sniper["signal"] == "SELL":
        sniper_prob = {"BUY": 0.0, "SELL": 0.9, "HOLD": 0.1}

    tech_probs = sniper_prob
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

    # 5. Risk Brief from sniper analysis
    risk_brief = None
    if decision in ("BUY","SELL") and sniper.get("entry_zone") and sniper.get("sl") and sniper.get("tp"):
        if decision == "BUY":
            entry = sniper["entry_zone"][0]
        else:
            entry = sniper["entry_zone"][1]
        risk_brief = {"entry": round(entry,2), "sl": sniper["sl"], "tp": sniper["tp"]}

    # 6. Log to trade journal
    if decision in ("BUY", "SELL") and risk_brief:
        con = sqlite3.connect(DB_PATH)
        con.execute("INSERT INTO trade_journal (timestamp, pair, signal, entry, sl, tp, strategy_used, gemini_insight) VALUES (?,?,?,?,?,?,?,?)",
                    (now.isoformat(), current_pair, decision, risk_brief["entry"], risk_brief["sl"], risk_brief["tp"], strategy["name"], gemini_advice.get("summary","")))
        con.commit()
        con.close()

    # 7. Agent activity log
    hour = now.strftime("%Y-%m-%d %H:00")
    con2 = sqlite3.connect(DB_PATH)
    for ag_name, probs in [("Tech", tech_probs), ("News", news_probs), ("Risk", risk_probs), ("Strategy", strat_probs)]:
        action = max(probs, key=probs.get)
        con2.execute("INSERT INTO agent_log (hour, agent, action, prob, details) VALUES (?,?,?,?,?)",
                     (hour, ag_name, action, probs[action], json.dumps(probs)))
    con2.commit()
    con2.close()

    # 8. Return with extra trend/sweep info for dashboard
    return {
        "decision": decision,
        "probabilities": final_probs,
        "risk_brief": risk_brief,
        "strategy_used": strategy["name"],
        "gemini_advice": gemini_advice,
        "market_trend": sniper.get("trend", "N/A"),
        "sweep_detected": sniper.get("sweep_detected", False),
        "sweep_type": sniper.get("sweep_type"),
        "mss": sniper.get("mss"),
        "fvg": sniper.get("fvg"),
        "current_price": sniper.get("current_price")
    }

@app.get("/agent-log")
def agent_log():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT hour, agent, action, prob, details FROM agent_log WHERE hour >= datetime('now','-1 hour') ORDER BY id DESC LIMIT 20").fetchall()
    con.close()
    return [{"hour": r[0], "agent": r[1], "action": r[2], "prob": r[3], "details": r[4]} for r in rows]

@app.get("/today-trades")
def today_trades():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT timestamp, pair, signal, entry, sl, tp, outcome, pnl FROM trade_journal WHERE date(timestamp) = date('now') ORDER BY timestamp DESC").fetchall()
    con.close()
    return [{"timestamp": r[0], "pair": r[1], "signal": r[2], "entry": r[3], "sl": r[4], "tp": r[5], "outcome": r[6], "pnl": r[7]} for r in rows]

@app.post("/auto-upgrade")
def auto_upgrade():
    best_strategy = backtest_engine.run(current_pair, strategy_selector.get_all())
    strategy_selector.promote(best_strategy)
    return {"new_top_strategy": best_strategy["name"]}

import schedule
def scheduler_thread():
    schedule.every().sunday.at("00:00").do(auto_upgrade)
    while True:
        schedule.run_pending()
        _time.sleep(3600)
threading.Thread(target=scheduler_thread, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))