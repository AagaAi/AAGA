# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository.

## What this is

**A.A.G.A** is an autonomous AI trading system for **XAUUSD (gold)**. It runs as a
FastAPI web service that continuously polls live prices from the **Deriv**
WebSocket API, aggregates BUY/SELL/HOLD signals from several strategy agents,
combines them with a weighted-voting "master" agent, and places trades — while a
self-optimizer tunes parameters over time using both rule-based logic and LLMs
(Gemini + Grok).

There is **no test suite, no CI, and no build step**. It is a single Python
service plus a static HTML dashboard.

## Running it

```bash
pip install -r requirements.txt
python main.py                    # starts uvicorn on 0.0.0.0:8000 with reload
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

On startup (`lifespan` in `main.py`) the service connects to Deriv and launches
the async `trading_loop()`. The loop polls a price every ~60s, builds a
(simplified) OHLC window, detects a coarse regime, collects signals, asks the
master agent to decide, and executes via `DerivBroker`.

### HTTP endpoints (`main.py`)
- `GET /` — service banner
- `GET /health` — health + Deriv connection state
- `GET /status` — strategies, latest cached signal, pause state
- `POST /pause` / `POST /resume` — toggle the trading loop
- `GET /report` — daily P&L report (`agents/pdf_report.py`)

### Environment variables
Everything sensitive is read from the environment (no secrets in the repo):
- `DERIV_API_TOKEN`, `DERIV_APP_ID` (defaults to `1089`) — `agents/broker_deriv.py`
- `GEMINI_API_KEY`, `GROK_API_KEY` — `agents/llm_router.py` (LLM features degrade
  gracefully to no-ops if unset)
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — notifications
- `METAAPI_TOKEN`, `METAAPI_ACCOUNT_ID` — **legacy**, only used by `trainer.py`

## Architecture

`main.py` is the composition root. It instantiates every agent as a module-level
singleton, wires them together, and owns the async trading loop and global state
(`latest_signals_cache`, `autonomous_trading_loop_paused`, etc.).

Data flow per loop tick:
```
DerivBroker.get_tick()  ->  ohlc window  ->  each strategy.get_signal()
        -> MasterAgent.decide(signals, regime)  -> execute_decision() -> DerivBroker.place_order()
        -> trade recorded in journal.db (trade_journal)
        -> periodically: SelfOptimizer adjusts params / applies LLM suggestions
```

### Agents (`agents/`)
- **`broker_deriv.py` — `DerivBroker`**: async Deriv WebSocket wrapper.
  `get_tick`, `place_order` (BUY→`MULTUP`, SELL→`MULTDOWN` multiplier contracts).
  The live symbol is `frxXAUUSD`.
- **`master_agent.py` — `MasterAgent`**: weighted voting across strategies with
  per-regime multipliers (`regime_weights`) and a `performance_memory` weight per
  strategy. Applies news + daily-loss overrides before voting.
- **Strategies** (all expose `.name` and `.get_signal(ohlc_dict) -> dict`):
  - `strategy_agent.py` — `EmaDmiStrategy` (name `"EMA+DMI+BOS"`): the "real"
    technical strategy (EMA stack + DMI/ADX + break-of-structure, with an ATR
    SL/TP and a trend-following fallback).
  - `ai_agent.py` — `RandomForestStrategy` (name `"RandomForestStrategy"`): a
    **stub that always returns BUY**. Also contains an unused `AIAgent` demo.
  - `ppo_agent.py` — `PPOAgent` (name `"PPO"`): reinforcement-learning agent
    (stable-baselines3 PPO + a custom `gymnasium` `TradingEnv`). Falls back to an
    EMA rule when not trained.
- **`self_optimizer.py` — `SelfOptimizer`**: the self-learning engine.
  Layer 1 (free, rule-based) auto-adjusts strategy params from recent
  win/loss stats; Layer 2 (LLM, rate-limited to ~1/hour) asks the router for
  parameter/regime suggestions and applies them. Reads/writes `journal.db`.
- **`llm_router.py` — `MultiLLMRouter`**: alternates Gemini 2.0 Flash and Grok
  per call with fallback + cooldown on errors/rate-limits. Central place for all
  LLM prompts (returns raw JSON that callers parse).
- **`gemini_agent.py` / `strategy_factory.py`**: Gemini-only path that can
  *generate new strategy classes at runtime* from LLM-returned params
  (`GeneratedStrategy` uses `eval()` on condition strings — treat as trusted-input
  only).
- **News/sentiment**: `news_agent.py` (RSS fetch) + `nlp_sentiment.py`
  (keyword sentiment). `MasterAgent` uses `news_agent.analyze` as a risk override.
- **Regime/classification**: `regime_detector.py`, `market_classifier.py`,
  `regime`-related logic in `main.py` (currently a crude price-threshold heuristic).
- **Persistence**: `memory.py` — `TradeMemory` (SQLite, RandomForest experience
  replay). Notifications: `telegram_notifier.py`, `telegram_bot.py`. Reporting:
  `pdf_report.py`. Genetic search: `genetic_optimizer.py`. GAN data augmentation:
  `gan_augmentor.py`. Weekly retrain orchestration: `weekly_pipeline.py`.

### Other files
- `backtest.py` — Deriv-based historical backtester (`fetch_candles_from_deriv`,
  `backtest_strategy`, plus a `fetch_candles_sync` wrapper).
- `trainer.py` — intended as a weekly cron job (genetic optimize, GAN, retrain).
  **Currently broken** — see Known issues.
- `dashboard.html` — standalone static dashboard (not served by FastAPI).
- `config.json` — capital, pair list, risk params, `agent_weights`, strategy pool.

## Data & persistence

State lives in a local SQLite file **`journal.db`** (created at runtime, git-ignored
in practice — do not commit it). Key tables:
- `trade_journal` — every executed trade (timestamp, pair, signal, entry, sl, tp,
  outcome, pnl, strategy_used, market_condition). Read by the optimizer, reports,
  and risk checks.
- `agent_log` — self-optimizer adjustment log.
- `trade_memory` — RF experience replay (the **only** table with a `CREATE TABLE`
  in code, in `memory.py`).

> **Gotcha:** `trade_journal` and `agent_log` are read/written throughout the code
> but never created by any `CREATE TABLE` statement in this repo. Inserts assume
> the schema already exists (columns above). If you touch the trade-recording path,
> you likely need to add schema initialization.

## Conventions

- **Signal contract**: every strategy's `get_signal()` returns a dict with at least
  `signal` ∈ `{"BUY","SELL","HOLD"}`, and usually `entry_zone`, `sl`, `tp`,
  `trend`, `current_price`. `MasterAgent` also reads `strategy` and `confidence`.
  Keep this shape when adding a strategy, and register it in `active_strategies` in
  `main.py`.
- **Regimes**: `UPTREND`, `DOWNTREND`, `SIDEWAYS`, `HIGH_VOLATILITY`, `UNKNOWN`.
  New strategies should be given weights in `MasterAgent.regime_weights` keyed by
  their `.name`.
- **Async vs sync**: the broker and loop are `asyncio`; strategies and optimizers
  are synchronous. Don't block the event loop from inside `trading_loop`.
- **Fail-soft everywhere**: agents wrap external calls (LLMs, RSS, WebSocket) in
  try/except and return safe defaults / print a `⚠️`/`❌` line rather than raising.
  Match this style — a failing agent must not kill the loop.
- **Logging** is `print()` with emoji prefixes (✅ ok, ⚠️ warning, ❌ error,
  🔧/🤖/🧬 optimizer). There is no logging framework.
- **Language**: several files (`ai_agent.py`, `news_agent.py`, `nlp_sentiment.py`)
  contain **Tamil** comments/docstrings. Preserve existing comments; new comments
  can be in English.
- **Strategy `.name` is the join key** across `main.py`, `MasterAgent`, and
  `SelfOptimizer`. Renaming a strategy silently breaks weighting/optimization —
  keep names consistent everywhere.

## Known issues / footguns (be careful here)

- **`trainer.py` is broken**: it imports `from metaapi_cloud_sdk import MetaApi`
  (removed from `requirements.txt`) and `run_comparison` from `backtest.py`
  (which no longer defines it after the Deriv refactor). Don't assume it runs.
- **Strategy-name mismatch**: `main.py` loads `RandomForestStrategy` with
  `.name = "RandomForestStrategy"`, but `MasterAgent.regime_weights` and
  `SelfOptimizer` refer to `"RandomForest"`, and the optimizer also expects RF
  attributes (`atr_multiplier_sl`, `atr_multiplier_tp`, `tp_reduction_points`)
  that the stub class doesn't have. So RF-specific weighting/tuning is effectively
  dead. Reconcile the name/attributes if you make RF real.
- The live `RandomForestStrategy` is a **stub that always votes BUY** — it biases
  the master vote. Keep this in mind when reasoning about decisions.
- `main.py`'s OHLC window and regime detection are heavily **simplified/simulated**
  (flat OHLC from a single tick; regime from raw price thresholds). Signals derived
  from indicators are therefore degenerate in the live loop.
- `StrategyFactory.GeneratedStrategy` runs `eval()` on LLM-produced condition
  strings — only safe because those come from your own Gemini prompt. Never feed
  untrusted text into it.

## Making changes

- **Add a strategy**: create a class with `.name` + `.get_signal(ohlc_dict)`,
  import it in `main.py`, append to `active_strategies`, and add its name to
  `MasterAgent.regime_weights` / `performance_memory` seeding.
- **Change trading behavior**: it almost always lives in `trading_loop()` and
  `execute_decision()` in `main.py`, or in `MasterAgent.decide()`.
- **Touch broker/API**: `agents/broker_deriv.py` is the only place that talks to
  Deriv; `backtest.py` also opens its own Deriv connection for history.
- Verify by importing/running the service locally (`python main.py`) — there are no
  automated tests to run. Guard any new external dependency behind an env var and a
  graceful fallback, matching the existing pattern.
