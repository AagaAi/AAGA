# agents/self_optimizer.py – Self-Learning, Self-Upgrading, Self-Strategy Engine
# Runs two layers:
#   Layer 1 (FREE): Rule-based auto-adjust from trade outcomes (every 30 min)
#   Layer 2 (LLM):  AI-powered deep analysis + new strategy generation (every 1 hour)
import json
import sqlite3
import datetime
import time as _time
from typing import Optional, Dict, Any

class SelfOptimizer:
    """
    Makes the trading system self-improving without human intervention:

    1. auto_adjust_parameters()   → Rule-based param tuning from win/loss patterns
    2. update_regime_weights()    → Data-driven regime→strategy weight mapping
    3. llm_suggest_upgrade()      → LLM deep analysis + parameter suggestions
    4. apply_llm_suggestions()    → Apply LLM suggestions to live strategies
    5. llm_generate_new_strategy()→ Ask LLM to create brand new strategy for current market
    """

    def __init__(self, db_path: str = "journal.db", llm_router=None):
        self.db  = db_path
        self.llm = llm_router

        self.last_llm_optimize   = 0
        self.llm_optimize_interval = 3600   # LLM analyze every 1 hour
        self.min_trades_for_llm  = 5        # need at least 5 trades before LLM analyzes

        self.performance_window  = 20       # last 20 trades for analysis
        self.rule_adjust_window  = 10       # last 10 trades for rule-based adjust

        # Track learning history
        self.param_adjustments   = []       # log of all adjustments made
        self.regime_performance  = {}       # {regime: {strategy: win_rate}}
        self.generated_strategies = []      # new strategies created by LLM
        self.upgrade_count       = 0        # total upgrades applied

        print("🧬 Self-Optimizer initialized (rule-based + LLM hybrid)")

    # ── Database Helpers ─────────────────────────────────────────────────────

    def get_recent_trades(self, limit: int = 20, pair: str = None) -> list:
        try:
            con = sqlite3.connect(self.db)
            q = """SELECT signal, entry, sl, tp, outcome, pnl, strategy_used,
                          market_condition, timestamp
                   FROM trade_journal WHERE outcome IS NOT NULL"""
            params = []
            if pair:
                q += " AND pair=?"; params.append(pair)
            q += " ORDER BY timestamp DESC LIMIT ?"; params.append(limit)
            rows = con.execute(q, params).fetchall()
            con.close()
            return [{"signal": r[0], "entry": r[1], "sl": r[2], "tp": r[3],
                     "outcome": r[4], "pnl": r[5], "strategy": r[6],
                     "regime": r[7] or "UNKNOWN", "timestamp": r[8]}
                    for r in rows]
        except:
            return []

    def _save_adjustment_log(self, action: str, details: str):
        try:
            con = sqlite3.connect(self.db)
            con.execute(
                "INSERT INTO agent_log (hour,agent,action,prob,details) VALUES (?,?,?,?,?)",
                (datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00"),
                 "SelfOptimizer", action, 1.0, details)
            )
            con.commit(); con.close()
        except: pass

    # ── Performance Calculators ───────────────────────────────────────────────

    def _compute_perf(self, trades: list) -> dict:
        if not trades:
            return {"win_rate": 0, "avg_pnl": 0, "total_pnl": 0, "trades": 0}
        wins = sum(1 for t in trades if t.get("outcome") == "WIN")
        pnls = [t.get("pnl") or 0 for t in trades]
        return {
            "win_rate": round(wins / len(trades) * 100, 1),
            "avg_pnl":  round(sum(pnls) / len(pnls), 2),
            "total_pnl": round(sum(pnls), 2),
            "trades":   len(trades)
        }

    def _by_strategy(self, trades: list) -> dict:
        grouped = {}
        for t in trades:
            s = t.get("strategy", "Unknown")
            grouped.setdefault(s, []).append(t)
        return {s: self._compute_perf(ts) for s, ts in grouped.items()}

    def _by_regime(self, trades: list) -> dict:
        grouped = {}
        for t in trades:
            r = t.get("regime", "UNKNOWN")
            grouped.setdefault(r, []).append(t)
        return {r: self._compute_perf(ts) for r, ts in grouped.items()}

    # ── Layer 1: Rule-Based Auto-Adjust (FREE, no LLM) ───────────────────────

    def auto_adjust_parameters(self, active_strategies_dict: dict) -> bool:
        """
        Rule-based parameter tuning from recent trade outcomes.
        NO LLM calls – runs frequently, zero cost.

        Rules:
          - Win rate < 40%  → tighten entry (higher ADX, tighter slope)
          - Win rate > 70%  → loosen slightly to get more trades
          - Avg PnL < 0 but WR > 50%  → widen TP (RR ratio up)
          - Consecutive losses > 3  → tighten SL multiplier
        """
        trades = self.get_recent_trades(self.rule_adjust_window)
        if len(trades) < 4:
            return False

        perf_by_strat = self._by_strategy(trades)
        adjustments = []

        for pair, strat_list in active_strategies_dict.items():
            for strat in strat_list:
                name = strat.name
                stats = perf_by_strat.get(name, {})
                if stats.get("trades", 0) < 3:
                    continue

                wr      = stats["win_rate"]
                avg_pnl = stats["avg_pnl"]

                # ── EMA+DMI+BOS rules ────────────────────────────────────
                if name == "EMA+DMI+BOS":
                    if wr < 40:
                        old = strat.adx_threshold
                        strat.adx_threshold  = min(strat.adx_threshold + 2, 28)
                        strat.slope_threshold = min(strat.slope_threshold * 1.3, 0.001)
                        adjustments.append(f"{name} ADX {old}→{strat.adx_threshold} (WR={wr}%↓ tighten)")
                    elif wr > 72 and avg_pnl > 0:
                        old = strat.adx_threshold
                        strat.adx_threshold  = max(strat.adx_threshold - 1, 12)
                        adjustments.append(f"{name} ADX {old}→{strat.adx_threshold} (WR={wr}%↑ relax)")
                    if avg_pnl < -2 and wr >= 50:
                        old = strat.rr_ratio
                        strat.rr_ratio = min(strat.rr_ratio + 0.15, 3.5)
                        adjustments.append(f"{name} RR {old:.2f}→{strat.rr_ratio:.2f} (PnL={avg_pnl} widen TP)")
                    if avg_pnl > 5 and wr > 60:
                        old = strat.atr_multiplier_sl
                        strat.atr_multiplier_sl = max(strat.atr_multiplier_sl - 0.1, 1.0)
                        adjustments.append(f"{name} ATR_SL {old:.1f}→{strat.atr_multiplier_sl:.1f} (tighten SL)")

                # ── RandomForest rules ────────────────────────────────────
                elif name == "RandomForest":
                    if wr < 40:
                        old = strat.atr_multiplier_sl
                        strat.atr_multiplier_sl = min(strat.atr_multiplier_sl + 0.1, 2.5)
                        adjustments.append(f"{name} ATR_SL {old:.1f}→{strat.atr_multiplier_sl:.1f} (WR↓ widen SL)")
                    elif wr > 65 and avg_pnl > 0:
                        old = strat.atr_multiplier_tp
                        strat.atr_multiplier_tp = min(strat.atr_multiplier_tp + 0.1, 3.5)
                        adjustments.append(f"{name} ATR_TP {old:.1f}→{strat.atr_multiplier_tp:.1f} (WR↑ extend TP)")
                    if avg_pnl < -3:
                        old = strat.tp_reduction_points
                        strat.tp_reduction_points = max(strat.tp_reduction_points - 0.1, 0.5)
                        adjustments.append(f"{name} TP_red {old:.1f}→{strat.tp_reduction_points:.1f}")

        if adjustments:
            self.upgrade_count += len(adjustments)
            self.param_adjustments.extend(adjustments)
            log_msg = " | ".join(adjustments)
            print(f"🔧 [SelfOptimizer] Rule-based adjust: {log_msg}")
            self._save_adjustment_log("RULE_ADJUST", log_msg)
            return True
        return False

    # ── Layer 2a: Data-Driven Regime Weight Update (FREE) ────────────────────

    def update_regime_weights(self, master_agents_dict: dict):
        """
        Update master agent's per-regime strategy weights from actual trade data.
        Uses exponential moving average to smoothly update weights.
        No LLM – pure data.
        """
        trades = self.get_recent_trades(50)
        if len(trades) < 8:
            return

        regime_trades = {}
        for t in trades:
            r = t.get("regime", "UNKNOWN")
            regime_trades.setdefault(r, []).append(t)

        for regime, rtrades in regime_trades.items():
            if len(rtrades) < 3:
                continue
            strat_perf = self._by_strategy(rtrades)
            if regime not in self.regime_performance:
                self.regime_performance[regime] = {}
            for sname, perf in strat_perf.items():
                wr = perf["win_rate"] / 100.0
                self.regime_performance[regime][sname] = round(wr, 3)

        updates = []
        for pair, master in master_agents_dict.items():
            for regime, strat_wrs in self.regime_performance.items():
                for sname, wr in strat_wrs.items():
                    if sname in master.performance_memory:
                        old = master.performance_memory[sname]
                        # EMA update: 70% keep old, 30% new data
                        new_w = round(0.7 * old + 0.3 * wr, 3)
                        master.performance_memory[sname] = new_w
                        if abs(new_w - old) > 0.05:
                            updates.append(f"{sname}@{regime}: {old:.2f}→{new_w:.2f}")

        if updates:
            print(f"🎯 [SelfOptimizer] Regime weights updated: {', '.join(updates[:4])}")

    # ── Layer 2b: LLM Deep Analysis + Parameter Suggestions ──────────────────

    def llm_suggest_upgrade(self, active_strategies_dict: dict) -> Optional[dict]:
        """
        Use LLM to analyze performance and suggest deeper improvements.
        Rate-limited: max once per hour.
        """
        if not self.llm or not self.llm.available:
            return None
        now = _time.time()
        if now - self.last_llm_optimize < self.llm_optimize_interval:
            remaining = int(self.llm_optimize_interval - (now - self.last_llm_optimize))
            print(f"⏳ [SelfOptimizer] LLM upgrade in {remaining}s")
            return None

        trades = self.get_recent_trades(self.performance_window)
        if len(trades) < self.min_trades_for_llm:
            return None

        self.last_llm_optimize = now

        perf_summary = {
            "overall":     self._compute_perf(trades),
            "by_strategy": self._by_strategy(trades),
            "by_regime":   self._by_regime(trades),
            "recent_adjustments": self.param_adjustments[-5:],
            "total_self_upgrades": self.upgrade_count
        }

        result = self.llm.suggest_parameter_upgrade(perf_summary)
        if result:
            verdict = result.get("verdict", "unknown")
            issue   = result.get("top_issue", "")
            print(f"🤖 [SelfOptimizer] LLM says: {verdict} – {issue}")
        return result

    # ── Apply LLM Suggestions ────────────────────────────────────────────────

    def apply_llm_suggestions(self, suggestions: dict,
                              active_strategies_dict: dict,
                              master_agents_dict: dict):
        """Apply LLM-suggested parameter + regime preference changes."""
        if not suggestions:
            return

        applied = []

        # 1. Parameter changes
        param_changes = suggestions.get("parameter_changes", {})
        for pair, strat_list in active_strategies_dict.items():
            for strat in strat_list:
                changes = param_changes.get(strat.name, {})
                for key, val in changes.items():
                    if hasattr(strat, key):
                        old = getattr(strat, key)
                        setattr(strat, key, val)
                        applied.append(f"{strat.name}.{key}: {old}→{val}")

        # 2. Regime preference → boost that strategy's weight in that regime
        regime_pref = suggestions.get("regime_preference", {})
        for pair, master in master_agents_dict.items():
            for regime, preferred in regime_pref.items():
                if regime in master.regime_weights and preferred in master.regime_weights[regime]:
                    master.regime_weights[regime][preferred] = min(
                        master.regime_weights[regime][preferred] * 1.25, 2.0
                    )
                    applied.append(f"RegimeBoost {preferred}@{regime}")

        if applied:
            self.upgrade_count += len(applied)
            log_msg = " | ".join(applied)
            self.param_adjustments.append(f"LLM[{suggestions.get('verdict','')}]: {log_msg}")
            print(f"✅ [SelfOptimizer] LLM applied: {log_msg}")
            self._save_adjustment_log("LLM_UPGRADE", log_msg)

    # ── LLM Generate New Strategy ─────────────────────────────────────────────

    def llm_generate_new_strategy(self, market_context: dict, trades: list) -> Optional[dict]:
        """Ask LLM to create a brand-new strategy for current market conditions."""
        if not self.llm or not self.llm.available:
            return None
        print("🌟 [SelfOptimizer] Generating new strategy via LLM...")
        return self.llm.generate_strategy_params(market_context, trades)

    # ── Status ───────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        trades = self.get_recent_trades(20)
        perf   = self._compute_perf(trades)
        now    = _time.time()
        next_llm = max(0, int(self.llm_optimize_interval - (now - self.last_llm_optimize)))
        return {
            "total_self_upgrades":    self.upgrade_count,
            "generated_strategies":   len(self.generated_strategies),
            "recent_performance":     perf,
            "regime_performance":     self.regime_performance,
            "recent_adjustments":     self.param_adjustments[-8:],
            "next_llm_analyze_in_sec": next_llm,
            "last_llm_optimize": (
                datetime.datetime.fromtimestamp(self.last_llm_optimize).isoformat()
                if self.last_llm_optimize > 0 else "Never"
            )
        }
