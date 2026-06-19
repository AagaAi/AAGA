# agents/master_agent.py
import numpy as np

class MasterAgent:
    def __init__(self, strategies, risk_evaluator, news_analyzer):
        """
        strategies: list of strategy instances (each must have .name)
        risk_evaluator: function that returns dict with 'halt' key (True/False)
        news_analyzer: function that returns dict with 'prob_hold' key
        """
        self.strategies = strategies
        self.risk_evaluator = risk_evaluator
        self.news_analyzer = news_analyzer
        # Performance memory: strategy name -> recent win rate (default 0.5)
        self.performance_memory = {s.name: 0.5 for s in strategies}
        # Regime‑specific weight multipliers (adjust based on backtest)
        self.regime_weights = {
            "UPTREND":        {"EMA+DMI+BOS": 1.0, "RandomForest": 0.8, "PPO": 1.2},
            "DOWNTREND":      {"EMA+DMI+BOS": 1.2, "RandomForest": 0.7, "PPO": 1.0},
            "SIDEWAYS":       {"EMA+DMI+BOS": 0.6, "RandomForest": 1.0, "PPO": 0.8},
            "HIGH_VOLATILITY": {"EMA+DMI+BOS": 0.5, "RandomForest": 0.5, "PPO": 0.4},
            "UNKNOWN":        {"EMA+DMI+BOS": 1.0, "RandomForest": 1.0, "PPO": 1.0}
        }

    def decide(self, signals, regime=None):
        """
        signals: list of dicts, each containing:
            - 'strategy' (name)
            - 'signal' (BUY/SELL/HOLD)
            - 'entry_zone' (optional)
            - 'sl', 'tp' (optional)
        regime: optional market regime string (UPTREND, DOWNTREND, SIDEWAYS, HIGH_VOLATILITY)
        Returns final decision dict with:
            signal, entry_zone, sl, tp, confidence, reason, strategy_used
        """
        # 1. News Risk Override
        news_data = self.news_analyzer()
        if news_data.get('prob_hold', 0) > 0.8:
            return {"signal": "HOLD", "confidence": 0.0,
                    "reason": "High news risk", "strategy_used": "Master"}

        # 2. Risk Manager Override (capital protection)
        risk_data = self.risk_evaluator()
        if risk_data.get('halt', False):
            return {"signal": "HOLD", "confidence": 0.0,
                    "reason": "Daily loss limit hit", "strategy_used": "Master"}

        # 3. Weighted voting – base weight from performance memory,
        #    then multiplied by regime-specific factor if regime is given.
        regime_mult = self.regime_weights.get(regime, self.regime_weights["UNKNOWN"])
        votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        total_weight = 0.0

        for sig in signals:
            name = sig.get("strategy")
            action = sig.get("signal", "HOLD")
            base_weight = self.performance_memory.get(name, 0.5)
            regime_factor = regime_mult.get(name, 1.0)
            weight = base_weight * regime_factor
            if action in votes:
                votes[action] += weight
            total_weight += weight

        if total_weight == 0:
            return {"signal": "HOLD", "confidence": 0.0,
                    "reason": "No valid signals", "strategy_used": "Master"}

        best_action = max(votes, key=votes.get)
        confidence = votes[best_action] / total_weight

        # 4. Select best SL/TP from a strategy that voted for the winning action
        best_sig = None
        for sig in signals:
            if sig.get("signal") == best_action and sig.get("entry_zone"):
                best_sig = sig
                break

        entry_zone = best_sig.get("entry_zone") if best_sig else None
        sl = best_sig.get("sl") if best_sig else None
        tp = best_sig.get("tp") if best_sig else None
        strategy_used = best_sig.get("strategy") if best_sig else "Master"

        return {
            "signal": best_action,
            "entry_zone": entry_zone,
            "sl": sl,
            "tp": tp,
            "confidence": round(confidence, 2),
            "reason": f"Weighted vote (regime={regime})" if regime else "Weighted agent vote",
            "strategy_used": strategy_used
        }

    def update_performance(self, strategy_name, win_rate):
        """Call after daily stats are computed to update performance memory."""
        self.performance_memory[strategy_name] = round(win_rate, 2)
