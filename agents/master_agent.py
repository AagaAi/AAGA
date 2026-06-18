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
        # Performance memory: strategy name -> recent win rate (0.5 default)
        self.performance_memory = {s.name: 0.5 for s in strategies}

    def decide(self, signals):
        """
        signals: list of dicts, each containing:
            - 'strategy' (name)
            - 'signal' (BUY/SELL/HOLD)
            - 'entry_zone' (optional)
            - 'sl', 'tp' (optional)
        Returns final decision dict with:
            signal, entry_zone, sl, tp, confidence, reason, strategy_used
        """
        # 1. Check News Risk
        news_data = self.news_analyzer()
        if news_data.get('prob_hold', 0) > 0.8:   # High risk news → HOLD
            return {"signal": "HOLD", "confidence": 0.0,
                    "reason": "High news risk", "strategy_used": "Master"}

        # 2. Check Risk Manager (capital protection)
        risk_data = self.risk_evaluator()
        if risk_data.get('halt', False):
            return {"signal": "HOLD", "confidence": 0.0,
                    "reason": "Daily loss limit hit", "strategy_used": "Master"}

        # 3. Weighted voting based on recent win rates
        votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        total_weight = 0.0
        for sig in signals:
            name = sig.get("strategy")
            action = sig.get("signal", "HOLD")
            weight = self.performance_memory.get(name, 0.5)
            if action in votes:
                votes[action] += weight
            total_weight += weight

        if total_weight == 0:
            return {"signal": "HOLD", "confidence": 0.0,
                    "reason": "No valid signals", "strategy_used": "Master"}

        best_action = max(votes, key=votes.get)
        confidence = votes[best_action] / total_weight

        # 4. Choose best SL/TP from a strategy that voted for the winning action
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
            "reason": "Weighted agent vote",
            "strategy_used": strategy_used
        }

    def update_performance(self, strategy_name, win_rate):
        """Call after daily stats are computed."""
        self.performance_memory[strategy_name] = round(win_rate, 2)