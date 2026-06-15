import json, os

class StrategySelector:
    def __init__(self, config):
        self.config = config
        self.strategies = self._load_strategies()

    def _load_strategies(self):
        pool = self.config.get("strategy_pool", [])
        # each strategy: {"name":"...", "prob_buy":..., "prob_sell":..., "prob_hold":..., "calc_sltp":...}
        return pool

    def select(self, pair, mtf_data):
        # Simple: choose the strategy with highest historical win rate (stored in meta)
        best = None
        best_wr = 0
        for s in self.strategies:
            wr = s.get("win_rate", 0)
            if wr > best_wr:
                best_wr = wr
                best = s
        if best:
            return best
        # fallback
        return {"name":"default", "prob_buy":0.5, "prob_sell":0.5, "prob_hold":0.0,
                "calculate_sltp": lambda dir, price: {"sl":price-2, "tp":price+4} if dir=="BUY" else {"sl":price+2, "tp":price-4}}

    def get_all(self):
        return self.strategies

    def promote(self, strategy):
        self.strategies.append(strategy)
        self.config["strategy_pool"] = self.strategies
        with open("config.json", "w") as f:
            json.dump(self.config, f, indent=2)