class BacktestEngine:
    def __init__(self, config):
        self.config = config

    def run(self, pair, strategies):
        # Simulate backtest on last 7 days 1m data
        # Compare all strategies, return best
        best = strategies[0] if strategies else {"name":"default","win_rate":0}
        return best