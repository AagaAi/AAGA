import datetime

class DailyRiskTracker:
    def __init__(self, config):
        self.balance = config["capital"]
        self.daily_pnl = 0
        self.loss_limit = config["capital"] * config["parameters"]["daily_loss_limit_pct"]
        self.last_reset = datetime.date.today()

    def update(self, pnl):
        self.daily_pnl += pnl
        if -self.daily_pnl >= self.loss_limit:
            return "HALT"
        return "OK"

    def check_reset(self):
        if datetime.date.today() != self.last_reset:
            self.daily_pnl = 0
            self.last_reset = datetime.date.today()

    def stats(self):
        return {"daily_pnl": self.daily_pnl, "balance": self.balance}