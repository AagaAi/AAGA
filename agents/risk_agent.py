class RiskManager:
    def __init__(self, config):
        self.config = config
    def evaluate(self):
        # dummy: if daily loss <5% then give neutral
        return {"prob_buy":0.33, "prob_sell":0.33, "prob_hold":0.34}