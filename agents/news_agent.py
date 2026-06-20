# agents/news_agent.py
from agents.nlp_sentiment import NLPSentimentAgent

class NewsSentimentAgent:
    """
    News sentiment agent that uses the advanced NLP engine (FinBERT)
    with automatic fallback to keyword analysis if the model fails.
    Provides the same interface (analyze() returning prob_hold etc.)
    as the old keyword-based agent.
    """
    def __init__(self):
        self.nlp_agent = NLPSentimentAgent()

    def analyze(self):
        """
        Returns dict with:
            - prob_hold: float 0-1 (for Master Agent override)
            - sentiment: BUY/SELL/HOLD
            - confidence: 0-1
            - expected_impact_pips: float
            - summary: string
            - headlines: list of analyzed headlines
        """
        result = self.nlp_agent.analyze()
        # Ensure backward‑compatible keys
        return {
            "prob_hold": result.get("prob_hold", 0.0),
            "sentiment": result.get("sentiment", "HOLD"),
            "confidence": result.get("confidence", 0.5),
            "expected_impact_pips": result.get("expected_impact_pips", 0),
            "summary": result.get("summary", ""),
            "headlines": result.get("headlines", [])
        }

    def fetch_headlines(self):
        """Return raw headlines for dashboard or other uses."""
        return self.nlp_agent.fetch_headlines()
