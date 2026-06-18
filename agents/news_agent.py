# agents/news_agent.py
import feedparser, json
from agents.gemini_agent import GeminiAnalyst

class NewsSentimentAgent:
    def __init__(self, gemini_analyst):
        self.gemini = gemini_analyst
        self.feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC%3DF&region=US&lang=en-US",
            "https://www.investing.com/rss/news_25.rss"
        ]

    def fetch_headlines(self, limit=5):
        headlines = []
        for url in self.feeds:
            try:
                parsed = feedparser.parse(url)
                for entry in parsed.entries[:limit]:
                    title = entry.get("title", "").strip()
                    if title and title not in headlines:
                        headlines.append(title)
            except:
                continue
        return headlines[:10]   # max 10 headlines

    def analyze(self):
        """
        Returns dict with:
            sentiment: "BUY"/"SELL"/"HOLD"
            confidence: 0-1
            expected_impact_pips: float
            summary: string
            prob_hold: 0-1  (for master agent override)
        """
        headlines = self.fetch_headlines()
        if not headlines:
            return {"sentiment": "HOLD", "confidence": 0.0, "prob_hold": 0.0, "summary": "No news"}

        # Fallback: keyword check if Gemini unavailable
        if not self.gemini.available:
            return self._keyword_analysis(headlines)

        # Ask Gemini
        prompt = f"""
You are a financial news analyst for XAUUSD. Analyze these headlines and return ONLY a JSON object with:
- "sentiment": "BUY", "SELL", or "HOLD"
- "confidence": float 0-1
- "expected_impact_pips": float (positive for BUY, negative for SELL, 0 for HOLD)
- "summary": one short sentence

Headlines:
{json.dumps(headlines)}
"""
        response = self.gemini._safe_generate(prompt)
        if not response:
            return self._keyword_analysis(headlines)

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            result = json.loads(response)
            # Convert to prob_hold for master agent
            prob_hold = 1.0 - result.get("confidence", 0.0)
            if result.get("sentiment") == "HOLD":
                prob_hold = 1.0
            result["prob_hold"] = prob_hold
            return result
        except:
            return self._keyword_analysis(headlines)

    def _keyword_analysis(self, headlines):
        # Old keyword method as fallback
        high_keywords = ["fomc","rate","cpi","nfp","gdp","war","recession"]
        med_keywords  = ["unemployment","retail","gold","dollar"]
        high_count = sum(1 for h in headlines for kw in high_keywords if kw in h.lower())
        med_count  = sum(1 for h in headlines for kw in med_keywords if kw in h.lower())
        if high_count > 0:
            prob_hold = 0.9
        elif med_count >= 3:
            prob_hold = 0.4
        else:
            prob_hold = 0.0
        return {"sentiment": "HOLD", "confidence": 0.0, "prob_hold": prob_hold, "summary": "Keyword analysis"}