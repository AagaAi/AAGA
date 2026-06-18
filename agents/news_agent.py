# agents/news_agent.py
import feedparser, json

class NewsSentimentAgent:
    """
    News sentiment using keyword analysis only.
    Fast, no API calls, no quota issues.
    Gemini is reserved for nightly deep analysis (if available).
    """
    def __init__(self, gemini_analyst=None):   # gemini not used, kept for compatibility
        self.feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC%3DF&region=US&lang=en-US",
            "https://www.investing.com/rss/news_25.rss"
        ]
        self.high_keywords = ["fomc","federal reserve","rate decision","rate hike","rate cut",
                              "cpi","inflation","nfp","non-farm","gdp","recession","war","conflict",
                              "geopolitical","sanctions","debt ceiling","government shutdown",
                              "credit rating","downgrade","surprise","emergency","shock"]
        self.med_keywords  = ["unemployment","jobless claims","ism","pmi","retail sales",
                              "trade balance","treasury","yield","bond","dollar","usd","dxy",
                              "gold","xau","precious metals","commodity","earnings","oecd","imf"]

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
        return headlines[:10]

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
            return {"sentiment": "HOLD", "confidence": 0.0, "prob_hold": 0.0, 
                    "expected_impact_pips": 0, "summary": "No news headlines"}

        high_count = sum(1 for h in headlines for kw in self.high_keywords if kw in h.lower())
        med_count  = sum(1 for h in headlines for kw in self.med_keywords if kw in h.lower())

        if high_count > 0:
            prob_hold = 0.9
            summary = f"High-impact news detected ({high_count} triggers). Exercise caution."
        elif med_count >= 3:
            prob_hold = 0.4
            summary = f"Moderate news activity ({med_count} triggers). Trade with reduced size."
        else:
            prob_hold = 0.0
            summary = "No major news events. Market technically driven."

        return {
            "sentiment": "HOLD",
            "confidence": 1.0 if prob_hold > 0 else 0.5,
            "expected_impact_pips": 0,
            "summary": summary,
            "prob_hold": prob_hold
        }