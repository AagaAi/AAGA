# agents/nlp_sentiment.py
import feedparser, json, datetime, time as _time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, Dict, Any

class NLPSentimentAgent:
    """
    Advanced sentiment analysis using FinBERT model.
    Falls back to keyword analysis if model fails to load.
    """
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.available = False
        self.pipeline = None
        self.feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC%3DF&region=US&lang=en-US",
            "https://www.investing.com/rss/news_25.rss"
        ]
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
            self.available = True
            print("✅ FinBERT sentiment model loaded")
        except Exception as e:
            print(f"⚠️ FinBERT loading failed: {e}. Using keyword fallback.")

    def fetch_headlines(self, limit: int = 5) -> list:
        headlines = []
        for url in self.feeds:
            try:
                parsed = feedparser.parse(url)
                for entry in parsed.entries[:limit]:
                    title = entry.get("title", "").strip()
                    if title and title not in [h['title'] for h in headlines]:
                        headlines.append({
                            "title": title,
                            "published": entry.get("published", "")
                        })
            except:
                continue
        return headlines[:10]

    def analyze(self) -> Dict[str, Any]:
        """
        Returns dict with:
          - sentiment: "BUY"/"SELL"/"HOLD"
          - confidence: 0-1
          - prob_hold: 0-1   (for Master Agent override)
          - expected_impact_pips: float
          - summary: string
          - headlines: list of analyzed headlines
        """
        headlines = self.fetch_headlines()
        if not headlines:
            return {
                "sentiment": "HOLD", "confidence": 0.0, "prob_hold": 0.0,
                "expected_impact_pips": 0, "summary": "No news", "headlines": []
            }

        if self.available and self.pipeline:
            # Analyze each headline with FinBERT
            analyzed = []
            bullish = bearish = neutral = 0
            total_conf = 0
            for h in headlines:
                result = self.pipeline(h["title"])[0]
                label = result["label"]
                score = result["score"]
                if label == "positive":
                    bullish += 1
                    total_conf += score
                elif label == "negative":
                    bearish += 1
                    total_conf += score
                else:
                    neutral += 1
                analyzed.append({
                    "title": h["title"],
                    "sentiment": label,
                    "confidence": round(score, 2)
                })
            # Determine overall sentiment
            n = len(analyzed)
            if bullish > bearish and bullish > neutral:
                sentiment = "BUY"
                prob_hold = 0.1
                impact = bullish * 50   # pips estimate
            elif bearish > bullish and bearish > neutral:
                sentiment = "SELL"
                prob_hold = 0.1
                impact = -bearish * 50
            else:
                sentiment = "HOLD"
                prob_hold = 0.5
                impact = 0
            confidence = round(total_conf / n, 2) if n > 0 else 0.5
            summary = f"NLP sentiment: {bullish} bullish, {bearish} bearish, {neutral} neutral."
        else:
            # Fallback to keyword analysis
            high_keywords = ["fomc","rate","cpi","nfp","gdp","war","recession","trump"]
            med_keywords  = ["unemployment","retail","gold","dollar","treasury","yield"]
            high_count = sum(1 for h in headlines for kw in high_keywords if kw in h['title'].lower())
            med_count  = sum(1 for h in headlines for kw in med_keywords if kw in h['title'].lower())
            if high_count > 0:
                prob_hold = 0.9
                sentiment = "HOLD"
                impact = 100
            elif med_count >= 3:
                prob_hold = 0.4
                sentiment = "HOLD"
                impact = 30
            else:
                prob_hold = 0.0
                sentiment = "HOLD"
                impact = 0
            confidence = 1.0 if prob_hold > 0 else 0.5
            summary = f"Keyword analysis: high={high_count}, med={med_count}"
            analyzed = headlines  # no sentiment per headline

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "prob_hold": prob_hold,
            "expected_impact_pips": impact,
            "summary": summary,
            "headlines": analyzed
        }
