# agents/news_agent.py
import feedparser, json, datetime, time as _time

class NewsSentimentAgent:
    def __init__(self, gemini_analyst=None):
        self.feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC%3DF&region=US&lang=en-US",
            "https://www.investing.com/rss/news_25.rss"
        ]
        self.high_keywords = ["fomc","federal reserve","rate decision","rate hike","rate cut",
                              "cpi","inflation","nfp","non-farm","gdp","recession","war","conflict",
                              "geopolitical","sanctions","debt ceiling","government shutdown",
                              "credit rating","downgrade","surprise","emergency","shock","trump"]
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
                    if not title:
                        continue
                    published = entry.get("published", entry.get("updated", ""))
                    headlines.append({"title": title, "published": published})
            except:
                continue
        return headlines[:10]

    def analyze(self):
        headlines = self.fetch_headlines()
        if not headlines:
            return {
                "sentiment": "HOLD", "confidence": 0.0, "prob_hold": 0.0,
                "expected_impact_pips": 0, "summary": "No news",
                "headlines": []
            }

        now = datetime.datetime.now(datetime.timezone.utc)
        high_count = 0
        recent_high = False
        recent_high_summary = ""
        detailed_headlines = []

        for h in headlines:
            title = h["title"]
            low_title = title.lower()
            high_hits = [kw for kw in self.high_keywords if kw in low_title]
            med_hits  = [kw for kw in self.med_keywords  if kw in low_title]

            # Determine time since published
            published_str = h.get("published", "")
            try:
                # Parse typical RSS dates
                pub_time = datetime.datetime.strptime(published_str, "%a, %d %b %Y %H:%M:%S %z")
            except:
                pub_time = None

            minutes_ago = None
            if pub_time:
                delta = now - pub_time
                minutes_ago = delta.total_seconds() / 60

            # Categorize
            impact = "low"
            if high_hits:
                impact = "high"
                high_count += 1
                # Check if within 15 minutes
                if minutes_ago is not None and minutes_ago <= 15:
                    recent_high = True
                    recent_high_summary = f"High-impact news ({', '.join(high_hits)}) published {int(minutes_ago)}min ago"
            elif med_hits:
                impact = "medium"

            detailed_headlines.append({
                "title": title,
                "impact": impact,
                "minutes_ago": round(minutes_ago, 1) if minutes_ago else "unknown",
                "matched_keywords": high_hits + med_hits
            })

        # Determine prob_hold: only if recent high-impact news
        prob_hold = 0.0
        summary = "No major news events. Market technically driven."
        if recent_high:
            prob_hold = 0.9
            summary = recent_high_summary + ". Exercise extreme caution – trading halted."
        elif high_count > 0:
            # High-impact but older than 15 min → moderate caution, trade allowed
            prob_hold = 0.3
            summary = f"High-impact news detected but older than 15 min ({high_count} triggers). Trade with reduced size."
        else:
            med_count = sum(1 for h in detailed_headlines if h["impact"] == "medium")
            if med_count >= 3:
                prob_hold = 0.2
                summary = f"Moderate news activity ({med_count} triggers). Some caution advised."
            else:
                prob_hold = 0.0

        # Estimate pips impact (simple)
        expected_impact_pips = 0
        if recent_high:
            expected_impact_pips = 100  # high impact
        elif high_count > 0:
            expected_impact_pips = 40
        elif med_count >= 3:
            expected_impact_pips = 15

        return {
            "sentiment": "HOLD",
            "confidence": 1.0 if prob_hold > 0.5 else 0.5,
            "prob_hold": prob_hold,
            "expected_impact_pips": expected_impact_pips,
            "summary": summary,
            "headlines": detailed_headlines
        }