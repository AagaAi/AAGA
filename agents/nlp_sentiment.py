# agents/nlp_sentiment.py
import feedparser, json, datetime, time as _time
from typing import Dict, Any


class NLPSentimentAnalyzer:
    """
    நிதிச் செய்திகளைப் படித்து சென்டிமென்ட்டைப் பகுப்பாய்வு செய்யும் லேசான ஏஜெண்ட்.
    இது எப்பொழுதும் "BUY" அல்லது "SELL" என்ற முடிவை மட்டுமே கொடுக்கும்.
    """
    def __init__(self):
        # பாசிட்டிவ் (Bullish) வார்த்தைகள்
        self.positive_keywords = ['growth', 'bullish', 'high', 'profit', 'up', 'soar', 'surge', 'buy', 'positive', 'gain', 'rally']
        # நெகட்டிவ் (Bearish) வார்த்தைகள்
        self.negative_keywords = ['crash', 'bearish', 'low', 'loss', 'down', 'drop', 'fall', 'sell', 'negative', 'decline', 'plunge']

    def analyze_text(self, text):
        """
        செய்திகளைப் படித்து இறுதி சிக்னலை உருவாக்குகிறது.
        """
        # செய்தி கிடைக்கவில்லை என்றால், ட்ரேடை நிறுத்தாமல் இருக்க டீஃபால்ட்டாக BUY கொடுக்கிறோம்
        if not text:
            return "BUY"

        text = str(text).lower()
        
        # வார்த்தைகளை எண்ணுதல்
        pos_score = sum(len(re.findall(r'\b' + word + r'\b', text)) for word in self.positive_keywords)
        neg_score = sum(len(re.findall(r'\b' + word + r'\b', text)) for word in self.negative_keywords)

        # பாசிட்டிவ் வார்த்தைகள் அதிகமாகவோ அல்லது சமமாகவோ இருந்தால் BUY
        if pos_score >= neg_score:
            return "BUY"
        # நெகட்டிவ் வார்த்தைகள் அதிகமாக இருந்தால் மட்டும் SELL
        else:
            return "SELL"

# லோக்கல் டெஸ்டிங்
if __name__ == "__main__":
    analyzer = NLPSentimentAnalyzer()
    sample_news = "Market is going up with high growth and massive profit"
    print(f"Sample NLP Decision: {analyzer.analyze_text(sample_news)}")

