import re

class NLPSentimentAgent:
    """
    செய்திகளைப் படித்து சென்டிமென்ட்டைப் பகுப்பாய்வு செய்யும் லேசான ஏஜெண்ட்.
    Dashboard மற்றும் main.py எதிர்பார்க்கும் Dictionary பார்மட்டில் இயங்கும்.
    """
    def __init__(self):
        self.positive_keywords = ['growth', 'bullish', 'high', 'profit', 'up', 'soar', 'surge', 'buy', 'positive']
        self.negative_keywords = ['crash', 'bearish', 'low', 'loss', 'down', 'drop', 'fall', 'sell', 'negative']

    def analyze(self, text=""):
        """
        Main.py எதிர்பார்க்கும் 'analyze' பங்க்ஷன் (Dictionary-ஐ அனுப்பும்)
        """
        # செய்தி வரவில்லை என்றால் டீஃபால்ட் சிக்னல்
        if not text:
            return {"signal": "BUY", "score": 1.0, "prob_hold": 0.0}

        text = str(text).lower()
        pos_score = sum(len(re.findall(r'\b' + word + r'\b', text)) for word in self.positive_keywords)
        neg_score = sum(len(re.findall(r'\b' + word + r'\b', text)) for word in self.negative_keywords)

        if pos_score >= neg_score:
            return {"signal": "BUY", "score": float(pos_score), "prob_hold": 0.0}
        else:
            return {"signal": "SELL", "score": float(neg_score), "prob_hold": 0.0}

    def analyze_text(self, text=""):
        """பழைய ஏஜெண்டுகள் கேட்டால் ஸ்ட்ரிங் பார்மட்டில் அனுப்பும்"""
        return self.analyze(text)["signal"]

