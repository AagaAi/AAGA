import urllib.request
import xml.etree.ElementTree as ET
from agents.nlp_sentiment import NLPSentimentAgent

class NewsAgent:
    """
    இணையத்திலிருந்து செய்திகளை எடுத்து NLP-க்கு அனுப்பும் ஏஜெண்ட்
    """
    def __init__(self):
        self.nlp_agent = NLPSentimentAgent()
        self.rss_url = "https://finance.yahoo.com/news/rssindex"
        
    def fetch_latest_news(self):
        try:
            req = urllib.request.Request(self.rss_url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=5)
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            news_items = []
            for item in root.findall('.//item')[:10]:
                title = item.find('title')
                if title is not None:
                    news_items.append(title.text)
            return " ".join(news_items).lower()
        except Exception as e:
            print(f"⚠️ செய்தி எடுப்பதில் பிழை: {e}")
            return ""

    def analyze(self):
        """
        Main.py இந்த analyze() ஃபங்ஷனைத்தான் அழைக்கிறது.
        இது Dictionary டேட்டாவை ரிட்டர்ன் செய்யும்.
        """
        news_text = self.fetch_latest_news()
        # NLP ஏஜெண்ட் மூலம் அகராதி (dict) வடிவத்தில் சிக்னலை அனுப்புகிறோம்
        return self.nlp_agent.analyze(news_text)

# (உங்கள் பிராஜெக்ட்டில் NewsSentimentAgent என்று பெயர் இருந்தால், 
# அதற்காக ஒரு டம்மி கிளாஸ்)
class NewsSentimentAgent(NewsAgent):
    pass
