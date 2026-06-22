from agents.nlp_sentiment import NLPSentimentAgent

# 1. Main.py எரர் அடிக்காமல் இருக்க உருவாக்கப்பட்ட டம்மி (Lightweight) ஸ்ட்ரேட்டஜி கிளாஸ்
class RandomForestStrategy:
    """
    உண்மையான Random Forest மாடல் 512MB RAM-ஐ க்ராஷ் செய்யும் என்பதால்,
    இது பெயரளவிற்காக மட்டுமே (Dummy Placeholder) உருவாக்கப்பட்டுள்ளது.
    """
    def __init__(self, *args, **kwargs):
        pass # மெமரியைச் சேமிக்க எதையும் லோட் செய்ய வேண்டாம்

    def train(self, *args, **kwargs):
        pass # ட்ரெய்னிங் செய்ய வேண்டாம்

    def predict(self, *args, **kwargs):
        return "BUY"

    def get_signal(self, market_data=None):
        return "BUY" # டீஃபால்ட் சிக்னல்


# 2. நமது முக்கிய AI மூளை
class AIAgent:
    """
    சந்தை தரவுகளையும் (Market Data) மற்றும் செய்திகளையும் (News) இணைத்து 
    இறுதி முடிவெடுக்கும் மூளை (Master Decision Maker).
    """
    def __init__(self, strategy_agent=None):
        self.nlp_analyzer = NLPSentimentAgent()
        self.strategy_agent = strategy_agent 

    def make_decision(self, market_data=None, news_text=""):
        """
        எந்தச் சூழ்நிலையிலும் ட்ரேடை நிறுத்தாமல், "BUY" அல்லது "SELL" 
        சிக்னலை மட்டும் உறுதியாகக் கொடுக்கும் ஃபங்ஷன்.
        """
        # 1. NLP சிக்னலை எடுப்பது (இது எப்போதுமே BUY அல்லது SELL மட்டுமே கொடுக்கும்)
        nlp_signal = self.nlp_analyzer.analyze_text(news_text)

        # 2. ஸ்ட்ரேட்டஜி சிக்னலை எடுப்பது
        strategy_signal = "BUY" # டீஃபால்ட் சிக்னல்
        
        if self.strategy_agent and hasattr(self.strategy_agent, 'get_signal'):
            try:
                strategy_signal = self.strategy_agent.get_signal(market_data)
                # ஒருவேளை ஸ்ட்ரேட்டஜி ஏஜெண்ட் HOLD அல்லது STOP கொடுத்தால், அதை BUY ஆக மாற்றிவிடுவோம்
                if strategy_signal not in ["BUY", "SELL"]:
                    strategy_signal = "BUY"
            except Exception as e:
                print(f"⚠️ Strategy பிழை: {e}. NLP சிக்னலைப் பயன்படுத்துகிறோம்.")
                strategy_signal = nlp_signal

        # 3. இறுதி முடிவெடுக்கும் லாஜிக் (No Halting)
        if strategy_signal == nlp_signal:
            final_signal = strategy_signal
        else:
            final_signal = strategy_signal 
            
        print(f"🧠 AI Decision -> Strategy: {strategy_signal} | News NLP: {nlp_signal} | FINAL ACTION: {final_signal}")
        
        return final_signal

# லோக்கல் டெஸ்டிங்
if __name__ == "__main__":
    ai = AIAgent()
    final_action = ai.make_decision(market_data={}, news_text="Market crash and huge loss")
    print(f"Test Execution Action: {final_action}")
