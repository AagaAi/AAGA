from agents.nlp_sentiment import NLPSentimentAgent

# 1. டம்மி (Lightweight) ஸ்ட்ரேட்டஜி கிளாஸ் (Dictionary Output)
class RandomForestStrategy:
    def __init__(self, *args, **kwargs):
        self.name = "RandomForestStrategy" 

    def train(self, *args, **kwargs):
        pass 

    def predict(self, *args, **kwargs):
        return {"signal": "BUY", "confidence": 1.0}

    def get_signal(self, market_data=None):
        # லூப் எதிர்பார்க்கும் Dictionary பார்மட்
        return {"signal": "BUY", "confidence": 1.0}


# 2. நமது முக்கிய AI மூளை
class AIAgent:
    def __init__(self, strategy_agent=None):
        self.nlp_analyzer = NLPSentimentAgent()
        self.strategy_agent = strategy_agent 

    def make_decision(self, market_data=None, news_text=""):
        # 1. NLP சிக்னலை எடுப்பது
        nlp_result = self.nlp_analyzer.analyze(news_text)
        # Dictionary ஆக வந்தால் அதில் உள்ளதை எடுக்கும், இல்லையென்றால் வார்த்தையை எடுக்கும்
        nlp_signal = nlp_result.get("signal", "BUY") if isinstance(nlp_result, dict) else nlp_result

        # 2. ஸ்ட்ரேட்டஜி சிக்னலை எடுப்பது
        strategy_signal = "BUY"
        if self.strategy_agent and hasattr(self.strategy_agent, 'get_signal'):
            try:
                strat_result = self.strategy_agent.get_signal(market_data)
                strategy_signal = strat_result.get("signal", "BUY") if isinstance(strat_result, dict) else strat_result
            except Exception as e:
                print(f"⚠️ Strategy பிழை: {e}")

        # தப்பித்தவறி வேறு வார்த்தைகள் வந்தால் BUY ஆக மாற்றிவிடும்
        if strategy_signal not in ["BUY", "SELL"]:
            strategy_signal = "BUY"
        if nlp_signal not in ["BUY", "SELL"]:
            nlp_signal = "BUY"

        # 3. இறுதி முடிவெடுக்கும் லாஜிக் 
        if strategy_signal == nlp_signal:
            final_signal = strategy_signal
        else:
            final_signal = strategy_signal 
            
        print(f"🧠 AI Decision -> Action: {final_signal}")
        
        # முக்கிய மாற்றம்: '.get()' எரரைத் தடுக்க DICTIONARY ஆக அனுப்புகிறோம்
        return {
            "signal": final_signal,
            "action": final_signal,
            "confidence": 1.0
        }

if __name__ == "__main__":
    ai = AIAgent()
    final_action = ai.make_decision(market_data={}, news_text="Market crash")
    print(f"Test Execution Action: {final_action}")

