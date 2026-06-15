import google.generativeai as genai

class GeminiAnalyst:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    def analyze(self, tech, news, mtf, daily_stats):
        prompt = f"""
        Market: XAUUSD, Price: {tech['current_price']}
        Daily P&L: {daily_stats}
        News: {news}
        MTF: {mtf}
        Why did previous trades lose? Suggest adjustments for SL/TP, entry logic.
        Return JSON: {{"summary":"...", "recommended_action":"BUY/SELL/HOLD", "confidence":0.8, "new_strategy_params":{{}}}}"""
        try:
            resp = self.model.generate_content(prompt)
            return eval(resp.text)  # careful, use json.loads in prod
        except:
            return {"summary": "Gemini unavailable"}