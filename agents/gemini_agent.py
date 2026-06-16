import os, json
import google.generativeai as genai

class GeminiAnalyst:
    def __init__(self, api_key=None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if key:
            genai.configure(api_key=key)
            # Use stable model "gemini-pro"
            self.model = genai.GenerativeModel("gemini-pro")
            self.available = True
        else:
            self.available = False

    def analyze(self, tech, news, mtf, daily_stats):
        if not self.available:
            return {"summary": "Gemini API key not set", "recommended_action": "HOLD", "confidence": 0.5}
        try:
            prompt = f"""
You are a senior trading analyst. Analyze this XAUUSD data and provide a short recommendation.

Tech Signal: {tech.get('signal', 'None')}
Price: {tech['current_price']}
Entry Zone: {tech.get('entry_zone')}
Stop Loss: {tech.get('stop_loss')}
Take Profit: {tech.get('take_profit')}
News Risk Level: {news}
MTF Bias: {mtf.get('htf_bias')}, Confluence Score: {mtf.get('confluence_score')}
Daily P&L stats: {daily_stats}

Return ONLY a JSON object (no other text) with keys:
"summary" (short string),
"recommended_action" (BUY/SELL/HOLD),
"confidence" (float 0-1),
"reasoning" (string).
"""
            resp = self.model.generate_content(prompt)
            # Extract JSON
            text = resp.text
            # Remove any markdown fences
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            parsed = json.loads(text)
            return parsed
        except Exception as e:
            return {"summary": f"Gemini error: {str(e)}", "recommended_action": "HOLD", "confidence": 0.5}