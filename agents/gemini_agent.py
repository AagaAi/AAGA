import os
import google.generativeai as genai

class GeminiAnalyst:
    def __init__(self, api_key=None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if key:
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            self.available = True
        else:
            self.available = False

    def analyze(self, tech, news, mtf, daily_stats):
        if not self.available:
            return {"summary": "Gemini API key not set"}
        try:
            prompt = f"""You are a senior trading analyst. Analyze this market data and provide a short recommendation with reasoning.

Market Data:
- Pair: XAUUSD
- Price: {tech['current_price']}
- Tech Signal: {tech.get('signal')}
- News Risk: {news}
- MTF Bias: {mtf.get('htf_bias')}, Confluence: {mtf.get('confluence_score')}
- Daily P&L: {daily_stats}

Return a JSON object with keys: "summary", "recommended_action" (BUY/SELL/HOLD), "confidence" (0-1), "reasoning"."""
            resp = self.model.generate_content(prompt)
            # Try to parse JSON from response
            import json
            return json.loads(resp.text.replace("```json","").replace("```",""))
        except Exception as e:
            return {"summary": f"Gemini error: {e}"}