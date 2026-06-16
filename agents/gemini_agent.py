# agents/gemini_agent.py
import os, json
import google.generativeai as genai

class GeminiAnalyst:
    def __init__(self, api_key=None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        self.available = False
        self.model = None
        if key:
            try:
                genai.configure(api_key=key)
                # Try to get available models and pick a text generation one
                models = genai.list_models()
                text_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
                if not text_models:
                    # Use a known fallback if list fails
                    self.model = genai.GenerativeModel("models/gemini-1.5-flash")
                else:
                    # Prefer flash for speed, else first available
                    preferred = [m for m in text_models if 'flash' in m]
                    if preferred:
                        self.model = genai.GenerativeModel(preferred[0])
                    else:
                        self.model = genai.GenerativeModel(text_models[0])
                self.available = True
            except Exception as e:
                self.error = str(e)
                self.available = False
        else:
            self.error = "No API key"

    def analyze(self, tech, news, mtf, daily_stats):
        if not self.available:
            return {
                "summary": f"Gemini unavailable: {getattr(self,'error','API key missing')}",
                "recommended_action": "HOLD",
                "confidence": 0.5
            }
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
            text = resp.text
            # Extract JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text)
        except Exception as e:
            return {
                "summary": f"Gemini error: {str(e)}",
                "recommended_action": "HOLD",
                "confidence": 0.5
            }