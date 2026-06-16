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
                models = genai.list_models()
                text_models = [m.name for m in models
                               if 'generateContent' in m.supported_generation_methods]
                if text_models:
                    preferred = [m for m in text_models if 'flash' in m]
                    self.model = genai.GenerativeModel(preferred[0] if preferred else text_models[0])
                    self.available = True
                else:
                    self.error = "No text model found"
            except Exception as e:
                self.error = str(e)
        else:
            self.error = "No API key"

    def analyze(self, tech, news, mtf, daily_stats):
        if not self.available:
            return {
                "summary": f"Gemini unavailable: {getattr(self,'error','')}",
                "recommended_action": "HOLD",
                "confidence": 0.5
            }
        try:
            # Build prompt with fallback text for missing values
            entry_text = f"{tech.get('entry_zone')}" if tech.get('entry_zone') else "No entry zone (market order)"
            sl_text = f"{tech.get('stop_loss')}" if tech.get('stop_loss') else "No SL suggested"
            tp_text = f"{tech.get('take_profit')}" if tech.get('take_profit') else "No TP suggested"

            prompt = f"""
You are a professional XAUUSD trading analyst.
- Current Price: {tech['current_price']}
- Technical Signal: {tech.get('signal', 'NONE')}
- Entry Zone: {entry_text}
- Stop Loss: {sl_text}
- Take Profit: {tp_text}
- News Risk: {news}
- Multi-Timeframe Bias: {mtf.get('htf_bias')}
- Confluence Score: {mtf.get('confluence_score')}
- Daily P&L: {daily_stats}

Provide a **market summary** and a **final recommendation** (BUY/SELL/HOLD).
Even if entry levels are missing, give your best judgement based on trend and risk.
Return ONLY a JSON object (no code fences) with keys:
"summary", "recommended_action", "confidence" (0-1), "reasoning".
"""
            resp = self.model.generate_content(prompt)
            text = resp.text
            # Clean JSON
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