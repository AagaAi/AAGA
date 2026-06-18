# agents/gemini_agent.py
import os, json, time as _time
import google.generativeai as genai
from typing import Optional

class GeminiAnalyst:
    def __init__(self, api_key: str = ""):
        self.available = False
        self.last_call = 0
        self.call_interval = 60   # seconds between calls (quota saver)
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            self.error = "GEMINI_API_KEY not set"
            return
        try:
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
            self.available = True
        except Exception as e:
            self.error = str(e)

    def _safe_generate(self, prompt: str) -> Optional[str]:
        now = _time.time()
        if now - self.last_call < self.call_interval:
            print("⏳ Gemini throttled (quota saver)")
            return None
        self.last_call = now
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None

    def analyze_daily_trades(self, trades: list) -> dict:
        if not self.available or not trades:
            return {"summary": "Gemini unavailable or no trades", "recommendations": []}
        prompt = f"""
You are a senior quantitative analyst. Below are today's completed XAUUSD trades:
{json.dumps(trades, default=str)}

Analyze them. Identify:
1. Main patterns leading to wins/losses.
2. Suggestions for parameter adjustments (SL buffer, TP multiplier, min FVG gap, sweep type preference).
Return ONLY a JSON object with keys: "summary", "patterns", "parameter_suggestions" (dict of param: value), "confidence" (0-1).
"""
        text = self._safe_generate(prompt)
        if not text:
            return {"summary": "Gemini call failed", "recommendations": []}
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text)
        except:
            return {"summary": text, "recommendations": []}
