# agents/gemini_agent.py
import os, json, time as _time
import google.generativeai as genai
from typing import Optional

class GeminiAnalyst:
    def __init__(self, api_key: str = ""):
        self.available = False
        self.consecutive_errors = 0
        self.max_errors = 3
        self.disabled_until = 0
        self.last_call = 0
        self.call_interval = 5
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            self.error = "GEMINI_API_KEY not set"
            return
        try:
            genai.configure(api_key=key)
            # Use full model path for v1beta (works with google-generativeai)
            self.model = genai.GenerativeModel("models/gemini-2.0-flash")
            self.available = True
            print("✅ Gemini 2.0 Flash connected")
        except Exception as e:
            self.error = str(e)

    def _should_throttle(self) -> bool:
        now = _time.time()
        if self.disabled_until > now:
            return True
        if now - self.last_call < self.call_interval:
            return True
        return False

    def _safe_generate(self, prompt: str) -> Optional[str]:
        if not self.available:
            return None
        if self._should_throttle():
            return None
        self.last_call = _time.time()
        try:
            response = self.model.generate_content(prompt)
            self.consecutive_errors = 0
            return response.text
        except Exception as e:
            self.consecutive_errors += 1
            print(f"Gemini API error: {e}")
            if self.consecutive_errors >= self.max_errors:
                self.disabled_until = _time.time() + 1800
                print("⚠️ Gemini disabled for 30 minutes")
            return None

    def analyze_daily_trades(self, trades: list) -> dict:
        if not self.available or not trades:
            return {"summary": "Gemini unavailable or no trades", "recommendations": []}
        prompt = f"""
You are a senior quantitative analyst. Below are today's completed XAUUSD trades:
{json.dumps(trades[:50], default=str)}

Analyze them. Identify:
1. Main patterns leading to wins/losses.
2. Suggestions for parameter adjustments.
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