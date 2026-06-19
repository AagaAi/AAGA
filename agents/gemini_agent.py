# agents/gemini_agent.py
import os, json, time as _time
from google import genai
from typing import Optional

class GeminiAnalyst:
    def __init__(self, api_key: str = ""):
        self.available = False
        self.last_call = 0
        self.call_interval = 10        # seconds (not used now, but keep for internal throttling)
        self.consecutive_errors = 0
        self.max_errors = 2
        self.disabled_until = 0
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            self.error = "GEMINI_API_KEY not set"
            return
        try:
            self.client = genai.Client(api_key=key)
            # Use 1.5 Flash (free tier available, high quota)
            self.model = "gemini-1.5-flash"
            self.available = True
        except Exception as e:
            self.error = str(e)

    def _should_throttle(self) -> bool:
        now = _time.time()
        if self.disabled_until > now:
            return True
        return False

    def _safe_generate(self, prompt: str) -> Optional[str]:
        if not self.available:
            return None
        if self._should_throttle():
            print("⏳ Gemini disabled due to earlier quota error")
            return None
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            self.consecutive_errors = 0
            return response.text
        except Exception as e:
            self.consecutive_errors += 1
            print(f"Gemini API error: {e}")
            if self.consecutive_errors >= self.max_errors:
                # Disable for 1 hour
                self.disabled_until = _time.time() + 3600
                print("⚠️ Gemini disabled for 1 hour due to repeated errors")
            return None

    def analyze_daily_trades(self, trades: list) -> dict:
        # (not used in hourly, but kept for compatibility)
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