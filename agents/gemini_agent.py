# agents/gemini_agent.py
import os, json, datetime, time as _time
import google.generativeai as genai
from typing import Dict, Any, Optional

class GeminiAnalyst:
    def __init__(self, api_key: str = ""):
        self.available = False
        self.last_call = 0
        self.call_interval = 60  # seconds between calls (quota saver)
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            self.error = "GEMINI_API_KEY not set"
            return
        try:
            genai.configure(api_key=key)
            # Use a stable model name
            self.model = genai.GenerativeModel("gemini-2.0-flash")
            self.available = True
        except Exception as e:
            self.error = str(e)
            self.available = False

    def _safe_generate(self, prompt: str) -> Optional[str]:
        """Call Gemini API with rate limiting."""
        now = _time.time()
        if now - self.last_call < self.call_interval:
            print("⏳ Gemini call throttled (quota saver)")
            return None
        self.last_call = now
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None

    def analyze_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Single trade analysis: why it won/lost."""
        if not self.available:
            return {"summary": "Gemini unavailable", "recommendation": "N/A"}
        prompt = f"""
You are a professional XAUUSD trading coach. Analyze this trade:

- Signal: {trade.get('signal')}
- Entry: {trade.get('entry')}
- Stop Loss: {trade.get('sl')}
- Take Profit: {trade.get('tp')}
- Outcome: {trade.get('outcome')}
- Profit/Loss: {trade.get('pnl')}
- Market trend at entry: {trade.get('trend', 'Unknown')}
- Sweep type: {trade.get('sweep_type')}
- MSS: {trade.get('mss')}
- FVG: {trade.get('fvg')}

Explain in simple terms WHY this trade won or lost. Give ONE actionable lesson to improve future trades.
Return ONLY a JSON object with keys: "summary", "lesson", "confidence" (0-1). No other text.
"""
        text = self._safe_generate(prompt)
        if not text:
            return {"summary": "Gemini call failed", "lesson": "N/A", "confidence": 0}
        try:
            # Clean potential markdown fences
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text)
        except:
            return {"summary": "Parse error", "lesson": text, "confidence": 0.5}

    def analyze_daily_trades(self, trades: list) -> Dict[str, Any]:
        """Daily summary: find patterns, suggest parameter adjustments."""
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