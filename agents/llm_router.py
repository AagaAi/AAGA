# agents/llm_router.py – Multi-LLM Router (Gemini + Grok alternating)
import os
import json
import time as _time
import requests
from typing import Optional

class MultiLLMRouter:
    """
    Routes LLM calls alternately between Gemini 2.0 Flash and Grok (xAI).
    - call 1 → Gemini, call 2 → Grok, call 3 → Gemini ...
    - If primary fails → auto-fallback to other LLM
    """

    def __init__(self):
        self.call_count   = 0
        self.last_call    = 0
        self.call_interval = 10

        # Gemini
        self.gemini_available    = False
        self.gemini_disabled_until = 0
        self.gemini_errors       = 0
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                self.gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
                self.gemini_available = True
                print("✅ Gemini 2.0 Flash ready")
            except Exception as e:
                print(f"⚠️ Gemini init error: {e}")

        # Grok (xAI)
        self.grok_available      = False
        self.grok_disabled_until = 0
        self.grok_errors         = 0
        self.grok_key  = os.environ.get("GROK_API_KEY", "")
        self.grok_url  = "https://api.x.ai/v1/chat/completions"
        self.grok_model = "grok-3-mini"
        if self.grok_key:
            self.grok_available = True
            print("✅ Grok xAI ready")

        if not self.gemini_available and not self.grok_available:
            print("⚠️ No LLM configured – intelligence features disabled")

    @property
    def available(self) -> bool:
        return self.gemini_available or self.grok_available

    @property
    def disabled_until(self) -> float:
        times = []
        if self.gemini_available:
            times.append(self.gemini_disabled_until)
        if self.grok_available:
            times.append(self.grok_disabled_until)
        return min(times) if times else _time.time() + 3600

    def _can_call(self) -> bool:
        return (_time.time() - self.last_call) >= self.call_interval

    def _call_gemini(self, prompt: str) -> Optional[str]:
        now = _time.time()
        if not self.gemini_available or now < self.gemini_disabled_until:
            return None
        try:
            resp = self.gemini_model.generate_content(prompt)
            self.gemini_errors = 0
            print("🟢 [Router] Gemini responded")
            return resp.text
        except Exception as e:
            self.gemini_errors += 1
            print(f"Gemini error #{self.gemini_errors}: {e}")
            if self.gemini_errors >= 2:
                self.gemini_disabled_until = _time.time() + 1800
            return None

    def _call_grok(self, prompt: str) -> Optional[str]:
        now = _time.time()
        if not self.grok_available or not self.grok_key or now < self.grok_disabled_until:
            return None
        try:
            resp = requests.post(
                self.grok_url,
                headers={
                    "Authorization": f"Bearer {self.grok_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.grok_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.1
                },
                timeout=30
            )
            if resp.status_code == 200:
                self.grok_errors = 0
                print("🟣 [Router] Grok responded")
                return resp.json()["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                self.grok_disabled_until = _time.time() + 3600
                print("⚠️ Grok rate limited – paused 1 hour")
            else:
                self.grok_errors += 1
                print(f"Grok HTTP {resp.status_code}")
                if self.grok_errors >= 3:
                    self.grok_disabled_until = _time.time() + 1800
        except Exception as e:
            self.grok_errors += 1
            print(f"Grok request error: {e}")
        return None

    def generate(self, prompt: str) -> Optional[str]:
        if not self.available or not self._can_call():
            return None
        self.last_call = _time.time()
        self.call_count += 1
        use_gemini_first = (self.call_count % 2 == 1)
        print(f"🔀 LLM call #{self.call_count} → {'Gemini first' if use_gemini_first else 'Grok first'}")
        if use_gemini_first:
            return self._call_gemini(prompt) or self._call_grok(prompt)
        else:
            return self._call_grok(prompt) or self._call_gemini(prompt)

    def _safe_generate(self, prompt: str) -> Optional[str]:
        return self.generate(prompt)

    def _clean_json(self, text: str) -> str:
        text = text.strip()
        for fence in ["```json", "```"]:
            if fence in text:
                parts = text.split(fence)
                text = parts[1] if len(parts) > 1 else parts[0]
                text = text.split("```")[0]
                break
        return text.strip()

    def _which_next(self) -> str:
        return "Gemini" if (self.call_count % 2 == 0) else "Grok"

    def analyze_daily_trades(self, trades: list) -> dict:
        if not self.available or not trades:
            return {"summary": "LLM unavailable", "recommendations": []}
        prompt = (
            "You are a senior quantitative analyst. Today's completed trades:\n"
            f"{json.dumps(trades[:10], default=str)}\n\n"
            "Analyze and return ONLY a JSON object:\n"
            '{"summary":"...","patterns":[],'
            '"parameter_suggestions":{"EMA+DMI+BOS":{"adx_threshold":18}},'
            '"confidence":0.8}\n'
            "No markdown. Only raw JSON."
        )
        text = self.generate(prompt)
        if not text:
            return {"summary": "LLM call failed", "recommendations": []}
        try:
            result = json.loads(self._clean_json(text))
            result["llm_used"] = "Gemini" if self.call_count % 2 == 0 else "Grok"
            return result
        except:
            return {"summary": text[:300], "recommendations": [],
                    "llm_used": "Gemini" if self.call_count % 2 == 0 else "Grok"}

    def generate_strategy_params(self, market_context: dict, recent_trades: list) -> Optional[dict]:
        prompt = (
            "Expert XAUUSD strategist. Return ONLY a JSON object (no markdown):\n"
            '{"name":"","description":"","buy_conditions":["ema8 > ema21","adx > 20"],'
            '"sell_conditions":["ema8 < ema21","adx > 20"],'
            '"sl_atr_multiplier":1.5,"tp_atr_multiplier":2.5,'
            '"min_stop_distance":1.1,"adx_threshold":15,"slope_threshold":0.00005}\n\n'
            f"Market context: {json.dumps(market_context)}\n"
            f"Recent trades: {json.dumps(recent_trades, default=str)}"
        )
        text = self.generate(prompt)
        if not text:
            return None
        try:
            result = json.loads(self._clean_json(text))
            result["llm_used"] = "Gemini" if self.call_count % 2 == 0 else "Grok"
            return result
        except:
            return None

    def suggest_parameter_upgrade(self, performance_summary: dict) -> Optional[dict]:
        prompt = (
            "You are an AI trading optimizer. Analyze this performance and return ONLY JSON:\n"
            f"{json.dumps(performance_summary, default=str)}\n\n"
            "Return:\n"
            '{"verdict":"improving/stable/declining","top_issue":"...",'
            '"parameter_changes":{"EMA+DMI+BOS":{"adx_threshold":18,"rr_ratio":2.2},'
            '"RandomForest":{"atr_multiplier_tp":2.5}},'
            '"regime_preference":{"UPTREND":"EMA+DMI+BOS","SIDEWAYS":"RandomForest"},'
            '"new_strategy_needed":false,"confidence":0.8}\n'
            "No markdown. Only raw JSON."
        )
        text = self.generate(prompt)
        if not text:
            return None
        try:
            return json.loads(self._clean_json(text))
        except:
            return None

    def status(self) -> dict:
        now = _time.time()
        return {
            "total_calls": self.call_count,
            "next_call_will_use": self._which_next(),
            "gemini": {
                "available": self.gemini_available,
                "active": self.gemini_available and now >= self.gemini_disabled_until,
                "cooldown_remaining_sec": max(0, round(self.gemini_disabled_until - now)),
                "errors": self.gemini_errors
            },
            "grok": {
                "available": self.grok_available,
                "active": self.grok_available and now >= self.grok_disabled_until,
                "cooldown_remaining_sec": max(0, round(self.grok_disabled_until - now)),
                "errors": self.grok_errors
            }
        }
