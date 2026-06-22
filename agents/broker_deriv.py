# agents/broker_deriv.py
import os
import json
import asyncio
import websockets
from typing import Optional, Dict, Any

class DerivBroker:
    """
    Deriv WebSocket API வழியாக:
    - நிகழ்நேர விலை பெறுதல் (ticks)
    - ஆர்டர் பிளேஸ் செய்தல் (BUY/SELL)
    - கணக்கு இருப்பு காணல்
    - திறந்த நிலைகள் காணல்
    """
    def __init__(self):
        self.api_token = os.getenv("DERIV_API_TOKEN")
        self.endpoint = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
        self.websocket = None
        self.connected = False
        self._price_cache = {}

        if not self.api_token:
            print("⚠️ DERIV_API_TOKEN கிடைக்கவில்லை. Environment Variables-ஐ சரிபார்க்கவும்.")
        else:
            print("✅ Deriv Broker தயார்.")

    # ---------- இணைப்பு மேலாண்மை ----------
    async def connect(self):
        """Deriv சர்வருடன் WebSocket இணைப்பு ஏற்படுத்தி, அங்கீகாரம் பெறுக."""
        if self.connected:
            return True
        try:
            self.websocket = await websockets.connect(self.endpoint)
            # அங்கீகாரம்
            auth_req = {"authorize": self.api_token}
            await self.websocket.send(json.dumps(auth_req))
            auth_res = json.loads(await self.websocket.recv())
            if "error" in auth_res:
                print(f"❌ Deriv Auth Error: {auth_res['error']['message']}")
                self.connected = False
                return False
            self.connected = True
            print("✅ Deriv API பாதுகாப்பாக இணைக்கப்பட்டது!")
            return True
        except Exception as e:
            print(f"⚠️ Deriv இணைப்பு பிழை: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            print("🔌 Deriv இணைப்பு மூடப்பட்டது.")

    # ---------- விலை தரவு ----------
    async def get_tick(self, symbol: str = "frxXAUUSD") -> Optional[float]:
        """குறிப்பிட்ட சின்னத்தின் தற்போதைய விலையைப் பெறுக (Tick)."""
        if not self.connected:
            if not await self.connect():
                return None

        # டிக் சப்ஸ்கிரைப் செய்து விலை பெறுக
        tick_req = {"ticks": symbol}
        await self.websocket.send(json.dumps(tick_req))
        response = json.loads(await self.websocket.recv())

        if "error" in response:
            print(f"❌ Tick Error: {response['error']['message']}")
            return None
        # பதிலில் 'tick' உள்ளது
        if "tick" in response:
            return float(response["tick"]["quote"])
        return None

    async def subscribe_ticks(self, symbol: str, callback):
        """தொடர்ந்து விலை புதுப்பிப்புகளைப் பெற callback-ஐ அழைக்கும்."""
        if not self.connected:
            if not await self.connect():
                return
        tick_req = {"ticks": symbol}
        await self.websocket.send(json.dumps(tick_req))
        while True:
            try:
                response = json.loads(await self.websocket.recv())
                if "tick" in response:
                    await callback(response["tick"])
                elif "error" in response:
                    print(f"Tick subscription error: {response['error']}")
                    break
            except Exception as e:
                print(f"Subscription error: {e}")
                break

    # ---------- ஆர்டர் பிளேஸ் ----------
    async def place_order(
        self,
        signal: str,               # "BUY" அல்லது "SELL"
        instrument: str = "frxXAUUSD",
        amount: float = 10.0,      # முதலீட்டுத் தொகை (USD)
        multiplier: int = 100,     # Leverage (100x)
        duration: int = 60,        # நிமிடங்களில் கால அவகாசம் (விரும்பினால்)
        take_profit: float = None, # TP விலை (optional)
        stop_loss: float = None    # SL விலை (optional)
    ) -> Optional[Dict[str, Any]]:
        """
        Deriv-ல் BUY/SELL ஆர்டரை பிளேஸ் செய்கிறது.
        """
        if not self.connected:
            if not await self.connect():
                return None

        # Deriv-ல் BUY -> MULTUP, SELL -> MULTDOWN
        contract_type = "MULTUP" if signal.upper() == "BUY" else "MULTDOWN"

        # அடிப்படை ஆர்டர் தரவு
        order_params = {
            "buy": 1,
            "price": amount,          # முதலீடு
            "parameters": {
                "amount": amount,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "multiplier": multiplier,
                "symbol": instrument,
                "duration": duration,
                "duration_unit": "minutes"
            }
        }

        # TP/SL இருந்தால் சேர்க்க (Deriv 'limit_order' மூலம்)
        if take_profit or stop_loss:
            # Deriv 'limit_order' பயன்படுத்தி TP/SL அமைக்கலாம், ஆனால் இங்கு எளிமைக்காக விடுகிறோம்
            # நீங்கள் விரும்பினால் இந்த பகுதியை விரிவாக்கலாம்
            print("ℹ️ TP/SL support coming soon.")

        await self.websocket.send(json.dumps(order_params))
        response = json.loads(await self.websocket.recv())

        if "error" in response:
            print(f"❌ Trade Error: {response['error']['message']}")
            return None

        trade_id = response.get("buy", {}).get("transaction_id")
        print(f"🚀 Deriv Trade Success! {signal} {instrument} | ID: {trade_id}")
        return response

    # ---------- கணக்கு இருப்பு ----------
    async def get_balance(self) -> Optional[float]:
        """கணக்கு இருப்பை (balance) பெறுக."""
        if not self.connected:
            if not await self.connect():
                return None
        bal_req = {"balance": 1}
        await self.websocket.send(json.dumps(bal_req))
        response = json.loads(await self.websocket.recv())
        if "error" in response:
            print(f"❌ Balance Error: {response['error']['message']}")
            return None
        return float(response.get("balance", {}).get("balance", 0.0))

    # ---------- செயல்பாட்டு உதவி ----------
    async def execute_trade_async(self, signal: str, instrument: str = "frxXAUUSD", amount: float = 10.0):
        """Async-ஆக வர்த்தகம் செய்ய."""
        return await self.place_order(signal, instrument, amount)

    def execute_trade(self, signal: str, instrument: str = "frxXAUUSD", amount: float = 10.0):
        """Sync-ஆக வர்த்தகம் செய்ய (முக்கிய லூப்பில் பயன்படுத்த)."""
        # asyncio.run() - ஏற்கனவே லூப் இருந்தால் conflict வரும், எனவே get_event_loop பயன்படுத்து
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # நடப்பு லூப்பில் task உருவாக்கு
            return asyncio.create_task(self.execute_trade_async(signal, instrument, amount))
        else:
            return asyncio.run(self.execute_trade_async(signal, instrument, amount))

    # ---------- சோதனை ----------
    @staticmethod
    def test():
        """எளிய சோதனை"""
        broker = DerivBroker()
        result = broker.execute_trade("BUY", "frxXAUUSD", 10)
        print(result)

if __name__ == "__main__":
    DerivBroker.test()
