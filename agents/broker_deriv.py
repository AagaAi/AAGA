# agents/broker_deriv.py
import os
import json
import asyncio
import websockets
from typing import Optional, Dict, Any

class DerivBroker:
    """
    Deriv WebSocket API wrapper – reads token & app ID from environment.
    """
    def __init__(self):
        self.api_token = os.getenv("DERIV_API_TOKEN")
        # Read App ID from env, fallback to 1089 if not set
        self.app_id = os.getenv("DERIV_APP_ID", "1089")
        self.endpoint = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        self.websocket = None
        self.connected = False
        self._price_cache = {}

        if not self.api_token:
            print("⚠️ DERIV_API_TOKEN not set.")
        else:
            print(f"✅ Deriv Broker ready (App ID: {self.app_id})")

    async def connect(self):
        if self.connected:
            return True
        try:
            self.websocket = await websockets.connect(self.endpoint)
            # Authorize with token
            auth_req = {"authorize": self.api_token}
            await self.websocket.send(json.dumps(auth_req))
            auth_res = json.loads(await self.websocket.recv())
            if "error" in auth_res:
                print(f"❌ Deriv Auth Error: {auth_res['error']['message']}")
                self.connected = False
                return False
            self.connected = True
            print("✅ Deriv API authenticated successfully!")
            return True
        except Exception as e:
            print(f"⚠️ Deriv connection error: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            print("🔌 Deriv disconnected.")

    async def get_tick(self, symbol: str = "frxXAUUSD") -> Optional[float]:
        if not self.connected:
            if not await self.connect():
                return None
        tick_req = {"ticks": symbol}
        await self.websocket.send(json.dumps(tick_req))
        response = json.loads(await self.websocket.recv())
        if "error" in response:
            print(f"❌ Tick error: {response['error']['message']}")
            return None
        if "tick" in response:
            return float(response["tick"]["quote"])
        return None

    async def place_order(
        self,
        signal: str,
        instrument: str = "frxXAUUSD",
        amount: float = 10.0,
        multiplier: int = 100,
        duration: int = 60
    ) -> Optional[Dict[str, Any]]:
        if not self.connected:
            if not await self.connect():
                return None

        contract_type = "MULTUP" if signal.upper() == "BUY" else "MULTDOWN"
        order_params = {
            "buy": 1,
            "price": amount,
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

        await self.websocket.send(json.dumps(order_params))
        response = json.loads(await self.websocket.recv())

        if "error" in response:
            print(f"❌ Trade error: {response['error']['message']}")
            return None

        trade_id = response.get("buy", {}).get("transaction_id")
        print(f"🚀 Trade executed: {signal} {instrument} | ID: {trade_id}")
        return response

    async def execute_trade_async(self, signal: str, instrument: str = "frxXAUUSD", amount: float = 10.0):
        return await self.place_order(signal, instrument, amount)

    def execute_trade(self, signal: str, instrument: str = "frxXAUUSD", amount: float = 10.0):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            return asyncio.create_task(self.execute_trade_async(signal, instrument, amount))
        else:
            return asyncio.run(self.execute_trade_async(signal, instrument, amount))
