# ============================================================
# execute_trade_sync — Place market order with SL/TP on MT5
# ============================================================
def execute_trade_sync(signal, sl, tp, lot=0.01):
    if not METAAPI_TOKEN or not METAAPI_ACCOUNT_ID:
        print("MetaApi credentials missing – cannot place order.")
        return None

    async def _place():
        account = await get_account()                # re‑use cached connection
        conn    = account.get_rpc_connection()
        await conn.connect()
        await conn.wait_synchronized()
        try:
            if signal == "BUY":
                order = await conn.create_market_buy_order(
                    symbol="XAUUSD",
                    volume=lot,
                    stop_loss=sl,
                    take_profit=tp,
                    comment="Tradevil AI"
                )
            else:
                order = await conn.create_market_sell_order(
                    symbol="XAUUSD",
                    volume=lot,
                    stop_loss=sl,
                    take_profit=tp,
                    comment="Tradevil AI"
                )
            return order
        finally:
            await conn.close()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_place())
        loop.close()
        return result
    except Exception as e:
        print(f"Order placement error: {e}")
        return None