# agents/telegram_bot.py
import os, json, sqlite3, datetime
from typing import Optional
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

DB_PATH = "journal.db"

class TelegramBotHandler:
    """
    Interactive Telegram bot for A.A.G.A AI.
    Commands: /start, /status, /performance, /pause, /resume
    """
    def __init__(self, token: str):
        self.token = token
        self.app = None
        self.paused = False

    async def start(self):
        if not self.token:
            print("Telegram token not set – bot disabled")
            return
        self.app = Application.builder().token(self.token).build()
        # Register commands
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("performance", self.cmd_performance))
        self.app.add_handler(CommandHandler("pause", self.cmd_pause))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CommandHandler("report", self.cmd_report))
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        print("🤖 Telegram bot started")

    async def stop(self):
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()

    async def send_message(self, chat_id: str, text: str):
        if self.app:
            await self.app.bot.send_message(chat_id=chat_id, text=text)

    # ---------- Command Handlers ----------
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 **A.A.G.A AI – Trading OS**\n\n"
            "Commands:\n"
            "/status – live trading status\n"
            "/performance – today's performance\n"
            "/report – daily PDF report\n"
            "/pause – pause autonomous trading\n"
            "/resume – resume autonomous trading"
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Fetch live data from global cache (we'll import from main)
        from main import latest_signals_cache, _metaapi_healthy
        msg = f"📡 **A.A.G.A AI Status**\n"
        msg += f"MetaApi: {'✅ Online' if _metaapi_healthy else '❌ Offline'}\n"
        for pair, cache in latest_signals_cache.items():
            msg += f"\n{pair}: {cache['decision']} @ {cache['current_price']:.2f} ({cache['regime']})\n"
        await update.message.reply_text(msg)

    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        con = sqlite3.connect(DB_PATH)
        rows = con.execute("""
            SELECT pair, COUNT(*) as total, SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins,
                   SUM(pnl) as pnl
            FROM trade_journal WHERE date(timestamp) = date('now') AND outcome IS NOT NULL
            GROUP BY pair
        """).fetchall()
        con.close()
        msg = "📊 **Today's Performance**\n"
        for r in rows:
            win_rate = (r[2] / r[1] * 100) if r[1] > 0 else 0
            msg += f"{r[0]}: {r[1]} trades, {win_rate:.0f}% WR, P&L ${r[3]:.2f}\n"
        if not rows:
            msg += "No completed trades today."
        await update.message.reply_text(msg)

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.paused = True
        # Tell main loop to pause
        from main import autonomous_trading_loop_paused
        autonomous_trading_loop_paused = True
        await update.message.reply_text("⏸ Autonomous trading PAUSED. Use /resume to continue.")

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.paused = False
        from main import autonomous_trading_loop_paused
        autonomous_trading_loop_paused = False
        await update.message.reply_text("▶️ Autonomous trading RESUMED.")

    async def cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Generate a simple text report (PDF would require reportlab, but text is enough for Telegram)
        from agents.pdf_report import generate_daily_report
        report = generate_daily_report()
        await update.message.reply_text(report)
