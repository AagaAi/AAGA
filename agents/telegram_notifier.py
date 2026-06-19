# agents/telegram_notifier.py
import os
import telegram

class TelegramNotifier:
    def __init__(self):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        self.bot = None
        if self.token:
            self.bot = telegram.Bot(token=self.token)

    def send_message(self, text):
        if self.bot and self.chat_id:
            try:
                self.bot.send_message(chat_id=self.chat_id, text=text)
            except Exception as e:
                print(f"Telegram send error: {e}")
