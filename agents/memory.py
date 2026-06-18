# agents/memory.py
import sqlite3, json, datetime, pandas as pd
from sklearn.ensemble import RandomForestClassifier

class TradeMemory:
    def __init__(self, db_path="journal.db"):
        self.db = db_path
        self._init_table()

    def _init_table(self):
        con = sqlite3.connect(self.db)
        con.execute("""
            CREATE TABLE IF NOT EXISTS trade_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                features TEXT,
                action INTEGER,
                reward REAL
            )
        """)
        con.commit()
        con.close()

    def add_experience(self, trade):
        con = sqlite3.connect(self.db)
        con.execute("INSERT INTO trade_memory (timestamp, features, action, reward) VALUES (?,?,?,?)",
                    (datetime.datetime.utcnow().isoformat(), json.dumps(trade['features']),
                     trade['action'], trade['reward']))
        con.commit()
        con.close()

    def get_all_experiences(self):
        con = sqlite3.connect(self.db)
        rows = con.execute("SELECT features, action, reward FROM trade_memory").fetchall()
        con.close()
        X, y = [], []
        for r in rows:
            X.append(json.loads(r[0]))
            y.append(r[1])
        return X, y

    def retrain_model(self, model):
        X, y = self.get_all_experiences()
        if len(set(y)) >= 2 and len(X) >= 10:
            model.fit(X, y)
            return True
        return False
