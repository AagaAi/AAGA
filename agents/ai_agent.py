# agents/ai_agent.py
import numpy as np
import pandas as pd
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class CNNLSTMModel:
    def __init__(self, input_size=4, lookback=10):
        self.lookback = lookback
        self.model = self._build_model(input_size)
    
    def _build_model(self, input_size):
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                   input_shape=(self.lookback, input_size)),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(units=50, return_sequences=True),
            LSTM(units=50),
            Flatten(),
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def prepare_data(self, prices):
        data = np.array(prices)
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            y.append(data[i, 3])  # close price
        return np.array(X), np.array(y)
    
    def train(self, prices, epochs=20):
        if len(prices) < self.lookback+10:
            return
        X, y = self.prepare_data(prices)
        self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    
    def predict(self, recent_prices):
        if len(recent_prices) < self.lookback:
            return 0.0
        X = np.array(recent_prices[-self.lookback:])
        X = X.reshape((1, self.lookback, X.shape[1]))
        return self.model.predict(X, verbose=0)[0,0]

class DQLAgent:
    def __init__(self, state_size=11, action_space=3):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target()
    
    def _build_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_size,)),
            Dense(64, activation='relu'),
            Dense(self.action_space)
        ])
        model.compile(optimizer=Adam(0.001), loss='mse')
        return model
    
    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state.reshape(1, -1), verbose=0)
            next_q = self.target_model.predict(next_state.reshape(1, -1), verbose=0)
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * np.amax(next_q[0])
            self.model.fit(state.reshape(1, -1), target, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class AIPureStrategy:
    def __init__(self, config=None):
        self.cnn_lstm = CNNLSTMModel(input_size=4, lookback=10)
        self.dql_agent = DQLAgent(state_size=11)
        self.trained = False
        self.last_trade_side = 0  # 0: none, 1: buy, -1: sell
        self.position = 0
        self.entry_price = 0.0
        self.balance = 10000.0
    
    def train_from_candles(self, candles):
        """Train CNN-LSTM on historical 1-minute data (OHLC)."""
        if len(candles) < 100:
            return
        # Format: open, high, low, close
        prices = [[c['open'], c['high'], c['low'], c['close']] for c in candles]
        self.cnn_lstm.train(prices, epochs=10)
        self.trained = True
    
    def get_signal(self, ohlc_dict):
        """
        Returns signal dict like strategy_agent:
        {"signal": "BUY"/"SELL"/"HOLD", "entry_zone", "sl", "tp", "trend", ...}
        Uses recent 1m candles to predict next price and DQL to decide action.
        """
        closes = pd.Series(ohlc_dict['closes'])
        highs  = pd.Series(ohlc_dict['highs'])
        lows   = pd.Series(ohlc_dict['lows'])
        opens  = pd.Series(ohlc_dict.get('opens', closes.shift(1).fillna(closes.iloc[0])))
        current_price = closes.iloc[-1]

        if not self.trained:
            # fallback to simple trend if model not trained
            trend = "UPTREND" if closes.iloc[-1] > closes.iloc[-20] else "DOWNTREND"
            return {"signal": "HOLD", "entry_zone": None, "sl": None, "tp": None,
                    "trend": trend, "current_price": current_price}

        # Prepare state from recent 100 bars
        lookback = 10
        if len(closes) < lookback:
            return {"signal": "HOLD", "entry_zone": None, "sl": None, "tp": None,
                    "trend": "Unknown", "current_price": current_price}
        
        # build state vector
        state = np.array([
            opens.iloc[-1] / 2000,
            highs.iloc[-1] / 2000,
            lows.iloc[-1] / 2000,
            closes.iloc[-1] / 2000,
            (closes.iloc[-1] - closes.iloc[-2]) / 10,
            (highs.iloc[-1] - lows.iloc[-1]) / 10,
            self.balance / 10000,
            self.position,
            closes.rolling(5).mean().iloc[-1] / 2000,
            closes.rolling(10).mean().iloc[-1] / 2000,
            closes.iloc[-1] / closes.rolling(20).mean().iloc[-1]
        ])

        action = self.dql_agent.act(state)  # 0=hold, 1=buy, 2=sell

        signal = "HOLD"
        entry_zone = sl = tp = None
        if action == 1:  # BUY
            signal = "BUY"
            # Use predicted price to set SL/TP
            pred_price = self.cnn_lstm.predict(opens.values.reshape(-1,1))  # simplified
            entry_zone = (current_price, current_price)
            atr = (highs.iloc[-14:].max() - lows.iloc[-14:].min()) / 2
            sl = current_price - 1.5 * atr
            tp = current_price + 2.0 * atr
        elif action == 2:  # SELL
            signal = "SELL"
            entry_zone = (current_price, current_price)
            atr = (highs.iloc[-14:].max() - lows.iloc[-14:].min()) / 2
            sl = current_price + 1.5 * atr
            tp = current_price - 2.0 * atr

        # Update DQL with reward if we closed a position? (simplified)
        # (Full reinforcement learning loop is done during separate training,
        # here we just use the trained agent for inference)

        trend = "UPTREND" if closes.iloc[-1] > closes.iloc[-20] else "DOWNTREND"
        return {
            "signal": signal,
            "entry_zone": entry_zone,
            "sl": round(sl,2) if sl else None,
            "tp": round(tp,2) if tp else None,
            "trend": trend,
            "current_price": current_price
        }

# Singleton for reuse
ai_strategy = AIPureStrategy()