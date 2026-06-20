# agents/portfolio_manager.py
import pandas as pd
import numpy as np
from typing import Dict, Any

class PortfolioManager:
    """
    Computes cross-asset correlation, volatility, and suggests capital allocation.
    Uses simple inverse volatility weighting.
    """
    def __init__(self, pairs: list, initial_capital: float = 10000.0):
        self.pairs = pairs
        self.total_capital = initial_capital
        self.allocations = {p: initial_capital / len(pairs) for p in pairs}

    def update_prices(self, current_prices: Dict[str, float]):
        """Store latest prices for each pair (not needed for allocation but used for display)."""
        self.current_prices = current_prices

    def compute_correlation(self, candles_dict: Dict[str, list]) -> Dict[str, Dict[str, float]]:
        """
        candles_dict: {pair: list of dicts with 'close'}
        Returns correlation matrix using last 200 common candles.
        """
        closes = {}
        for pair in self.pairs:
            if pair in candles_dict and candles_dict[pair] and len(candles_dict[pair]) >= 200:
                df = pd.DataFrame(candles_dict[pair])
                closes[pair] = df['close'].pct_change().dropna()
        if len(closes) < 2:
            return {}
        df = pd.DataFrame(closes)
        corr = df.corr()
        return corr.to_dict()

    def compute_volatility(self, candles_dict: Dict[str, list]) -> Dict[str, float]:
        """Return annualized volatility for each pair (based on 1h candles)."""
        vols = {}
        for pair in self.pairs:
            if pair in candles_dict and candles_dict[pair] and len(candles_dict[pair]) >= 50:
                df = pd.DataFrame(candles_dict[pair])
                returns = df['close'].pct_change().dropna()
                vols[pair] = returns.std() * np.sqrt(252 * 24)  # annualized (hourly)
        return vols

    def allocate_capital(self, volatilities: Dict[str, float], max_per_pair: float = 0.5):
        """
        Allocate capital inversely proportional to volatility.
        max_per_pair limits concentration.
        """
        inv_vol = {p: 1.0 / (volatilities.get(p, 0.01) + 1e-8) for p in self.pairs}
        total = sum(inv_vol.values())
        weights = {p: inv_vol[p] / total for p in self.pairs}
        # Apply max limit
        for p in self.pairs:
            if weights[p] > max_per_pair:
                excess = weights[p] - max_per_pair
                weights[p] = max_per_pair
                redist = excess / (len(self.pairs) - 1)
                for q in self.pairs:
                    if q != p:
                        weights[q] += redist
        self.allocations = {p: self.total_capital * weights[p] for p in self.pairs}
        return weights

    def get_portfolio_stats(self, candles_dict: Dict[str, list], current_prices: Dict[str, float]):
        """Main method: compute correlation, volatility, allocation, and return summary."""
        corr = self.compute_correlation(candles_dict)
        vols = self.compute_volatility(candles_dict)
        weights = self.allocate_capital(vols)
        # Allocate risk per pair to individual risk managers? We'll just return weights.
        # Actual risk per trade will be handled by RiskManager per pair using allocated capital.
        return {
            "correlation": corr,
            "volatilities": vols,
            "allocation_weights": weights,
            "allocated_capital": {p: round(self.total_capital * weights[p], 2) for p in self.pairs}
        }
