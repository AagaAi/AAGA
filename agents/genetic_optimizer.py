# agents/genetic_optimizer.py
import json, random, copy, datetime, asyncio, pandas as pd
from agents.strategy_agent import EmaDmiStrategy
from agents.ai_agent import RandomForestStrategy
from backtest import fetch_last_month_candles, backtest_strategy

class GeneticOptimizer:
    """
    Evolves strategy parameters using a simple genetic algorithm.
    Fitness = win_rate * profit_factor (from backtest on last 30 days)
    """
    def __init__(self, db_path="journal.db"):
        self.db = db_path
        self.population_size = 10
        self.generations = 5
        self.mutation_rate = 0.2

    def _random_params(self):
        """Generate random parameters for a strategy."""
        return {
            "adx_threshold": random.randint(10, 30),
            "slope_threshold": round(random.uniform(0.00001, 0.0005), 6),
            "atr_multiplier_sl": round(random.uniform(1.0, 2.5), 2),
            "rr_ratio": round(random.uniform(1.5, 3.0), 2),
            "min_stop_distance": round(random.uniform(0.5, 1.5), 2),
        }

    def _mutate(self, params):
        """Randomly tweak one parameter."""
        new = copy.deepcopy(params)
        key = random.choice(list(new.keys()))
        if key == "adx_threshold":
            new[key] = max(10, min(30, new[key] + random.choice([-2, 2])))
        elif key == "slope_threshold":
            new[key] = round(new[key] * random.choice([0.8, 1.2]), 6)
        elif key in ("atr_multiplier_sl", "rr_ratio", "min_stop_distance"):
            new[key] = round(new[key] * random.uniform(0.8, 1.2), 2)
        return new

    async def optimize(self, strategy_class, base_cfg=None):
        """
        Run genetic algorithm for a given strategy class.
        Returns best parameters and fitness.
        """
        candles = await fetch_last_month_candles()
        if not candles or len(candles) < 200:
            print("Not enough data for genetic optimization")
            return None

        # Initial population
        population = [self._random_params() for _ in range(self.population_size)]
        if base_cfg:
            population[0] = base_cfg  # keep current best in gene pool

        best_params = None
        best_fitness = -999

        for gen in range(self.generations):
            fitnesses = []
            for params in population:
                strat = strategy_class(params)
                result = backtest_strategy(strat, candles)
                # fitness = win_rate * profit_factor (balance/10000)
                win_rate = result['win_rate'] / 100
                profit_factor = result['final_balance'] / 10000
                fitness = win_rate * profit_factor
                fitnesses.append((fitness, params))
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = params

            # Sort by fitness
            fitnesses.sort(key=lambda x: x[0], reverse=True)
            # Keep top half, crossover to create new half
            survivors = [p for _, p in fitnesses[:self.population_size//2]]
            new_pop = survivors[:]
            while len(new_pop) < self.population_size:
                p1, p2 = random.sample(survivors, 2)
                child = {}
                for key in p1:
                    child[key] = random.choice([p1[key], p2[key]])
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                new_pop.append(child)
            population = new_pop

        print(f"🧬 Genetic optimizer complete. Best fitness: {best_fitness:.2f}")
        return best_params

    def apply_to_strategy(self, strategy, params):
        """Update strategy instance with new parameters."""
        for key, val in params.items():
            if hasattr(strategy, key):
                setattr(strategy, key, val)
        print(f"🔧 Updated {strategy.name} with evolved parameters")
