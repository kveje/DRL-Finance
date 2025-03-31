"""Unit tests for the TradingEnv class."""

import unittest
from typing import Dict, List
import sys
import os
import numpy as np
import pandas as pd
#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the function to test
from environments.trading_env import TradingEnv

class TestTradingEnv(unittest.TestCase):
    """Test cases for the TradingEnv class."""

    def setUp(self):
        """Set up test fixtures."""
        self.raw_data = pd.read_csv("tests/environments/trading_env/raw_test.csv", skipinitialspace=True, index_col="day")
        self.processed_data = pd.read_csv("tests/environments/trading_env/processed_test.csv", skipinitialspace=True, index_col="day")
        self.columns = {
            "ticker": "ticker",
            "price": "close",
            "day": "day",
            "ohlcv": ["open", "high", "low", "close", "volume"],
            "tech_cols": ["RSI", "MACD", "Bollinger Bands"]
        }
        self.env_params = {
            "initial_balance": 100000.0,
            "window_size": 3
        }
        self.friction_params = {
            "slippage": {"slippage_mean": 0.0, "slippage_std": 0.001},
            "commission": {"commission_rate": 0.001}
        }
        self.reward_params = ("returns_based", {"scale": 1.0})
        self.constraint_params = {
            "position_limits": {"min": -100, "max": 100}
        }
        self.ohlcv_dim = 5
        self.tech_dim = 3
        self.n_assets = len(self.processed_data["ticker"].unique())
        self.window_size = self.env_params["window_size"]
        self.market_cols = self.columns["ohlcv"] + self.columns["tech_cols"]
        self.asset_list = list(self.processed_data["ticker"].unique())

    
    def test_trading_env_initialization(self):
        """Test the initialization of the TradingEnv class."""
        env = TradingEnv(
            processed_data=self.processed_data,
            raw_data=self.raw_data,
            columns=self.columns,
            env_params=self.env_params,
            friction_params=self.friction_params,
            reward_params=self.reward_params,
            constraint_params=self.constraint_params,
        )
        self.assertEqual(env.initial_balance, self.env_params["initial_balance"])
        self.assertEqual(env.window_size, self.env_params["window_size"])
        self.assertEqual(env.tic_col, self.columns["ticker"])
        self.assertEqual(env.price_col, self.columns["price"])
        self.assertEqual(env.day_col, self.columns["day"])
        self.assertEqual(env.ohlcv_cols, self.columns["ohlcv"])
        self.assertEqual(env.tech_cols, self.columns["tech_cols"])
        self.assertEqual(env.asset_list, list(self.processed_data["ticker"].unique()))
        self.assertEqual(env.n_assets, len(self.processed_data["ticker"].unique()))
        self.assertEqual(env.current_cash, self.env_params["initial_balance"])
        self.assertTrue(np.array_equal(env.positions, np.zeros(len(self.processed_data["ticker"].unique()))))
        self.assertTrue(np.array_equal(env.asset_values, np.zeros(len(self.processed_data["ticker"].unique()))))
        self.assertEqual(env.portfolio_value_history, [self.env_params["initial_balance"]])
        
        # Action space
        self.assertEqual(env.action_space.shape, (self.n_assets,))
        
        # Observation space
        self.assertEqual(list(env.observation_space.keys()), sorted(["data", "positions", "portfolio_info"]))
        self.assertEqual(env.observation_space["data"].shape, (self.n_assets, self.ohlcv_dim + self.tech_dim, self.window_size))

        # Initialize state
        self.assertEqual(env.current_step, self.env_params["window_size"])
        self.assertFalse(env.done)
        
    def test_trading_env_reset(self):
        """Test the reset method of the TradingEnv class."""
        env = TradingEnv(
            processed_data=self.processed_data,
            raw_data=self.raw_data,
            columns=self.columns,
            env_params=self.env_params,
            friction_params=self.friction_params,
            reward_params=self.reward_params,
            constraint_params=self.constraint_params,
        )
        _, _, _, _ = env.step(np.ones_like(env.action_space, dtype=np.int32))
        
        _ = env.reset()

        self.assertEqual(env.current_step, self.env_params["window_size"])
        self.assertEqual(env.current_cash, self.env_params["initial_balance"])
        self.assertTrue(np.array_equal(env.positions, np.zeros(len(self.processed_data["ticker"].unique()))))
        self.assertTrue(np.array_equal(env.asset_values, np.zeros(len(self.processed_data["ticker"].unique()))))
        self.assertEqual(env.portfolio_value_history, [self.env_params["initial_balance"]])
        self.assertEqual(env.done, False)
        self.assertEqual(env.info, {})
    
    def test_trading_env_step(self):
        """Test the step method of the TradingEnv class."""
        env = TradingEnv(
            processed_data=self.processed_data,
            raw_data=self.raw_data,
            columns=self.columns,
            env_params=self.env_params,
            friction_params=self.friction_params,
            reward_params=self.reward_params,
            constraint_params=self.constraint_params,
        )

        # Test case 1: Action is within the constraints
        action = np.array([0, 0, 0])
        observation, reward, done, info = env.step(action)
        self.assertEqual(reward, 0.0)
        self.assertEqual(done, False)
        self.assertTrue(np.array_equal(observation["positions"], np.zeros(shape=(self.n_assets))))
        self.assertTrue(np.array_equal(observation["portfolio_info"], np.array([self.env_params["initial_balance"], self.env_params["initial_balance"]])))

        # Test case 2: Action violates the constraints
        action = np.array([1000, -5, 0])
        observation, reward, done, info = env.step(action)
        self.assertEqual(done, True)
        self.assertEqual(reward, -1000.0)
        self.assertTrue(np.array_equal(observation["positions"], np.zeros(shape=(self.n_assets))))

        env.reset()
        self.assertFalse(env.done)

        # Test case 3: Action is within the constraints
        action = np.array([1, 1, 1])
        observation, reward, done, info = env.step(action)
        self.assertEqual(done, False)
        self.assertTrue(np.array_equal(observation["positions"], np.ones(shape=(self.n_assets))))
        

    def test_observation_space(self):
        """Test the observation space of the TradingEnv class."""
        env = TradingEnv(
            processed_data=self.processed_data,
            raw_data=self.raw_data,
            columns=self.columns,
            env_params=self.env_params,
            friction_params=self.friction_params,
            reward_params=self.reward_params,
            constraint_params=self.constraint_params,
        )

        observation = env.reset()
        # Shapes
        self.assertEqual(observation["data"].shape, (self.n_assets, self.ohlcv_dim + self.tech_dim, self.window_size))
        self.assertEqual(observation["positions"].shape, (self.n_assets,))
        self.assertEqual(observation["portfolio_info"].shape, (2,))

        # Values
        self.assertTrue(np.array_equal(observation["positions"], np.zeros(shape=(self.n_assets))))
        self.assertTrue(np.array_equal(observation["portfolio_info"], np.array([self.env_params["initial_balance"], self.env_params["initial_balance"]])))
        days = [0,1,2]
        for i, asset in enumerate(self.asset_list):
            for j, col in enumerate(self.market_cols):
                for k, day in enumerate(days):
                    self.assertEqual(observation["data"][i, j, k], self.processed_data[self.processed_data["ticker"] == asset].iloc[day][col])

    

if __name__ == "__main__":
    unittest.main()
