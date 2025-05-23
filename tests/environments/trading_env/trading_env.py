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
        self.raw_data = pd.read_csv("tests/environments/trading_env/raw_test.csv", skipinitialspace=True)
        self.processed_data = pd.read_csv("tests/environments/trading_env/processed_test.csv", skipinitialspace=True)
        # self.raw_data.set_index("day", inplace=True) # This is also handled as an edge case in the environment (does not affect the test)
        # self.processed_data.set_index("day", inplace=True) # This is also handled as an edge case in the environment (does not affect the test)
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
            "position_limits": {"min": 0, "max": 100},
            "cash_limit": {"min": 0, "max": 100000.0}
        }
        self.ohlcv_dim = 5
        self.tech_dim = 3
        self.n_assets = len(self.processed_data["ticker"].unique())
        self.window_size = self.env_params["window_size"]
        self.market_cols = self.columns["ohlcv"] + self.columns["tech_cols"]
        self.asset_list = list(self.processed_data["ticker"].unique())

        # Define default processor configs for testing
        self.processor_configs = [
            {
                'type': 'price',
                'data_name': 'market_data',
                'kwargs': {
                    'window_size': self.window_size,
                    'asset_list': self.asset_list
                }
            },
            {
                'type': 'cash',
                'data_name': 'cash_data',
                'kwargs': {
                    'cash_limit': self.constraint_params["cash_limit"]["max"]
                }
            },
            {
                'type': 'position',
                'data_name': 'position_data',
                'kwargs': {
                    'position_limits': self.constraint_params["position_limits"],
                    'asset_list': self.asset_list
                }
            }
        ]
    
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
            processor_configs=self.processor_configs
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
        
        # Observation space - now based on processors
        self.assertEqual(list(env.observation_space.keys()), sorted(["price", "cash", "position"]))
        
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
            processor_configs=self.processor_configs
        )
        _, _, _, _ = env.step(np.ones_like(env.action_space, dtype=np.int32))
        
        observation = env.reset()

        self.assertEqual(env.current_step, self.env_params["window_size"])
        self.assertEqual(env.current_cash, self.env_params["initial_balance"])
        self.assertTrue(np.array_equal(env.positions, np.zeros(len(self.processed_data["ticker"].unique()))))
        self.assertTrue(np.array_equal(env.asset_values, np.zeros(len(self.processed_data["ticker"].unique()))))
        self.assertEqual(env.portfolio_value_history, [self.env_params["initial_balance"]])
        self.assertEqual(env.done, False)
        self.assertEqual(env.info, {})
        
        # Check observation after reset
        self.assertEqual(sorted(list(observation.keys())), sorted(["price", "cash", "position"]))
        self.assertTrue(np.array_equal(observation["position"], np.zeros(self.n_assets)))
        self.assertEqual(observation["cash"].shape, (2,))  # [cash, portfolio_value]
    
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
            processor_configs=self.processor_configs
        )
        env.reset()

        # Test case 1: Action is within the constraints
        action = np.array([0, 0, 0])
        observation, reward, done, info = env.step(action)
        self.assertEqual(reward, 0.0)
        self.assertEqual(done, False)
        self.assertTrue(np.array_equal(observation["position"], np.zeros(shape=(self.n_assets))))
        self.assertEqual(observation["cash"].shape, (2,))

        # Test case 2: Action violates the constraints
        env.reset()
        action = np.array([1000, -5, 0])
        observation, reward, done, info = env.step(action)
        self.assertEqual(done, False)
        self.assertLess(reward, 0.0)
       
        # Test case 3: Action is within the constraints
        env.reset()
        self.assertFalse(env.done)
        action = np.array([1, 1, 1])
        observation, reward, done, info = env.step(action)
        self.assertEqual(done, False)

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
            processor_configs=self.processor_configs
        )

        observation = env.reset()
        
        # Check observation structure
        self.assertEqual(sorted(list(observation.keys())), sorted(["price", "cash", "position"]))
        
        # Check shapes
        self.assertEqual(observation["price"].shape, (self.n_assets, self.window_size))
        self.assertEqual(observation["position"].shape, (self.n_assets,))
        self.assertEqual(observation["cash"].shape, (2,))

        # Check initial values
        self.assertTrue(np.array_equal(observation["position"], np.zeros(shape=(self.n_assets))))
        self.assertAlmostEqual(observation["cash"][0], 1.0)  # Initial cash ratio
        self.assertAlmostEqual(observation["cash"][1], 0.0)  # Initial portfolio value ratio

        # Check price data
        for i, asset in enumerate(self.asset_list):
            for j in range(self.window_size):
                self.assertEqual(
                    observation["price"][i, j],
                    self.processed_data[self.processed_data["ticker"] == asset].iloc[j]["close"]
                )

    def test_done(self):
        """Test that the environment is done when no more data is left."""
        env = TradingEnv(
            processed_data=self.processed_data,
            raw_data=self.raw_data,
            columns=self.columns,
            env_params=self.env_params,
            friction_params=self.friction_params,
            reward_params=self.reward_params,
            constraint_params=self.constraint_params,
            processor_configs=self.processor_configs
        )
        i = 1
        observation = env.reset()
        while not env.done:
            i += 1
            env.step(np.zeros(shape=(self.n_assets,)))

        self.assertTrue(env.done)
        self.assertEqual(env.current_step, env.max_step)
        self.assertEqual(i, 3)

    def test_custom_processor_config(self):
        """Test the environment with a custom processor configuration."""
        custom_configs = [
            {
                'type': 'price',
                'data_name': 'market_data',
                'kwargs': {
                    'window_size': self.window_size,
                    'asset_list': self.asset_list
                }
            },
            {
                'type': 'tech',
                'data_name': 'market_data',
                'kwargs': {
                    'tech_cols': self.columns["tech_cols"],
                    'asset_list': self.asset_list
                }
            },
            {
                'type': 'cash',
                'data_name': 'cash_data',
                'kwargs': {
                    'cash_limit': self.constraint_params["cash_limit"]["max"]
                }
            }
        ]

        env = TradingEnv(
            processed_data=self.processed_data,
            raw_data=self.raw_data,
            columns=self.columns,
            env_params=self.env_params,
            friction_params=self.friction_params,
            reward_params=self.reward_params,
            constraint_params=self.constraint_params,
            processor_configs=custom_configs
        )

        observation = env.reset()
        
        # Check observation structure with custom config
        self.assertEqual(sorted(list(observation.keys())), sorted(["price", "tech", "cash"]))
        
        # Check shapes
        self.assertEqual(observation["price"].shape, (self.n_assets, self.window_size))
        self.assertEqual(observation["tech"].shape, (self.n_assets, len(self.columns["tech_cols"])))
        self.assertEqual(observation["cash"].shape, (2,))

if __name__ == "__main__":
    unittest.main()
