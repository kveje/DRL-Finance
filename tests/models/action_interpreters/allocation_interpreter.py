"""Unit tests for the AllocationInterpreter class."""

import unittest
from typing import Dict, List
import sys
import os
import numpy as np
import pandas as pd

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the function to test
from models.action_interpreters.allocation_interpreter import AllocationInterpreter
from environments.trading_env import TradingEnv

class TestAllocationInterpreter(unittest.TestCase):
    """Unit tests for the AllocationInterpreter class."""

    def setUp(self):
        """Set up test fixtures."""
        # Set up the environment
        self.raw_data = pd.read_csv("tests/environments/trading_env/raw_test.csv", skipinitialspace=True)
        self.processed_data = pd.read_csv("tests/environments/trading_env/processed_test.csv", skipinitialspace=True)
        self.columns = {
            "ticker": "ticker",
            "price": "close",
            "day": "day",
            "ohlcv": ["open", "high", "low", "close", "volume"],
            "tech_cols": ["RSI", "MACD", "Bollinger Bands"]
        }
        self.env = TradingEnv(self.raw_data, self.processed_data, self.columns)

        # Set up the interpreter
        self.interpreter = AllocationInterpreter(self.env)

    def test_allocation_interpreter_initialization(self):
        """Test that the AllocationInterpreter class initializes correctly."""
        self.assertIsNotNone(self.interpreter)

    def test_get_beta_distribution_params(self):
        """Test that the get_beta_distribution_params method returns the correct parameters."""
        # No batch dimension
        raw_alpha = np.random.randn(10)
        raw_beta = np.random.randn(10)
        alpha, beta = self.interpreter._get_beta_distribution_params(raw_alpha, raw_beta)
        
        self.assertEqual(alpha.shape, raw_alpha.shape)
        self.assertEqual(beta.shape, raw_beta.shape)
        self.assertTrue(np.all(alpha > 0))
        self.assertTrue(np.all(beta > 0))

        # Batch dimension
        raw_alpha = np.random.randn(10, 10)
        raw_beta = np.random.randn(10, 10)
        alpha, beta = self.interpreter._get_beta_distribution_params(raw_alpha, raw_beta)

        self.assertEqual(alpha.shape, raw_alpha.shape)
        self.assertEqual(beta.shape, raw_beta.shape)

    def test_calculate_risk_aware_objective_params(self):
        """Test that the calculate_risk_aware_objective_params method returns the correct parameters."""
        # No batch dimension
        raw_alpha = np.random.randn(10)
        raw_beta = np.random.randn(10)
        alpha, beta = self.interpreter._get_beta_distribution_params(raw_alpha, raw_beta)
        position_limits = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        target_mean, target_variance = self.interpreter._calculate_risk_aware_objective_params(alpha, beta, position_limits)

        self.assertEqual(target_mean.shape, raw_alpha.shape)
        self.assertEqual(target_variance.shape, raw_beta.shape)
        self.assertTrue(np.all(target_mean <= position_limits))
        self.assertTrue(np.all(target_variance >= 0))

        # Batch dimension
        raw_alpha = np.random.randn(10, 10)
        raw_beta = np.random.randn(10, 10)
        alpha, beta = self.interpreter._get_beta_distribution_params(raw_alpha, raw_beta)
        position_limits = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        target_mean, target_variance = self.interpreter._calculate_risk_aware_objective_params(alpha, beta, position_limits)

        self.assertEqual(target_mean.shape, raw_alpha.shape)
        self.assertEqual(target_variance.shape, raw_beta.shape)
        self.assertTrue(np.all(target_mean <= position_limits))
        self.assertTrue(np.all(target_variance >= 0))

    def test_determine_intents(self):
        """Test that the determine_intents method returns the correct intents."""
        # Intentions: 0 = Buy, 1 = Sell, 2 = Hold

        # No batch dimension
        direction_logits = np.array([[0.2, 0.4, 0.5], # Hold
                                   [0.2, 0.4, 0.5], # Hold
                                   [0.2, 0.4, 0.5], # Hold
                                   [0.1, 0.5, 0.2], # Sell
                                   [0.1, 0.5, 0.2], # Sell
                                   [0.1, 0.5, 0.2], # Sell
                                   [0.8, 0.6, 0.1], # Buy
                                   [0.8, 0.6, 0.1], # Buy
                                   [0.8, 0.6, 0.1], # Buy
                                   [0.5, 0.5, 0.5], # Hold
                                   [0.5, 0.5, 0.5], # Hold
                                   [0.5, 0.5, 0.5]]) # Hold
        target_mean = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        current_position = np.array([0, 10, 20, 0, 10, 20, 0, 10, 20, 0, 10, 20])
        intents, max_trade_size = self.interpreter._determine_intents(direction_logits, target_mean, current_position)

        self.assertEqual(intents.shape, (12,))
        self.assertEqual(max_trade_size.shape, (12,))
        
        ### Test the intents ###
        # All intentions of the first 3 are hold (no matter relationship between target and current position)
        self.assertTrue(np.all(intents[:3] == 2)) # Hold

        # The next 3 have selling intent, but relationship between target and current position is varying
        self.assertTrue(np.all(intents[3] == 2)) # Current < Target: Hold
        self.assertTrue(np.all(intents[4] == 2)) # Current = Target: Hold
        self.assertTrue(np.all(intents[5] == 1)) # Current > Target: Sell

        # The next 3 have buying intent, but relationship between target and current position is varying
        self.assertTrue(np.all(intents[6] == 0)) # Current < Target: Buy
        self.assertTrue(np.all(intents[7] == 2)) # Current = Target: Hold
        self.assertTrue(np.all(intents[8] == 2)) # Current > Target: Hold

        # The last 3 have identical intent logits, but relationship between target and current position is varying
        # Default is to buy, as long as desired change is positive
        self.assertTrue(np.all(intents[9] == 0)) # Current < Target: Buy
        self.assertTrue(np.all(intents[10] == 2)) # Current = Target: Hold
        self.assertTrue(np.all(intents[11] == 2)) # Current > Target: Hold

        ### Test the max trade size ###
        # Calculate masks
        buy_mask = intents == 0
        sell_mask = intents == 1
        hold_mask = intents == 2

        # Test the max trade size
        self.assertTrue(np.all(max_trade_size[buy_mask] == target_mean[buy_mask] - current_position[buy_mask]))
        self.assertTrue(np.all(max_trade_size[sell_mask] == current_position[sell_mask] - target_mean[sell_mask]))
        self.assertTrue(np.all(max_trade_size[hold_mask] == 0))

        # Test that all max trade sizes are non-negative
        self.assertTrue(np.all(max_trade_size >= 0))

    def test_determine_intents_batch(self):
        """Test that the determine_intents method returns the correct intents for a batch of data."""
        # Intentions: 0 = Buy, 1 = Sell, 2 = Hold

        # With batch dimension (batch size = 2)
        direction_logits = np.array([[
            [0.2, 0.4, 0.5], # Hold
            [0.2, 0.4, 0.5], # Hold
            [0.2, 0.4, 0.5], # Hold
            [0.1, 0.5, 0.2], # Sell
            [0.1, 0.5, 0.2], # Sell
            [0.1, 0.5, 0.2], # Sell
            [0.8, 0.6, 0.1], # Buy
            [0.8, 0.6, 0.1], # Buy
            [0.8, 0.6, 0.1], # Buy
            [0.5, 0.5, 0.5], # Equal
            [0.5, 0.5, 0.5], # Equal
            [0.5, 0.5, 0.5]  # Equal
            ], [
            [0.2, 0.4, 0.5], # Hold
            [0.2, 0.4, 0.5], # Hold
            [0.2, 0.4, 0.5], # Hold
            [0.1, 0.5, 0.2], # Sell
            [0.1, 0.5, 0.2], # Sell
            [0.1, 0.5, 0.2], # Sell
            [0.8, 0.6, 0.1], # Buy
            [0.8, 0.6, 0.1], # Buy
            [0.8, 0.6, 0.1], # Buy
            [0.5, 0.5, 0.5], # Equal
            [0.5, 0.5, 0.5], # Equal
            [0.5, 0.5, 0.5]  # Equal
            ]])
        target_mean = np.array([[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                              [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]])
        current_position = np.array([[0, 10, 20, 0, 10, 20, 0, 10, 20, 0, 10, 20],
                                   [0, 10, 20, 0, 10, 20, 0, 10, 20, 0, 10, 20]])
        intents, max_trade_size = self.interpreter._determine_intents(direction_logits, target_mean, current_position)

        self.assertEqual(intents.shape, (2, 12))
        self.assertEqual(max_trade_size.shape, (2, 12))

        # All intentions of the first 3 are hold (no matter relationship between target and current position)
        self.assertTrue(np.all(intents[:, :3] == 2)) # Hold

        # The next 3 have selling intent, but relationship between target and current position is varying
        self.assertTrue(np.all(intents[:, 3] == 2)) # Current < Target: Hold
        self.assertTrue(np.all(intents[:, 4] == 2)) # Current = Target: Hold
        self.assertTrue(np.all(intents[:, 5] == 1)) # Current > Target: Sell

        # The next 3 have buying intent, but relationship between target and current position is varying
        self.assertTrue(np.all(intents[:, 6] == 0)) # Current < Target: Buy
        self.assertTrue(np.all(intents[:, 7] == 2)) # Current = Target: Hold
        self.assertTrue(np.all(intents[:, 8] == 2)) # Current > Target: Hold

        # The last 3 have identical intent logits, but relationship between target and current position is varying
        # Default is to buy, as long as desired change is positive
        self.assertTrue(np.all(intents[:, 9] == 0)) # Current < Target: Buy
        self.assertTrue(np.all(intents[:, 10] == 2)) # Current = Target: Hold
        self.assertTrue(np.all(intents[:, 11] == 2)) # Current > Target: Hold

        ### Test the max trade size ###
        # Calculate masks
        buy_mask = intents == 0
        sell_mask = intents == 1
        hold_mask = intents == 2

        # Test the max trade size
        self.assertTrue(np.all(max_trade_size[buy_mask] == target_mean[buy_mask] - current_position[buy_mask]))
        self.assertTrue(np.all(max_trade_size[sell_mask] == current_position[sell_mask] - target_mean[sell_mask]))
        self.assertTrue(np.all(max_trade_size[hold_mask] == 0))

        # Test that all max trade sizes are non-negative
        self.assertTrue(np.all(max_trade_size >= 0))

if __name__ == "__main__":
    unittest.main()

