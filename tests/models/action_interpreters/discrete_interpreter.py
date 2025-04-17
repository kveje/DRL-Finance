"""Unit tests for the DiscreteInterpreter class."""

import unittest
from typing import Dict, List
import sys
import os
import numpy as np
import pandas as pd
import torch

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the function to test
from models.action_interpreters.discrete_interpreter import DiscreteInterpreter
from environments.trading_env import TradingEnv

CONSTRAINT_PARAMS = {
    'position_limits': {
        'min': 0,
        'max': 100
    },
    'cash_limit': {
        'min': 0,
        'max': 100000 * 2
    }
}

class TestDiscreteInterpreter(unittest.TestCase):
    """Unit tests for the DiscreteInterpreter class."""

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
        self.env = TradingEnv(self.raw_data, self.processed_data, self.columns, constraint_params=CONSTRAINT_PARAMS)

        # Set up the interpreter with custom thresholds
        self.interpreter = DiscreteInterpreter(self.env)
        
        # Get position limits from environment
        self.position_limits = self.env.get_position_limits()
        self.min_position = self.position_limits["min"]
        self.max_position = self.position_limits["max"]
        
        # Get number of assets
        self.n_assets = self.env.get_action_dim()
        print(f"Number of assets: {self.n_assets}")

    def test_discrete_interpreter_initialization(self):
        """Test that the DiscreteInterpreter class initializes correctly."""
        self.assertIsNotNone(self.interpreter)
        self.assertEqual(self.interpreter.sell_threshold, -0.2)
        self.assertEqual(self.interpreter.buy_threshold, 0.2)
        self.assertEqual(self.interpreter.min_position, self.min_position)
        self.assertEqual(self.interpreter.max_position, self.max_position)

    def test_interpret(self):
        """Test that the interpret method correctly handles q_values in a dictionary."""
        # Create a test case with q_values in a dictionary
        model_output = {'q_values': np.array([-0.8, 0.0, 0.8])} # sell, hold, buy
        current_position = np.array([50.0, 50.0, 50.0])
        
        actions = self.interpreter.interpret(model_output["q_values"], current_position)
        print(f"Actions: {actions}")
        
        # Check shape
        self.assertEqual(actions.shape, current_position.shape)
        
        # Check actions: negative (sell), zero (hold), positive (buy)
        self.assertLess(actions[0], 0)  # Sell
        self.assertEqual(actions[1], 0)  # Hold
        self.assertGreater(actions[2], 0)  # Buy

        # Check that actions are integers
        self.assertTrue(np.all(actions == np.round(actions)))

    def test_interpret_sell_action(self):
        """Test the interpret method with sell actions."""
        # Create test values below sell threshold
        q_values = np.array([-1.0, -0.8, -0.6, -0.4, -0.3])
        current_position = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        
        actions = self.interpreter.interpret(q_values, current_position)
        
        # All should be sell actions (negative values)
        for i in range(len(actions)):
            self.assertLess(actions[i], 0)
            
        # Check that values closer to -1 result in selling more shares
        self.assertLess(actions[0], actions[1])  # -1.0 should sell more than -0.8
        self.assertLess(actions[1], actions[2])  # -0.8 should sell more than -0.6
        self.assertLess(actions[2], actions[3])  # -0.6 should sell more than -0.4
        self.assertLess(actions[3], actions[4])  # -0.4 should sell more than -0.3
        
        # Check extreme case: -1.0 should sell almost all shares
        self.assertAlmostEqual(actions[0], self.min_position - current_position[0], delta=1.0)

    def test_interpret_buy_action(self):
        """Test the interpret method with buy actions."""
        # Create test values above buy threshold
        q_values = np.array([0.3, 0.4, 0.6, 0.8, 1.0])
        current_position = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        
        actions = self.interpreter.interpret(q_values, current_position)
        
        # All should be buy actions (positive values)
        for i in range(len(actions)):
            self.assertGreater(actions[i], 0)
            
        # Check that values closer to 1 result in buying more shares
        self.assertGreater(actions[4], actions[3])  # 1.0 should buy more than 0.8
        self.assertGreater(actions[3], actions[2])  # 0.8 should buy more than 0.6
        self.assertGreater(actions[2], actions[1])  # 0.6 should buy more than 0.4
        self.assertGreater(actions[1], actions[0])  # 0.4 should buy more than 0.3
        
        # Check extreme case: 1.0 should buy up to max position
        self.assertAlmostEqual(actions[4], self.max_position - current_position[4], delta=1.0)

    def test_interpret_hold_action(self):
        """Test the interpret method with hold actions."""
        # Create test values in hold range
        q_values = np.array([-0.1, 0.0, 0.1])
        current_position = np.array([50.0, 50.0, 50.0])
        
        actions = self.interpreter.interpret(q_values, current_position)
        
        # All should be hold actions (zero values)
        for i in range(len(actions)):
            self.assertEqual(actions[i], 0)

    def test_interpret_with_different_current_positions(self):
        """Test the interpret method with different current positions."""
        # Create test values
        q_values = np.array([-0.8, 0.0, 0.8])  # Sell, Hold, Buy
        current_position = np.array([0.0, 50.0, 100.0])
        
        actions = self.interpreter.interpret(q_values, current_position)
        
        # Check that actions match expectations based on current position
        self.assertLessEqual(actions[0], 0)  # Sell (but already at min)
        self.assertEqual(actions[1], 0)      # Hold
        self.assertGreaterEqual(actions[2], 0)  # Buy (but already at max)
        
        # For position at 0, sell action should be 0 or close (can't sell what you don't have)
        self.assertAlmostEqual(actions[0], 0, delta=1.0)
        
        # For position at max, buy action should be 0 or close (can't buy beyond max)
        self.assertAlmostEqual(actions[2], 0, delta=1.0)
        
        # Test with middle positions
        q_values = np.array([-0.8, 0.0, 0.8])  # Sell, Hold, Buy
        current_position = np.array([25.0, 50.0, 75.0])
        
        actions = self.interpreter.interpret(q_values, current_position)
        
        # Check that actions match expectations
        self.assertLess(actions[0], 0)       # Sell
        self.assertEqual(actions[1], 0)      # Hold
        self.assertGreater(actions[2], 0)    # Buy


if __name__ == "__main__":
    unittest.main() 