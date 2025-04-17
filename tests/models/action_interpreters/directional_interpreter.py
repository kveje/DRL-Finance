"""Unit tests for the DirectionalActionInterpreter class."""

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
from models.action_interpreters.directional_interpreter import DirectionalActionInterpreter
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

class TestDirectionalActionInterpreter(unittest.TestCase):
    """Unit tests for the DirectionalActionInterpreter class."""

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

        # Set up the interpreter
        self.interpreter = DirectionalActionInterpreter(self.env)
        
        # Get position limits from environment
        self.position_limits = self.env.get_position_limits()
        self.min_position = self.position_limits["min"]
        self.max_position = self.position_limits["max"]
        
        # Get number of assets
        self.n_assets = self.env.get_action_dim()
        print(f"Number of assets: {self.n_assets}")

    def test_directional_interpreter_initialization(self):
        """Test that the DirectionalActionInterpreter class initializes correctly."""
        self.assertIsNotNone(self.interpreter)
        self.assertEqual(self.interpreter.min_action_scale, 0.1)
        self.assertEqual(self.interpreter.min_position, self.min_position)
        self.assertEqual(self.interpreter.max_position, self.max_position)

    def test_softmax(self):
        """Test that the softmax function works correctly."""
        # Create sample logits for 3 assets
        logits = np.array([
            [1.0, 0.0, -1.0],  # First asset: sell=1, hold=0, buy=-1
            [0.0, 2.0, 0.0],   # Second asset: hold has highest logit
            [-1.0, 0.0, 1.0]   # Third asset: buy=1, hold=0, sell=-1
        ])
        
        # Apply softmax
        probs = self.interpreter._softmax(logits)
        
        # Check shape
        self.assertEqual(probs.shape, logits.shape)
        
        # Check that all probabilities sum to 1 for each asset
        for i in range(probs.shape[0]):
            self.assertAlmostEqual(np.sum(probs[i]), 1.0, places=5)
        
        # Check that the relative order of probabilities matches logits
        self.assertGreater(probs[0, 0], probs[0, 1])  # First asset: sell > hold
        self.assertGreater(probs[0, 1], probs[0, 2])  # First asset: hold > buy
        self.assertGreater(probs[1, 1], probs[1, 0])  # Second asset: hold > sell
        self.assertGreater(probs[1, 1], probs[1, 2])  # Second asset: hold > buy
        self.assertGreater(probs[2, 2], probs[2, 1])  # Third asset: buy > hold
        self.assertGreater(probs[2, 1], probs[2, 0])  # Third asset: hold > sell

    def test_interpret_deterministic(self):
        """Test the deterministic interpretation of direction logits."""
        # Create sample logits with clear directions
        direction_logits = np.array([
            [5.0, 0.0, -5.0],  # First asset: strongly sell
            [0.0, 5.0, 0.0],   # Second asset: strongly hold
            [-5.0, 0.0, 5.0]   # Third asset: strongly buy
        ])
        
        # Current positions at midpoint
        current_position = np.array([50.0, 50.0, 50.0])
        
        # Interpret actions deterministically
        actions = self.interpreter.interpret(direction_logits, current_position, deterministic=True)
        
        # Check shape
        self.assertEqual(actions.shape, current_position.shape)
        
        # First asset should sell (negative action)
        self.assertLess(actions[0], 0)
        
        # Second asset should hold (zero action)
        self.assertEqual(actions[1], 0)
        
        # Third asset should buy (positive action)
        self.assertGreater(actions[2], 0)
        
        # Check that actions are integers
        self.assertTrue(np.all(actions == np.round(actions)))

    def test_interpret_nondeterministic(self):
        """Test the non-deterministic interpretation of direction logits."""
        # Set a seed for reproducibility
        np.random.seed(42)
        
        # Create sample logits with clear preferences
        direction_logits = np.array([
            [10.0, 0.0, 0.0],  # First asset: strongly sell
            [0.0, 10.0, 0.0],  # Second asset: strongly hold
            [0.0, 0.0, 10.0]   # Third asset: strongly buy
        ])
        
        # Current positions at midpoint
        current_position = np.array([50.0, 50.0, 50.0])
        
        # Interpret actions non-deterministically with fixed seed
        # With very strong preferences, the sampling should still mostly follow the preferences
        for _ in range(5):  # Run a few times to ensure consistency
            actions = self.interpreter.interpret(direction_logits, current_position, deterministic=False)
            
            # Check shape
            self.assertEqual(actions.shape, current_position.shape)
            
            # With very high logits, the actions should almost always follow the intended direction
            self.assertLessEqual(actions[0], 0)  # First asset should sell or hold
            self.assertGreaterEqual(actions[2], 0)  # Third asset should buy or hold
            
            # Check that actions are integers
            self.assertTrue(np.all(actions == np.round(actions)))

    def test_action_scaling_by_confidence(self):
        """Test that actions are properly scaled by confidence."""
        # Create logits with varying confidence
        direction_logits = np.array([
            [1.0, 0.0, 0.0],   # Low confidence sell
            [10.0, 0.0, 0.0],  # High confidence sell
            [0.0, 0.0, 1.0],   # Low confidence buy
            [0.0, 0.0, 10.0]   # High confidence buy
        ])
        
        # Current positions at midpoint
        current_position = np.array([50.0, 50.0, 50.0, 50.0])
        
        # Interpret actions
        actions = self.interpreter.interpret(direction_logits, current_position, deterministic=True)
        
        # Check that higher confidence results in stronger actions
        self.assertLess(actions[0], 0)     # Low confidence sell: negative but small
        self.assertLess(actions[1], 0)     # High confidence sell: more negative
        self.assertLess(actions[1], actions[0])  # High confidence sell should sell more
        
        self.assertGreater(actions[2], 0)  # Low confidence buy: positive but small
        self.assertGreater(actions[3], 0)  # High confidence buy: more positive
        self.assertGreater(actions[3], actions[2])  # High confidence buy should buy more

    def test_interpret_with_different_positions(self):
        """Test interpretation with different current positions."""
        # Create logits with clear directions
        direction_logits = np.array([
            [5.0, 0.0, 0.0],   # Sell
            [0.0, 5.0, 0.0],   # Hold
            [0.0, 0.0, 5.0]    # Buy
        ])
        
        # Test with positions at min, mid, and max
        current_position = np.array([0.0, 50.0, 100.0])
        
        # Interpret actions
        actions = self.interpreter.interpret(direction_logits, current_position, deterministic=True)
        
        # Check that actions respect position limits
        self.assertEqual(actions[0], 0)  # Can't sell from min position
        self.assertEqual(actions[1], 0)  # Hold remains 0
        self.assertEqual(actions[2], 0)  # Can't buy from max position
        
        # Test with positions in between
        current_position = np.array([25.0, 50.0, 75.0])
        
        # Interpret actions
        actions = self.interpreter.interpret(direction_logits, current_position, deterministic=True)
        
        # Check expected behavior
        self.assertLess(actions[0], 0)     # Can sell from 25
        self.assertEqual(actions[1], 0)    # Hold remains 0
        self.assertGreater(actions[2], 0)  # Can buy from 75

    def test_interpret_with_log_prob(self):
        """Test interpret_with_log_prob method."""
        # Set a seed for reproducibility
        np.random.seed(42)
        
        # Create logits with clear directions
        direction_logits = np.array([
            [5.0, 0.0, 0.0],   # Sell preference
            [0.0, 5.0, 0.0],   # Hold preference
            [0.0, 0.0, 5.0]    # Buy preference
        ])
        
        # Current positions at midpoint
        current_position = np.array([50.0, 50.0, 50.0])
        
        # Interpret actions with log probs
        actions, log_probs = self.interpreter.interpret_with_log_prob(direction_logits, current_position)
        
        # Check shapes
        self.assertEqual(actions.shape, current_position.shape)
        self.assertEqual(log_probs.shape, (current_position.shape[0],))
        
        # Log probs should be negative (or zero for probability 1)
        self.assertTrue(np.all(log_probs <= 0))
        
        # With high logits, the log probs should be close to 0 (probability close to 1)
        for log_prob in log_probs:
            self.assertGreaterEqual(log_prob, -0.1)  # Very close to 0 with high logits

if __name__ == "__main__":
    unittest.main() 