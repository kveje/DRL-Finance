"""Unit tests for the LogReturnsReward class."""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from environments.rewards.log_returns import LogReturnsReward

class TestLogReturnsReward(unittest.TestCase):
    """Test cases for the LogReturnsReward class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {"scale": 1.0}
        self.reward = LogReturnsReward(self.config)

    def test_initialization(self):
        """Test initialization of LogReturnsReward."""
        self.assertEqual(self.reward.scale, 1.0)

    def test_positive_return(self):
        """Test reward calculation for positive return."""
        portfolio_value = 110000.0
        previous_portfolio_value = 100000.0
        expected_return = np.log(portfolio_value / previous_portfolio_value)
        expected_reward = expected_return * self.reward.scale

        reward = self.reward.calculate(portfolio_value, previous_portfolio_value)
        self.assertAlmostEqual(reward, expected_reward)

    def test_negative_return(self):
        """Test reward calculation for negative return."""
        portfolio_value = 90000.0
        previous_portfolio_value = 100000.0
        expected_return = np.log(portfolio_value / previous_portfolio_value)
        expected_reward = expected_return * self.reward.scale

        reward = self.reward.calculate(portfolio_value, previous_portfolio_value)
        self.assertAlmostEqual(reward, expected_reward)

    def test_zero_return(self):
        """Test reward calculation for zero return."""
        portfolio_value = 100000.0
        previous_portfolio_value = 100000.0
        expected_reward = 0.0

        reward = self.reward.calculate(portfolio_value, previous_portfolio_value)
        self.assertAlmostEqual(reward, expected_reward)

    def test_small_portfolio_value(self):
        """Test reward calculation with very small portfolio value."""
        portfolio_value = 1e-6
        previous_portfolio_value = 1e-6
        expected_reward = 0.0

        reward = self.reward.calculate(portfolio_value, previous_portfolio_value)
        self.assertAlmostEqual(reward, expected_reward)

    def test_asymptotic_return(self):
        """Test reward calculation for small returns."""
        portfolio_value = 100000.0
        expected_return = 0.01
        n = 100

        for _ in range(n):
            # Update portfolio value
            previous_portfolio_value = portfolio_value
            portfolio_value = portfolio_value * (1 + expected_return)

            # Calculate reward
            reward = self.reward.calculate(portfolio_value, previous_portfolio_value)

            # For small returns, log(1 + r) ≈ r
            self.assertAlmostEqual(reward, expected_return, delta=0.001)

    def test_different_scale(self):
        """Test reward calculation with different scale factor."""
        config = {"scale": 2.0}
        reward = LogReturnsReward(config)
        
        portfolio_value = 110000.0
        previous_portfolio_value = 100000.0
        expected_return = np.log(portfolio_value / previous_portfolio_value)
        expected_reward = expected_return * reward.scale

        calculated_reward = reward.calculate(portfolio_value, previous_portfolio_value)
        self.assertAlmostEqual(calculated_reward, expected_reward)

    def test_get_parameters(self):
        """Test getting reward parameters."""
        params = self.reward.get_parameters()
        self.assertEqual(params["scale"], 1.0)

if __name__ == '__main__':
    unittest.main() 