"""Unit tests for the Returns Based Reward class."""

import unittest
from typing import Dict, List
import sys
import os
import numpy as np

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the function to test
from environments.rewards.returns_based import ReturnsBasedReward

class TestReturnsBasedReward(unittest.TestCase):
    """Test cases for the Returns Based Reward class."""

    def setUp(self):
        self.config = {
            "scale": 1.0
        }
        self.reward = ReturnsBasedReward(self.config)

    def test_initialization(self):
        self.assertEqual(self.reward.name, "returns_based")
        self.assertEqual(self.reward.scale, self.config["scale"])

    def test_zero_return(self):
        portfolio_value = 100.0
        previous_portfolio_value = 100.0
        
        reward = self.reward.calculate(portfolio_value, previous_portfolio_value)
        self.assertEqual(reward, 0.0)

    def test_positive_return(self):
        portfolio_value = 110.0
        previous_portfolio_value = 100.0
        
        reward = self.reward.calculate(portfolio_value, previous_portfolio_value)
        self.assertEqual(reward, 0.1)

    def test_negative_return(self):
        portfolio_value = 90.0
        previous_portfolio_value = 100.0
        
        reward = self.reward.calculate(portfolio_value, previous_portfolio_value)
        self.assertEqual(reward, -0.1)
    
    def test_asymptotic_return(self):
        portfolio_value = 100.0
        expected_reward = 0.01
        n = 100

        for _ in range(n):
            # Update portfolio value
            previous_portfolio_value = portfolio_value
            portfolio_value = portfolio_value * (1 + expected_reward)

            # Calculate reward
            reward = self.reward.calculate(portfolio_value, previous_portfolio_value)

            # Assert that the reward is close to the expected reward
            self.assertAlmostEqual(reward, expected_reward, delta=0.001)

    def test_scale(self):
        config = {
            "scale": 2.0    
        }
        reward_class = ReturnsBasedReward(config)
        portfolio_value = 100.0
        previous_portfolio_value = 100.0
        
        reward = reward_class.calculate(portfolio_value, previous_portfolio_value)
        self.assertEqual(reward, 0.0)

        portfolio_value = 110.0
        previous_portfolio_value = 100.0
        
        reward = reward_class.calculate(portfolio_value, previous_portfolio_value)
        self.assertEqual(reward, 0.2)

        portfolio_value = 90.0
        previous_portfolio_value = 100.0
        
        reward = reward_class.calculate(portfolio_value, previous_portfolio_value)
        self.assertEqual(reward, -0.2)

if __name__ == "__main__":
    unittest.main()
