"""Unit tests for the Sharpe Based Reward class."""

import unittest
from typing import Dict, List
import sys
import os
import numpy as np

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the function to test
from environments.rewards.sharpe_based import SharpeBasedReward

class TestSharpeBasedReward(unittest.TestCase):
    """Test cases for the Sharpe Based Reward class."""

    def setUp(self):
        self.config = {
            "annual_risk_free_rate": 0.01,
            "annualization_factor": 252,
            "window_size": 20,
            "min_history_size": 10,
            "scale": 1.0
        }
        self.daily_risk_free_rate = (1 + self.config["annual_risk_free_rate"]) ** (1/self.config["annualization_factor"]) - 1
        

    def test_initialization(self):
        reward_class = SharpeBasedReward(self.config)
        self.assertEqual(reward_class.annual_risk_free_rate, self.config["annual_risk_free_rate"])
        self.assertEqual(reward_class.annualization_factor, self.config["annualization_factor"])
        self.assertEqual(reward_class.window_size, self.config["window_size"])
        self.assertEqual(reward_class.min_history_size, self.config["min_history_size"])
        self.assertEqual(reward_class.scale, self.config["scale"])

    def test_zero_return(self):
        reward_class = SharpeBasedReward(self.config)
        portfolio_value = 100.0
        previous_portfolio_value = 100.0
        
        reward = reward_class.calculate(portfolio_value, previous_portfolio_value)
        expected_reward = 0 - self.daily_risk_free_rate * np.sqrt(self.config["annualization_factor"])
        self.assertEqual(reward, expected_reward)

    def test_zero_excess_return(self):
        reward_class = SharpeBasedReward(self.config)
        portfolio_value = 100 * (1 + self.daily_risk_free_rate)
        previous_portfolio_value = 100.0

        reward = reward_class.calculate(portfolio_value, previous_portfolio_value)
        self.assertAlmostEqual(reward, 0.0)

    def test_positive_excess_return(self):
        reward_class = SharpeBasedReward(self.config)
        portfolio_value = 100 * (1 + self.daily_risk_free_rate + 0.01)
        previous_portfolio_value = 100.0

        reward = reward_class.calculate(portfolio_value, previous_portfolio_value)
        self.assertGreater(reward, 0.0)

    def test_negative_excess_return(self):
        reward_class = SharpeBasedReward(self.config)
        portfolio_value = 100 * (1 + self.daily_risk_free_rate - 0.01)
        previous_portfolio_value = 100.0

        reward = reward_class.calculate(portfolio_value, previous_portfolio_value)
        self.assertLess(reward, 0.0)

    def test_scale(self):
        config = {
            "annual_risk_free_rate": 0.01,
            "annualization_factor": 252,
            "window_size": 20,
            "min_history_size": 10,
            "scale": 2.0
        }
        reward_class = SharpeBasedReward(config)
        portfolio_value = 102.0
        previous_portfolio_value = 100.0

        reward = reward_class.calculate(portfolio_value, previous_portfolio_value)
        expected_reward = 2 * (0.02 - self.daily_risk_free_rate) * np.sqrt(self.config["annualization_factor"])
        self.assertAlmostEqual(reward, expected_reward)

    def test_sequence_of_returns(self):
        config = {
            "annual_risk_free_rate": 0.01,
            "annualization_factor": 252,
            "window_size": 20,
            "min_history_size": 10,
            "scale": 1.0
        }
        n = 100
        reward_class = SharpeBasedReward(config)
        portfolio_value = 100.0
        previous_portfolio_value = 100.0
        percentage_return = np.random.uniform(-0.02, 0.02, n)

        # Initialize the returns history
        returns_history = []
        
        # Calculate the reward for each period
        for i in range(n):
            # Calculate the portfolio value
            portfolio_value = previous_portfolio_value * (1 + percentage_return[i])

            # Update the returns history
            returns_history.append(percentage_return[i])
            if len(returns_history) > self.config["window_size"]:
                returns_history.pop(0)

            # Calculate the reward
            reward = reward_class.calculate(portfolio_value, previous_portfolio_value)

            # Case 1: Only one period
            if i == 0:
                expected_reward = (percentage_return[i] - self.daily_risk_free_rate) * np.sqrt(self.config["annualization_factor"])
            # Case 2: More than one period, fewer than min_history_size returns
            elif i+1 < self.config["min_history_size"]:
                expected_reward = (percentage_return[i] - self.daily_risk_free_rate) / (np.std(returns_history) + 1e-9) * np.sqrt(self.config["annualization_factor"])
            # Case 3: More than one period, more than min_history_size returns
            else:
                returns_array = np.array(returns_history)
                excess_returns = returns_array - self.daily_risk_free_rate
                sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-9) * np.sqrt(self.config["annualization_factor"])
                expected_reward = sharpe_ratio

            # Assert the reward is as expected
            self.assertAlmostEqual(reward, expected_reward)

            # Update the previous portfolio value
            previous_portfolio_value = portfolio_value

if __name__ == "__main__":
    unittest.main()
