"""Unit tests for the RewardManager class."""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from environments.rewards.manager import RewardManager

class TestRewardManager(unittest.TestCase):
    """Test cases for the RewardManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.reward_params = {
            "returns": {"scale": 1.0},
            "sharpe_based": {"scale": 1.0},
            "log_returns": {"scale": 1.0},
            "constraint_violation": {"scale": 1.0},
            "zero_action": {"scale": 0.001, "window_size": 5}
        }
        self.manager = RewardManager(self.reward_params)

    def test_initialization(self):
        """Test initialization of RewardManager."""
        self.assertEqual(len(self.manager.rewards), 5)
        self.assertEqual(len(self.manager.reward_history), 0)
        self.assertTrue("returns" in self.manager.rewards)
        self.assertTrue("sharpe_based" in self.manager.rewards)
        self.assertTrue("log_returns" in self.manager.rewards)
        self.assertTrue("constraint_violation" in self.manager.rewards)
        self.assertTrue("zero_action" in self.manager.rewards)

    def test_calculate_reward(self):
        """Test reward calculation with all components."""
        portfolio_value = 110000.0
        previous_portfolio_value = 100000.0
        intended_action = np.array([100, -50])
        feasible_action = np.array([80, -40])

        reward = self.manager.calculate(
            portfolio_value=portfolio_value,
            previous_portfolio_value=previous_portfolio_value,
            intended_action=intended_action,
            feasible_action=feasible_action
        )

        # Check that reward is calculated
        self.assertIsInstance(reward, float)
        
        # Check that reward history is updated
        self.assertEqual(len(self.manager.reward_history), 5)
        for reward_type in self.reward_params.keys():
            self.assertEqual(len(self.manager.reward_history[reward_type]), 1)

    def test_reward_history_tracking(self):
        """Test that reward history is properly tracked."""
        # Calculate rewards for multiple steps
        for i in range(3):
            self.manager.calculate(
                portfolio_value=100000.0 + i * 1000,
                previous_portfolio_value=100000.0 + (i-1) * 1000,
                intended_action=np.array([100, -50]),
                feasible_action=np.array([80, -40])
            )

        # Check history length
        for reward_type in self.reward_params.keys():
            self.assertEqual(len(self.manager.reward_history[reward_type]), 3)

    def test_reward_statistics(self):
        """Test reward statistics calculation."""
        # Calculate rewards for multiple steps
        for i in range(3):
            self.manager.calculate(
                portfolio_value=100000.0 + i * 1000,
                previous_portfolio_value=100000.0 + (i-1) * 1000,
                intended_action=np.array([100, -50]),
                feasible_action=np.array([80, -40])
            )

        stats = self.manager.get_reward_statistics()
        
        # Check that statistics are calculated for each reward type
        self.assertEqual(len(stats), 5)
        for reward_type in self.reward_params.keys():
            self.assertIn(reward_type, stats)
            self.assertIn("mean", stats[reward_type])
            self.assertIn("std", stats[reward_type])
            self.assertIn("min", stats[reward_type])
            self.assertIn("max", stats[reward_type])
            self.assertIn("total", stats[reward_type])

    def test_reset(self):
        """Test resetting the reward manager."""
        # Calculate some rewards
        self.manager.calculate(
            portfolio_value=110000.0,
            previous_portfolio_value=100000.0,
            intended_action=np.array([100, -50]),
            feasible_action=np.array([80, -40])
        )

        # Reset
        self.manager.reset()

        # Check that history is cleared
        self.assertEqual(len(self.manager.reward_history), 0)

    def test_missing_actions(self):
        """Test reward calculation with missing actions."""
        # Test with None actions
        reward = self.manager.calculate(
            portfolio_value=110000.0,
            previous_portfolio_value=100000.0,
            intended_action=None,
            feasible_action=None
        )

        # Check that reward is still calculated
        self.assertIsInstance(reward, float)
        
        # Check that action-dependent rewards are zero
        self.assertEqual(self.manager.reward_history["constraint_violation"][-1], 0.0)
        self.assertEqual(self.manager.reward_history["zero_action"][-1], 0.0)

    def test_get_parameters(self):
        """Test getting reward parameters."""
        params = self.manager.get_parameters()
        self.assertEqual(len(params), 7)  # 5 reward types with their parameters
        self.assertIn("scale", params)
        self.assertIn("window_size", params)

if __name__ == '__main__':
    unittest.main()
