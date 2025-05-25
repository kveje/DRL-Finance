"""Unit tests for the ZeroActionReward class."""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from environments.rewards.zero_action import ZeroActionReward

class TestZeroActionReward(unittest.TestCase):
    """Test cases for the ZeroActionReward class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "scale": 0.001,
            "window_size": 5,
            "min_consecutive_days": 3
        }
        self.reward = ZeroActionReward(self.config)

    def test_initialization(self):
        """Test initialization of ZeroActionReward."""
        self.assertEqual(self.reward.scale, 0.001)
        self.assertEqual(self.reward.window_size, 5)
        self.assertEqual(self.reward.min_consecutive_days, 3)
        self.assertEqual(self.reward.not_clipped_zero_action_history, [])

    def test_clipped_zero_action(self):
        """Test when there is a clipped zero action (should not be tracked)."""
        intended_action = np.array([100, -50])  # Non-zero intended
        feasible_action = np.array([0, 0])     # Zero feasible (clipped)
        reward = self.reward.calculate(intended_action, feasible_action)
        self.assertEqual(reward, 0.0)  # No penalty for clipped zero action
        self.assertEqual(self.reward.not_clipped_zero_action_history, [False])  # Not tracked as it was clipped

    def test_consecutive_non_clipped_zero_actions(self):
        """Test when there are consecutive non-clipped zero actions."""
        # First non-clipped zero action
        intended_action = np.array([0, 0])     # Zero intended
        feasible_action = np.array([0, 0])     # Zero feasible (not clipped)
        reward = self.reward.calculate(intended_action, feasible_action)
        self.assertEqual(reward, 0.0)  # No penalty for first non-clipped zero action
        self.assertEqual(self.reward.not_clipped_zero_action_history, [True])  # Tracked as it was not clipped

        # Second non-clipped zero action
        reward = self.reward.calculate(intended_action, feasible_action)
        self.assertEqual(reward, 0.0)  # No penalty yet as we need min_consecutive_days
        self.assertEqual(self.reward.not_clipped_zero_action_history, [True, True])

        # Third non-clipped zero action (minimum required)
        reward = self.reward.calculate(intended_action, feasible_action)
        self.assertEqual(reward, -0.003)  # Penalty for three consecutive non-clipped zeros
        self.assertEqual(self.reward.not_clipped_zero_action_history, [True, True, True])

        # Fourth non-clipped zero action
        reward = self.reward.calculate(intended_action, feasible_action)
        self.assertEqual(reward, -0.004)  # Penalty for four consecutive non-clipped zeros
        self.assertEqual(self.reward.not_clipped_zero_action_history, [True, True, True, True])

    def test_non_zero_action_resets(self):
        """Test that non-zero action resets the consecutive non-clipped zero count."""
        # Three non-clipped zero actions (minimum required)
        intended_action = np.array([0, 0])     # Zero intended
        feasible_action = np.array([0, 0])     # Zero feasible (not clipped)
        self.reward.calculate(intended_action, feasible_action)
        self.reward.calculate(intended_action, feasible_action)
        self.reward.calculate(intended_action, feasible_action)

        # Non-zero action (both intended and feasible)
        intended_action = np.array([1, 0])
        feasible_action = np.array([1, 0])
        reward = self.reward.calculate(intended_action, feasible_action)
        self.assertEqual(reward, 0.0)  # No penalty after non-zero action
        self.assertEqual(self.reward.not_clipped_zero_action_history, [True, True, True, False])  # Reset by non-zero action

        # Another non-clipped zero action
        intended_action = np.array([0, 0])     # Zero intended
        feasible_action = np.array([0, 0])     # Zero feasible (not clipped)
        reward = self.reward.calculate(intended_action, feasible_action)
        self.assertEqual(reward, 0.0)  # No penalty as we need min_consecutive_days again
        self.assertEqual(self.reward.not_clipped_zero_action_history, [True, True, True, False, True])

    def test_window_size_limit(self):
        """Test that the window size limits the history."""
        intended_action = np.array([0, 0])     # Zero intended
        feasible_action = np.array([0, 0])     # Zero feasible (not clipped)
        
        # Fill the window with non-clipped zero actions
        for _ in range(6):  # One more than window_size
            self.reward.calculate(intended_action, feasible_action)
        
        self.assertEqual(len(self.reward.not_clipped_zero_action_history), 5)  # Should be limited to window_size
        self.assertEqual(self.reward.not_clipped_zero_action_history, [True] * 5)  # All should be True

    def test_mixed_actions(self):
        """Test mixing of clipped and non-clipped zero actions."""
        # Non-clipped zero action
        intended_action = np.array([0, 0])     # Zero intended
        feasible_action = np.array([0, 0])     # Zero feasible (not clipped)
        reward = self.reward.calculate(intended_action, feasible_action)
        self.assertEqual(reward, 0.0)  # No penalty for first non-clipped zero action
        self.assertEqual(self.reward.not_clipped_zero_action_history, [True])

        # Clipped zero action
        intended_action = np.array([100, -50])  # Non-zero intended
        feasible_action = np.array([0, 0])     # Zero feasible (clipped)
        reward = self.reward.calculate(intended_action, feasible_action)
        self.assertEqual(reward, 0.0)  # No penalty for clipped zero action
        self.assertEqual(self.reward.not_clipped_zero_action_history, [True, False])  # Not tracked as it was clipped

        # Another non-clipped zero action
        intended_action = np.array([0, 0])     # Zero intended
        feasible_action = np.array([0, 0])     # Zero feasible (not clipped)
        reward = self.reward.calculate(intended_action, feasible_action)
        self.assertEqual(reward, 0.0)  # No penalty as previous action was clipped
        self.assertEqual(self.reward.not_clipped_zero_action_history, [True, False, True])

    def test_custom_min_consecutive_days(self):
        """Test with a custom minimum consecutive days parameter."""
        config = {
            "scale": 0.001,
            "window_size": 5,
            "min_consecutive_days": 2  # Lower minimum for testing
        }
        reward = ZeroActionReward(config)

        # First non-clipped zero action
        intended_action = np.array([0, 0])     # Zero intended
        feasible_action = np.array([0, 0])     # Zero feasible (not clipped)
        reward_value = reward.calculate(intended_action, feasible_action)
        self.assertEqual(reward_value, 0.0)  # No penalty for first non-clipped zero action

        # Second non-clipped zero action (minimum required with custom config)
        reward_value = reward.calculate(intended_action, feasible_action)
        self.assertEqual(reward_value, -0.002)  # Penalty for two consecutive non-clipped zeros

    def test_reset(self):
        """Test resetting the reward."""
        intended_action = np.array([0, 0])     # Zero intended
        feasible_action = np.array([0, 0])     # Zero feasible (not clipped)
        self.reward.calculate(intended_action, feasible_action)
        self.reward.calculate(intended_action, feasible_action)
        self.reward.calculate(intended_action, feasible_action)
        
        self.reward.reset()
        self.assertEqual(self.reward.not_clipped_zero_action_history, [])
        
        # After reset, should start counting from zero again
        reward = self.reward.calculate(intended_action, feasible_action)
        self.assertEqual(reward, 0.0)
        self.assertEqual(self.reward.not_clipped_zero_action_history, [True])

    def test_get_parameters(self):
        """Test getting reward parameters."""
        params = self.reward.get_parameters()
        self.assertEqual(params["scale"], 0.001)
        self.assertEqual(params["window_size"], 5)
        self.assertEqual(params["min_consecutive_days"], 3)

if __name__ == '__main__':
    unittest.main() 