"""Unit tests for the PositionLimits class."""

import unittest
from typing import Dict, List
import sys
import os
import numpy as np

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the function to test
from environments.constraints.position_limits import PositionLimits

class TestPositionLimits(unittest.TestCase):
    """Test cases for the PositionLimits class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "min": 0,
            "max": 100
        }
        self.position_limits = PositionLimits(self.config)

    def test_initialization(self):
        """Test initialization of PositionLimits."""
        self.assertEqual(self.position_limits.min_position, 0)
        self.assertEqual(self.position_limits.max_position, 100)
        self.assertEqual(self.position_limits.name, "position_limits")
        self.assertFalse(self.position_limits.is_violated())

    def test_valid_action(self):
        """Test when action is within position limits."""
        action = np.array([10, -5])
        current_positions = np.array([0, 0])
        current_cash = 10000
        current_prices = np.array([10, 10])

        feasible_action, violation_info = self.position_limits.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )
        expected_action = np.array([10, 0])
        self.assertTrue(np.array_equal(feasible_action, expected_action))
        self.assertNotEqual(violation_info, {})
        self.assertTrue(self.position_limits.is_violated())

    def test_min_position_violation(self):
        """Test when action would result in position below minimum limit."""
        action = np.array([-150, 0])  # Trying to sell 150 shares
        current_positions = np.array([0, 0])
        current_cash = 10000
        current_prices = np.array([10, 10])

        feasible_action, violation_info = self.position_limits.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        # Should be clipped to 0 (min_position)
        self.assertTrue(np.array_equal(feasible_action, np.array([0, 0])))
        self.assertIn("min_violations", violation_info)
        self.assertEqual(violation_info["min_limit"], 0)
        self.assertTrue(self.position_limits.is_violated())

    def test_max_position_violation(self):
        """Test when action would result in position above maximum limit."""
        action = np.array([150, 0])  # Trying to buy 150 shares
        current_positions = np.array([0, 0])
        current_cash = 10000
        current_prices = np.array([10, 10])

        feasible_action, violation_info = self.position_limits.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        # Should be clipped to 100 (max_position)
        self.assertTrue(np.array_equal(feasible_action, np.array([100, 0])))
        self.assertIn("max_violations", violation_info)
        self.assertEqual(violation_info["max_limit"], 100)
        self.assertTrue(self.position_limits.is_violated())

    def test_multiple_violations(self):
        """Test when multiple assets violate position limits."""
        action = np.array([150, -150])  # Trying to buy 150 and sell 150
        current_positions = np.array([0, 0])
        current_cash = 10000
        current_prices = np.array([10, 10])

        feasible_action, violation_info = self.position_limits.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        # Should be clipped to limits
        self.assertTrue(np.array_equal(feasible_action, np.array([100, 0])))
        self.assertIn("min_violations", violation_info)
        self.assertIn("max_violations", violation_info)
        self.assertTrue(self.position_limits.is_violated())

    def test_edge_cases(self):
        """Test edge cases of position limits."""
        # Test at exact limits
        action = np.array([100, 0])
        current_positions = np.array([0, 0])
        current_cash = 10000
        current_prices = np.array([10, 10])

        feasible_action, violation_info = self.position_limits.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        self.assertTrue(np.array_equal(feasible_action, action))
        self.assertEqual(violation_info, {})
        self.assertFalse(self.position_limits.is_violated())

        # Test with current positions at limits
        action = np.array([1, -1])
        current_positions = np.array([100, 0])
        
        feasible_action, violation_info = self.position_limits.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        # Should be clipped to 0 (can't exceed limits)
        self.assertTrue(np.array_equal(feasible_action, np.array([0, 0])))
        self.assertIn("max_violations", violation_info)
        self.assertIn("min_violations", violation_info)

    def test_update_limits(self):
        """Test updating position limits."""
        new_limits = (0, 200)
        self.position_limits.update_limits(new_limits)
        
        self.assertEqual(self.position_limits.min_position, 0)
        self.assertEqual(self.position_limits.max_position, 200)

        # Test with new limits
        action = np.array([150, -50])
        current_positions = np.array([0, 0])
        current_cash = 10000
        current_prices = np.array([10, 10])

        feasible_action, violation_info = self.position_limits.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        # Should now be valid with new limits
        expected_action = np.array([150, 0])
        self.assertTrue(np.array_equal(feasible_action, expected_action))
        self.assertNotEqual(violation_info, {})
        self.assertTrue(self.position_limits.is_violated())

    def test_get_parameters(self):
        """Test getting constraint parameters."""
        params = self.position_limits.get_parameters()
        self.assertEqual(params["min"], 0)
        self.assertEqual(params["max"], 100)

if __name__ == '__main__':
    unittest.main()