"""Unit tests for the ActionValidator class."""

import unittest
import numpy as np

import sys
import os

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from environments.constraints.action_validator import ActionValidator
from environments.constraints.manager import ConstraintManager

class TestActionValidator(unittest.TestCase):
    """Test cases for the ActionValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "position_limits": {
                "min": 0,
                "max": 100
            },
            "cash_limit": {
                "min": 1000,
                "max": 100000
            }
        }
        self.constraint_manager = ConstraintManager(self.config)
        self.action_validator = ActionValidator(self.constraint_manager)

    def test_position_limits(self):
        """Test position limit constraints."""
        # Test case 1: Action within limits
        action = np.array([10, -5])
        current_positions = np.array([0, 0])
        current_cash = 10000
        current_prices = np.array([10, 10])

        feasible_action, violation_info = self.action_validator.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        expected_action = np.array([10, 0])  # Negative actions should be clipped to 0
        self.assertTrue(np.array_equal(feasible_action, expected_action))
        self.assertNotEqual(violation_info, {})
        self.assertTrue("position_limits" in violation_info)

        # Test case 2: Action exceeding position limits
        action = np.array([150, 0])
        current_positions = np.array([0, 0])

        feasible_action, violation_info = self.action_validator.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        expected_action = np.array([100, 0])  # Should be clipped to max position
        self.assertTrue(np.array_equal(feasible_action, expected_action))
        self.assertNotEqual(violation_info, {})
        self.assertTrue("position_limits" in violation_info)

    def test_cash_constraints(self):
        """Test cash constraints."""
        # Test case 1: Action within cash limits
        action = np.array([10, 0])
        current_positions = np.array([0, 0])
        current_cash = 10000
        current_prices = np.array([10, 10])

        feasible_action, violation_info = self.action_validator.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        self.assertTrue(np.array_equal(feasible_action, action))
        self.assertEqual(violation_info, {})
        self.assertFalse("cash_limit" in violation_info)

        # Test case 2: Action exceeding cash limits
        action = np.array([1000, 0])  # Trying to buy 1000 shares at $10 each
        current_positions = np.array([0, 0])
        current_cash = 5000  # Only $5000 available
        current_prices = np.array([100, 100])

        feasible_action, violation_info = self.action_validator.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        # Should scale down the buy to fit within cash constraints
        # Available cash for trade = 5000 - 1000 = 4000
        # Scaling factor = 4000 / 10000 = 0.4
        # New buy amount = 1000 * 0.4 = 400
        expected_action = np.array([39, 0])
        self.assertTrue(np.array_equal(feasible_action, expected_action))
        self.assertNotEqual(violation_info, {})
        self.assertTrue("cash_limit" in violation_info)

    def test_combined_constraints(self):
        """Test when both position and cash constraints are violated."""
        action = np.array([1000, 0])  # Trying to buy 1000 shares
        current_positions = np.array([0, 0])
        current_cash = 5000  # Only $5000 available
        current_prices = np.array([100, 100])  # Higher price to ensure cash limit is violated

        feasible_action, violation_info = self.action_validator.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        # Should be limited by both constraints:
        # 1. Position limit: max 100 shares
        # 2. Cash limit: can only buy 39 shares with available cash (5000 - 1000 = 4000, 4000/100 = 39)
        # The more restrictive constraint (cash limit) should apply
        expected_action = np.array([39, 0])
        self.assertTrue(np.array_equal(feasible_action, expected_action))
        self.assertNotEqual(violation_info, {})
        self.assertTrue("position_limits" in violation_info)
        self.assertTrue("cash_limit" in violation_info)

    def test_get_constraint_parameters(self):
        """Test getting constraint parameters."""
        params = self.action_validator.get_constraint_parameters()
        self.assertEqual(params["position_limits"]["min"], 0)
        self.assertEqual(params["position_limits"]["max"], 100)
        self.assertEqual(params["cash_limit"]["min"], 1000)
        self.assertEqual(params["cash_limit"]["max"], 100000)

if __name__ == '__main__':
    unittest.main() 