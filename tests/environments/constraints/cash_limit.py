"""Unit tests for the CashLimit class."""

import unittest
import numpy as np
import sys
import os

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the function to test
from environments.constraints.cash_limit import CashLimit

class TestCashLimit(unittest.TestCase):
    """Test cases for the CashLimit class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "min": 1000,
            "max": 100000
        }
        self.cash_limit = CashLimit(self.config)

    def test_initialization(self):
        """Test initialization of CashLimit."""
        self.assertEqual(self.cash_limit.min_cash, 1000)
        self.assertEqual(self.cash_limit.max_cash, 100000)
        self.assertEqual(self.cash_limit.name, "cash_limit")
        self.assertFalse(self.cash_limit.is_violated())

    def test_valid_action(self):
        """Test when action is within cash limits."""
        action = np.array([10, -5])
        current_positions = np.array([0, 0])
        current_cash = 10000
        current_prices = np.array([10, 10])

        feasible_action, violation_info = self.cash_limit.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        self.assertTrue(np.array_equal(feasible_action, action))
        self.assertEqual(violation_info, {})
        self.assertFalse(self.cash_limit.is_violated())

    def test_cash_limit_violation(self):
        """Test when action would result in cash below minimum limit."""
        action = np.array([1000, 0])  # Trying to buy 1000 shares at $10 each
        current_positions = np.array([0, 0])
        current_cash = 5000  # Only $5000 available
        current_prices = np.array([10, 10])

        feasible_action, violation_info = self.cash_limit.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        # Should scale down the buy to fit within cash constraints
        # Available cash for trade = 5000 - 1000 = 4000
        # Scaling factor = 4000 / 10000 = 0.4
        # New buy amount = 1000 * 0.4 = 400
        expected_action = np.array([399, 0])
        self.assertTrue(np.array_equal(feasible_action, expected_action))
        self.assertIn("required_cash", violation_info)
        self.assertIn("available_cash", violation_info)
        self.assertEqual(violation_info["min_cash_limit"], 1000)
        self.assertTrue(self.cash_limit.is_violated())

    def test_multiple_assets_cash_limit(self):
        """Test cash limit with multiple assets."""
        action = np.array([100, 100])  # Trying to buy 100 shares of each
        current_positions = np.array([0, 0])
        current_cash = 5000  # Only $5000 available
        current_prices = np.array([10, 20])  # Different prices

        feasible_action, violation_info = self.cash_limit.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        # Total cost would be 100 * 10 + 100 * 20 = 3000
        # Available cash for trade = 5000 - 1000 = 4000
        # Scaling factor = 4000 / 3000 ≈ 1.33
        # Since we floor the result, we should get the original action
        self.assertTrue(np.array_equal(feasible_action, action))
        self.assertEqual(violation_info, {})
        self.assertFalse(self.cash_limit.is_violated())

    def test_edge_cases(self):
        """Test edge cases of cash limits."""
        # Test at exact minimum cash limit
        action = np.array([100, 0])
        current_positions = np.array([0, 0])
        current_cash = 2000  # Exactly minimum + cost of action
        current_prices = np.array([10, 10])

        feasible_action, violation_info = self.cash_limit.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        self.assertTrue(np.array_equal(feasible_action, action))
        self.assertEqual(violation_info, {})
        self.assertFalse(self.cash_limit.is_violated())

        # Test with very small scaling factor
        action = np.array([1000, 0])
        current_positions = np.array([0, 0])
        current_cash = 1001  # Just above minimum
        
        feasible_action, violation_info = self.cash_limit.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        # Should scale down to almost nothing
        self.assertTrue(np.array_equal(feasible_action, np.array([0, 0])))
        self.assertIn("required_cash", violation_info)
        self.assertTrue(self.cash_limit.is_violated())

    def test_sell_actions(self):
        """Test that sell actions don't trigger cash limit violations."""
        action = np.array([-1000, -1000])  # Large sell orders
        current_positions = np.array([1000, 1000])
        current_cash = 1000  # Very low cash
        current_prices = np.array([10, 10])

        feasible_action, violation_info = self.cash_limit.validate_and_adjust_action(
            action, current_positions, current_cash, current_prices
        )

        # Sells should be allowed as they increase cash
        self.assertTrue(np.array_equal(feasible_action, action))
        self.assertEqual(violation_info, {})
        self.assertFalse(self.cash_limit.is_violated())

    def test_get_parameters(self):
        """Test getting constraint parameters."""
        params = self.cash_limit.get_parameters()
        self.assertEqual(params["min"], 1000)
        self.assertEqual(params["max"], 100000)

if __name__ == '__main__':
    unittest.main()
