"""Unit tests for the CashLimit class."""

import unittest
from typing import Dict, List
import sys
import os
import numpy as np

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the function to test
from environments.constraints.cash_limit import CashLimit


class TestCashLimit(unittest.TestCase):
    """Test cases for the CashLimit class."""

    def setUp(self):
        """Set up test fixtures."""
        config = {"min": 0, "max": np.inf}
        self.cash_limit = CashLimit(config)

    def test_cash_limit_initialization(self):
        """Test initialization of CashLimit."""
        self.assertEqual(self.cash_limit.min_cash, 0)
        self.assertEqual(self.cash_limit.max_cash, np.inf)
        self.assertEqual(self.cash_limit.name, "cash_limit")
        self.assertEqual(self.cash_limit.violated, False)
        self.assertEqual(self.cash_limit.violation_message, "")

    def test_cash_limit_check(self):
        """Test check method of CashLimit."""
        # Test case 1: Action is within the cash limit
        action = np.array([10, -5])
        current_positions = np.array([10, -5])
        current_cash = 100
        current_prices = np.array([10, 10])
        self.assertEqual(self.cash_limit.check(action, current_positions, current_cash, current_prices), True)

        # Test case 2: Action is outside the cash limit
        action = np.array([1000, -5])
        current_positions = np.array([10, 5])
        current_cash = 100
        current_prices = np.array([10, 10])
        self.assertEqual(self.cash_limit.check(action, current_positions, current_cash, current_prices), False)

        # Test case 3: Action is on the limit
        action = np.array([1, 0])
        current_positions = np.array([1, 0])
        current_cash = 10
        current_prices = np.array([10, 10])
        self.assertEqual(self.cash_limit.check(action, current_positions, current_cash, current_prices), True)

        # Test case 4: Action is on the limit
        action = np.array([1, 0])
        current_positions = np.array([1, 0])
        current_cash = 9.999999999999999
        current_prices = np.array([10, 10])
        self.assertEqual(self.cash_limit.check(action, current_positions, current_cash, current_prices), False)

if __name__ == "__main__":
    unittest.main()
