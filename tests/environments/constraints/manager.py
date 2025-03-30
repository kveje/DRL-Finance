"""Unit tests for the ConstraintManager class."""

import unittest
from typing import Dict, List
import sys
import os
import numpy as np

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the function to test
from environments.constraints.manager import ConstraintManager
from environments.constraints.position_limits import PositionLimits
from environments.constraints.cash_limit import CashLimit


class TestConstraintManager(unittest.TestCase):
    """Test cases for the ConstraintManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "position_limits": {
                "min": 0,
                "max": 10000
            },
            "cash_limit": {
                "min": 0,
                "max": np.inf
            }
        }

    def test_constraint_manager_initialization(self):
        """Test initialization of ConstraintManager."""
        constraint_manager = ConstraintManager(self.config)
        self.assertEqual(len(constraint_manager.constraints), 2)
        self.assertIsInstance(constraint_manager.constraints["position_limits"], PositionLimits)
        self.assertIsInstance(constraint_manager.constraints["cash_limit"], CashLimit)

    def test_constraint_manager_check_constraints(self):
        """Test check_constraints method of ConstraintManager."""
        constraint_manager = ConstraintManager(self.config)
        # Case 1: Action is within the constraints
        action = np.array([10, -5])
        current_positions = np.array([10, 10])
        current_cash = 100
        current_prices = np.array([10, 10])
        self.assertTrue(constraint_manager.check_constraints(action, current_positions, current_cash, current_prices))

        # Case 2: Action violates the constraints (cash limit)
        action = np.array([1000, -5])
        current_positions = np.array([10, 5])
        current_cash = 100
        current_prices = np.array([10, 10])
        expected_cash = 100 + np.sum(-1 * action * current_prices)
        self.assertFalse(constraint_manager.check_constraints(action, current_positions, current_cash, current_prices))
        details = constraint_manager.constraints["cash_limit"].get_violation_details()
        self.assertEqual(details["constraint"], "cash_limit")
        self.assertEqual(details["violation_details"]["cash"], expected_cash)
        self.assertEqual(details["violation_details"]["min_cash"], 0)

        # Case 3: Action violates the constraints (position limits)
        action = np.array([1000, -50])
        current_positions = np.array([10, 5])
        current_cash = 1000000
        current_prices = np.array([10, 10])
        self.assertFalse(constraint_manager.check_constraints(action, current_positions, current_cash, current_prices))
        details = constraint_manager.constraints["position_limits"].get_violation_details()
        self.assertEqual(details["constraint"], "position_limits")
        self.assertEqual(details["violation_details"]["position"], 5-50)
        self.assertEqual(details["violation_details"]["min_position"], 0)

        



if __name__ == "__main__":
    unittest.main()

