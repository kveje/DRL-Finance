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
        config = {"min": -100, "max": 100}
        self.position_limits = PositionLimits(config)

    def test_position_limits_initialization(self):
        """Test initialization of PositionLimits."""
        self.assertEqual(self.position_limits.min_position, -100)
        self.assertEqual(self.position_limits.max_position, 100)
        self.assertEqual(self.position_limits.name, "position_limits")
        self.assertEqual(self.position_limits.violated, False)
        self.assertEqual(self.position_limits.violation_message, "")

    def test_position_limits_check(self):
        """Test check method of PositionLimits."""
        cash = 100
        prices = np.array([10, 10])
        position = np.array([10, -5])
        action = np.array([10, -5])
        self.assertEqual(self.position_limits.check(action, position, cash, prices), True)

        position = np.array([10, 5])
        action = np.array([-10, 5])
        self.assertEqual(self.position_limits.check(action, position, cash, prices), True)
        
        # Edge case: position at limits
        position = np.array([99, -99])
        action = np.array([1, -1])
        self.assertEqual(self.position_limits.check(action, position, cash, prices), True)

        # Edge case: position at limits
        position = np.array([-99, 99])
        action = np.array([-1, 1])
        self.assertEqual(self.position_limits.check(action, position, cash, prices), True)

        # False edge case: position at limits
        position = np.array([99, -99])
        action = np.array([2, -2])
        self.assertEqual(self.position_limits.check(action, position, cash, prices), False)

        # False edge case: position at limits
        position = np.array([-99, 99])
        action = np.array([-2, 2])
        self.assertEqual(self.position_limits.check(action, position, cash, prices), False)

    def test_position_limits_update(self):
        """Test update method of PositionLimits."""
        self.position_limits.update_limits((-200, 200))
        self.assertEqual(self.position_limits.min_position, -200)
        self.assertEqual(self.position_limits.max_position, 200)

    def test_position_limits_str(self):
        """Test str method of PositionLimits."""
        self.assertEqual(str(self.position_limits), "PositionLimits(limits=-100, 100)")

    def test_position_limits_repr(self):
        """Test repr method of PositionLimits."""
        self.assertEqual(repr(self.position_limits), "PositionLimits(limits=-100, 100)")
    

if __name__ == "__main__":
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    unittest.main()