"""Unit tests for the Commission Frictionclass."""

import unittest
from typing import Dict, List
import sys
import os
import numpy as np

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the function to test
from environments.market_friction.commission import CommissionFriction

class TestCommission(unittest.TestCase):
    """Test cases for the Commission class."""

    def setUp(self):
        """Set up test fixtures."""
        config = {
            "commission_rate": 0.001
        }
        self.commission = CommissionFriction(config)

    def test_commission_initialization(self):
        """Test initialization of Commission."""
        self.assertEqual(self.commission.rate, 0.001)
        self.assertEqual(self.commission.name, "commission")

    def test_commission(self):
        """Test apply method of Commission."""
        action = np.array([1, -1])
        price = np.array([100.0, 100.0])
        new_price = self.commission.apply(action, price)
        self.assertEqual(new_price[0], 100.1)
        self.assertEqual(new_price[1], 99.9)

    def test_commission_zero_action(self):
        """Test apply method of Commission with zero action."""
        action = np.array([0, 0])
        price = np.array([100.0, 100.0])
        new_price = self.commission.apply(action, price)
        self.assertEqual(new_price[0], 100.0)
        self.assertEqual(new_price[1], 100.0)

    def test_commision_mixed_action(self):
        """Test apply method of Commission with mixed action."""
        action = np.array([0, -1])
        price = np.array([100.0, 100.0])
        new_price = self.commission.apply(action, price)
        self.assertEqual(new_price[0], 100.0)
        self.assertEqual(new_price[1], 99.9)

    def test_commission_str(self):
        """Test str method of Commission."""
        self.assertEqual(str(self.commission), "CommissionFriction(rate=0.001)")

    def test_commission_repr(self):
        """Test repr method of Commission."""
        self.assertEqual(repr(self.commission), "CommissionFriction(rate=0.001)")

    def test_asymptotic_commission(self):
        """Test asymptotic behavior of Commission."""
        n = 10000
        action = np.array([1, -1])
        price = np.array([100.0, 100.0])
        new_price = np.zeros(shape=(n, 2))

        for i in range(n):
            new_price[i,:] = self.commission.apply(action, price)

        mean_price = np.mean(new_price, axis=0)
        std_price = np.std(new_price, axis=0)

        expected_mean_price = np.array([100.0 + 1 * self.commission.rate, 100.0 - 1 * self.commission.rate])
        expected_std_price = np.array([0, 0])

        self.assertAlmostEqual(mean_price[0], expected_mean_price[0], delta=0.1)
        self.assertAlmostEqual(mean_price[1], expected_mean_price[1], delta=0.1)
        self.assertAlmostEqual(std_price[0], expected_std_price[0], delta=0.1)
        self.assertAlmostEqual(std_price[1], expected_std_price[1], delta=0.1)
        
        
        

if __name__ == "__main__":
    unittest.main()
