"""Unit tests for the Slippage Frictionclass."""

import unittest
from typing import Dict, List
import sys
import os
import numpy as np

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the function to test
from environments.market_friction.slippage import SlippageFriction

class TestSlippage(unittest.TestCase):
    """Test cases for the Slippage class."""

    def setUp(self):
        """Set up test fixtures.""" 
        config = {
            "slippage_mean": 0.0,
            "slippage_std": 0.001
        }
        self.slippage = SlippageFriction(config)
    
    def test_slippage_initialization(self):
        """Test initialization of Slippage."""
        self.assertEqual(self.slippage.mean, 0.0)
        self.assertEqual(self.slippage.std, 0.001)
        self.assertEqual(self.slippage.name, "slippage")

    def test_asymptotic_slippage(self):
        """Test apply method of Slippage."""
        n = 100000
        action = np.array([1, -1])
        price = np.array([100.0, 100.0])
        
        new_price = np.zeros(shape=(n, 2))

        for i in range(n):
            new_price[i,:] = self.slippage.apply(action, price)

        mean_price = np.mean(new_price)
        std_price = np.std(new_price)
        
        self.assertAlmostEqual(mean_price, 100.0, delta=0.01)
        self.assertAlmostEqual(std_price, 0.1, delta=0.01) # 0.1 = 0.001 * 100.0

    def test_slippage_zero_action(self):
        """Test apply method of Slippage with zero action."""
        action = np.array([0, 0])
        price = np.array([100.0, 100.0])
        new_price = self.slippage.apply(action, price)
        self.assertEqual(new_price[0], 100.0)
        self.assertEqual(new_price[1], 100.0)

    def test_slippage_mixed_action(self):
        """Test apply method of Slippage with mixed action."""
        action = np.array([0, -1])
        price = np.array([100.0, 100.0])
        new_price = self.slippage.apply(action, price)
        self.assertEqual(new_price[0], 100.0)

    def test_multiple_assets(self):
        """Test apply method of Slippage with multiple assets."""
        action = np.array([0, 0, 5, 10, -3])
        price = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        new_price = self.slippage.apply(action, price)
        self.assertEqual(new_price[0], 100.0)
        self.assertEqual(new_price[1], 100.0)

        

if __name__ == "__main__":
    unittest.main()


