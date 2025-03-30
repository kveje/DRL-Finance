"""Unit tests for the Commission Manager."""

import unittest
from typing import Dict, List
import sys
import os
import numpy as np

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the function to test
from environments.market_friction.manager import MarketFrictionManager

class TestMarketFrictionManager(unittest.TestCase):
    """Test cases for the MarketFrictionManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
                'slippage': {
                    'slippage_mean': 0.0,
                    'slippage_std': 0.001
                },
                'commission': {
                    'commission_rate': 0.001
                }
            }
        self.manager = MarketFrictionManager(self.config)

    def test_asymptotic_frictions(self):
        """Test apply_frictions method of MarketFrictionManager."""
        n = 10000
        action = np.array([1, -1])
        price = np.array([100.0, 100.0])
        new_price = np.zeros(shape=(n, 2))
        for i in range(n):
            new_price[i,:] = self.manager.apply_frictions(action, price)
        
        mean_price = np.mean(new_price, axis=0)
        std_price = np.std(new_price, axis=0)

        expected_mean_price = np.array([price[0] * (1 + self.config['commission']['commission_rate']), price[1] * (1 - self.config['commission']['commission_rate'])])
        expected_std_price = np.array([0.1, 0.1])

        self.assertAlmostEqual(mean_price[0], expected_mean_price[0], delta=0.1)
        self.assertAlmostEqual(mean_price[1], expected_mean_price[1], delta=0.1)
        self.assertAlmostEqual(std_price[0], expected_std_price[0], delta=0.01)
        self.assertAlmostEqual(std_price[1], expected_std_price[1], delta=0.01)

    def test_zero_action(self):
        """Test apply_frictions method of MarketFrictionManager with zero action."""
        action = np.array([0, 0])
        price = np.array([100.0, 100.0])
        new_price = self.manager.apply_frictions(action, price)
        self.assertEqual(new_price[0], 100.0)
        self.assertEqual(new_price[1], 100.0)

    def test_mixed_action(self):
        """Test apply_frictions method of MarketFrictionManager with mixed action."""
        action = np.array([1, 0])
        price = np.array([100.0, 100.0])
        new_price = self.manager.apply_frictions(action, price)
        self.assertEqual(new_price[1], 100.0)
            
if __name__ == "__main__":
    unittest.main()

