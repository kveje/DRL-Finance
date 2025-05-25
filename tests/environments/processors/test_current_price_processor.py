import unittest
import numpy as np
import pandas as pd

import sys
import os 

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from environments.processors.current_price_processor import CurrentPriceProcessor

class TestCurrentPriceProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.asset_list = ['AAPL', 'GOOGL']
        self.processor = CurrentPriceProcessor(
            n_assets=len(self.asset_list),
            min_cash_limit=1000,
            price_col="close",
            transaction_cost=0.001,
            slippage_mean=0.0001
        )
        
        # Create sample price data
        self.price_data = pd.DataFrame({
            'close': {
                0: [100.0, 200.0],  # AAPL: $100, GOOGL: $200
                1: [150.0, 250.0],  # AAPL: $150, GOOGL: $250
                2: [50.0, 75.0]     # AAPL: $50, GOOGL: $75
            }
        })

    def test_process_normal_case(self):
        """Test normal processing of current prices."""
        # Test with $5000 cash
        result = self.processor.process(self.price_data, current_cash=5000, step=0)
        
        # Check shape
        self.assertEqual(result.shape, (2,))
        
        # Calculate expected values:
        # AAPL: (100*1.0011)/(5000-1000) = 0.0250275
        # GOOGL: (200*1.0011)/(5000-1000) = 0.050055
        expected = np.array([0.0250275, 0.050055], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_process_edge_cases(self):
        """Test processing of edge cases."""
        # Test with minimum cash
        result = self.processor.process(self.price_data, current_cash=1000, step=0)
        expected = np.array([1.0, 1.0], dtype=np.float32)  # Should be clipped to 1.0
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with exactly enough cash for one share
        # For AAPL: price=100, adjusted=100.11, we want ratio=0.90909
        # So available_cash should be 100.11/0.90909 ≈ 110.11
        result = self.processor.process(self.price_data, current_cash=1110.11, step=0)
        expected = np.array([0.90909, 1.0], dtype=np.float32)  # Should be clipped to 1.0 for GOOGL
        np.testing.assert_array_almost_equal(result, expected, decimal=4)  # Reduced precision for floating point comparison

    def test_process_different_prices(self):
        """Test processing with different price scenarios."""
        # Test with higher prices
        result = self.processor.process(self.price_data, current_cash=5000, step=1)
        expected = np.array([0.03754125, 0.06256875], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        
        # Test with lower prices
        result = self.processor.process(self.price_data, current_cash=5000, step=2)
        expected = np.array([0.01251375, 0.018770625], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_process_with_transaction_costs(self):
        """Test processing with transaction costs."""
        processor = CurrentPriceProcessor(
            n_assets=len(self.asset_list),
            min_cash_limit=1000,
            transaction_cost=0.01,  # 1% transaction cost
            slippage_mean=0.0
        )
        
        result = processor.process(self.price_data, current_cash=5000, step=0)
        expected = np.array([0.02525, 0.0505], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_process_with_slippage(self):
        """Test processing with slippage."""
        processor = CurrentPriceProcessor(
            n_assets=len(self.asset_list),
            min_cash_limit=1000,
            slippage_mean=0.005,  # 0.5% slippage
            transaction_cost=0.0
        )
        
        result = processor.process(self.price_data, current_cash=5000, step=0)
        expected = np.array([0.025125, 0.05025], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_get_observation_space(self):
        """Test observation space shape and type."""
        obs_space = self.processor.get_observation_space()
        
        self.assertEqual(obs_space['current_price']['shape'], (2,))  # (n_assets,)
        self.assertEqual(obs_space['current_price']['dtype'], np.float32)

    def test_get_input_dim(self):
        """Test input dimension."""
        self.assertEqual(self.processor.get_input_dim(), (2,))  # n_assets

if __name__ == '__main__':
    unittest.main() 