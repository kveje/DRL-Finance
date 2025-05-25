import unittest
import numpy as np
import pandas as pd

import sys
import os 

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from environments.processors.affordability_processor import AffordabilityProcessor

class TestAffordabilityProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.asset_list = ['AAPL', 'GOOGL']
        self.processor = AffordabilityProcessor(
            n_assets=len(self.asset_list),
            min_cash_limit=1000,
            max_trade_size=10,
            price_col="close",
            transaction_cost=0.001,
            slippage_mean=0.0
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
        """Test normal processing of affordability."""
        # Test with $5000 cash
        result = self.processor.process(self.price_data, current_cash=5000, step=0)
        
        # Check shape
        self.assertEqual(result.shape, (2,))
        
        # Calculate expected values:
        # AAPL: (5000-1000)/(100*1.001) = 40 shares possible, normalized to 1.0 (capped at max_trade_size)
        # GOOGL: (5000-1000)/(200*1.001) = 20 shares possible, normalized to 1.0 (capped at max_trade_size)
        expected = np.array([1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_process_edge_cases(self):
        """Test processing of edge cases."""
        # Test with minimum cash
        result = self.processor.process(self.price_data, current_cash=1000, step=0)
        expected = np.array([0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with exactly enough cash for one share (including transaction cost)
        result = self.processor.process(self.price_data, current_cash=1200, step=0)
        expected = np.array([0.1, 0.0], dtype=np.float32)  # 1/10 of max trade size
        np.testing.assert_array_almost_equal(result, expected)

    def test_process_different_prices(self):
        """Test processing with different price scenarios."""
        # Test with higher prices
        result = self.processor.process(self.price_data, current_cash=5000, step=1)
        expected = np.array([1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with lower prices
        result = self.processor.process(self.price_data, current_cash=5000, step=2)
        expected = np.array([1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_process_with_transaction_costs(self):
        """Test processing with transaction costs."""
        processor = AffordabilityProcessor(
            n_assets=len(self.asset_list),
            min_cash_limit=1000,
            max_trade_size=10,
            transaction_cost=0.01,  # 1% transaction cost
            slippage_mean=0.0
        )
        
        result = processor.process(self.price_data, current_cash=5000, step=0)
        expected = np.array([1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_process_with_slippage(self):
        """Test processing with slippage."""
        processor = AffordabilityProcessor(
            n_assets=len(self.asset_list),
            min_cash_limit=1000,
            max_trade_size=10,
            slippage_mean=0.005,  # 0.5% slippage
            transaction_cost=0.0
        )
        
        result = processor.process(self.price_data, current_cash=5000, step=0)
        expected = np.array([1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_observation_space(self):
        """Test observation space shape and type."""
        obs_space = self.processor.get_observation_space()
        
        self.assertEqual(obs_space['affordability']['shape'], (2,))  # (n_assets,)
        self.assertEqual(obs_space['affordability']['dtype'], np.float32)

    def test_get_input_dim(self):
        """Test input dimension."""
        self.assertEqual(self.processor.get_input_dim(), (2,))  # n_assets

if __name__ == '__main__':
    unittest.main() 