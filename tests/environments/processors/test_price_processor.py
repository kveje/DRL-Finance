import unittest
import numpy as np
import pandas as pd
import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from environments.processors.price_processor import PriceProcessor

class TestPriceProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create sample data with multiple assets and days
        self.data = pd.DataFrame({
            'day': [1, 1, 2, 2, 3, 3],
            'ticker': ['AAPL', 'GOOGL', 'AAPL', 'GOOGL', 'AAPL', 'GOOGL'],
            'close': [100, 200, 101, 201, 102, 202],
            'open': [99, 199, 100, 200, 101, 201]
        })
        
        # Set day to index
        self.data = self.data.set_index('day')
        
        self.processor = PriceProcessor(
            price_col='close',
            window_size=2,
            asset_list=['AAPL', 'GOOGL']
        )

    def test_process_normal_case(self):
        """Test normal processing of price data."""
        # Process data for day 3 (should get prices for days 2 and 3)
        result = self.processor.process(self.data, current_step=3)
        
        # Check shape
        self.assertEqual(result.shape, (2, 2))  # (n_assets, window_size)
        
        # Check values for AAPL
        expected_aapl = np.array([101, 102], dtype=np.float32)
        np.testing.assert_array_equal(result[0], expected_aapl)
        
        # Check values for GOOGL
        expected_googl = np.array([201, 202], dtype=np.float32)
        np.testing.assert_array_equal(result[1], expected_googl)

    def test_process_insufficient_data(self):
        """Test processing with insufficient data."""
        with self.assertRaises(ValueError):
            self.processor.process(self.data, current_step=1)  # Not enough days for window_size=2

    def test_process_wrong_column_names(self):
        """Test processing with wrong column names."""
        processor = PriceProcessor(
            price_col='wrong_col',
            window_size=2,
            asset_list=['AAPL', 'GOOGL']
        )
        with self.assertRaises(KeyError):
            processor.process(self.data, current_step=3)

    def test_get_observation_space(self):
        """Test observation space shape and type."""
        obs_space = self.processor.get_observation_space()
        
        self.assertEqual(obs_space['price']['shape'], (2, 2))  # (n_assets, window_size)
        self.assertEqual(obs_space['price']['dtype'], np.float32)

    def test_get_input_dim(self):
        """Test input dimension."""
        self.assertEqual(self.processor.get_input_dim(), (2, 2))  # n_assets * window_size

    def test_process_different_window_size(self):
        """Test processing with different window size."""
        processor = PriceProcessor(
            price_col='close',
            window_size=3,
            asset_list=['AAPL', 'GOOGL']
        )
        
        result = processor.process(self.data, current_step=3)
        
        # Check shape
        self.assertEqual(result.shape, (2, 3))  # (n_assets, window_size)
        
        # Check values for AAPL
        expected_aapl = np.array([100, 101, 102], dtype=np.float32)
        np.testing.assert_array_equal(result[0], expected_aapl)
        
        # Check values for GOOGL
        expected_googl = np.array([200, 201, 202], dtype=np.float32)
        np.testing.assert_array_equal(result[1], expected_googl)

    def test_process_different_price_column(self):
        """Test processing with different price column."""
        processor = PriceProcessor(
            price_col='open',
            window_size=2,
            asset_list=['AAPL', 'GOOGL']
        )
        
        result = processor.process(self.data, current_step=3)
        
        # Check values for AAPL
        expected_aapl = np.array([100, 101], dtype=np.float32)
        np.testing.assert_array_equal(result[0], expected_aapl)
        
        # Check values for GOOGL
        expected_googl = np.array([200, 201], dtype=np.float32)
        np.testing.assert_array_equal(result[1], expected_googl)

if __name__ == '__main__':
    unittest.main() 