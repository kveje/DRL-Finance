import unittest
import numpy as np
import pandas as pd


import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from environments.processors.ohlcv_processor import OHLCVProcessor

class TestOHLCVProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create sample data with multiple assets and days
        self.data = pd.DataFrame({
            'day': [1, 1, 2, 2, 3, 3],
            'ticker': ['AAPL', 'GOOGL', 'AAPL', 'GOOGL', 'AAPL', 'GOOGL'],
            'open': [100, 200, 101, 201, 102, 202],
            'high': [105, 205, 106, 206, 107, 207],
            'low': [95, 195, 96, 196, 97, 197],
            'close': [102, 202, 103, 203, 104, 204],
            'volume': [1000, 2000, 1100, 2100, 1200, 2200]
        })
        
        # Set day to index
        self.data = self.data.set_index('day')
        
        self.ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        self.asset_list = ['AAPL', 'GOOGL']
        self.processor = OHLCVProcessor(
            ohlcv_cols=self.ohlcv_cols,
            window_size=2,
            asset_list=self.asset_list
        )

    def test_process_normal_case(self):
        """Test normal processing of OHLCV data."""
        # Process data for day 3 (should get OHLCV for days 2 and 3)
        result = self.processor.process(self.data, current_step=3)
        
        # Check shape
        self.assertEqual(result.shape, (2, 5, 2))  # (n_assets, ohlcv_dim, window_size)
        
        # Check values for AAPL
        expected_aapl = np.array([
            [101, 102],  # open
            [106, 107],  # high
            [96, 97],    # low
            [103, 104],  # close
            [1100, 1200] # volume
        ], dtype=np.float32)
        np.testing.assert_array_equal(result[0], expected_aapl)
        
        # Check values for GOOGL
        expected_googl = np.array([
            [201, 202],  # open
            [206, 207],  # high
            [196, 197],  # low
            [203, 204],  # close
            [2100, 2200] # volume
        ], dtype=np.float32)
        np.testing.assert_array_equal(result[1], expected_googl)

    def test_process_insufficient_data(self):
        """Test processing with insufficient data."""
        with self.assertRaises(ValueError):
            self.processor.process(self.data, current_step=1)

    def test_get_observation_space(self):
        """Test observation space shape and type."""
        obs_space = self.processor.get_observation_space()
        
        self.assertEqual(obs_space['ohlcv']['shape'], (2, 5, 2))  # (n_assets, ohlcv_dim, window_size)
        self.assertEqual(obs_space['ohlcv']['dtype'], np.float32)

    def test_get_input_dim(self):
        """Test input dimension."""
        self.assertEqual(self.processor.get_input_dim(), (2, 5, 2))  # n_assets * ohlcv_dim * window_size

    def test_process_different_window_size(self):
        """Test processing with different window size."""
        processor = OHLCVProcessor(
            ohlcv_cols=self.ohlcv_cols,
            window_size=3,
            asset_list=self.asset_list
        )
        
        result = processor.process(self.data, current_step=3)
        
        # Check shape
        self.assertEqual(result.shape, (2, 5, 3))  # (n_assets, ohlcv_dim, window_size)
        
        # Check values for AAPL
        expected_aapl = np.array([
            [100, 101, 102],  # open
            [105, 106, 107],  # high
            [95, 96, 97],     # low
            [102, 103, 104],  # close
            [1000, 1100, 1200] # volume
        ], dtype=np.float32)
        np.testing.assert_array_equal(result[0], expected_aapl)

    def test_process_subset_ohlcv_cols(self):
        """Test processing with subset of OHLCV columns."""
        processor = OHLCVProcessor(
            ohlcv_cols=['open', 'close'],
            window_size=2,
            asset_list=self.asset_list
        )
        
        result = processor.process(self.data, current_step=3)
        
        # Check shape
        self.assertEqual(result.shape, (2, 2, 2))  # (n_assets, ohlcv_dim, window_size)
        
        # Check values for AAPL
        expected_aapl = np.array([
            [101, 102],  # open
            [103, 104]   # close
        ], dtype=np.float32)
        np.testing.assert_array_equal(result[0], expected_aapl)

if __name__ == '__main__':
    unittest.main() 