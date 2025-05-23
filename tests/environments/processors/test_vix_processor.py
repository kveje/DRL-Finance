import unittest
import numpy as np
import pandas as pd
import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from environments.processors.vix_processor import VIXProcessor

class TestVIXProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create sample data with multiple assets and days
        self.data = pd.DataFrame({
            'day': [1, 1, 2, 2, 3, 3],
            'ticker': ['AAPL', 'GOOGL', 'AAPL', 'GOOGL', 'AAPL', 'GOOGL'],
            'price': [100, 200, 101, 201, 102, 202],
            'vix': [20, 20, 21, 21, 22, 22]  # VIX is same for all assets on same day
        })
        
        # Set day to index
        self.data = self.data.set_index('day')
        
        self.processor = VIXProcessor(
            vix_col='vix',
            window_size=2
        )

    def test_process_normal_case(self):
        """Test normal processing of VIX data."""
        # Process data for day 3 (should get VIX values for days 2 and 3)
        result = self.processor.process(self.data, current_step=3)
        
        # Check shape
        self.assertEqual(result.shape, (2,))
        
        # Check values (should be VIX values for days 2 and 3)
        expected = np.array([21, 22], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_process_insufficient_data(self):
        """Test processing with insufficient data."""
        with self.assertRaises(ValueError):
            self.processor.process(self.data, current_step=1)  # Not enough days for window_size=2

    def test_get_observation_space(self):
        """Test observation space shape and type."""
        obs_space = self.processor.get_observation_space()
        
        self.assertEqual(obs_space['vix']['shape'], (2,))
        self.assertEqual(obs_space['vix']['dtype'], np.float32)

    def test_get_input_dim(self):
        """Test input dimension."""
        self.assertEqual(self.processor.get_input_dim(), (2,))

    def test_process_different_window_size(self):
        """Test processing with different window size."""
        processor = VIXProcessor(
            vix_col='vix',
            window_size=3
        )
        
        result = processor.process(self.data, current_step=3)
        
        # Check shape
        self.assertEqual(result.shape, (3,))
        
        # Check values (should be VIX values for days 1, 2, and 3)
        expected = np.array([20, 21, 22], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main() 