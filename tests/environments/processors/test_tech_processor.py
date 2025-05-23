import unittest
import numpy as np
import pandas as pd

import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from environments.processors.tech_processor import TechProcessor

class TestTechProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create sample data with multiple assets and days
        self.data = pd.DataFrame({
            'day': [1, 1, 2, 2, 3, 3],
            'ticker': ['AAPL', 'GOOGL', 'AAPL', 'GOOGL', 'AAPL', 'GOOGL'],
            'RSI': [50, 55, 52, 57, 54, 59],
            'MACD': [0.1, 0.2, 0.15, 0.25, 0.2, 0.3],
            'BB_upper': [110, 210, 111, 211, 112, 212],
            'BB_lower': [90, 190, 91, 191, 92, 192]
        })
        
        # Set day to index
        self.data = self.data.set_index('day')
        
        self.tech_cols = ['RSI', 'MACD', 'BB_upper', 'BB_lower']
        self.asset_list = ['AAPL', 'GOOGL']
        self.processor = TechProcessor(
            tech_cols=self.tech_cols,
            asset_list=self.asset_list
        )

    def test_process_normal_case(self):
        """Test normal processing of technical indicators."""
        # Process data for day 3 (should get current tech indicators)
        result = self.processor.process(self.data, current_step=3)
        
        # Check shape
        self.assertEqual(result.shape, (2, 4))  # (n_assets, tech_dim)
        
        # Check values for AAPL
        expected_aapl = np.array([54, 0.2, 112, 92], dtype=np.float32)  # RSI, MACD, BB_upper, BB_lower
        np.testing.assert_array_equal(result[0], expected_aapl)
        
        # Check values for GOOGL
        expected_googl = np.array([59, 0.3, 212, 192], dtype=np.float32)  # RSI, MACD, BB_upper, BB_lower
        np.testing.assert_array_equal(result[1], expected_googl)


    def test_get_observation_space(self):
        """Test observation space shape and type."""
        obs_space = self.processor.get_observation_space()
        
        self.assertEqual(obs_space['tech']['shape'], (2, 4))  # (n_assets, tech_dim)
        self.assertEqual(obs_space['tech']['dtype'], np.float32)

    def test_get_input_dim(self):
        """Test input dimension."""
        self.assertEqual(self.processor.get_input_dim(), (2, 4))  # n_assets * tech_dim

    def test_process_subset_tech_cols(self):
        """Test processing with subset of technical indicators."""
        processor = TechProcessor(
            tech_cols=['RSI', 'MACD'],
            asset_list=self.asset_list
        )
        
        result = processor.process(self.data, current_step=3)
        
        # Check shape
        self.assertEqual(result.shape, (2, 2))  # (n_assets, tech_dim)
        
        # Check values for AAPL
        expected_aapl = np.array([54, 0.2], dtype=np.float32)  # RSI, MACD
        np.testing.assert_array_equal(result[0], expected_aapl)

    def test_process_single_asset(self):
        """Test processing for a single asset."""
        processor = TechProcessor(
            tech_cols=self.tech_cols,
            asset_list=['AAPL']
        )
        
        result = processor.process(self.data, current_step=3)
        
        # Check shape
        self.assertEqual(result.shape, (1, 4))  # (n_assets, tech_dim)


        # Check values
        expected = np.array([54, 0.2, 112, 92], dtype=np.float32)  # RSI, MACD, BB_upper, BB_lower
        np.testing.assert_array_equal(result[0], expected)

if __name__ == '__main__':
    unittest.main() 