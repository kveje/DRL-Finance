import unittest
import numpy as np
import pandas as pd

import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from environments.processors.position_processor import PositionProcessor

class TestPositionProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.position_limits = {
            'min': -1000,
            'max': 1000
        }
        
        self.asset_list = ['AAPL', 'GOOGL']
        self.processor = PositionProcessor(
            position_limits=self.position_limits,
            asset_list=self.asset_list
        )

    def test_process_normal_case(self):
        """Test normal processing of positions."""
        # Test various position scenarios
        positions = np.array([500, -500])  # AAPL long, GOOGL short
        result = self.processor.process(positions)
        
        # Check shape
        self.assertEqual(result.shape, (2,))
        
        # Check values
        expected = np.array([0.75, 0.25], dtype=np.float32)  # Normalized positions
        np.testing.assert_array_almost_equal(result, expected)

    def test_process_edge_cases(self):
        """Test processing of edge cases."""
        # Test max long position
        positions = np.array([1000, 0])
        result = self.processor.process(positions)
        expected = np.array([1.0, 0.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test max short position
        positions = np.array([0, -1000])
        result = self.processor.process(positions)
        expected = np.array([0.5, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test zero positions
        positions = np.array([0, 0])
        result = self.processor.process(positions)
        expected = np.array([0.5, 0.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_process_out_of_bounds(self):
        """Test processing of out-of-bounds positions."""
        # Test positions beyond limits
        positions = np.array([1500, -1500])
        result = self.processor.process(positions)
        
        # Should be clipped to [0, 1]
        expected = np.array([1.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_observation_space(self):
        """Test observation space shape and type."""
        obs_space = self.processor.get_observation_space()
        
        self.assertEqual(obs_space['position']['shape'], (2,))  # (n_assets,)
        self.assertEqual(obs_space['position']['dtype'], np.float32)

    def test_get_input_dim(self):
        """Test input dimension."""
        self.assertEqual(self.processor.get_input_dim(), (2,))  # n_assets

    def test_process_different_limits(self):
        """Test processing with different position limits."""
        processor = PositionProcessor(
            position_limits={'min': -500, 'max': 500},
            asset_list=self.asset_list
        )
        
        positions = np.array([250, -250])
        result = processor.process(positions)
        
        # Check values
        expected = np.array([0.75, 0.25], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_process_single_asset(self):
        """Test processing for a single asset."""
        processor = PositionProcessor(
            position_limits=self.position_limits,
            asset_list=['AAPL']
        )
        
        positions = np.array([500])
        result = processor.process(positions)
        
        # Check shape
        self.assertEqual(result.shape, (1,))
        
        # Check value
        expected = np.array([0.75], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_process_asymmetric_limits(self):
        """Test processing with asymmetric position limits."""
        processor = PositionProcessor(
            position_limits={'min': -500, 'max': 1000},
            asset_list=self.asset_list
        )
        
        # Test long position
        positions = np.array([750, 0])
        result = processor.process(positions)
        expected = np.array([5/6, 1/3], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test short position
        positions = np.array([0, -250])
        result = processor.process(positions)
        expected = np.array([1/3, 1/6], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_lower_limit(self):
        """Test processing with zero lower limit."""
        processor = PositionProcessor(
            position_limits={'min': 0, 'max': 1000},
            asset_list=self.asset_list
        )

        # Position 1
        positions = np.array([1000, 0])
        result = processor.process(positions)
        expected = np.array([1.0, 0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

        # Position 2
        positions = np.array([250, 750])
        result = processor.process(positions)
        expected = np.array([1/4, 3/4], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

        # Position 3
        positions = np.array([0, 0])
        result = processor.process(positions)
        expected = np.array([0,0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == '__main__':
    unittest.main() 