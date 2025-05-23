import unittest
import numpy as np
import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from environments.processors.cash_processor import CashProcessor

class TestCashProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.cash_limit = 100000.0
        self.processor = CashProcessor(self.cash_limit)
        
    def test_initialization(self):
        """Test processor initialization"""
        self.assertEqual(self.processor.cash_limit, self.cash_limit)
        self.assertEqual(self.processor.get_input_dim(), (2,))
        
    def test_normalization(self):
        """Test cash balance normalization"""
        # Test zero balance
        result = self.processor.process(0.0)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result[0], 0.0)  # Normalized balance
        self.assertEqual(result[1], 1.0)  # Normalized remaining
        
        # Test half balance
        result = self.processor.process(50000.0)
        self.assertEqual(result[0], 0.5)  # Normalized balance
        self.assertEqual(result[1], 0.5)  # Normalized remaining
        
        # Test full balance
        result = self.processor.process(100000.0)
        self.assertEqual(result[0], 1.0)  # Normalized balance
        self.assertEqual(result[1], 0.0)  # Normalized remaining
        
        # Test exceeding balance
        result = self.processor.process(150000.0)
        self.assertEqual(result[0], 1.0)  # Normalized balance (capped at 1.0)
        self.assertEqual(result[1], 0.0)  # Normalized remaining
        
    def test_edge_cases(self):
        """Test edge cases"""
        # Test negative balance
        result = self.processor.process(-10000.0)
        self.assertEqual(result[0], -0.1)  # Normalized balance
        self.assertEqual(result[1], 1.1)  # Normalized remaining (can exceed 1.0 in this case)
        
        # Test very small balance
        result = self.processor.process(0.01)
        self.assertGreater(result[0], 0.0)  # Should be a very small positive number
        self.assertLess(result[1], 1.0)  # Should be very close to 1.0
        
    def test_output_type(self):
        """Test output type and dtype"""
        result = self.processor.process(50000.0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        
    def test_observation_space(self):
        """Test observation space definition"""
        obs_space = self.processor.get_observation_space()
        self.assertIn('cash', obs_space)
        self.assertEqual(obs_space['cash']['low'], 0.0)
        self.assertEqual(obs_space['cash']['high'], 1.0)
        self.assertEqual(obs_space['cash']['shape'], (2,))
        self.assertEqual(obs_space['cash']['dtype'], np.float32)

if __name__ == '__main__':
    unittest.main() 