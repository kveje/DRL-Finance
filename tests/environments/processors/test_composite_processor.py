import unittest
import numpy as np
import pandas as pd
import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from environments.processors.composite_processor import CompositeProcessor

class TestCompositeProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.asset_list = ['AAPL', 'MSFT', 'GOOGL']
        self.window_size = 10
        self.tech_cols = ['bb_upper', 'bb_lower', 'rsi', 'macd']
        self.ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        self.position_limits = {'min': 0, 'max': 100}
        self.cash_limit = 100000.0
        self.current_step = 20  # Add current step
        
        # Create processor configurations
        self.processor_configs = [
            {
                'type': 'price',
                'data_name': 'market_data',
                'kwargs': {
                    'window_size': self.window_size,
                    'asset_list': self.asset_list
                }
            },
            {
                'type': 'cash',
                'data_name': 'cash_data',
                'kwargs': {
                    'cash_limit': self.cash_limit
                }
            },
            {
                'type': 'position',
                'data_name': 'position_data',
                'kwargs': {
                    'position_limits': self.position_limits,
                    'asset_list': self.asset_list
                }
            },
            {
                'type': 'vix',
                'data_name': 'market_data',
                'kwargs': {
                    'window_size': self.window_size
                }
            },
            {
                'type': 'ohlcv',
                'data_name': 'market_data',
                'kwargs': {
                    'window_size': self.window_size,
                    'ohlcv_cols': self.ohlcv_cols,
                    'asset_list': self.asset_list
                }
            },
            {
                'type': 'tech',
                'data_name': 'market_data',
                'kwargs': {
                    'tech_cols': self.tech_cols,
                    'asset_list': self.asset_list
                }
            }
        ]
        
        # Create test data
        days = 25
        
        market_data = pd.DataFrame({
            "day": np.repeat(np.arange(1, days+1), len(self.asset_list)),
            "ticker": np.tile(self.asset_list, days),
            "close": np.random.randn(len(self.asset_list) * days),
            "open": np.random.randn(len(self.asset_list) * days),
            "high": np.random.randn(len(self.asset_list) * days),
            "low": np.random.randn(len(self.asset_list) * days),
            "volume": np.random.randn(len(self.asset_list) * days),
            "bb_upper": np.random.randn(len(self.asset_list) * days),
            "bb_lower": np.random.randn(len(self.asset_list) * days),
            "rsi": np.random.randn(len(self.asset_list) * days),
            "macd": np.random.randn(len(self.asset_list) * days),
            "vix": np.repeat(np.random.randn(days), len(self.asset_list))
        })
        market_data = market_data.set_index('day')
        
        position_data = np.array([50, -30, 20])
        cash_data = 50000.0

        self.test_data = {
            'market_data': market_data,
            'cash_data': cash_data,
            'position_data': position_data
        }
        
    def test_initialization(self):
        """Test processor initialization"""
        processor = CompositeProcessor(self.processor_configs)
        self.assertEqual(len(processor.processors), 6)
        self.assertIn('price', processor.processors)
        self.assertIn('cash', processor.processors)
        self.assertIn('position', processor.processors)
        self.assertIn('vix', processor.processors)
        self.assertIn('ohlcv', processor.processors)
        self.assertIn('tech', processor.processors)
        
    def test_process(self):
        """Test processing data through all processors"""
        processor = CompositeProcessor(self.processor_configs)
        result = processor.process(self.test_data, self.current_step)
        # Check that all processors' outputs are present
        self.assertIn('price', result)
        self.assertIn('cash', result)
        self.assertIn('position', result)
        self.assertIn('vix', result)
        self.assertIn('ohlcv', result)
        self.assertIn('tech', result)
        
        # Check shapes of processed data
        self.assertEqual(result['price'].shape, (len(self.asset_list), self.window_size))
        self.assertEqual(result['cash'].shape, (2,))
        self.assertEqual(result['position'].shape, (len(self.asset_list),))
        self.assertEqual(result['vix'].shape, (self.window_size,))
        self.assertEqual(result['ohlcv'].shape, (len(self.asset_list), len(self.ohlcv_cols), self.window_size))
        self.assertEqual(result['tech'].shape, (len(self.asset_list), len(self.tech_cols)))
        
    def test_missing_data(self):
        """Test handling of missing data"""
        processor = CompositeProcessor(self.processor_configs)
        incomplete_data = {
            'market_data': self.test_data['market_data'],
            'cash_data': self.test_data['cash_data']
            # Missing position_data
        }
        
        with self.assertRaises(KeyError):
            processor.process(incomplete_data, self.current_step)
            
    def test_observation_space(self):
        """Test observation space definition"""
        processor = CompositeProcessor(self.processor_configs)
        obs_space = processor.get_observation_space()
        
        self.assertIn('price', obs_space)
        self.assertIn('cash', obs_space)
        self.assertIn('position', obs_space)
        self.assertIn('vix', obs_space)
        self.assertIn('ohlcv', obs_space)
        self.assertIn('tech', obs_space)
        
    def test_input_dim(self):
        """Test input dimension calculation"""
        processor = CompositeProcessor(self.processor_configs)
        input_dim = processor.get_input_dim()
        
        # Calculate expected total dimension
        expected_dim = {
            'price': (len(self.asset_list), self.window_size),
            'cash': (2,),
            'position': (len(self.asset_list),),
            'vix': (self.window_size,),
            'ohlcv': (len(self.asset_list), len(self.ohlcv_cols), self.window_size),
            'tech': (len(self.asset_list), len(self.tech_cols))
        }

        self.assertEqual(input_dim, expected_dim)

if __name__ == '__main__':
    unittest.main() 