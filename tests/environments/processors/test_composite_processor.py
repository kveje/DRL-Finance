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
        self.current_step = 20
        
        # Create processor configurations
        self.processor_configs = [
            {
                'type': 'price',
                'kwargs': {
                    'window_size': self.window_size,
                    'asset_list': self.asset_list
                }
            },
            {
                'type': 'cash',
                'kwargs': {
                    'cash_limit': self.cash_limit
                }
            },
            {
                'type': 'position',
                'kwargs': {
                    'position_limits': self.position_limits,
                    'asset_list': self.asset_list
                }
            },
            {
                'type': 'vix',
                'kwargs': {
                    'window_size': self.window_size
                }
            },
            {
                'type': 'ohlcv',
                'kwargs': {
                    'window_size': self.window_size,
                    'ohlcv_cols': self.ohlcv_cols,
                    'asset_list': self.asset_list
                }
            },
            {
                'type': 'tech',
                'kwargs': {
                    'tech_cols': self.tech_cols,
                    'asset_list': self.asset_list
                }
            },
            {
                'type': 'affordability',
                'kwargs': {
                    'n_assets': len(self.asset_list)
                }
            },
            {
                'type': 'current_price',
                'kwargs': {
                    'n_assets': len(self.asset_list)
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
        raw_data = pd.DataFrame({
            "day": np.repeat(np.arange(1, days+1), len(self.asset_list)),
            "ticker": np.tile(self.asset_list, days),
            "close": np.random.randn(len(self.asset_list) * days),
            "open": np.random.randn(len(self.asset_list) * days),
            "high": np.random.randn(len(self.asset_list) * days),
            "low": np.random.randn(len(self.asset_list) * days),
            "volume": np.random.randn(len(self.asset_list) * days)
        })
        raw_data = raw_data.set_index('day')

        self.test_data = {
            'market': market_data,
            'cash': cash_data,
            'position': position_data,
            'raw': raw_data,
            'step': self.current_step
        }
        
    def test_initialization(self):
        """Test processor initialization"""
        processor = CompositeProcessor(self.processor_configs)
        self.assertEqual(len(processor.processors), 8)  # Updated count
        self.assertIn('price', processor.processors)
        self.assertIn('cash', processor.processors)
        self.assertIn('position', processor.processors)
        self.assertIn('vix', processor.processors)
        self.assertIn('ohlcv', processor.processors)
        self.assertIn('tech', processor.processors)
        self.assertIn('affordability', processor.processors)
        self.assertIn('current_price', processor.processors)
        
    def test_process(self):
        """Test processing data through all processors"""
        processor = CompositeProcessor(self.processor_configs)
        result = processor.process(self.test_data)
        
        # Check that all processors' outputs are present
        self.assertIn('price', result)
        self.assertIn('cash', result)
        self.assertIn('position', result)
        self.assertIn('vix', result)
        self.assertIn('ohlcv', result)
        self.assertIn('tech', result)
        self.assertIn('affordability', result)
        self.assertIn('current_price', result)
        
        # Check shapes of processed data
        self.assertEqual(result['price'].shape, (len(self.asset_list), self.window_size))
        self.assertEqual(result['cash'].shape, (2,))
        self.assertEqual(result['position'].shape, (len(self.asset_list),))
        self.assertEqual(result['vix'].shape, (self.window_size,))
        self.assertEqual(result['ohlcv'].shape, (len(self.asset_list), len(self.ohlcv_cols), self.window_size))
        self.assertEqual(result['tech'].shape, (len(self.asset_list), len(self.tech_cols)))
        self.assertEqual(result['affordability'].shape, (len(self.asset_list),))
        self.assertEqual(result['current_price'].shape, (len(self.asset_list),))
        
    def test_missing_data(self):
        """Test handling of missing data"""
        processor = CompositeProcessor(self.processor_configs)
        incomplete_data = {
            'market': self.test_data['market'],
            'cash': self.test_data['cash']
            # Missing position and raw data
        }
        
        with self.assertRaises(KeyError):
            processor.process(incomplete_data)
            
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
        self.assertIn('affordability', obs_space)
        self.assertIn('current_price', obs_space)
        
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
            'tech': (len(self.asset_list), len(self.tech_cols)),
            'affordability': (len(self.asset_list),),
            'current_price': (len(self.asset_list),)
        }

        self.assertEqual(input_dim, expected_dim)

if __name__ == '__main__':
    unittest.main() 