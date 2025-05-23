import unittest
import numpy as np
import pandas as pd
import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from environments.processors.processor_factory import ProcessorFactory
from environments.processors.base_processor import BaseProcessor

class TestProcessorFactory(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.asset_list = ['AAPL', 'MSFT', 'GOOGL']
        self.window_size = 10
        self.tech_cols = ['rsi', 'macd', 'bollinger']
        self.ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        self.position_limits = {'min': -100, 'max': 100}
        self.cash_limit = 100000.0
        
    def test_create_price_processor(self):
        """Test creating a price processor"""
        processor = ProcessorFactory.create_processor(
            'price',
            window_size=self.window_size,
            asset_list=self.asset_list
        )
        self.assertIsInstance(processor, BaseProcessor)
        self.assertEqual(processor.get_input_dim(), (len(self.asset_list), self.window_size))
        
    def test_create_ohlcv_processor(self):
        """Test creating an OHLCV processor"""
        processor = ProcessorFactory.create_processor(
            'ohlcv',
            ohlcv_cols=self.ohlcv_cols,
            window_size=self.window_size,
            asset_list=self.asset_list
        )
        self.assertIsInstance(processor, BaseProcessor)
        self.assertEqual(processor.get_input_dim(), (len(self.asset_list), len(self.ohlcv_cols), self.window_size))
        
    def test_create_tech_processor(self):
        """Test creating a technical indicators processor"""
        processor = ProcessorFactory.create_processor(
            'tech',
            tech_cols=self.tech_cols,
            asset_list=self.asset_list
        )
        self.assertIsInstance(processor, BaseProcessor)
        self.assertEqual(processor.get_input_dim(), (len(self.asset_list), len(self.tech_cols)))
        
    def test_create_position_processor(self):
        """Test creating a position processor"""
        processor = ProcessorFactory.create_processor(
            'position',
            position_limits=self.position_limits,
            asset_list=self.asset_list
        )
        self.assertIsInstance(processor, BaseProcessor)
        self.assertEqual(processor.get_input_dim(), (len(self.asset_list),))
        
    def test_create_vix_processor(self):
        """Test creating a VIX processor"""
        processor = ProcessorFactory.create_processor(
            'vix',
            window_size=self.window_size
        )
        self.assertIsInstance(processor, BaseProcessor)
        self.assertEqual(processor.get_input_dim(), (self.window_size,))
        
    def test_create_cash_processor(self):
        """Test creating a cash processor"""
        processor = ProcessorFactory.create_processor(
            'cash',
            cash_limit=self.cash_limit
        )
        self.assertIsInstance(processor, BaseProcessor)
        self.assertEqual(processor.get_input_dim(), (2,))
        
    def test_invalid_processor_type(self):
        """Test creating an invalid processor type"""
        with self.assertRaises(ValueError):
            ProcessorFactory.create_processor('invalid_type')
            
    def test_register_new_processor(self):
        """Test registering a new processor type"""
        class TestProcessor(BaseProcessor):
            def process(self, *args, **kwargs):
                return np.array([1.0])
                
            def get_observation_space(self):
                return {'data': {'shape': (1,), 'dtype': np.float32}}
                
            def get_input_dim(self):
                return (1,)
                
        ProcessorFactory.register_processor('test', TestProcessor)
        processor = ProcessorFactory.create_processor('test')
        self.assertIsInstance(processor, TestProcessor)
        self.assertEqual(processor.get_input_dim(), (1,))

if __name__ == '__main__':
    unittest.main() 