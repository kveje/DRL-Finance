import unittest
import torch
import numpy as np
import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from models.networks.processors.ohlcv_processor import OHLCVProcessor

class TestOHLCVProcessor(unittest.TestCase):
    """Test cases for OHLCVProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.window_size = 20
        self.hidden_dim = 64
        self.n_heads = 4
        self.n_assets = 2
        
        # Initialize processor
        self.processor = OHLCVProcessor(
            window_size=self.window_size,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            device=self.device
        )
        
    def create_sample_input(self, batch_size=1):
        """Create a sample OHLCV input tensor"""
        return torch.randn(
            batch_size, 
            self.n_assets, 
            self.window_size, 
            5,  # OHLCV
            device=self.device
        )
        
    def assert_valid_processed_tensor(self, tensor, expected_shape):
        """Helper method to validate processed tensor format"""
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.device.type, self.device)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.shape, expected_shape)
        
    def test_initialization(self):
        """Test processor initialization"""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.window_size, self.window_size)
        self.assertEqual(self.processor.n_heads, self.n_heads)
        
    def test_forward_shape_single_asset(self):
        """Test forward pass with single asset"""
        x = torch.randn(1, self.window_size, 5, device=self.device)  # Single asset
        output = self.processor(x)
        self.assert_valid_processed_tensor(output, (1, self.hidden_dim))
        
    def test_forward_shape_multiple_assets(self):
        """Test forward pass with multiple assets"""
        x = self.create_sample_input()
        output = self.processor(x)
        self.assert_valid_processed_tensor(output, (1, self.hidden_dim))
        
    def test_forward_shape_batch(self):
        """Test forward pass with batch processing"""
        batch_size = 4
        x = self.create_sample_input(batch_size)
        output = self.processor(x)
        self.assert_valid_processed_tensor(output, (batch_size, self.hidden_dim))
        
    def test_attention_mechanism(self):
        """Test that attention mechanism is working"""
        x = self.create_sample_input()
        output = self.processor(x)
        
        # Check that output is not just zeros
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
        
        # Check that output values are reasonable
        self.assertTrue(torch.all(torch.isfinite(output)))
        
    def test_cnn_processing(self):
        """Test that CNN processing is working"""
        x = self.create_sample_input()
        output = self.processor(x)
        
        # Check that output has been processed through CNN
        self.assertFalse(torch.allclose(output, x.view(1, -1)[:, :self.hidden_dim]))
        
    def test_output_range(self):
        """Test that output values are in reasonable range"""
        x = self.create_sample_input()
        output = self.processor(x)
        
        # Check that output values are not too extreme
        self.assertTrue(torch.all(torch.abs(output) < 100))
        
    def test_gradient_flow(self):
        """Test that gradients can flow through the network"""
        x = self.create_sample_input()
        x.requires_grad = True
        
        output = self.processor(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.allclose(x.grad, torch.zeros_like(x.grad)))

if __name__ == "__main__":
    unittest.main()