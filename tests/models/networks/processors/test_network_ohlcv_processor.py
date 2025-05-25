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
        self.n_assets = 2
        
        # Initialize processor
        self.processor = OHLCVProcessor(
            window_size=self.window_size,
            hidden_dim=self.hidden_dim,
            device=self.device,
            n_assets=self.n_assets
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
        self.assertEqual(self.processor.n_assets, self.n_assets)
        
    def test_different_n_assets(self):
        """Test processor with different number of assets"""
        # Test with default n_assets (3)
        processor_default = OHLCVProcessor(
            window_size=self.window_size,
            hidden_dim=self.hidden_dim,
            device=self.device
        )
        self.assertEqual(processor_default.n_assets, 3)
        
        # Test with custom n_assets
        custom_n_assets = 5
        processor_custom = OHLCVProcessor(
            window_size=self.window_size,
            hidden_dim=self.hidden_dim,
            device=self.device,
            n_assets=custom_n_assets
        )
        self.assertEqual(processor_custom.n_assets, custom_n_assets)
        
        # Test processing with different n_assets
        x_default = torch.randn(1, 3, self.window_size, 5, device=self.device)
        x_custom = torch.randn(1, custom_n_assets, self.window_size, 5, device=self.device)
        
        output_default = processor_default(x_default)
        output_custom = processor_custom(x_custom)
        
        self.assert_valid_processed_tensor(output_default, (1, self.hidden_dim))
        self.assert_valid_processed_tensor(output_custom, (1, self.hidden_dim))
        
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
        # self.assertFalse(torch.allclose(x.grad, torch.zeros_like(x.grad)))

    def test_non_batched_input(self):
        """Test processing of non-batched input thoroughly"""
        # Test with multiple assets but no batch dimension
        x = torch.randn(self.n_assets, self.window_size, 5, device=self.device)  # Shape: [n_assets, window_size, 5]
        output = self.processor(x)
        
        # Check output shape
        self.assert_valid_processed_tensor(output, (1, self.hidden_dim))
        
        # Test that the output is deterministic
        output2 = self.processor(x)
        self.assertTrue(torch.allclose(output, output2))
        
        # Test that different inputs give different outputs
        x2 = torch.randn(self.n_assets, self.window_size, 5, device=self.device)
        output3 = self.processor(x2)
        self.assertFalse(torch.allclose(output, output3))
        
        # Test gradient flow with non-batched input
        x.requires_grad = True
        output = self.processor(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist and are correct shape
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        # self.assertFalse(torch.allclose(x.grad, torch.zeros_like(x.grad)))
        
        # Test that the processor maintains asset relationships
        x3 = x.clone()
        x3[0] = x3[0] * 2  # Double the first asset's OHLCV data
        output4 = self.processor(x3)
        self.assertFalse(torch.allclose(output, output4))
        self.assertTrue(torch.all(torch.isfinite(output4)))  # Should still be reasonable
        
        # Test that the processor maintains temporal relationships
        x4 = x.clone()
        x4[:, 0, :] = x4[:, 0, :] * 2  # Double the first timestep's data for all assets
        output5 = self.processor(x4)
        self.assertFalse(torch.allclose(output, output5))  # Should be different
        self.assertTrue(torch.all(torch.isfinite(output5)))  # Should still be reasonable

if __name__ == "__main__":
    unittest.main()