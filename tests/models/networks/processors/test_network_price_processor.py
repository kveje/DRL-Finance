import unittest
import torch
import numpy as np
import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from models.networks.processors.price_processor import PriceProcessor

class TestPriceProcessor(unittest.TestCase):
    """Test cases for PriceProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.window_size = 20
        self.hidden_dim = 64
        self.n_assets = 3
        
        # Initialize processor
        self.processor = PriceProcessor(
            window_size=self.window_size,
            n_assets=self.n_assets,
            hidden_dim=self.hidden_dim,
            device=self.device
        )
        
    def create_sample_input(self, batch_size=1):
        """Create a sample price input tensor"""
        return torch.randn(
            batch_size,
            self.n_assets,
            self.window_size,
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
        
    def test_price_processing(self):
        """Test that price data is properly processed"""
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
        # First average across assets, then pad/truncate to match hidden_dim
        x_reshaped = x.mean(dim=1)  # Shape: (batch_size, window_size)
        if x_reshaped.shape[1] < self.hidden_dim:
            # Pad with zeros if window_size < hidden_dim
            padding = torch.zeros(x_reshaped.shape[0], self.hidden_dim - x_reshaped.shape[1], device=self.device)
            x_reshaped = torch.cat([x_reshaped, padding], dim=1)
        else:
            # Truncate if window_size > hidden_dim
            x_reshaped = x_reshaped[:, :self.hidden_dim]
            
        self.assertFalse(torch.allclose(output, x_reshaped, rtol=1e-3))
        
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
        
    def test_price_sensitivity(self):
        """Test that different price patterns produce different outputs"""
        x1 = self.create_sample_input()
        x2 = x1.clone()
        
        # Modify one price series
        x2[:, 0, :] = torch.randn_like(x2[:, 0, :])
        
        # Process both inputs
        output1 = self.processor(x1)
        output2 = self.processor(x2)
        
        # Check that outputs are different
        self.assertFalse(torch.allclose(output1, output2))
        
    def test_constant_prices(self):
        """Test processing of constant price series"""
        x = torch.ones(1, self.n_assets, self.window_size, device=self.device)
        output = self.processor(x)
        
        # Check that output is not just zeros
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
        
    def test_extreme_prices(self):
        """Test processing of extreme price values"""
        x = torch.randn(1, self.n_assets, self.window_size, device=self.device) * 1000
        output = self.processor(x)
        
        # Check that output is still reasonable
        self.assertTrue(torch.all(torch.isfinite(output)))
        self.assertTrue(torch.all(torch.abs(output) < 100))
        
    def test_negative_prices(self):
        """Test processing of negative price values"""
        x = -torch.rand(1, self.n_assets, self.window_size, device=self.device)
        output = self.processor(x)
        
        # Check that output is still reasonable
        self.assertTrue(torch.all(torch.isfinite(output)))
        self.assertTrue(torch.all(torch.abs(output) < 100))
        
    def test_non_batched_input(self):
        """Test processing of non-batched input thoroughly"""        
        # Test with multiple assets but no batch dimension
        x_multi = torch.randn(self.n_assets, self.window_size, device=self.device)  # Shape: [n_assets, window_size]
        output_multi = self.processor(x_multi)
        self.assert_valid_processed_tensor(output_multi, (1, self.hidden_dim))
        
        # Test that different assets produce different outputs
        x_multi2 = x_multi.clone()
        x_multi2[0] = torch.randn_like(x_multi2[0])  # Change first asset's prices
        output_multi2 = self.processor(x_multi2)
        self.assertFalse(torch.allclose(output_multi, output_multi2))

if __name__ == "__main__":
    unittest.main()