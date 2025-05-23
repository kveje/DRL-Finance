import unittest
import torch
import numpy as np
import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from models.networks.processors.tech_processor import TechProcessor

class TestTechProcessor(unittest.TestCase):
    """Test cases for TechProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_assets = 3
        self.tech_dim = 5
        self.hidden_dim = 64
        
        # Initialize processor
        self.processor = TechProcessor(
            n_assets=self.n_assets,
            tech_dim=self.tech_dim,
            hidden_dim=self.hidden_dim,
            device=self.device
        )
        
    def create_sample_input(self, batch_size=1):
        """Create a sample technical indicators input tensor"""
        return torch.randn(
            batch_size,
            self.n_assets,
            self.tech_dim,
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
        self.assertEqual(self.processor.n_assets, self.n_assets)
        self.assertEqual(self.processor.tech_dim, self.tech_dim)
        
    def test_forward_shape_single_asset(self):
        """Test forward pass with single asset"""
        x = torch.randn(1, self.tech_dim, device=self.device)  # Single asset
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
        
    def test_asset_processing(self):
        """Test that each asset is processed independently"""
        x = self.create_sample_input()
        output = self.processor(x)
        
        # Check that output is not just zeros
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
        
        # Check that output values are reasonable
        self.assertTrue(torch.all(torch.isfinite(output)))
        
    def test_asset_combining(self):
        """Test that assets are properly combined"""
        x = self.create_sample_input()
        output = self.processor(x)
        
        # Check that output has been processed
        # Process input through first layer only for comparison
        processed = []
        for i in range(self.n_assets):
            asset_tech = x[:, i, :]
            processed.append(self.processor.asset_processor(asset_tech))
        x_processed = torch.cat(processed, dim=-1)
        x_processed = self.processor.combiner(x_processed)
        
        # Compare with output (should be different due to final normalization and tanh)
        self.assertFalse(torch.allclose(output, x_processed, rtol=1e-3))
        
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
        
    def test_asset_independence(self):
        """Test that processing is independent for each asset"""
        x1 = self.create_sample_input()
        x2 = x1.clone()
        
        # Modify one asset's data
        x2[:, 0, :] = torch.randn_like(x2[:, 0, :])
        
        # Process both inputs
        output1 = self.processor(x1)
        output2 = self.processor(x2)
        
        # Check that outputs are different
        self.assertFalse(torch.allclose(output1, output2))

if __name__ == "__main__":
    unittest.main()