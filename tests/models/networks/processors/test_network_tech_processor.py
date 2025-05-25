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
        
    def test_non_batched_input(self):
        """Test processing of non-batched input thoroughly"""
        # Test with multiple assets but no batch dimension
        x = torch.randn(self.n_assets, self.tech_dim, device=self.device)  # Shape: [n_assets, tech_dim]
        output = self.processor(x)
        
        # Check output shape
        self.assert_valid_processed_tensor(output, (1, self.hidden_dim))
        
        # Test that the output is deterministic
        output2 = self.processor(x)
        self.assertTrue(torch.allclose(output, output2))
        
        # Test that different inputs give different outputs
        x2 = torch.randn(self.n_assets, self.tech_dim, device=self.device)
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
        self.assertFalse(torch.allclose(x.grad, torch.zeros_like(x.grad)))
        
        # Test that the processor maintains asset relationships
        x3 = x.clone()
        x3[0] = x3[0] * 2  # Double the first asset's indicators
        output4 = self.processor(x3)
        self.assertFalse(torch.allclose(output, output4))  # Should be different
        self.assertTrue(torch.all(torch.isfinite(output4)))  # Should still be reasonable
        

if __name__ == "__main__":
    unittest.main()