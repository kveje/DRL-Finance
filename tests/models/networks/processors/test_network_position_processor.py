import unittest
import torch
import numpy as np
import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from models.networks.processors.position_processor import PositionProcessor

class TestPositionProcessor(unittest.TestCase):
    """Test cases for PositionProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_assets = 5
        self.hidden_dim = 64
        
        # Initialize processor
        self.processor = PositionProcessor(
            n_assets=self.n_assets,
            hidden_dim=self.hidden_dim,
            device=self.device
        )
        
    def create_sample_input(self, batch_size=1):
        """Create a sample position input tensor"""
        return torch.randn(
            batch_size,
            self.n_assets,
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
        
    def test_forward_shape_single_position(self):
        """Test forward pass with single position"""
        x = torch.randn(self.n_assets, device=self.device)  # Single position
        output = self.processor(x)
        self.assert_valid_processed_tensor(output, (1, self.hidden_dim))
        
    def test_forward_shape_batch(self):
        """Test forward pass with batch processing"""
        batch_size = 4
        x = self.create_sample_input(batch_size)
        output = self.processor(x)
        self.assert_valid_processed_tensor(output, (batch_size, self.hidden_dim))
        
    def test_position_processing(self):
        """Test that positions are properly processed"""
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
        
    def test_position_sensitivity(self):
        """Test that different positions produce different outputs"""
        x1 = self.create_sample_input()
        x2 = x1.clone()
        
        # Modify one position
        x2[:, 0] = torch.randn_like(x2[:, 0])
        
        # Process both inputs
        output1 = self.processor(x1)
        output2 = self.processor(x2)
        
        # Check that outputs are different
        self.assertFalse(torch.allclose(output1, output2))
        
    def test_zero_positions(self):
        """Test processing of zero positions"""
        x = torch.zeros(1, self.n_assets, device=self.device)
        output = self.processor(x)
        
        # Check that output is not just zeros
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
        
    def test_extreme_positions(self):
        """Test processing of extreme position values"""
        x = torch.randn(1, self.n_assets, device=self.device) * 1000  # Very large values
        output = self.processor(x)
        
        # Check that output is still reasonable
        self.assertTrue(torch.all(torch.isfinite(output)))
        self.assertTrue(torch.all(torch.abs(output) < 100))
        
    def test_non_batched_input(self):
        """Test processing of non-batched input thoroughly"""
        # Test with single position data (no batch dimension)
        x = torch.randn(self.n_assets, device=self.device)  # Shape: [n_assets]
        output = self.processor(x)
        
        # Check output shape
        self.assert_valid_processed_tensor(output, (1, self.hidden_dim))
        
        # Test that the output is deterministic
        output2 = self.processor(x)
        self.assertTrue(torch.allclose(output, output2))
        
        # Test that different inputs give different outputs
        x2 = torch.randn(self.n_assets, device=self.device)
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
        x3[0] = x3[0] * 2  # Double the first asset's position
        output4 = self.processor(x3)
        self.assertFalse(torch.allclose(output, output4))  # Should be different
        self.assertTrue(torch.all(torch.isfinite(output4)))  # Should still be reasonable

if __name__ == "__main__":
    unittest.main()