import unittest
import torch
import numpy as np
import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from models.networks.processors.cash_processor import CashProcessor

class TestCashProcessor(unittest.TestCase):
    """Test cases for CashProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = 2  # [cash_balance, portfolio_value]
        self.hidden_dim = 32
        
        # Initialize processor
        self.processor = CashProcessor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            device=self.device
        )
        
    def create_sample_input(self, batch_size=1):
        """Create a sample cash input tensor"""
        return torch.randn(
            batch_size,
            self.input_dim,
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
        self.assertEqual(self.processor.input_dim, self.input_dim)
        self.assertEqual(self.processor.hidden_dim, self.hidden_dim)
        self.assertEqual(self.processor.device, self.device)
        
    def test_forward_shape(self):
        """Test forward pass output shape"""
        x = self.create_sample_input()
        output = self.processor(x)
        self.assert_valid_processed_tensor(output, (1, self.processor.get_output_dim()))
        
    def test_batch_processing(self):
        """Test processing multiple samples at once"""
        batch_size = 4
        x = self.create_sample_input(batch_size)
        output = self.processor(x)
        self.assert_valid_processed_tensor(output, (batch_size, self.processor.get_output_dim()))
        
    def test_output_dim(self):
        """Test output dimension is valid"""
        output_dim = self.processor.get_output_dim()
        self.assertIsInstance(output_dim, int)
        self.assertGreater(output_dim, 0)
        
    def test_forward_shape_single_cash(self):
        """Test forward pass with single cash data"""
        x = torch.randn(self.input_dim, device=self.device)  # Single cash data
        output = self.processor(x)
        self.assert_valid_processed_tensor(output, (1, self.hidden_dim))
        
    def test_cash_processing(self):
        """Test that cash data is properly processed"""
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
        
    def test_cash_sensitivity(self):
        """Test that different cash values produce different outputs"""
        x1 = self.create_sample_input()
        x2 = x1.clone()
        
        # Modify one cash value
        x2[:, 0] = torch.randn_like(x2[:, 0])
        
        # Process both inputs
        output1 = self.processor(x1)
        output2 = self.processor(x2)
        
        # Check that outputs are different
        self.assertFalse(torch.allclose(output1, output2))
        
    def test_zero_cash(self):
        """Test processing of zero cash values"""
        x = torch.zeros(1, self.input_dim, device=self.device)
        output = self.processor(x)
        
        # Check that output is not just zeros
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
        
    def test_extreme_cash(self):
        """Test processing of extreme cash values"""
        x = torch.randn(1, self.input_dim, device=self.device) * 1000000  # Very large values
        output = self.processor(x)
        
        # Check that output is still reasonable
        self.assertTrue(torch.all(torch.isfinite(output)))
        self.assertTrue(torch.all(torch.abs(output) < 100))
        
    def test_negative_cash(self):
        """Test processing of negative cash values"""
        x = -torch.rand(1, self.input_dim, device=self.device) * 1000  # Negative values
        output = self.processor(x)
        
        # Check that output is still reasonable
        self.assertTrue(torch.all(torch.isfinite(output)))
        self.assertTrue(torch.all(torch.abs(output) < 100))

if __name__ == "__main__":
    unittest.main()