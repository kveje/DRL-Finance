"""Unit tests for network backbones."""
import unittest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from models.networks.backbones import BaseBackbone, MLPBackbone

class DummyBackbone(BaseBackbone):
    """Dummy backbone for testing the base class."""
    
    def __init__(self, input_dim: int, config: dict, device: str = "cuda"):
        super().__init__(input_dim, config, device)
        self.layer = nn.Linear(input_dim, config.get("output_dim", 64)).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
    
    def get_output_dim(self) -> int:
        return self.config.get("output_dim", 64)

class TestBaseBackbone(unittest.TestCase):
    """Test suite for the base backbone class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dummy_config = {"output_dim": 64}
        self.dummy_backbone = DummyBackbone(input_dim=32, config=self.dummy_config, device=self.device)
    
    def test_initialization(self):
        """Test backbone initialization."""
        self.assertEqual(self.dummy_backbone.input_dim, 32)
        self.assertEqual(self.dummy_backbone.config, self.dummy_config)
        self.assertEqual(self.dummy_backbone.device.type, self.device.type)
    
    def test_forward(self):
        """Test forward pass."""
        x = torch.randn(16, 32, device=self.device)  # batch_size=16, input_dim=32
        output = self.dummy_backbone(x)
        self.assertEqual(output.shape, (16, 64))  # batch_size=16, output_dim=64
        self.assertEqual(output.device.type, self.device.type)
    
    def test_get_output_dim(self):
        """Test output dimension retrieval."""
        self.assertEqual(self.dummy_backbone.get_output_dim(), 64)

class TestMLPBackbone(unittest.TestCase):
    """Test suite for the MLP backbone class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp_config = {
            "hidden_dims": [256, 128],
            "dropout": 0.1,
            "use_layer_norm": True
        }
        self.mlp_backbone = MLPBackbone(input_dim=32, config=self.mlp_config, device=self.device)
    
    def test_initialization(self):
        """Test MLP backbone initialization."""
        self.assertEqual(self.mlp_backbone.input_dim, 32)
        self.assertEqual(self.mlp_backbone.config, self.mlp_config)
        self.assertEqual(self.mlp_backbone.device.type, self.device.type)
        self.assertEqual(self.mlp_backbone.hidden_dims, [256, 128])
        self.assertEqual(self.mlp_backbone.dropout, 0.1)
        self.assertTrue(self.mlp_backbone.use_layer_norm)
    
    def test_forward(self):
        """Test forward pass through MLP."""
        x = torch.randn(16, 32, device=self.device)  # batch_size=16, input_dim=32
        output = self.mlp_backbone(x)
        self.assertEqual(output.shape, (16, 128))  # batch_size=16, last_hidden_dim=128
        self.assertEqual(output.device.type, self.device.type)
    
    def test_get_output_dim(self):
        """Test output dimension retrieval."""
        self.assertEqual(self.mlp_backbone.get_output_dim(), 128)
    
    def test_default_config(self):
        """Test MLP backbone with default configuration."""
        backbone = MLPBackbone(input_dim=32, config={}, device=self.device)
        self.assertEqual(backbone.hidden_dims, [256, 128])
        self.assertEqual(backbone.dropout, 0.1)
        self.assertTrue(backbone.use_layer_norm)
    
    def test_no_layer_norm(self):
        """Test MLP backbone without layer normalization."""
        config = {
            "hidden_dims": [256, 128],
            "dropout": 0.1,
            "use_layer_norm": False
        }
        backbone = MLPBackbone(input_dim=32, config=config, device=self.device)
        x = torch.randn(16, 32, device=self.device)
        output = backbone(x)
        self.assertEqual(output.shape, (16, 128))
        self.assertEqual(output.device.type, self.device.type)
    
    def test_custom_hidden_dims(self):
        """Test MLP backbone with custom hidden dimensions."""
        config = {
            "hidden_dims": [512, 256, 128],
            "dropout": 0.1,
            "use_layer_norm": True
        }
        backbone = MLPBackbone(input_dim=32, config=config, device=self.device)
        x = torch.randn(16, 32, device=self.device)
        output = backbone(x)
        self.assertEqual(output.shape, (16, 128))
        self.assertEqual(output.device.type, self.device.type)
    
    def test_dropout_effect(self):
        """Test that dropout is active during training."""
        self.mlp_backbone.train()
        x = torch.randn(16, 32, device=self.device)
        output1 = self.mlp_backbone(x)
        output2 = self.mlp_backbone(x)
        # Outputs should be different due to dropout
        self.assertFalse(torch.allclose(output1, output2))
    
    def test_dropout_inactive_during_eval(self):
        """Test that dropout is inactive during evaluation."""
        self.mlp_backbone.eval()
        x = torch.randn(16, 32, device=self.device)
        output1 = self.mlp_backbone(x)
        output2 = self.mlp_backbone(x)
        # Outputs should be identical due to no dropout
        self.assertTrue(torch.allclose(output1, output2))
    
    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        x = torch.randn(16, 32, device=self.device, requires_grad=True)
        output = self.mlp_backbone(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.allclose(x.grad, torch.zeros_like(x.grad)))

if __name__ == '__main__':
    unittest.main() 