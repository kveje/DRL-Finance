"""Unit tests for the unified network framework."""
import unittest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.networks.unified_network import UnifiedNetwork

class TestUnifiedNetwork(unittest.TestCase):
    """Test suite for the unified network class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {
            "n_assets": 30,  # Top-level configuration for number of assets
            "window_size": 20,  # Top-level configuration for observation window
            "processors": {
                "ohlcv": {
                    "enabled": True,
                    "hidden_dim": 256,
                    "asset_embedding_dim": 32,
                },
                "technical": {
                    "enabled": True,
                    "tech_dim": 20,
                    "hidden_dim": 128
                },
                "price": {
                    "enabled": True,
                    "hidden_dim": 256,
                    "asset_embedding_dim": 16
                },
                "position": {
                    "enabled": True,
                    "hidden_dim": 32
                },
                "cash": {
                    "enabled": True,
                    "input_dim": 2,  # [cash_balance, portfolio_value]
                    "hidden_dim": 32
                },
                "affordability": {
                    "enabled": True,
                    "hidden_dim": 32
                },
                "current_price": {
                    "enabled": True,
                    "hidden_dim": 32
                }
            },
            "backbone": {
                "type": "mlp",
                "hidden_dims": [256, 128],
                "dropout": 0.0,
                "use_layer_norm": True
            },
            "heads": {
                "value": {
                    "enabled": True,
                    "type": "parametric",
                    "hidden_dim": 128
                },
                "discrete": {
                    "enabled": True,
                    "type": "parametric",
                    "hidden_dim": 128
                },
                "confidence": {
                    "enabled": True,
                    "type": "parametric",
                    "hidden_dim": 128
                }
            }
        }
        self.network = UnifiedNetwork(config=self.config, device=self.device)
    
    def test_initialization(self):
        """Test network initialization."""
        # Check that all components are initialized
        self.assertIsInstance(self.network.processors, nn.ModuleDict)
        self.assertIsInstance(self.network.backbone, nn.Module)
        self.assertIsInstance(self.network.heads, nn.ModuleDict)
        
        # Check that all processors are present
        expected_processors = ["ohlcv", "technical", "price", "position", "cash", "affordability", "current_price"]
        for name in expected_processors:
            self.assertIn(name, self.network.processors)
        
        # Check that all heads are present
        expected_heads = ["value", "discrete", "confidence"]
        for name in expected_heads:
            self.assertIn(name, self.network.heads)
        
        # Check that n_assets and window_size are properly set
        self.assertEqual(self.network.n_assets, self.config["n_assets"])
        self.assertEqual(self.network.window_size, self.config["window_size"])
    
    def test_forward(self):
        """Test forward pass with all processors."""
        # Create test observations
        observations = {
            "ohlcv": torch.randn(2, 30, 20, 5, device=self.device),  # [batch, assets, window, features]
            "technical": torch.randn(2, 30, 20, device=self.device),  # [batch, assets, features]
            "price": torch.randn(2, 30, 20, device=self.device),  # [batch, assets, window]
            "position": torch.randn(2, 30, device=self.device),  # [batch, assets]
            "cash": torch.randn(2, 2, device=self.device),  # [batch, features]
            "affordability": torch.randn(2, 30, device=self.device),  # [batch, assets]
            "current_price": torch.randn(2, 30, device=self.device),  # [batch, assets]
        }
        
        # Forward pass
        outputs = self.network(observations)
        
        # Check outputs
        self.assertIn("value", outputs)
        self.assertIn("action_probs", outputs)
        self.assertIn("confidences", outputs)
        
        # Check shapes
        self.assertEqual(outputs["value"].shape, (2, 1))  # [batch, 1]
        self.assertEqual(outputs["action_probs"].shape, (2, 30, 3))  # [batch, assets, actions]
        self.assertEqual(outputs["confidences"].shape, (2, 30))  # [batch, assets]
        
        # Check value ranges
        self.assertTrue(torch.all(outputs["value"] >= -1e6) and torch.all(outputs["value"] <= 1e6))
        self.assertTrue(torch.all(outputs["action_probs"] >= 0) and torch.all(outputs["action_probs"] <= 1))
        self.assertTrue(torch.all(outputs["confidences"] >= 0) and torch.all(outputs["confidences"] <= 1))
        
        # Check action probabilities sum to 1
        action_sums = outputs["action_probs"].sum(dim=-1)
        self.assertTrue(torch.allclose(action_sums, torch.ones_like(action_sums), atol=1e-6))
    
    def test_bayesian_heads(self):
        """Test network with Bayesian heads."""
        # Create network with Bayesian heads
        config = self.config.copy()
        config["heads"]["value"]["type"] = "bayesian"
        config["heads"]["discrete"]["type"] = "bayesian"
        config["heads"]["confidence"]["type"] = "bayesian"
        network = UnifiedNetwork(config, device=self.device)
        
        # Create test observations
        observations = {
            "ohlcv": torch.randn(2, 30, 20, 5, device=self.device),
            "technical": torch.randn(2, 30, 20, device=self.device),
            "price": torch.randn(2, 30, 20, device=self.device),
            "position": torch.randn(2, 30, device=self.device),
            "cash": torch.randn(2, 2, device=self.device),
            "affordability": torch.randn(2, 30, device=self.device),
            "current_price": torch.randn(2, 30, device=self.device),
        }
        
        # Test with sampling
        outputs = network(observations, use_sampling=True)
        
        # Check outputs
        self.assertIn("value", outputs)
        self.assertIn("action_probs", outputs)
        self.assertIn("confidences", outputs)
        
        # Check shapes
        self.assertEqual(outputs["value"].shape, (2, 1))  # [batch, 1]
        self.assertEqual(outputs["action_probs"].shape, (2, 30, 3))  # [batch, assets, actions]
        self.assertEqual(outputs["confidences"].shape, (2, 30))  # [batch, assets]
        
        # Check value ranges
        self.assertTrue(torch.all(outputs["value"] >= -1e6) and torch.all(outputs["value"] <= 1e6))
        self.assertTrue(torch.all(outputs["action_probs"] >= 0) and torch.all(outputs["action_probs"] <= 1))
        self.assertTrue(torch.all(outputs["confidences"] >= 0) and torch.all(outputs["confidences"] <= 1))
        
        # Check action probabilities sum to 1
        action_sums = outputs["action_probs"].sum(dim=-1)
        self.assertTrue(torch.allclose(action_sums, torch.ones_like(action_sums), atol=1e-6))
        
        # Test without sampling (using mean)
        outputs_mean = network(observations, use_sampling=False)
        
        # Check outputs
        self.assertIn("value", outputs_mean)
        self.assertIn("action_probs", outputs_mean)
        self.assertIn("confidences", outputs_mean)
        
        # Check shapes
        self.assertEqual(outputs_mean["value"].shape, (2, 1))  # [batch, 1]
        self.assertEqual(outputs_mean["action_probs"].shape, (2, 30, 3))  # [batch, assets, actions]
        self.assertEqual(outputs_mean["confidences"].shape, (2, 30))  # [batch, assets]
        
        # Check value ranges
        self.assertTrue(torch.all(outputs_mean["value"] >= -1e6) and torch.all(outputs_mean["value"] <= 1e6))
        self.assertTrue(torch.all(outputs_mean["action_probs"] >= 0) and torch.all(outputs_mean["action_probs"] <= 1))
        self.assertTrue(torch.all(outputs_mean["confidences"] >= 0) and torch.all(outputs_mean["confidences"] <= 1))
        
        # Check action probabilities sum to 1
        action_sums = outputs_mean["action_probs"].sum(dim=-1)
        self.assertTrue(torch.allclose(action_sums, torch.ones_like(action_sums), atol=1e-6))
        
        # Check that sampling and mean give different results
        self.assertFalse(torch.allclose(outputs["value"], outputs_mean["value"]))
        self.assertFalse(torch.allclose(outputs["action_probs"], outputs_mean["action_probs"]))
        self.assertFalse(torch.allclose(outputs["confidences"], outputs_mean["confidences"]))
    
    def test_partial_observations(self):
        """Test network with partial observations."""
        # Create sample observations with only some processors
        batch_size = 16
        observations = {
            "ohlcv": torch.randn(batch_size, self.config["n_assets"], self.config["window_size"], 5, device=self.device),
            "technical": torch.randn(batch_size, self.config["n_assets"], 20, device=self.device)
        }

        config = self.config.copy()
        config["processors"]["price"]["enabled"] = False
        config["processors"]["position"]["enabled"] = False
        config["processors"]["cash"]["enabled"] = False
        config["processors"]["affordability"]["enabled"] = False
        config["processors"]["current_price"]["enabled"] = False
        config["backbone"]["hidden_dims"] = [384, 128]

        network = UnifiedNetwork(config, device=self.device)
        
        # Forward pass should still work
        outputs = network(observations)
        
        # Check outputs
        self.assertIn("value", outputs)
        self.assertIn("action_probs", outputs)
        self.assertIn("confidences", outputs)
    
    def test_invalid_observations(self):
        """Test network with invalid observations."""
        # Empty observations should raise ValueError
        with self.assertRaises(ValueError):
            self.network({})
        
        # Invalid observation shape should raise RuntimeError
        batch_size = 16
        observations = {
            "ohlcv": torch.randn(batch_size, self.config["n_assets"], 10, 5, device=self.device),  # Wrong window size
            "technical": torch.randn(batch_size, self.config["n_assets"], 20, device=self.device)
        }
        with self.assertRaises(RuntimeError):
            self.network(observations)
    
    def test_eval_mode(self):
        """Test network behavior in evaluation mode."""
        self.network.eval()
        
        # Create sample observations
        batch_size = 16
        observations = {
            "ohlcv": torch.randn(batch_size, self.config["n_assets"], self.config["window_size"], 5, device=self.device),
            "technical": torch.randn(batch_size, self.config["n_assets"], 20, device=self.device),
            "price": torch.randn(batch_size, self.config["n_assets"], self.config["window_size"], device=self.device),
            "position": torch.randn(batch_size, self.config["n_assets"], device=self.device),
            "cash": torch.randn(batch_size, 2, device=self.device),
            "affordability": torch.randn(batch_size, self.config["n_assets"], device=self.device),
            "current_price": torch.randn(batch_size, self.config["n_assets"], device=self.device)
        }
        
        # Forward pass
        outputs1 = self.network(observations)
        outputs2 = self.network(observations)
        
        # Outputs should be identical due to no dropout
        self.assertTrue(torch.allclose(outputs1["value"], outputs2["value"]))

    def test_single_sample_inference(self):
        """Test network behavior with single sample (no batch dimension) for agent action selection."""
        self.network.eval()
        
        # Create single sample observations (no batch dimension)
        observations = {
            "ohlcv": torch.randn(self.config["n_assets"], self.config["window_size"], 5, device=self.device),  # [assets, window, features]
            "technical": torch.randn(self.config["n_assets"], 20, device=self.device),  # [assets, features]
            "price": torch.randn(self.config["n_assets"], self.config["window_size"], device=self.device),  # [assets, window]
            "position": torch.randn(self.config["n_assets"], device=self.device),  # [assets]
            "cash": torch.randn(2, device=self.device),  # [features]
            "affordability": torch.randn(self.config["n_assets"], device=self.device),  # [assets]
            "current_price": torch.randn(self.config["n_assets"], device=self.device)  # [assets]
        }
        
        # Forward pass
        outputs = self.network(observations)
        
        # Check outputs
        self.assertIn("value", outputs)
        self.assertIn("action_probs", outputs)
        self.assertIn("confidences", outputs)
        
        # Check shapes for single sample
        self.assertEqual(outputs["value"].shape, (1,))  # [1]
        self.assertEqual(outputs["action_probs"].shape, (self.config["n_assets"], 3))  # [assets, actions]
        self.assertEqual(outputs["confidences"].shape, (self.config["n_assets"],))  # [assets]
        
        # Check value ranges
        self.assertTrue(torch.all(outputs["value"] >= -1e6) and torch.all(outputs["value"] <= 1e6))
        self.assertTrue(torch.all(outputs["action_probs"] >= 0) and torch.all(outputs["action_probs"] <= 1))
        self.assertTrue(torch.all(outputs["confidences"] >= 0) and torch.all(outputs["confidences"] <= 1))
        
        # Check action probabilities sum to 1
        action_sums = outputs["action_probs"].sum(dim=-1)
        self.assertTrue(torch.allclose(action_sums, torch.ones_like(action_sums), atol=1e-6))

if __name__ == '__main__':
    unittest.main() 