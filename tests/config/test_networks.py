"""Unit tests for network configurations."""
import unittest
import torch
import sys
import os

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.networks.unified_network import UnifiedNetwork
from config.networks import (
    SIMPLE_DISCRETE_PARAMETRIC_CONFIG,
    SIMPLE_DISCRETE_BAYESIAN_CONFIG,
    ADVANCED_DISCRETE_PARAMETRIC_CONFIG,
    ADVANCED_DISCRETE_BAYESIAN_CONFIG,
    ADVANCED_CONFIDENCE_PARAMETRIC_CONFIG,
    ADVANCED_CONFIDENCE_BAYESIAN_CONFIG,
    ADVANCED_VALUE_PARAMETRIC_CONFIG,
    ADVANCED_VALUE_BAYESIAN_CONFIG,
    ADVANCED_FULL_PARAMETRIC_CONFIG,
    ADVANCED_FULL_BAYESIAN_CONFIG
)

class TestNetworkConfigs(unittest.TestCase):
    """Test suite for network configurations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.n_assets = 3
        self.window_size = 20
        
    def create_simple_observations(self):
        """Create observations for simple networks."""
        return {
            "price": torch.randn(self.batch_size, self.n_assets, self.window_size, device=self.device),
            "position": torch.randn(self.batch_size, self.n_assets, device=self.device),
            "cash": torch.randn(self.batch_size, 2, device=self.device)  # [cash_balance, portfolio_value]
        }
        
    def create_advanced_observations(self):
        """Create observations for advanced networks."""
        return {
            "ohlcv": torch.randn(self.batch_size, self.n_assets, self.window_size, 5, device=self.device),
            "technical": torch.randn(self.batch_size, self.n_assets, 20, device=self.device),
            "position": torch.randn(self.batch_size, self.n_assets, device=self.device),
            "cash": torch.randn(self.batch_size, 2, device=self.device)  # [cash_balance, portfolio_value]
        }
    
    def test_simple_discrete_parametric(self):
        """Test simple discrete parametric network."""
        network = UnifiedNetwork(SIMPLE_DISCRETE_PARAMETRIC_CONFIG, device=self.device)
        observations = self.create_simple_observations()
        outputs = network(observations)
        
        # Check outputs
        self.assertIn("discrete", outputs)
        self.assertEqual(outputs["discrete"].shape, (self.batch_size, self.n_assets, 3))
        
        # Check action probabilities sum to 1
        action_sums = outputs["discrete"].sum(dim=-1)
        self.assertTrue(torch.allclose(action_sums, torch.ones_like(action_sums), atol=1e-6))
    
    def test_simple_discrete_bayesian(self):
        """Test simple discrete bayesian network."""
        network = UnifiedNetwork(SIMPLE_DISCRETE_BAYESIAN_CONFIG, device=self.device)
        observations = self.create_simple_observations()
        outputs = network(observations)
        
        # Check outputs
        self.assertIn("discrete", outputs)
        self.assertEqual(outputs["discrete"].shape, (self.batch_size, self.n_assets, 3))
        
        # Check action probabilities sum to 1
        action_sums = outputs["discrete"].sum(dim=-1)
        self.assertTrue(torch.allclose(action_sums, torch.ones_like(action_sums), atol=1e-6))
    
    def test_advanced_discrete_parametric(self):
        """Test advanced discrete parametric network."""
        network = UnifiedNetwork(ADVANCED_DISCRETE_PARAMETRIC_CONFIG, device=self.device)
        observations = self.create_advanced_observations()
        outputs = network(observations)
        
        # Check outputs
        self.assertIn("discrete", outputs)
        self.assertEqual(outputs["discrete"].shape, (self.batch_size, self.n_assets, 3))
        
        # Check action probabilities sum to 1
        action_sums = outputs["discrete"].sum(dim=-1)
        self.assertTrue(torch.allclose(action_sums, torch.ones_like(action_sums), atol=1e-6))
    
    def test_advanced_discrete_bayesian(self):
        """Test advanced discrete bayesian network."""
        network = UnifiedNetwork(ADVANCED_DISCRETE_BAYESIAN_CONFIG, device=self.device)
        observations = self.create_advanced_observations()
        outputs = network(observations)
        
        # Check outputs
        self.assertIn("discrete", outputs)
        self.assertEqual(outputs["discrete"].shape, (self.batch_size, self.n_assets, 3))
        
        # Check action probabilities sum to 1
        action_sums = outputs["discrete"].sum(dim=-1)
        self.assertTrue(torch.allclose(action_sums, torch.ones_like(action_sums), atol=1e-6))
    
    def test_advanced_confidence_parametric(self):
        """Test advanced confidence parametric network."""
        network = UnifiedNetwork(ADVANCED_CONFIDENCE_PARAMETRIC_CONFIG, device=self.device)
        observations = self.create_advanced_observations()
        outputs = network(observations)
        
        # Check outputs
        self.assertIn("discrete", outputs)
        self.assertIn("confidence", outputs)
        self.assertEqual(outputs["discrete"].shape, (self.batch_size, self.n_assets, 3))
        self.assertEqual(outputs["confidence"].shape, (self.batch_size, self.n_assets))
        
        # Check action probabilities sum to 1
        action_sums = outputs["discrete"].sum(dim=-1)
        self.assertTrue(torch.allclose(action_sums, torch.ones_like(action_sums), atol=1e-6))
        
        # Check confidence values are between 0 and 1
        self.assertTrue(torch.all(outputs["confidence"] >= 0) and torch.all(outputs["confidence"] <= 1))
    
    def test_advanced_confidence_bayesian(self):
        """Test advanced confidence bayesian network."""
        network = UnifiedNetwork(ADVANCED_CONFIDENCE_BAYESIAN_CONFIG, device=self.device)
        observations = self.create_advanced_observations()
        outputs = network(observations)
        
        # Check outputs
        self.assertIn("discrete", outputs)
        self.assertIn("confidence", outputs)
        self.assertEqual(outputs["discrete"].shape, (self.batch_size, self.n_assets, 3))
        self.assertEqual(outputs["confidence"].shape, (self.batch_size, self.n_assets))
        
        # Check action probabilities sum to 1
        action_sums = outputs["discrete"].sum(dim=-1)
        self.assertTrue(torch.allclose(action_sums, torch.ones_like(action_sums), atol=1e-6))
        
        # Check confidence values are between 0 and 1
        self.assertTrue(torch.all(outputs["confidence"] >= 0) and torch.all(outputs["confidence"] <= 1))
    
    def test_advanced_value_parametric(self):
        """Test advanced value parametric network."""
        network = UnifiedNetwork(ADVANCED_VALUE_PARAMETRIC_CONFIG, device=self.device)
        observations = self.create_advanced_observations()
        outputs = network(observations)
        
        # Check outputs
        self.assertIn("discrete", outputs)
        self.assertIn("value", outputs)
        self.assertEqual(outputs["discrete"].shape, (self.batch_size, self.n_assets, 3))
        self.assertEqual(outputs["value"].shape, (self.batch_size, 1))
        
        # Check action probabilities sum to 1
        action_sums = outputs["discrete"].sum(dim=-1)
        self.assertTrue(torch.allclose(action_sums, torch.ones_like(action_sums), atol=1e-6))
    
    def test_advanced_value_bayesian(self):
        """Test advanced value bayesian network."""
        network = UnifiedNetwork(ADVANCED_VALUE_BAYESIAN_CONFIG, device=self.device)
        observations = self.create_advanced_observations()
        outputs = network(observations)
        
        # Check outputs
        self.assertIn("discrete", outputs)
        self.assertIn("value", outputs)
        self.assertEqual(outputs["discrete"].shape, (self.batch_size, self.n_assets, 3))
        self.assertEqual(outputs["value"].shape, (self.batch_size, 1))
        
        # Check action probabilities sum to 1
        action_sums = outputs["discrete"].sum(dim=-1)
        self.assertTrue(torch.allclose(action_sums, torch.ones_like(action_sums), atol=1e-6))
    
    def test_advanced_full_parametric(self):
        """Test advanced full parametric network."""
        network = UnifiedNetwork(ADVANCED_FULL_PARAMETRIC_CONFIG, device=self.device)
        observations = self.create_advanced_observations()
        outputs = network(observations)
        
        # Check outputs
        self.assertIn("discrete", outputs)
        self.assertIn("value", outputs)
        self.assertIn("confidence", outputs)
        self.assertEqual(outputs["discrete"].shape, (self.batch_size, self.n_assets, 3))
        self.assertEqual(outputs["value"].shape, (self.batch_size, 1))
        self.assertEqual(outputs["confidence"].shape, (self.batch_size, self.n_assets))
        
        # Check action probabilities sum to 1
        action_sums = outputs["discrete"].sum(dim=-1)
        self.assertTrue(torch.allclose(action_sums, torch.ones_like(action_sums), atol=1e-6))
        
        # Check confidence values are between 0 and 1
        self.assertTrue(torch.all(outputs["confidence"] >= 0) and torch.all(outputs["confidence"] <= 1))
    
    def test_advanced_full_bayesian(self):
        """Test advanced full bayesian network."""
        network = UnifiedNetwork(ADVANCED_FULL_BAYESIAN_CONFIG, device=self.device)
        observations = self.create_advanced_observations()
        outputs = network(observations)
        
        # Check outputs
        self.assertIn("discrete", outputs)
        self.assertIn("value", outputs)
        self.assertIn("confidence", outputs)
        self.assertEqual(outputs["discrete"].shape, (self.batch_size, self.n_assets, 3))
        self.assertEqual(outputs["value"].shape, (self.batch_size, 1))
        self.assertEqual(outputs["confidence"].shape, (self.batch_size, self.n_assets))
        
        # Check action probabilities sum to 1
        action_sums = outputs["discrete"].sum(dim=-1)
        self.assertTrue(torch.allclose(action_sums, torch.ones_like(action_sums), atol=1e-6))
        
        # Check confidence values are between 0 and 1
        self.assertTrue(torch.all(outputs["confidence"] >= 0) and torch.all(outputs["confidence"] <= 1))

if __name__ == '__main__':
    unittest.main() 