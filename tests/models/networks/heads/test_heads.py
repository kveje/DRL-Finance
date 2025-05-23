import unittest
import torch
import numpy as np

import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from models.networks.heads.value_head import ParametricValueHead, BayesianValueHead
from models.networks.heads.discrete_head import ParametricDiscreteHead, BayesianDiscreteHead
from models.networks.heads.confidence_head import ParametricConfidenceHead, BayesianConfidenceHead

class TestValueHeads(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_dim = 64
        self.hidden_dim = 32
        self.n_assets = 3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create test input
        self.x = torch.randn(self.batch_size, self.input_dim, device=self.device)
        
        # Initialize heads
        self.parametric_head = ParametricValueHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_assets=self.n_assets,
            device=self.device
        )
        
        self.bayesian_head = BayesianValueHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_assets=self.n_assets,
            device=self.device
        )
    
    def test_parametric_value_head(self):
        # Test forward pass
        outputs = self.parametric_head(self.x)
        self.assertIn("value", outputs)
        self.assertEqual(outputs["value"].shape, (self.batch_size, 1))
        
        # Test output dimension
        self.assertEqual(self.parametric_head.get_output_dim(), 1)  # Single value estimate
    
    def test_bayesian_value_head(self):
        # Test forward pass
        outputs = self.bayesian_head(self.x)
        self.assertIn("mean", outputs)
        self.assertIn("std", outputs)
        self.assertEqual(outputs["mean"].shape, (self.batch_size, 1))
        self.assertEqual(outputs["std"].shape, (self.batch_size, 1))
        
        # Test sampling strategies
        strategies = ["thompson", "optimistic", "ucb"]
        for strategy in strategies:
            samples = self.bayesian_head.sample(self.x, strategy=strategy)
            self.assertIn("value", samples)
            self.assertEqual(samples["value"].shape, (self.batch_size, 1))
        
        # Test invalid strategy
        with self.assertRaises(ValueError):
            self.bayesian_head.sample(self.x, strategy="invalid")
        
        # Test output dimension
        self.assertEqual(self.bayesian_head.get_output_dim(), 2)  # Mean and std for single value

class TestDiscreteHeads(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_dim = 64
        self.hidden_dim = 32
        self.n_assets = 3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create test input
        self.x = torch.randn(self.batch_size, self.input_dim, device=self.device)
        
        # Initialize heads
        self.parametric_head = ParametricDiscreteHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_assets=self.n_assets,
            device=self.device
        )
        
        self.bayesian_head = BayesianDiscreteHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_assets=self.n_assets,
            device=self.device
        )
    
    def test_parametric_discrete_head(self):
        # Test forward pass
        outputs = self.parametric_head(self.x)
        self.assertIn("action_probs", outputs)
        self.assertEqual(outputs["action_probs"].shape, (self.batch_size, self.n_assets, 3))
        
        # Test probability properties
        probs = outputs["action_probs"]
        self.assertTrue(torch.all(probs >= 0) and torch.all(probs <= 1))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(self.batch_size, self.n_assets, device=self.device)))
        
        # Test output dimension
        self.assertEqual(self.parametric_head.get_output_dim(), self.n_assets * 3)
    
    def test_bayesian_discrete_head(self):
        # Test forward pass
        outputs = self.bayesian_head(self.x)
        self.assertIn("alphas", outputs)
        self.assertIn("betas", outputs)
        self.assertEqual(outputs["alphas"].shape, (self.batch_size, self.n_assets, 3))
        self.assertEqual(outputs["betas"].shape, (self.batch_size, self.n_assets, 3))
        
        # Test sampling strategies
        strategies = ["thompson", "entropy"]
        for strategy in strategies:
            samples = self.bayesian_head.sample(self.x, strategy=strategy)
            self.assertIn("action_probs", samples)
            self.assertEqual(samples["action_probs"].shape, (self.batch_size, self.n_assets, 3))
            
            # Test probability properties
            probs = samples["action_probs"]
            self.assertTrue(torch.all(probs >= 0) and torch.all(probs <= 1))
            self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(self.batch_size, self.n_assets, device=self.device)))
        
        # Test invalid strategy
        with self.assertRaises(ValueError):
            self.bayesian_head.sample(self.x, strategy="invalid")
        
        # Test output dimension
        self.assertEqual(self.bayesian_head.get_output_dim(), self.n_assets * 3 * 2)

class TestConfidenceHeads(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_dim = 64
        self.hidden_dim = 32
        self.n_assets = 3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create test input
        self.x = torch.randn(self.batch_size, self.input_dim, device=self.device)
        
        # Initialize heads
        self.parametric_head = ParametricConfidenceHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_assets=self.n_assets,
            device=self.device
        )
        
        self.bayesian_head = BayesianConfidenceHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_assets=self.n_assets,
            device=self.device
        )
    
    def test_parametric_confidence_head(self):
        # Test forward pass
        outputs = self.parametric_head(self.x)
        self.assertIn("confidences", outputs)
        self.assertEqual(outputs["confidences"].shape, (self.batch_size, self.n_assets))
        
        # Test confidence range
        confidences = outputs["confidences"]
        self.assertTrue(torch.all(confidences >= 0) and torch.all(confidences <= 1))
        
        # Test output dimension
        self.assertEqual(self.parametric_head.get_output_dim(), self.n_assets)
    
    def test_bayesian_confidence_head(self):
        # Test forward pass
        outputs = self.bayesian_head(self.x)
        self.assertIn("alphas", outputs)
        self.assertIn("betas", outputs)
        self.assertEqual(outputs["alphas"].shape, (self.batch_size, self.n_assets))
        self.assertEqual(outputs["betas"].shape, (self.batch_size, self.n_assets))
        
        # Test sampling strategies
        strategies = ["thompson", "entropy"]
        for strategy in strategies:
            samples = self.bayesian_head.sample(self.x, strategy=strategy)
            self.assertIn("confidences", samples)
            self.assertEqual(samples["confidences"].shape, (self.batch_size, self.n_assets))
            
            # Test confidence range
            confidences = samples["confidences"]
            self.assertTrue(torch.all(confidences >= 0) and torch.all(confidences <= 1))
        
        # Test invalid strategy
        with self.assertRaises(ValueError):
            self.bayesian_head.sample(self.x, strategy="invalid")
        
        # Test output dimension
        self.assertEqual(self.bayesian_head.get_output_dim(), self.n_assets * 2)

if __name__ == "__main__":
    unittest.main() 