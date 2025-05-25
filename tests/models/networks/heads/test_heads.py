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
        
        # Create test inputs
        self.x_batch = torch.randn(self.batch_size, self.input_dim, device=self.device)
        self.x_single = torch.randn(self.input_dim, device=self.device)
        
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
        # Test batched forward pass
        outputs = self.parametric_head(self.x_batch)
        self.assertIn("value", outputs)
        self.assertEqual(outputs["value"].shape, (self.batch_size, 1))
        
        # Test single sample forward pass
        outputs_single = self.parametric_head(self.x_single)
        self.assertIn("value", outputs_single)
        self.assertEqual(outputs_single["value"].shape, (1,))
        
        # Test output dimension
        self.assertEqual(self.parametric_head.get_output_dim(), 1)  # Single value estimate
    
    def test_bayesian_value_head(self):
        # Test batched forward pass
        outputs = self.bayesian_head(self.x_batch)
        self.assertIn("mean", outputs)
        self.assertIn("std", outputs)
        self.assertEqual(outputs["mean"].shape, (self.batch_size, 1))
        self.assertEqual(outputs["std"].shape, (self.batch_size, 1))
        
        # Test single sample forward pass
        outputs_single = self.bayesian_head(self.x_single)
        self.assertIn("mean", outputs_single)
        self.assertIn("std", outputs_single)
        self.assertEqual(outputs_single["mean"].shape, (1,))
        self.assertEqual(outputs_single["std"].shape, (1,))
        
        # Test sampling strategies for batched input
        strategies = ["thompson", "optimistic", "ucb"]
        for strategy in strategies:
            samples = self.bayesian_head.sample(self.x_batch, strategy=strategy)
            self.assertIn("value", samples)
            self.assertEqual(samples["value"].shape, (self.batch_size, 1))
        
        # Test sampling strategies for single sample
        for strategy in strategies:
            samples = self.bayesian_head.sample(self.x_single, strategy=strategy)
            self.assertIn("value", samples)
            self.assertEqual(samples["value"].shape, (1,))
        
        # Test invalid strategy
        with self.assertRaises(ValueError):
            self.bayesian_head.sample(self.x_batch, strategy="invalid")
        
        # Test output dimension
        self.assertEqual(self.bayesian_head.get_output_dim(), 2)  # Mean and std for single value

class TestDiscreteHeads(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_dim = 64
        self.hidden_dim = 32
        self.n_assets = 3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create test inputs
        self.x_batch = torch.randn(self.batch_size, self.input_dim, device=self.device)
        self.x_single = torch.randn(self.input_dim, device=self.device)
        
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
        # Test batched forward pass
        outputs = self.parametric_head(self.x_batch)
        self.assertIn("action_probs", outputs)
        self.assertEqual(outputs["action_probs"].shape, (self.batch_size, self.n_assets, 3))
        
        # Test single sample forward pass
        outputs_single = self.parametric_head(self.x_single)
        self.assertIn("action_probs", outputs_single)
        self.assertEqual(outputs_single["action_probs"].shape, (self.n_assets, 3))
        
        # Test probability properties for batched input
        probs = outputs["action_probs"]
        self.assertTrue(torch.all(probs >= 0) and torch.all(probs <= 1))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(self.batch_size, self.n_assets, device=self.device)))
        
        # Test probability properties for single sample
        probs_single = outputs_single["action_probs"]
        self.assertTrue(torch.all(probs_single >= 0) and torch.all(probs_single <= 1))
        self.assertTrue(torch.allclose(probs_single.sum(dim=-1), torch.ones(self.n_assets, device=self.device)))
        
        # Test output dimension
        self.assertEqual(self.parametric_head.get_output_dim(), self.n_assets * 3)
    
    def test_bayesian_discrete_head(self):
        # Test batched forward pass
        outputs = self.bayesian_head(self.x_batch)
        self.assertIn("alphas", outputs)
        self.assertIn("betas", outputs)
        self.assertEqual(outputs["alphas"].shape, (self.batch_size, self.n_assets, 3))
        self.assertEqual(outputs["betas"].shape, (self.batch_size, self.n_assets, 3))
        
        # Test single sample forward pass
        outputs_single = self.bayesian_head(self.x_single)
        self.assertIn("alphas", outputs_single)
        self.assertIn("betas", outputs_single)
        self.assertEqual(outputs_single["alphas"].shape, (self.n_assets, 3))
        self.assertEqual(outputs_single["betas"].shape, (self.n_assets, 3))
        
        # Test sampling strategies for batched input
        strategies = ["thompson", "entropy"]
        for strategy in strategies:
            samples = self.bayesian_head.sample(self.x_batch, strategy=strategy)
            self.assertIn("action_probs", samples)
            self.assertEqual(samples["action_probs"].shape, (self.batch_size, self.n_assets, 3))
            
            # Test probability properties
            probs = samples["action_probs"]
            self.assertTrue(torch.all(probs >= 0) and torch.all(probs <= 1))
            self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(self.batch_size, self.n_assets, device=self.device)))
        
        # Test sampling strategies for single sample
        for strategy in strategies:
            samples = self.bayesian_head.sample(self.x_single, strategy=strategy)
            self.assertIn("action_probs", samples)
            self.assertEqual(samples["action_probs"].shape, (self.n_assets, 3))
            
            # Test probability properties
            probs = samples["action_probs"]
            self.assertTrue(torch.all(probs >= 0) and torch.all(probs <= 1))
            self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(self.n_assets, device=self.device)))
        
        # Test invalid strategy
        with self.assertRaises(ValueError):
            self.bayesian_head.sample(self.x_batch, strategy="invalid")
        
        # Test output dimension
        self.assertEqual(self.bayesian_head.get_output_dim(), self.n_assets * 3 * 2)

class TestConfidenceHeads(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_dim = 64
        self.hidden_dim = 32
        self.n_assets = 3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create test inputs
        self.x_batch = torch.randn(self.batch_size, self.input_dim, device=self.device)
        self.x_single = torch.randn(self.input_dim, device=self.device)
        
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
        # Test batched forward pass
        outputs = self.parametric_head(self.x_batch)
        self.assertIn("confidences", outputs)
        self.assertEqual(outputs["confidences"].shape, (self.batch_size, self.n_assets))
        
        # Test single sample forward pass
        outputs_single = self.parametric_head(self.x_single)
        self.assertIn("confidences", outputs_single)
        self.assertEqual(outputs_single["confidences"].shape, (self.n_assets,))
        
        # Test confidence range for batched input
        confidences = outputs["confidences"]
        self.assertTrue(torch.all(confidences >= 0) and torch.all(confidences <= 1))
        
        # Test confidence range for single sample
        confidences_single = outputs_single["confidences"]
        self.assertTrue(torch.all(confidences_single >= 0) and torch.all(confidences_single <= 1))
        
        # Test output dimension
        self.assertEqual(self.parametric_head.get_output_dim(), self.n_assets)
    
    def test_bayesian_confidence_head(self):
        # Test batched forward pass
        outputs = self.bayesian_head(self.x_batch)
        self.assertIn("alphas", outputs)
        self.assertIn("betas", outputs)
        self.assertEqual(outputs["alphas"].shape, (self.batch_size, self.n_assets))
        self.assertEqual(outputs["betas"].shape, (self.batch_size, self.n_assets))
        
        # Test single sample forward pass
        outputs_single = self.bayesian_head(self.x_single)
        self.assertIn("alphas", outputs_single)
        self.assertIn("betas", outputs_single)
        self.assertEqual(outputs_single["alphas"].shape, (self.n_assets,))
        self.assertEqual(outputs_single["betas"].shape, (self.n_assets,))
        
        # Test sampling strategies for batched input
        strategies = ["thompson", "entropy"]
        for strategy in strategies:
            samples = self.bayesian_head.sample(self.x_batch, strategy=strategy)
            self.assertIn("confidences", samples)
            self.assertEqual(samples["confidences"].shape, (self.batch_size, self.n_assets))
            
            # Test confidence range
            confidences = samples["confidences"]
            self.assertTrue(torch.all(confidences >= 0) and torch.all(confidences <= 1))
        
        # Test sampling strategies for single sample
        for strategy in strategies:
            samples = self.bayesian_head.sample(self.x_single, strategy=strategy)
            self.assertIn("confidences", samples)
            self.assertEqual(samples["confidences"].shape, (self.n_assets,))
            
            # Test confidence range
            confidences = samples["confidences"]
            self.assertTrue(torch.all(confidences >= 0) and torch.all(confidences <= 1))
        
        # Test invalid strategy
        with self.assertRaises(ValueError):
            self.bayesian_head.sample(self.x_batch, strategy="invalid")
        
        # Test output dimension
        self.assertEqual(self.bayesian_head.get_output_dim(), self.n_assets * 2)

if __name__ == "__main__":
    unittest.main() 