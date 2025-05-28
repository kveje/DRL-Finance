import unittest
import torch
import numpy as np

import sys
import os 

#Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from models.networks.heads.sampling import BayesianSampler

class TestBayesianSampler(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.n_assets = 5
        self.means = torch.randn(self.batch_size, self.n_assets)
        self.stds = torch.rand(self.batch_size, self.n_assets) + 0.1  # avoid zero std
        self.alphas = torch.rand(self.batch_size, self.n_assets) * 2 + 1  # >1 for stability
        self.betas = torch.rand(self.batch_size, self.n_assets) * 2 + 1
        self.n_samples = 10000  # For empirical statistics

    def test_thompson_sample(self):
        samples = BayesianSampler.thompson_sample(self.means, self.stds)
        self.assertEqual(samples.shape, (self.batch_size, self.n_assets))
        self.assertFalse(torch.allclose(samples, self.means))
        # Empirical mean/variance test
        means = torch.zeros_like(self.means)
        stds = torch.ones_like(self.stds)
        for i in range(self.n_samples):
            s = BayesianSampler.thompson_sample(self.means, self.stds)
            means += s
        means /= self.n_samples
        # Should be close to theoretical mean
        self.assertTrue(torch.allclose(means, self.means, atol=0.05))

    def test_beta_sample(self):
        samples = BayesianSampler.beta_sample(self.alphas, self.betas)
        self.assertEqual(samples.shape, (self.batch_size, self.n_assets))
        self.assertTrue(torch.all(samples >= 0) and torch.all(samples <= 1))
        # Empirical mean test
        empirical_means = torch.zeros_like(self.alphas)
        for i in range(self.n_samples):
            s = BayesianSampler.beta_sample(self.alphas, self.betas)
            empirical_means += s
        empirical_means /= self.n_samples
        theoretical_means = self.alphas / (self.alphas + self.betas)
        self.assertTrue(torch.allclose(empirical_means, theoretical_means, atol=0.05))

    def test_optimistic_sample(self):
        optimism_factor = 2.0
        samples = BayesianSampler.optimistic_sample(self.means, self.stds, optimism_factor=optimism_factor)
        self.assertEqual(samples.shape, (self.batch_size, self.n_assets))
        thompson_samples = BayesianSampler.thompson_sample(self.means, self.stds)
        self.assertTrue(samples.mean() > thompson_samples.mean() - 0.1)
        # Empirical mean test: optimistic mean should be higher than normal mean
        normal_means = torch.zeros_like(self.means)
        optimistic_means = torch.zeros_like(self.means)
        for i in range(self.n_samples):
            normal_means += BayesianSampler.thompson_sample(self.means, self.stds)
            optimistic_means += BayesianSampler.optimistic_sample(self.means, self.stds, optimism_factor=optimism_factor)
        normal_means /= self.n_samples
        optimistic_means /= self.n_samples
        self.assertTrue(torch.all(optimistic_means > normal_means))

    def test_entropy_sample(self):
        entropy_weight = 0.5
        samples = BayesianSampler.entropy_sample(self.alphas, self.betas, entropy_weight=entropy_weight)
        self.assertEqual(samples.shape, (self.batch_size, self.n_assets))
        self.assertTrue(torch.all(samples < 2.0) and torch.all(samples > -1.0))
        # Empirical mean test: with zero entropy_weight, should match beta mean
        empirical_means = torch.zeros_like(self.alphas)
        for i in range(self.n_samples):
            s = BayesianSampler.entropy_sample(self.alphas, self.betas, entropy_weight=0.0)
            empirical_means += s
        empirical_means /= self.n_samples
        theoretical_means = self.alphas / (self.alphas + self.betas)
        self.assertTrue(torch.allclose(empirical_means, theoretical_means, atol=0.05))
        # With positive entropy_weight, mean should be shifted up (since entropy is always positive for Beta)
        empirical_means_entropy = torch.zeros_like(self.alphas)
        for i in range(self.n_samples):
            s = BayesianSampler.entropy_sample(self.alphas, self.betas, entropy_weight=entropy_weight)
            empirical_means_entropy += s
        empirical_means_entropy /= self.n_samples

        fraction_greater = (empirical_means_entropy > empirical_means).float().mean().item()
        self.assertTrue(fraction_greater > 0.95)

    def test_ucb_sample(self):
        exploration_factor = 2.0
        samples = BayesianSampler.ucb_sample(self.means, self.stds, exploration_factor=exploration_factor)
        self.assertEqual(samples.shape, (self.batch_size, self.n_assets))
        # UCB samples should be means + exploration_factor * stds
        expected = self.means + exploration_factor * self.stds
        self.assertTrue(torch.allclose(samples, expected, atol=1e-6))

if __name__ == "__main__":
    unittest.main() 