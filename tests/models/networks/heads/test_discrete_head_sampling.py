import unittest
import torch
import numpy as np

import sys
import os

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from models.networks.heads.discrete_head import BayesianDiscreteHead

class TestBayesianDiscreteHeadSampling(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.input_dim = 16
        self.hidden_dim = 8
        self.n_assets = 4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.head = BayesianDiscreteHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_assets=self.n_assets,
            device=self.device
        )
        self.x_batch = torch.randn(self.batch_size, self.input_dim, device=self.device)
        self.x_single = torch.randn(self.input_dim, device=self.device)
        self.n_samples = 10
        self.n_actions = 3

    def test_thompson_sampling_batch(self):
        samples_list = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_batch, strategy="thompson")
            samples_list.append(out["action_probs"].detach().cpu().numpy())
        samples_arr = np.stack(samples_list, axis=0)  # (n_samples, batch_size, n_assets, n_actions)
        empirical_means = samples_arr.mean(axis=0)  # (batch_size, n_assets, n_actions)
        with torch.no_grad():
            params = self.head(self.x_batch)
            alphas = params["alphas"].detach().cpu().numpy()
            betas = params["betas"].detach().cpu().numpy()
            theoretical_means = alphas / (alphas + betas)
        self.assertEqual(empirical_means.shape, (self.batch_size, self.n_assets, self.n_actions))
        print(empirical_means)
        print(theoretical_means)
        self.assertTrue(np.allclose(empirical_means, theoretical_means, atol=0.05))
        self.assertTrue(np.all((samples_arr >= 0) & (samples_arr <= 1)))
        # Probabilities should sum to 1 along last axis
        self.assertTrue(np.allclose(samples_arr.sum(axis=-1), 1, atol=1e-2))

    def test_thompson_sampling_single(self):
        samples_list = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_single, strategy="thompson")
            samples_list.append(out["action_probs"].detach().cpu().numpy())
        samples_arr = np.stack(samples_list, axis=0)  # (n_samples, n_assets, n_actions)
        empirical_means = samples_arr.mean(axis=0)  # (n_assets, n_actions)
        with torch.no_grad():
            params = self.head(self.x_single)
            alphas = params["alphas"].detach().cpu().numpy()
            betas = params["betas"].detach().cpu().numpy()
            theoretical_means = alphas / (alphas + betas)
        self.assertEqual(empirical_means.shape, (self.n_assets, self.n_actions))
        print(empirical_means)
        print(theoretical_means)
        self.assertTrue(np.allclose(empirical_means, theoretical_means, atol=0.05))
        self.assertTrue(np.all((samples_arr >= 0) & (samples_arr <= 1)))
        self.assertTrue(np.allclose(samples_arr.sum(axis=-1), 1, atol=1e-2))

    def test_entropy_sampling_batch(self):
        entropy_weight = 0.5
        samples_list = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_batch, strategy="entropy", entropy_weight=entropy_weight)
            samples_list.append(out["action_probs"].detach().cpu().numpy())
        samples_arr = np.stack(samples_list, axis=0)
        empirical_means = samples_arr.mean(axis=0)
        thompson_list = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_batch, strategy="thompson")
            thompson_list.append(out["action_probs"].detach().cpu().numpy())
        thompson_arr = np.stack(thompson_list, axis=0)
        thompson_means = thompson_arr.mean(axis=0)
        self.assertEqual(empirical_means.shape, (self.batch_size, self.n_assets, self.n_actions))
        self.assertTrue(np.all(samples_arr < 2.0) and np.all(samples_arr > -1.0))
        self.assertTrue(empirical_means.mean() > thompson_means.mean() - 1e-3)
        self.assertTrue(np.allclose(samples_arr.sum(axis=-1), 1, atol=1e-2))

    def test_entropy_sampling_single(self):
        entropy_weight = 0.5
        samples_list = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_single, strategy="entropy", entropy_weight=entropy_weight)
            samples_list.append(out["action_probs"].detach().cpu().numpy())
        samples_arr = np.stack(samples_list, axis=0)
        empirical_means = samples_arr.mean(axis=0)
        thompson_list = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_single, strategy="thompson")
            thompson_list.append(out["action_probs"].detach().cpu().numpy())
        thompson_arr = np.stack(thompson_list, axis=0)
        thompson_means = thompson_arr.mean(axis=0)
        self.assertEqual(empirical_means.shape, (self.n_assets, self.n_actions))
        self.assertTrue(np.all(samples_arr < 2.0) and np.all(samples_arr > -1.0))
        self.assertTrue(empirical_means.mean() > thompson_means.mean() - 1e-3)
        self.assertTrue(np.allclose(samples_arr.sum(axis=-1), 1, atol=1e-2))

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            self.head.sample(self.x_batch, strategy="invalid")

if __name__ == "__main__":
    unittest.main() 