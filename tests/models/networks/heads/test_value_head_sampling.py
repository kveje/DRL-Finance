import unittest
import torch
import numpy as np

import sys
import os

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from models.networks.heads.value_head import BayesianValueHead

class TestBayesianValueHeadSampling(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.input_dim = 16
        self.hidden_dim = 8
        self.n_assets = 1  # Value head outputs a single value
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.head = BayesianValueHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_assets=self.n_assets,
            device=self.device
        )
        self.x_batch = torch.randn(self.batch_size, self.input_dim, device=self.device)
        self.x_single = torch.randn(self.input_dim, device=self.device)
        self.n_samples = 5000

    def test_thompson_sampling_batch(self):
        samples_list = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_batch, strategy="thompson")
            samples_list.append(out["value"].detach().cpu().numpy())
        samples_arr = np.stack(samples_list, axis=0)  # (n_samples, batch_size, 1)
        empirical_means = samples_arr.mean(axis=0).squeeze(-1)  # (batch_size,)
        with torch.no_grad():
            params = self.head(self.x_batch)
            means = params["mean"].detach().cpu().numpy().squeeze(-1)
        self.assertEqual(empirical_means.shape, (self.batch_size,))
        self.assertTrue(np.allclose(empirical_means, means, atol=0.05))

    def test_thompson_sampling_single(self):
        samples_list = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_single, strategy="thompson")
            samples_list.append(out["value"].detach().cpu().numpy())
        samples_arr = np.stack(samples_list, axis=0)  # (n_samples, 1)
        empirical_mean = samples_arr.mean(axis=0).squeeze(-1)  # ()
        with torch.no_grad():
            params = self.head(self.x_single)
            mean = params["mean"].detach().cpu().numpy().squeeze(-1)
        self.assertTrue(np.allclose(empirical_mean, mean, atol=0.05))

    def test_optimistic_sampling_batch(self):
        optimism_factor = 2.0
        samples_list = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_batch, strategy="optimistic", optimism_factor=optimism_factor)
            samples_list.append(out["value"].detach().cpu().numpy())
        samples_arr = np.stack(samples_list, axis=0)
        optimistic_means = samples_arr.mean(axis=0).squeeze(-1)
        # Compare to thompson mean
        thompson_list = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_batch, strategy="thompson")
            thompson_list.append(out["value"].detach().cpu().numpy())
        thompson_arr = np.stack(thompson_list, axis=0)
        thompson_means = thompson_arr.mean(axis=0).squeeze(-1)
        self.assertTrue((optimistic_means > thompson_means - 1e-3).mean() > 0.95)

    def test_ucb_sampling_batch(self):
        exploration_factor = 2.0
        out = self.head(self.x_batch)
        means = out["mean"].detach().cpu().numpy().squeeze(-1)
        stds = out["std"].detach().cpu().numpy().squeeze(-1)
        ucb = means + exploration_factor * stds
        ucb_samples = self.head.sample(self.x_batch, strategy="ucb", exploration_factor=exploration_factor)["value"].detach().cpu().numpy().squeeze(-1)
        self.assertTrue(np.allclose(ucb_samples, ucb, atol=1e-5))

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            self.head.sample(self.x_batch, strategy="invalid")

if __name__ == "__main__":
    unittest.main() 