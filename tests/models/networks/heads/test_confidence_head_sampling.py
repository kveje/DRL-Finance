import unittest
import torch
import numpy as np

import sys
import os

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from models.networks.heads.confidence_head import BayesianConfidenceHead

class TestBayesianConfidenceHeadSampling(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.input_dim = 16
        self.hidden_dim = 8
        self.n_assets = 4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.head = BayesianConfidenceHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_assets=self.n_assets,
            device=self.device
        )
        self.x_batch = torch.randn(self.batch_size, self.input_dim, device=self.device)
        self.x_single = torch.randn(self.input_dim, device=self.device)
        self.n_samples = 5000

    def test_thompson_sampling_batch(self):
        # Collect many samples to check empirical mean
        samples_list = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_batch, strategy="thompson")
            samples_list.append(out["confidences"].detach().cpu().numpy())
        samples_arr = np.stack(samples_list, axis=0)  # (n_samples, batch_size, n_assets)
        empirical_means = samples_arr.mean(axis=0)
        # Theoretical mean
        with torch.no_grad():
            params = self.head(self.x_batch)
            alphas = params["alphas"].detach().cpu().numpy()
            betas = params["betas"].detach().cpu().numpy()
            theoretical_means = alphas / (alphas + betas)
        # Check shape
        self.assertEqual(empirical_means.shape, (self.batch_size, self.n_assets))

        # Check empirical mean close to theoretical mean
        self.assertTrue(np.allclose(empirical_means, theoretical_means, atol=0.05))
        # Check range
        self.assertTrue(np.all((samples_arr >= 0) & (samples_arr <= 1)))

    def test_thompson_sampling_single(self):
        samples_list = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_single, strategy="thompson")
            samples_list.append(out["confidences"].detach().cpu().numpy())
        samples_arr = np.stack(samples_list, axis=0)  # (n_samples, n_assets)
        empirical_means = samples_arr.mean(axis=0)
        with torch.no_grad():
            params = self.head(self.x_single)
            alphas = params["alphas"].detach().cpu().numpy()
            betas = params["betas"].detach().cpu().numpy()
            theoretical_means = alphas / (alphas + betas)
        self.assertEqual(empirical_means.shape, (self.n_assets,))
        self.assertTrue(np.allclose(empirical_means, theoretical_means, atol=0.05))
        self.assertTrue(np.all((samples_arr >= 0) & (samples_arr <= 1)))

    def test_entropy_sampling_batch(self):
        entropy_weight = 0.5
        samples_list = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_batch, strategy="entropy", entropy_weight=entropy_weight)
            samples_list.append(out["confidences"].detach().cpu().numpy())
        samples_arr = np.stack(samples_list, axis=0)
        empirical_means = samples_arr.mean(axis=0)
        # With positive entropy_weight, mean should be higher than thompson
        thompson_means = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_batch, strategy="thompson")
            thompson_means.append(out["confidences"].detach().cpu().numpy())
        thompson_arr = np.stack(thompson_means, axis=0)
        thompson_empirical = thompson_arr.mean(axis=0)
        # Check shape
        self.assertEqual(empirical_means.shape, (self.batch_size, self.n_assets))
        # Check range (allowing for some out-of-bounds due to entropy term)
        self.assertTrue(np.all(samples_arr < 2.0) and np.all(samples_arr > -1.0))
        # Mean should be higher (on average)
        self.assertTrue(empirical_means.mean() > thompson_empirical.mean() - 1e-3)

    def test_entropy_sampling_single(self):
        entropy_weight = 0.5
        samples_list = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_single, strategy="entropy", entropy_weight=entropy_weight)
            samples_list.append(out["confidences"].detach().cpu().numpy())
        samples_arr = np.stack(samples_list, axis=0)
        empirical_means = samples_arr.mean(axis=0)
        thompson_means = []
        for _ in range(self.n_samples):
            out = self.head.sample(self.x_single, strategy="thompson")
            thompson_means.append(out["confidences"].detach().cpu().numpy())
        thompson_arr = np.stack(thompson_means, axis=0)
        thompson_empirical = thompson_arr.mean(axis=0)
        self.assertEqual(empirical_means.shape, (self.n_assets,))
        self.assertTrue(np.all(samples_arr < 2.0) and np.all(samples_arr > -1.0))
        self.assertTrue(empirical_means.mean() > thompson_empirical.mean() - 1e-3)

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            self.head.sample(self.x_batch, strategy="invalid")

if __name__ == "__main__":
    unittest.main() 