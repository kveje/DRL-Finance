import unittest
import numpy as np
import torch
from unittest.mock import MagicMock
from tensordict import TensorDict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) )

from models.agents.sac_agent import SACAgent
from models.action_interpreters.discrete_interpreter import DiscreteInterpreter
from models.networks.unified_network import UnifiedNetwork
from models.agents.temperature_manager import TemperatureManager

class TestSACAgent(unittest.TestCase):
    """Test cases for the SAC agent."""
    def setUp(self):
        self.n_assets = 3
        self.window_size = 20
        self.batch_size = 16
        self.max_trade_size = 10
        self.device = "cpu"
        # Minimal network config
        self.network_config = {
            'n_assets': self.n_assets,
            'window_size': self.window_size,
            'processors': {
                'price': {'enabled': True, 'hidden_dim': 32},
                'technical': {'enabled': True, 'tech_dim': 8, 'hidden_dim': 16}
            },
            'backbone': {'type': 'mlp', 'hidden_dims': [32]},
            'heads': {
                'discrete': {'enabled': True, 'type': 'parametric', 'hidden_dim': 16}
            }
        }
        self.interpreter = DiscreteInterpreter(
            n_assets=self.n_assets,
            max_trade_size=self.max_trade_size
        )
        self.temperature_manager = TemperatureManager(
            update_frequency=1,
            head_configs={
                "discrete": {"initial_temp": 1.0, "final_temp": 0.1, "decay_rate": 1.0, "decay_fraction": 0.8}
            }
        )
        self.agent = SACAgent(
            network_config=self.network_config,
            interpreter=self.interpreter,
            temperature_manager=self.temperature_manager,
            update_frequency=1,
            device=self.device
        )
    def test_initialization(self):
        self.assertIsInstance(self.agent.network, UnifiedNetwork)
        self.assertIsInstance(self.agent.critic_1, UnifiedNetwork)
        self.assertIsInstance(self.agent.critic_2, UnifiedNetwork)
        self.assertEqual(self.agent.steps, 0)
    def test_get_intended_action(self):
        observations = {
            'price': np.random.randn(self.n_assets, self.window_size),
            'technical': np.random.randn(self.n_assets, 8)
        }
        # Mock actor output
        mock_output = {'action_probs': torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])}
        self.agent.actor = MagicMock(return_value=mock_output)
        # Deterministic
        scaled_actions, action_choices = self.agent.get_intended_action(observations, deterministic=True)
        self.assertEqual(scaled_actions.shape, (self.n_assets,))
        self.assertTrue(np.all(np.isin(action_choices, [-1, 0, 1])))
        # Stochastic
        self.agent.actor = MagicMock(return_value=mock_output)
        self.agent.interpreter.interpret_with_log_prob = MagicMock(return_value=(torch.tensor([0,0,0]), torch.tensor([0,0,0]), torch.tensor([0.0,0.0,0.0])))
        scaled_actions, action_choices = self.agent.get_intended_action(observations, deterministic=False)
        self.assertEqual(scaled_actions.shape, (3,))
    def test_update(self):
        # Create a real batch
        batch = {
            'observations': {
                'price': torch.randn(self.batch_size, self.n_assets, self.window_size, requires_grad=True),
                'technical': torch.randn(self.batch_size, self.n_assets, 8, requires_grad=True)
            },
            'actions': torch.randint(-1, 2, (self.batch_size, self.n_assets)),
            'rewards': torch.randn(self.batch_size),
            'next_observations': {
                'price': torch.randn(self.batch_size, self.n_assets, self.window_size, requires_grad=True),
                'technical': torch.randn(self.batch_size, self.n_assets, 8, requires_grad=True)
            },
            'dones': torch.zeros(self.batch_size)
        }
        batch = TensorDict(batch)
        # Use real networks and interpreter (no mocks)
        metrics = self.agent.update(batch)
        self.assertIn('critic_1_loss', metrics)
        self.assertIn('critic_2_loss', metrics)
        self.assertIn('actor_loss', metrics)
        self.assertIn('critic_1_grad_norm', metrics)
        self.assertIn('critic_2_grad_norm', metrics)
        self.assertIn('actor_grad_norm', metrics)
    def test_save_load(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            self.agent.save(tmpdir)
            # Create new agent with same config
            new_agent = SACAgent(
                network_config=self.network_config,
                interpreter=self.interpreter,
                temperature_manager=self.temperature_manager,
                update_frequency=1,
                device=self.device
            )
            new_agent.load(tmpdir)
            self.assertEqual(new_agent.alpha, self.agent.alpha)
            self.assertEqual(new_agent.steps, self.agent.steps)
            # Check if actor weights match
            for p1, p2 in zip(new_agent.network.parameters(), self.agent.network.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

if __name__ == '__main__':
    unittest.main() 