"""Unit tests for the PPO agent."""
import unittest
import numpy as np
import torch
from unittest.mock import MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) )

from models.agents.ppo_agent import PPOAgent
from models.action_interpreters.discrete_interpreter import DiscreteInterpreter
from models.action_interpreters.confidence_scaled_interpreter import ConfidenceScaledInterpreter
from models.networks.unified_network import UnifiedNetwork
from models.agents.temperature_manager import TemperatureManager
from config.temperature import get_temperature_config

class TestPPOAgent(unittest.TestCase):
    """Test cases for the PPO agent."""
    def setUp(self):
        """Set up test fixtures."""
        self.n_assets = 3
        self.window_size = 20
        self.max_position_size = 10
        self.update_frequency = 4
        # Base network configuration
        self.base_network_config = {
            'n_assets': self.n_assets,
            'window_size': self.window_size,
            'processors': {
                'price': {
                    'enabled': True,
                    'hidden_dim': 64
                },
                'technical': {
                    'enabled': True,
                    'tech_dim': 15,
                    'hidden_dim': 64
                }
            },
            'backbone': {
                'type': 'mlp',
                'hidden_dims': [128, 64],
                'dropout': 0.1
            }
        }
        # Create discrete network config
        self.discrete_network_config = self.base_network_config.copy()
        self.discrete_network_config['heads'] = {
            'discrete': {
                'enabled': True,
                'type': 'parametric',
                'hidden_dim': 64
            },
            'value': {
                'enabled': True,
                'type': 'parametric',
                'hidden_dim': 64
            }
        }
        # Create confidence network config
        self.confidence_network_config = self.base_network_config.copy()
        self.confidence_network_config['heads'] = {
            'discrete': {
                'enabled': True,
                'type': 'parametric',
                'hidden_dim': 64
            },
            'confidence': {
                'enabled': True,
                'type': 'parametric',
                'hidden_dim': 64
            },
            'value': {
                'enabled': True,
                'type': 'parametric',
                'hidden_dim': 64
            }
        }
        # Create interpreters
        self.discrete_interpreter = DiscreteInterpreter(
            n_assets=self.n_assets,
            max_trade_size=self.max_position_size
        )
        self.confidence_interpreter = ConfidenceScaledInterpreter(
            n_assets=self.n_assets,
            max_trade_size=self.max_position_size
        )
        # Create temperature managers
        self.discrete_temp_manager = TemperatureManager(
            update_frequency=self.update_frequency,
            head_configs=get_temperature_config(self.discrete_network_config)['head_configs']
        )
        self.confidence_temp_manager = TemperatureManager(
            update_frequency=self.update_frequency,
            head_configs=get_temperature_config(self.confidence_network_config)['head_configs']
        )
        # Create agents with their respective configs
        self.discrete_agent = PPOAgent(
            network_config=self.discrete_network_config,
            interpreter=self.discrete_interpreter,
            temperature_manager=self.discrete_temp_manager,
            update_frequency=self.update_frequency,
            device="cpu"
        )
        self.confidence_agent = PPOAgent(
            network_config=self.confidence_network_config,
            interpreter=self.confidence_interpreter,
            temperature_manager=self.confidence_temp_manager,
            update_frequency=self.update_frequency,
            device="cpu"
        )
    def test_initialization(self):
        """Test agent initialization."""
        self.assertIsInstance(self.discrete_agent.network, UnifiedNetwork)
        self.assertEqual(self.discrete_agent.steps, 0)
        self.assertEqual(self.discrete_agent.gamma, 0.99)
        self.assertEqual(self.discrete_agent.entropy_coef, 0.01)
        self.assertEqual(self.discrete_agent.value_coef, 0.5)
        self.assertIsInstance(self.confidence_agent.network, UnifiedNetwork)
        self.assertEqual(self.confidence_agent.steps, 0)
        self.assertEqual(self.confidence_agent.gamma, 0.99)
        self.assertEqual(self.confidence_agent.entropy_coef, 0.01)
        self.assertEqual(self.confidence_agent.value_coef, 0.5)
    def test_get_intended_action_discrete(self):
        """Test action selection for discrete interpreter."""
        observations = {
            'price': np.random.randn(self.n_assets, self.window_size),
            'technical': np.random.randn(self.n_assets, 15)
        }
        mock_output = {
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]
            ]),
            'value': torch.tensor([0.5]),
            'entropy': torch.tensor([0.5])
        }
        self.discrete_agent.network = MagicMock(return_value=mock_output)
        scaled_actions, action_choices = self.discrete_agent.get_intended_action(
            observations=observations,
            deterministic=True
        )
        self.assertEqual(scaled_actions.shape, (self.n_assets,))
        self.assertTrue(np.all(np.isin(action_choices, [-1, 0, 1])))
        scaled_actions, action_choices = self.discrete_agent.get_intended_action(
            observations=observations,
            deterministic=False
        )
        self.assertEqual(scaled_actions.shape, (self.n_assets,))
        self.assertTrue(np.all(np.isin(action_choices, [-1, 0, 1])))
    def test_get_intended_action_confidence(self):
        """Test action selection for confidence-scaled interpreter."""
        observations = {
            'price': np.random.randn(self.n_assets, self.window_size),
            'technical': np.random.randn(self.n_assets, 15)
        }
        mock_output = {
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]
            ]),
            'confidences': torch.tensor([0.9, 0.5, 0.7]),
            'value': torch.tensor([0.5]),
            'entropy': torch.tensor([0.5])
        }
        self.confidence_agent.network = MagicMock(return_value=mock_output)
        scaled_actions, action_choices = self.confidence_agent.get_intended_action(
            observations=observations,
            deterministic=True
        )
        self.assertEqual(scaled_actions.shape, (self.n_assets,))
        self.assertTrue(np.all(scaled_actions >= -self.max_position_size) and np.all(scaled_actions <= self.max_position_size))
        self.assertTrue(np.all(np.isin(action_choices, [-1, 0, 1])))
        scaled_actions, action_choices = self.confidence_agent.get_intended_action(
            observations=observations,
            deterministic=False
        )
        self.assertEqual(scaled_actions.shape, (self.n_assets,))
        self.assertTrue(np.all(scaled_actions >= -self.max_position_size) and np.all(scaled_actions <= self.max_position_size))
        self.assertTrue(np.all(np.isin(action_choices, [-1, 0, 1])))
    def test_add_to_rollout(self):
        """Test adding transitions to rollout buffer."""
        observations = {
            'price': np.random.randn(self.n_assets, self.window_size),
            'technical': np.random.randn(self.n_assets, 15)
        }
        next_observations = {
            'price': np.random.randn(self.n_assets, self.window_size),
            'technical': np.random.randn(self.n_assets, 15)
        }
        action = np.array([self.max_position_size, 0, -self.max_position_size])
        action_choice = np.array([1, 0, -1])
        reward = 1.0
        done = False
        mock_output = {
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]
            ]),
            'value': torch.tensor([0.5]),
            'entropy': torch.tensor([0.5])
        }
        self.discrete_agent.network = MagicMock(return_value=mock_output)
        self.discrete_agent.get_intended_action(observations, deterministic=False)
        self.discrete_agent.add_to_rollout(
            observation=observations,
            action=action,
            action_choice=action_choice,
            reward=reward,
            next_observation=next_observations,
            done=done
        )
        self.assertEqual(len(self.discrete_agent.rollout), 1)
        transition = self.discrete_agent.rollout[0]
        self.assertIn('obs', transition.keys())
        self.assertIn('next_obs', transition.keys())
        self.assertIn('action_choices', transition.keys())
        self.assertIn('log_probs', transition.keys())
        self.assertIn('rewards', transition.keys())
        self.assertIn('dones', transition.keys())
        self.assertIn('values', transition.keys())
    def test_update(self):
        """Test network update using rollout data."""
        for i in range(5):
            observations = {
                'price': np.random.randn(self.n_assets, self.window_size),
                'technical': np.random.randn(self.n_assets, 15)
            }
            next_observations = {
                'price': np.random.randn(self.n_assets, self.window_size),
                'technical': np.random.randn(self.n_assets, 15)
            }
            action = np.random.choice([-self.max_position_size, 0, self.max_position_size], size=self.n_assets)
            action_choice = np.sign(action)
            reward = np.random.randn()
            done = np.array([1]) if i == 4 else np.array([0])
            # Use real network
            self.discrete_agent.get_intended_action(observations, deterministic=False)
            self.discrete_agent.add_to_rollout(
                observation=observations,
                action=action,
                action_choice=action_choice,
                reward=reward,
                next_observation=next_observations,
                done=done
            )
        # Use real network for update
        metrics = self.discrete_agent.update()
        self.assertIn('policy_loss', metrics)
        self.assertIn('value_loss', metrics)
        self.assertIn('entropy_loss', metrics)
        self.assertIn('total_loss', metrics)
        self.assertEqual(len(self.discrete_agent.rollout), 0)
    def test_save_load(self):
        """Test saving and loading agent state."""
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            self.discrete_agent.save(tmpdir)
            new_temp_manager = TemperatureManager(
                update_frequency=self.update_frequency,
                head_configs=get_temperature_config(self.discrete_network_config)['head_configs']
            )
            new_agent = PPOAgent(
                network_config=self.discrete_network_config.copy(),
                interpreter=self.discrete_interpreter,
                temperature_manager=new_temp_manager,
                update_frequency=self.update_frequency,
                device="cpu"
            )
            new_agent.load(tmpdir)
            self.assertEqual(new_agent.steps, self.discrete_agent.steps)
            for p1, p2 in zip(new_agent.network.parameters(), self.discrete_agent.network.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
    def test_save_load_confidence(self):
        """Test saving and loading agent state for confidence agent."""
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            self.confidence_agent.save(tmpdir)
            new_temp_manager = TemperatureManager(
                update_frequency=self.update_frequency,
                head_configs=get_temperature_config(self.confidence_network_config)['head_configs']
            )
            new_agent = PPOAgent(
                network_config=self.confidence_network_config.copy(),
                interpreter=self.confidence_interpreter,
                temperature_manager=new_temp_manager,
                update_frequency=self.update_frequency,
                device="cpu"
            )
            new_agent.load(tmpdir)
            self.assertEqual(new_agent.steps, self.confidence_agent.steps)
            for p1, p2 in zip(new_agent.network.parameters(), self.confidence_agent.network.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
    def test_bayesian_outputs(self):
        """Test agent with Bayesian network outputs."""
        bayesian_config = self.confidence_network_config.copy()
        bayesian_config['heads'] = {
            'discrete': {
                'enabled': True,
                'type': 'bayesian',
                'hidden_dim': 64
            },
            'confidence': {
                'enabled': True,
                'type': 'bayesian',
                'hidden_dim': 64
            },
            'value': {
                'enabled': True,
                'type': 'parametric',
                'hidden_dim': 64
            }
        }
        bayesian_temp_manager = TemperatureManager(
            update_frequency=self.update_frequency,
            head_configs=get_temperature_config(bayesian_config)['head_configs']
        )
        bayesian_agent = PPOAgent(
            network_config=bayesian_config,
            interpreter=self.confidence_interpreter,
            temperature_manager=bayesian_temp_manager,
            update_frequency=self.update_frequency,
            device="cpu"
        )
        observations = {
            'price': np.random.randn(self.n_assets, self.window_size),
            'technical': np.random.randn(self.n_assets, 15)
        }

if __name__ == '__main__':
    unittest.main() 