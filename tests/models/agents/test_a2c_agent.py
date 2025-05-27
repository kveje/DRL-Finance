"""Unit tests for the A2C agent."""
import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.agents.a2c_agent import A2CAgent
from models.action_interpreters.discrete_interpreter import DiscreteInterpreter
from models.action_interpreters.confidence_scaled_interpreter import ConfidenceScaledInterpreter
from models.networks.unified_network import UnifiedNetwork

class TestA2CAgent(unittest.TestCase):
    """Test cases for the A2C agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_assets = 3
        self.window_size = 20
        self.max_position_size = 10
        
        # Create mock environment
        self.env = MagicMock()
        self.env.n_assets = self.n_assets
        
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
        
        # Create agents with their respective configs
        self.discrete_agent = A2CAgent(
            env=self.env,
            network_config=self.discrete_network_config,
            interpreter=self.discrete_interpreter,
            device="cpu"
        )
        
        self.confidence_agent = A2CAgent(
            env=self.env,
            network_config=self.confidence_network_config,
            interpreter=self.confidence_interpreter,
            device="cpu"
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        # Test discrete agent
        self.assertIsInstance(self.discrete_agent.network, UnifiedNetwork)
        self.assertEqual(self.discrete_agent.steps, 0)
        self.assertEqual(self.discrete_agent.gamma, 0.99)
        self.assertEqual(self.discrete_agent.entropy_coef, 0.01)
        self.assertEqual(self.discrete_agent.value_coef, 0.5)
        
        # Test confidence agent
        self.assertIsInstance(self.confidence_agent.network, UnifiedNetwork)
        self.assertEqual(self.confidence_agent.steps, 0)
        self.assertEqual(self.confidence_agent.gamma, 0.99)
        self.assertEqual(self.confidence_agent.entropy_coef, 0.01)
        self.assertEqual(self.confidence_agent.value_coef, 0.5)
    
    def test_get_intended_action_discrete(self):
        """Test action selection for discrete interpreter."""
        # Create mock observations
        observations = {
            'price': np.random.randn(self.n_assets, self.window_size),  # (n_assets, window_size)
            'technical': np.random.randn(self.n_assets, 15)  # (n_assets, tech_dim)
        }
        current_position = np.zeros(self.n_assets)
        
        # Mock network output
        mock_output = {
            'action_probs': torch.tensor([[
                [0.8, 0.1, 0.1],  # First asset: strongly sell
                [0.1, 0.8, 0.1],  # Second asset: strongly hold
                [0.1, 0.1, 0.8]   # Third asset: strongly buy
            ]]),
            'value': torch.tensor([[0.5]])  # Value prediction
        }
        self.discrete_agent.network = MagicMock(return_value=mock_output)
        
        # Test deterministic action selection
        scaled_actions, action_choices = self.discrete_agent.get_intended_action(
            observations=observations,
            current_position=current_position,
            deterministic=True
        )
        self.assertEqual(scaled_actions.shape, (self.n_assets,))
        self.assertTrue(np.all(np.isin(scaled_actions, [-self.max_position_size, 0, self.max_position_size])))
        self.assertTrue(np.all(np.isin(action_choices, [-1, 0, 1])))
        
        # Test stochastic action selection
        scaled_actions, action_choices = self.discrete_agent.get_intended_action(
            observations=observations,
            current_position=current_position,
            deterministic=False
        )
        self.assertEqual(scaled_actions.shape, (self.n_assets,))
        self.assertTrue(np.all(np.isin(scaled_actions, [-self.max_position_size, 0, self.max_position_size])))
        self.assertTrue(np.all(np.isin(action_choices, [-1, 0, 1])))
    
    def test_get_intended_action_confidence(self):
        """Test action selection for confidence-scaled interpreter."""
        # Create mock observations
        observations = {
            'price': np.random.randn(self.n_assets, self.window_size),  # (n_assets, window_size)
            'technical': np.random.randn(self.n_assets, 15)  # (n_assets, tech_dim)
        }
        current_position = np.zeros(self.n_assets)
        
        # Mock network output
        mock_output = {
            'action_probs': torch.tensor([[
                [0.8, 0.1, 0.1],  # First asset: strongly sell
                [0.1, 0.8, 0.1],  # Second asset: strongly hold
                [0.1, 0.1, 0.8]   # Third asset: strongly buy
            ]]),
            'confidences': torch.tensor([[
                0.9,  # High confidence in first asset
                0.5,  # Medium confidence in second asset
                0.7   # High confidence in third asset
            ]]),
            'value': torch.tensor([[0.5]])  # Value prediction
        }
        self.confidence_agent.network = MagicMock(return_value=mock_output)
        
        # Test deterministic action selection
        scaled_actions, action_choices = self.confidence_agent.get_intended_action(
            observations=observations,
            current_position=current_position,
            deterministic=True
        )
        self.assertEqual(scaled_actions.shape, (self.n_assets,))
        self.assertTrue(np.all(scaled_actions >= -self.max_position_size) and np.all(scaled_actions <= self.max_position_size))
        self.assertTrue(np.all(np.isin(action_choices, [-1, 0, 1])))
        
        # Test stochastic action selection
        scaled_actions, action_choices = self.confidence_agent.get_intended_action(
            observations=observations,
            current_position=current_position,
            deterministic=False
        )
        self.assertEqual(scaled_actions.shape, (self.n_assets,))
        self.assertTrue(np.all(scaled_actions >= -self.max_position_size) and np.all(scaled_actions <= self.max_position_size))
        self.assertTrue(np.all(np.isin(action_choices, [-1, 0, 1])))
    
    def test_add_to_rollout(self):
        """Test adding transitions to rollout buffer."""
        # Create mock observations and actions
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
        
        # Mock network output for value prediction
        mock_output = {
            'action_probs': torch.tensor([[
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]
            ]]),
            'value': torch.tensor([[0.5]])
        }
        self.discrete_agent.network = MagicMock(return_value=mock_output)
        
        # Get action to set last_value and last_log_probs
        self.discrete_agent.get_intended_action(observations, np.zeros(self.n_assets), deterministic=False)
        
        # Add to rollout
        self.discrete_agent.add_to_rollout(
            observation=observations,
            action=action,
            action_choice=action_choice,
            reward=reward,
            next_observation=next_observations,
            done=done
        )
        
        # Check if data was added correctly
        self.assertEqual(len(self.discrete_agent.rollout_states), 1)
        self.assertEqual(len(self.discrete_agent.rollout_actions), 1)
        self.assertEqual(len(self.discrete_agent.rollout_rewards), 1)
        self.assertEqual(len(self.discrete_agent.rollout_dones), 1)
        self.assertEqual(len(self.discrete_agent.rollout_values), 1)
        self.assertEqual(len(self.discrete_agent.rollout_log_probs), 1)
    
    def test_update(self):
        """Test network update using rollout data."""
        # Add some transitions to rollout
        for _ in range(5):  # Add 5 transitions
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
            done = np.random.choice([True, False])
            
            # Mock network output
            mock_output = {
                'action_probs': torch.tensor([[
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8]
                ]]),
                'value': torch.tensor([[0.5]])
            }
            self.discrete_agent.network = MagicMock(return_value=mock_output)
            
            # Get action to set last_value and last_log_probs
            self.discrete_agent.get_intended_action(observations, np.zeros(self.n_assets), deterministic=False)
            
            # Add to rollout
            self.discrete_agent.add_to_rollout(
                observation=observations,
                action=action,
                action_choice=action_choice,
                reward=reward,
                next_observation=next_observations,
                done=done
            )
        
        # Mock network outputs for update
        mock_output = {
            'action_probs': torch.softmax(torch.randn(5, self.n_assets, 3, requires_grad=True), dim=-1),
            'value': torch.randn(5, 1, requires_grad=True),
            'entropy': torch.tensor(0.5)
        }
        self.discrete_agent.network = MagicMock(return_value=mock_output)
        
        # Test update
        metrics = self.discrete_agent.update({})  # Empty batch as A2C uses rollout data
        self.assertIn('policy_loss', metrics)
        self.assertIn('value_loss', metrics)
        self.assertIn('entropy_loss', metrics)
        self.assertIn('total_loss', metrics)
        
        # Check if rollout buffers were cleared
        self.assertEqual(len(self.discrete_agent.rollout_states), 0)
        self.assertEqual(len(self.discrete_agent.rollout_actions), 0)
        self.assertEqual(len(self.discrete_agent.rollout_rewards), 0)
        self.assertEqual(len(self.discrete_agent.rollout_dones), 0)
        self.assertEqual(len(self.discrete_agent.rollout_values), 0)
        self.assertEqual(len(self.discrete_agent.rollout_log_probs), 0)
    
    def test_save_load(self):
        """Test saving and loading agent state."""
        # Create temporary directory for saving
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save agent state
            self.discrete_agent.save(tmpdir)
            
            # Create new agent with EXACTLY the same config
            new_agent = A2CAgent(
                env=self.env,
                network_config=self.discrete_network_config.copy(),
                interpreter=self.discrete_interpreter,
                device="cpu"
            )
            
            # Load state
            new_agent.load(tmpdir)
            
            # Check if states match
            self.assertEqual(new_agent.steps, self.discrete_agent.steps)
            
            # Check if networks match
            for p1, p2 in zip(new_agent.network.parameters(), self.discrete_agent.network.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
    
    def test_save_load_confidence(self):
        """Test saving and loading agent state for confidence agent."""
        # Create temporary directory for saving
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save agent state
            self.confidence_agent.save(tmpdir)
            
            # Create new agent with EXACTLY the same config
            new_agent = A2CAgent(
                env=self.env,
                network_config=self.confidence_network_config.copy(),
                interpreter=self.confidence_interpreter,
                device="cpu"
            )
            
            # Load state
            new_agent.load(tmpdir)
            
            # Check if states match
            self.assertEqual(new_agent.steps, self.confidence_agent.steps)
            
            # Check if networks match
            for p1, p2 in zip(new_agent.network.parameters(), self.confidence_agent.network.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
    
    def test_bayesian_outputs(self):
        """Test agent with Bayesian network outputs."""
        # Update network config for Bayesian outputs
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
        
        # Create agent with Bayesian outputs
        bayesian_agent = A2CAgent(
            env=self.env,
            network_config=bayesian_config,
            interpreter=self.confidence_interpreter,
            device="cpu"
        )
        
        # Create mock observations
        observations = {
            'price': np.random.randn(self.n_assets, self.window_size),
            'technical': np.random.randn(self.n_assets, 15)
        }
        current_position = np.zeros(self.n_assets)
        
        # Mock network output
        mock_output = {
            'alphas': torch.tensor([[
                [8.0, 1.0, 1.0],  # Strong sell
                [1.0, 8.0, 1.0],  # Strong hold
                [1.0, 1.0, 8.0]   # Strong buy
            ]]),
            'betas': torch.tensor([[
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0]
            ]]),
            'value': torch.tensor([[0.5]])
        }
        bayesian_agent.network = MagicMock(return_value=mock_output)
        
        # Test action selection
        scaled_actions, action_choices = bayesian_agent.get_intended_action(
            observations=observations,
            current_position=current_position,
            deterministic=False
        )
        self.assertEqual(scaled_actions.shape, (self.n_assets,))
        self.assertTrue(np.all(scaled_actions >= -self.max_position_size) and np.all(scaled_actions <= self.max_position_size))

if __name__ == '__main__':
    unittest.main() 