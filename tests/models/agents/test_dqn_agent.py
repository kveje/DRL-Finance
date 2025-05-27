"""Unit tests for the DQN agent."""
import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.agents.dqn_agent import DQNAgent
from models.action_interpreters.discrete_interpreter import DiscreteInterpreter
from models.action_interpreters.confidence_scaled_interpreter import ConfidenceScaledInterpreter
from models.networks.unified_network import UnifiedNetwork

class TestDQNAgent(unittest.TestCase):
    """Test cases for the DQN agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_assets = 3
        self.window_size = 20
        self.batch_size = 32
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
        self.discrete_agent = DQNAgent(
            env=self.env,
            network_config=self.discrete_network_config,
            interpreter=self.discrete_interpreter,
            device="cpu"
        )
        
        self.confidence_agent = DQNAgent(
            env=self.env,
            network_config=self.confidence_network_config,
            interpreter=self.confidence_interpreter,
            device="cpu"
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        # Test discrete agent
        self.assertIsInstance(self.discrete_agent.network, UnifiedNetwork)
        self.assertIsInstance(self.discrete_agent.target_network, UnifiedNetwork)
        self.assertEqual(self.discrete_agent.epsilon, 1.0)
        self.assertEqual(self.discrete_agent.steps, 0)
        
        # Test confidence agent
        self.assertIsInstance(self.confidence_agent.network, UnifiedNetwork)
        self.assertIsInstance(self.confidence_agent.target_network, UnifiedNetwork)
        self.assertEqual(self.confidence_agent.epsilon, 1.0)
        self.assertEqual(self.confidence_agent.steps, 0)
    
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
            ]])
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
            ]])
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
    
    def test_update_discrete(self):
        """Test network update for discrete interpreter."""
        # Create mock batch
        batch = {
            'observations': {
                'price': torch.randn(self.batch_size, self.n_assets, self.window_size, requires_grad=True),  # (batch_size, n_assets, window_size)
                'technical': torch.randn(self.batch_size, self.n_assets, 15, requires_grad=True)  # (batch_size, n_assets, tech_dim)
            },
            'actions': torch.randint(-1, 2, (self.batch_size, self.n_assets)) * self.max_position_size,  # Scale actions
            'rewards': torch.randn(self.batch_size),
            'next_observations': {
                'price': torch.randn(self.batch_size, self.n_assets, self.window_size, requires_grad=True),
                'technical': torch.randn(self.batch_size, self.n_assets, 15, requires_grad=True)
            },
            'dones': torch.zeros(self.batch_size)
        }
        
        # Mock network outputs with proper batch handling and gradients
        mock_current_output = {
            'action_probs': torch.softmax(torch.randn(self.batch_size, self.n_assets, 3, requires_grad=True), dim=-1)  # (batch_size, n_assets, actions)
        }
        mock_target_output = {
            'action_probs': torch.softmax(torch.randn(self.batch_size, self.n_assets, 3, requires_grad=True), dim=-1)  # (batch_size, n_assets, actions)
        }
        
        # Mock the network to return the same outputs for all inputs
        def mock_forward(*args, **kwargs):
            return mock_current_output
        
        def mock_target_forward(*args, **kwargs):
            return mock_target_output
        
        self.discrete_agent.network.forward = mock_forward
        self.discrete_agent.target_network.forward = mock_target_forward
        
        # Test update
        metrics = self.discrete_agent.update(batch)
        self.assertIn('loss', metrics)
        self.assertIn('epsilon', metrics)
        self.assertIsInstance(metrics['loss'], float)
        self.assertIsInstance(metrics['epsilon'], float)
    
    def test_update_confidence(self):
        """Test network update for confidence-scaled interpreter."""
        # Create mock batch
        batch = {
            'observations': {
                'price': torch.randn(self.batch_size, self.n_assets, self.window_size, requires_grad=True),  # (batch_size, n_assets, window_size)
                'technical': torch.randn(self.batch_size, self.n_assets, 15, requires_grad=True)  # (batch_size, n_assets, tech_dim)
            },
            'actions': torch.randint(-1, 2, (self.batch_size, self.n_assets)),  # Use unscaled actions [-1,0,1]
            'rewards': torch.randn(self.batch_size),
            'next_observations': {
                'price': torch.randn(self.batch_size, self.n_assets, self.window_size, requires_grad=True),
                'technical': torch.randn(self.batch_size, self.n_assets, 15, requires_grad=True)
            },
            'dones': torch.zeros(self.batch_size)
        }
        
        # Mock network outputs with proper batch handling and gradients
        mock_current_output = {
            'action_probs': torch.softmax(torch.randn(self.batch_size, self.n_assets, 3, requires_grad=True), dim=-1),  # (batch_size, n_assets, actions)
            'confidences': torch.sigmoid(torch.randn(self.batch_size, self.n_assets, requires_grad=True))  # (batch_size, n_assets)
        }
        mock_target_output = {
            'action_probs': torch.softmax(torch.randn(self.batch_size, self.n_assets, 3, requires_grad=True), dim=-1),  # (batch_size, n_assets, actions)
            'confidences': torch.sigmoid(torch.randn(self.batch_size, self.n_assets, requires_grad=True))  # (batch_size, n_assets)
        }
        
        # Mock the network to return the same outputs for all inputs
        def mock_forward(*args, **kwargs):
            return mock_current_output
        
        def mock_target_forward(*args, **kwargs):
            return mock_target_output
        
        self.confidence_agent.network.forward = mock_forward
        self.confidence_agent.target_network.forward = mock_target_forward
        
        # Test update
        metrics = self.confidence_agent.update(batch)
        self.assertIn('loss', metrics)
        self.assertIn('epsilon', metrics)
        self.assertIsInstance(metrics['loss'], float)
        self.assertIsInstance(metrics['epsilon'], float)
    
    def test_save_load(self):
        """Test saving and loading agent state."""
        # Create temporary directory for saving
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save agent state
            self.discrete_agent.save(tmpdir)
            
            # Create new agent with EXACTLY the same config
            new_agent = DQNAgent(
                env=self.env,
                network_config=self.discrete_network_config.copy(),  # Make a copy to ensure it's the same
                interpreter=self.discrete_interpreter,
                device="cpu"
            )
            
            # Load state
            new_agent.load(tmpdir)
            
            # Check if states match
            self.assertEqual(new_agent.epsilon, self.discrete_agent.epsilon)
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
            new_agent = DQNAgent(
                env=self.env,
                network_config=self.confidence_network_config.copy(),  # Make a copy to ensure it's the same
                interpreter=self.confidence_interpreter,
                device="cpu"
            )
            
            # Load state
            new_agent.load(tmpdir)
            
            # Check if states match
            self.assertEqual(new_agent.epsilon, self.confidence_agent.epsilon)
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
            }
        }
        
        # Create agent with Bayesian outputs
        bayesian_agent = DQNAgent(
            env=self.env,
            network_config=bayesian_config,
            interpreter=self.confidence_interpreter,
            device="cpu",
            use_bayesian=True
        )
        
        # Create mock observations
        observations = {
            'price': np.random.randn(self.n_assets, self.window_size),  # (n_assets, window_size)
            'technical': np.random.randn(self.n_assets, 15)  # (n_assets, tech_dim)
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
            ]])
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