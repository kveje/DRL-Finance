import unittest
import numpy as np
import torch
from unittest.mock import MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.action_interpreters.discrete_interpreter import DiscreteInterpreter

class TestDiscreteInterpreter(unittest.TestCase):
    """Unit tests for the DiscreteInterpreter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create interpreter with 3 assets
        self.interpreter = DiscreteInterpreter(
            n_assets=3,
            max_position_size=10,
            temperature=1.0,
            temperature_decay=0.995,
            min_temperature=0.1
        )
    
    def test_initialization(self):
        """Test that the interpreter initializes correctly."""
        self.assertIsNotNone(self.interpreter)
        self.assertEqual(self.interpreter.n_assets, 3)
        self.assertEqual(self.interpreter.max_position_size, 10)
        self.assertEqual(self.interpreter.temperature, 1.0)
    
    def test_interpret_deterministic(self):
        """Test deterministic interpretation of action probabilities."""
        # Create sample network outputs
        network_outputs = {
            'action_probs': np.array([[
                [0.8, 0.1, 0.1],  # First asset: strongly sell
                [0.1, 0.8, 0.1],  # Second asset: strongly hold
                [0.1, 0.1, 0.8]   # Third asset: strongly buy
            ]])
        }
        
        # Current positions
        current_position = np.array([50.0, 50.0, 50.0])
        
        # Interpret actions deterministically
        actions = self.interpreter.interpret(network_outputs, current_position, deterministic=True)
        
        # Check shape
        self.assertEqual(actions.shape, current_position.shape)
        
        # Check expected actions (scaled by max_position_size)
        self.assertEqual(actions[0], -10)  # Sell
        self.assertEqual(actions[1], 0)    # Hold
        self.assertEqual(actions[2], 10)   # Buy
    
    def test_interpret_stochastic(self):
        """Test stochastic interpretation of action probabilities."""
        # Create network outputs with equal distribution
        network_outputs = {
            'action_probs': np.array([[
                [0.33, 0.34, 0.33],  # Equal probabilities
                [0.33, 0.34, 0.33],  # Equal probabilities
                [0.33, 0.34, 0.33]   # Equal probabilities
            ]])
        }
        
        # Current positions
        current_position = np.array([50.0, 50.0, 50.0])
        
        # Run multiple times to ensure stochastic behavior
        actions_list = []
        for _ in range(10):
            actions = self.interpreter.interpret(network_outputs, current_position, deterministic=False)
            actions_list.append(actions)
        
        # Check that we get different actions (not all deterministic)
        unique_actions = set(tuple(actions) for actions in actions_list)
        self.assertGreater(len(unique_actions), 1)
        
        # Check that all actions are valid (-max_position_size, 0, or max_position_size)
        for actions in actions_list:
            self.assertTrue(np.all(np.isin(actions, [-10, 0, 10])))
    
    def test_handle_tensor_input(self):
        """Test handling of tensor inputs."""
        # Create tensor network outputs
        network_outputs = {
            'action_probs': torch.tensor([[
                [0.8, 0.1, 0.1],  # Sell
                [0.1, 0.8, 0.1],  # Hold
                [0.1, 0.1, 0.8]   # Buy
            ]])
        }
        
        current_position = np.array([50.0, 50.0, 50.0])
        
        # Should handle tensor input
        actions = self.interpreter.interpret(network_outputs, current_position)
        
        # Check that we get valid actions
        self.assertTrue(np.all(np.isfinite(actions)))
        self.assertEqual(actions.shape, current_position.shape)
        
        # Check expected actions (scaled by max_position_size)
        self.assertEqual(actions[0], -10)  # Sell
        self.assertEqual(actions[1], 0)    # Hold
        self.assertEqual(actions[2], 10)   # Buy
    
    def test_handle_batch_input(self):
        """Test handling of batch inputs."""
        # Create batch of network outputs
        network_outputs = {
            'action_probs': np.array([
                [  # First batch
                    [0.8, 0.1, 0.1],  # Sell
                    [0.1, 0.8, 0.1],  # Hold
                    [0.1, 0.1, 0.8]   # Buy
                ],
                [  # Second batch (different actions)
                    [0.1, 0.1, 0.8],  # Buy
                    [0.8, 0.1, 0.1],  # Sell
                    [0.1, 0.8, 0.1]   # Hold
                ]
            ])
        }
        
        current_position = np.array([50.0, 50.0, 50.0])
        
        # Should handle batch input and use first batch
        actions = self.interpreter.interpret(network_outputs, current_position)
        
        # Check that we get valid actions
        self.assertTrue(np.all(np.isfinite(actions)))
        self.assertEqual(actions.shape, current_position.shape)
        
        # Check expected actions from first batch (scaled by max_position_size)
        self.assertEqual(actions[0], -10)  # Sell
        self.assertEqual(actions[1], 0)    # Hold
        self.assertEqual(actions[2], 10)   # Buy
    
    def test_handle_bayesian_output(self):
        """Test handling of Bayesian head outputs."""
        # Create Bayesian network outputs
        network_outputs = {
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
        
        current_position = np.array([50.0, 50.0, 50.0])
        
        # Test deterministic interpretation
        actions = self.interpreter.interpret(network_outputs, current_position, deterministic=True)
        self.assertEqual(actions[0], -10)  # Strong sell
        self.assertEqual(actions[1], 0)    # Hold
        self.assertEqual(actions[2], 10)   # Strong buy
        
        # Test stochastic interpretation
        actions = self.interpreter.interpret(network_outputs, current_position, deterministic=False)
        self.assertTrue(np.all(np.isfinite(actions)))
        self.assertEqual(actions.shape, current_position.shape)
        self.assertTrue(np.all(np.isin(actions, [-10, 0, 10])))
    
    def test_get_q_values(self):
        """Test Q-value extraction."""
        # Create network outputs
        network_outputs = {
            'action_probs': np.array([[
                [0.8, 0.1, 0.1],  # Sell
                [0.1, 0.8, 0.1],  # Hold
                [0.1, 0.1, 0.8]   # Buy
            ]])
        }
        
        # Test actions (scaled by max_position_size)
        actions = np.array([-10, 0, 10])  # sell, hold, buy
        
        # Get Q-values
        q_values = self.interpreter.get_q_values(network_outputs, actions)
        
        # Check shape
        self.assertEqual(q_values.shape, (3,))
        
        # Check expected Q-values
        self.assertAlmostEqual(q_values[0].item(), 0.8)  # Sell Q-value
        self.assertAlmostEqual(q_values[1].item(), 0.8)  # Hold Q-value
        self.assertAlmostEqual(q_values[2].item(), 0.8)  # Buy Q-value
    
    def test_get_max_q_values(self):
        """Test maximum Q-value extraction."""
        # Create network outputs
        network_outputs = {
            'action_probs': np.array([[
                [0.8, 0.1, 0.1],  # Sell
                [0.1, 0.8, 0.1],  # Hold
                [0.1, 0.1, 0.8]   # Buy
            ]])
        }
        
        # Get max Q-values
        max_q_values = self.interpreter.get_max_q_values(network_outputs)
        
        # Check shape
        self.assertEqual(max_q_values.shape, (3,))
        
        # Check expected max Q-values
        self.assertAlmostEqual(max_q_values[0].item(), 0.8)  # Max Q-value for first asset
        self.assertAlmostEqual(max_q_values[1].item(), 0.8)  # Max Q-value for second asset
        self.assertAlmostEqual(max_q_values[2].item(), 0.8)  # Max Q-value for third asset
    
    def test_interpret_with_log_prob(self):
        """Test interpretation with log probabilities."""
        # Create sample network outputs
        network_outputs = {
            'action_probs': np.array([[
                [0.8, 0.1, 0.1],  # First asset: strongly sell
                [0.1, 0.8, 0.1],  # Second asset: strongly hold
                [0.1, 0.1, 0.8]   # Third asset: strongly buy
            ]])
        }
        
        current_position = np.array([50.0, 50.0, 50.0])
        
        # Get actions and log probabilities
        scaled_actions, log_probs = self.interpreter.interpret_with_log_prob(network_outputs, current_position)
        
        # Check shapes
        self.assertEqual(scaled_actions.shape, current_position.shape)
        self.assertEqual(log_probs.shape, current_position.shape)
        
        # Check that actions are valid
        self.assertTrue(np.all(np.isin(scaled_actions, [-10, 0, 10])))
        
        # Check that log probabilities are valid (negative numbers)
        self.assertTrue(np.all(log_probs <= 0))
        
        # Check that log probabilities correspond to actions
        for i, action in enumerate(scaled_actions):
            action_idx = int(action / 10) + 1  # Convert -10,0,10 to 0,1,2
            expected_log_prob = np.log(network_outputs['action_probs'][0, i, action_idx])
            self.assertAlmostEqual(log_probs[i], expected_log_prob)
    
    def test_interpret_with_log_prob_bayesian(self):
        """Test interpretation with log probabilities for Bayesian outputs."""
        # Create Bayesian network outputs
        network_outputs = {
            'alphas': np.array([[
                [8.0, 1.0, 1.0],  # Strong sell
                [1.0, 8.0, 1.0],  # Strong hold
                [1.0, 1.0, 8.0]   # Strong buy
            ]]),
            'betas': np.array([[
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0]
            ]])
        }
        
        current_position = np.array([50.0, 50.0, 50.0])
        
        # Get actions and log probabilities
        scaled_actions, log_probs = self.interpreter.interpret_with_log_prob(network_outputs, current_position)
        
        # Check shapes
        self.assertEqual(scaled_actions.shape, current_position.shape)
        self.assertEqual(log_probs.shape, current_position.shape)
        
        # Check that actions are valid
        self.assertTrue(np.all(np.isin(scaled_actions, [-10, 0, 10])))
        
        # Check that log probabilities are valid (negative numbers)
        self.assertTrue(np.all(log_probs <= 0))
        
        # Check that log probabilities correspond to actions
        probs = network_outputs['alphas'] / (network_outputs['alphas'] + network_outputs['betas'])
        # Normalize probabilities to match implementation
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        
        for i, action in enumerate(scaled_actions):
            action_idx = int(action / 10) + 1  # Convert -10,0,10 to 0,1,2
            expected_log_prob = np.log(probs[0, i, action_idx])
            self.assertAlmostEqual(log_probs[i], expected_log_prob)
    
    def test_scale_actions(self):
        """Test scaling of actions."""
        # Test single action
        actions = torch.tensor([-1, 0, 1])
        scaled_actions = self.interpreter.scale_actions(actions)
        
        # Check shape and values
        self.assertEqual(scaled_actions.shape, (3,))
        self.assertEqual(scaled_actions[0].item(), -10)  # -1 * max_position_size
        self.assertEqual(scaled_actions[1].item(), 0)    # 0 * max_position_size
        self.assertEqual(scaled_actions[2].item(), 10)   # 1 * max_position_size
        
        # Test batch of actions
        batch_actions = torch.tensor([[-1, 0, 1], [1, -1, 0]])
        batch_scaled = self.interpreter.scale_actions(batch_actions)
        
        # Check shape and values
        self.assertEqual(batch_scaled.shape, (2, 3))
        self.assertTrue(torch.all(batch_scaled[0] == torch.tensor([-10, 0, 10])))
        self.assertTrue(torch.all(batch_scaled[1] == torch.tensor([10, -10, 0])))
    
    def test_evaluate_actions_log_probs(self):
        """Test evaluation of action log probabilities."""
        # Create sample network outputs
        network_outputs = {
            'action_probs': torch.tensor([[
                [0.8, 0.1, 0.1],  # First asset: strongly sell
                [0.1, 0.8, 0.1],  # Second asset: strongly hold
                [0.1, 0.1, 0.8]   # Third asset: strongly buy
            ]])
        }
        
        # Test single action
        actions = torch.tensor([-1, 0, 1])  # sell, hold, buy
        scaled_actions, log_probs = self.interpreter.evaluate_actions_log_probs(network_outputs, actions)
        
        # Check shapes
        self.assertEqual(scaled_actions.shape, (3,))
        self.assertEqual(log_probs.shape, (3,))
        
        # Check scaled actions
        self.assertEqual(scaled_actions[0].item(), -10)  # sell
        self.assertEqual(scaled_actions[1].item(), 0)    # hold
        self.assertEqual(scaled_actions[2].item(), 10)   # buy
        
        # Check log probabilities
        expected_log_probs = torch.log(torch.tensor([0.8, 0.8, 0.8]))
        self.assertTrue(torch.allclose(log_probs, expected_log_probs))
        
        # Test batch of actions
        batch_actions = torch.tensor([[-1, 0, 1], [1, -1, 0]])
        batch_scaled, batch_log_probs = self.interpreter.evaluate_actions_log_probs(network_outputs, batch_actions)
        
        # Check shapes
        self.assertEqual(batch_scaled.shape, (2, 3))
        self.assertEqual(batch_log_probs.shape, (2, 3))
        
        # Check scaled actions
        self.assertTrue(torch.all(batch_scaled[0] == torch.tensor([-10, 0, 10])))
        self.assertTrue(torch.all(batch_scaled[1] == torch.tensor([10, -10, 0])))
        
        # Check log probabilities
        expected_batch_log_probs = torch.log(torch.tensor([[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]]))
        self.assertTrue(torch.allclose(batch_log_probs, expected_batch_log_probs))
    
    def test_evaluate_actions_log_probs_bayesian(self):
        """Test evaluation of action log probabilities with Bayesian outputs."""
        # Create Bayesian network outputs
        network_outputs = {
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
        
        # Test single action
        actions = torch.tensor([-1, 0, 1])  # sell, hold, buy
        scaled_actions, log_probs = self.interpreter.evaluate_actions_log_probs(network_outputs, actions)
        
        # Check shapes
        self.assertEqual(scaled_actions.shape, (3,))
        self.assertEqual(log_probs.shape, (3,))
        
        # Check scaled actions
        self.assertEqual(scaled_actions[0].item(), -10)  # sell
        self.assertEqual(scaled_actions[1].item(), 0)    # hold
        self.assertEqual(scaled_actions[2].item(), 10)   # buy
        
        # Check log probabilities
        probs = network_outputs['alphas'] / (network_outputs['alphas'] + network_outputs['betas'])
        expected_log_probs = torch.log(probs[0, torch.arange(3), torch.tensor([0, 1, 2])])
        self.assertTrue(torch.allclose(log_probs, expected_log_probs))
        
        # Test batch of actions
        batch_actions = torch.tensor([[-1, 0, 1], [1, -1, 0]])
        batch_scaled, batch_log_probs = self.interpreter.evaluate_actions_log_probs(network_outputs, batch_actions)
        
        # Check shapes
        self.assertEqual(batch_scaled.shape, (2, 3))
        self.assertEqual(batch_log_probs.shape, (2, 3))
        
        # Check scaled actions
        self.assertTrue(torch.all(batch_scaled[0] == torch.tensor([-10, 0, 10])))
        self.assertTrue(torch.all(batch_scaled[1] == torch.tensor([10, -10, 0])))
        
        # Check log probabilities
        expected_batch_log_probs = torch.log(probs[0, torch.arange(3), torch.tensor([0, 1, 2])]).repeat(2, 1)
        self.assertTrue(torch.allclose(batch_log_probs, expected_batch_log_probs))

if __name__ == '__main__':
    unittest.main() 