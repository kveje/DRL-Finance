import unittest
import numpy as np
import torch
from unittest.mock import MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.action_interpreters.confidence_scaled_interpreter import ConfidenceScaledInterpreter

class TestConfidenceScaledInterpreter(unittest.TestCase):
    """Unit tests for the ConfidenceScaledInterpreter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create interpreter with 3 assets
        self.interpreter = ConfidenceScaledInterpreter(
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
        """Test deterministic interpretation of action probabilities and confidences."""
        # Create sample network outputs
        network_outputs = {
            'action_probs': np.array([[
                [0.8, 0.1, 0.1],  # First asset: strongly sell
                [0.1, 0.8, 0.1],  # Second asset: strongly hold
                [0.1, 0.1, 0.8]   # Third asset: strongly buy
            ]]),
            'confidences': np.array([[
                0.9,  # High confidence in first asset
                0.5,  # Medium confidence in second asset
                0.7   # High confidence in third asset
            ]])
        }
        
        # Current positions
        current_position = np.array([50.0, 50.0, 50.0])
        
        # Interpret actions deterministically
        actions = self.interpreter.interpret(network_outputs, current_position, deterministic=True)
        
        # Check shape
        self.assertEqual(actions.shape, current_position.shape)
        
        # Check expected actions (scaled by confidence)
        self.assertAlmostEqual(actions[0], -9.0)  # Sell with 0.9 confidence
        self.assertAlmostEqual(actions[1], 0.0)   # Hold with 0.5 confidence
        self.assertAlmostEqual(actions[2], 7.0)   # Buy with 0.7 confidence
    
    def test_interpret_stochastic(self):
        """Test stochastic interpretation of action probabilities and confidences."""
        # Create network outputs with equal distribution
        network_outputs = {
            'action_probs': np.array([[
                [0.33, 0.34, 0.33],  # Equal probabilities
                [0.33, 0.34, 0.33],  # Equal probabilities
                [0.33, 0.34, 0.33]   # Equal probabilities
            ]]),
            'confidences': np.array([[
                0.5,  # Medium confidence
                0.5,  # Medium confidence
                0.5   # Medium confidence
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
        
        # Check that all actions are within valid range
        for actions in actions_list:
            self.assertTrue(np.all(np.abs(actions) <= self.interpreter.max_position_size))
    
    def test_handle_tensor_input(self):
        """Test handling of tensor inputs."""
        # Create tensor network outputs
        network_outputs = {
            'action_probs': torch.tensor([[
                [0.8, 0.1, 0.1],  # Sell
                [0.1, 0.8, 0.1],  # Hold
                [0.1, 0.1, 0.8]   # Buy
            ]]),
            'confidences': torch.tensor([[
                0.9,  # High confidence
                0.5,  # Medium confidence
                0.7   # High confidence
            ]])
        }
        
        current_position = np.array([50.0, 50.0, 50.0])
        
        # Should handle tensor input
        actions = self.interpreter.interpret(network_outputs, current_position)
        
        # Check that we get valid actions
        self.assertTrue(np.all(np.isfinite(actions)))
        self.assertEqual(actions.shape, current_position.shape)
        
        # Check expected actions (scaled by confidence)
        self.assertAlmostEqual(actions[0], -9.0)  # Sell with 0.9 confidence
        self.assertAlmostEqual(actions[1], 0.0)   # Hold with 0.5 confidence
        self.assertAlmostEqual(actions[2], 7.0)   # Buy with 0.7 confidence
    
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
            ]),
            'confidences': np.array([
                [  # First batch
                    0.9,  # High confidence
                    0.5,  # Medium confidence
                    0.7   # High confidence
                ],
                [  # Second batch
                    0.7,  # High confidence
                    0.9,  # High confidence
                    0.5   # Medium confidence
                ]
            ])
        }
        
        current_position = np.array([50.0, 50.0, 50.0])
        
        # Should handle batch input and use first batch
        actions = self.interpreter.interpret(network_outputs, current_position)
        
        # Check that we get valid actions
        self.assertTrue(np.all(np.isfinite(actions)))
        self.assertEqual(actions.shape, current_position.shape)
        
        # Check expected actions from first batch (scaled by confidence)
        self.assertAlmostEqual(actions[0], -9.0)  # Sell with 0.9 confidence
        self.assertAlmostEqual(actions[1], 0.0)   # Hold with 0.5 confidence
        self.assertAlmostEqual(actions[2], 7.0)   # Buy with 0.7 confidence
    
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
        self.assertTrue(np.all(np.isfinite(actions)))
        self.assertEqual(actions.shape, current_position.shape)
        
        # Test stochastic interpretation
        actions = self.interpreter.interpret(network_outputs, current_position, deterministic=False)
        self.assertTrue(np.all(np.isfinite(actions)))
        self.assertEqual(actions.shape, current_position.shape)
    
    def test_get_q_values(self):
        """Test Q-value extraction."""
        # Create network outputs
        network_outputs = {
            'action_probs': np.array([[
                [0.8, 0.1, 0.1],  # Sell
                [0.1, 0.8, 0.1],  # Hold
                [0.1, 0.1, 0.8]   # Buy
            ]]),
            'confidences': np.array([[
                0.9,  # High confidence
                0.5,  # Medium confidence
                0.7   # High confidence
            ]])
        }
        
        # Test actions
        actions = np.array([-1, 0, 1])  # sell, hold, buy
        
        # Get Q-values
        q_values = self.interpreter.get_q_values(network_outputs, actions)
        
        # Check shape
        self.assertEqual(q_values.shape, (3,))
        
        # Check expected Q-values (scaled by confidence)
        self.assertAlmostEqual(q_values[0].item(), 0.8 * 0.9)  # Sell Q-value * confidence
        self.assertAlmostEqual(q_values[1].item(), 0.8 * 0.5)  # Hold Q-value * confidence
        self.assertAlmostEqual(q_values[2].item(), 0.8 * 0.7)  # Buy Q-value * confidence
    
    def test_get_max_q_values(self):
        """Test maximum Q-value extraction."""
        # Create network outputs
        network_outputs = {
            'action_probs': np.array([[
                [0.8, 0.1, 0.1],  # Sell
                [0.1, 0.8, 0.1],  # Hold
                [0.1, 0.1, 0.8]   # Buy
            ]]),
            'confidences': np.array([[
                0.9,  # High confidence
                0.5,  # Medium confidence
                0.7   # High confidence
            ]])
        }
        
        # Get max Q-values
        max_q_values = self.interpreter.get_max_q_values(network_outputs)
        
        # Check shape
        self.assertEqual(max_q_values.shape, (3,))
        
        # Check expected max Q-values (scaled by confidence)
        self.assertAlmostEqual(max_q_values[0].item(), 0.8 * 0.9)  # Max Q-value * confidence for first asset
        self.assertAlmostEqual(max_q_values[1].item(), 0.8 * 0.5)  # Max Q-value * confidence for second asset
        self.assertAlmostEqual(max_q_values[2].item(), 0.8 * 0.7)  # Max Q-value * confidence for third asset
    
    def test_interpret_with_log_prob(self):
        """Test interpretation with log probabilities."""
        # Create sample network outputs
        network_outputs = {
            'action_probs': np.array([[
                [0.8, 0.1, 0.1],  # First asset: strongly sell
                [0.1, 0.8, 0.1],  # Second asset: strongly hold
                [0.1, 0.1, 0.8]   # Third asset: strongly buy
            ]]),
            'confidences': np.array([[
                0.9,  # High confidence in first asset
                0.5,  # Medium confidence in second asset
                0.7   # High confidence in third asset
            ]])
        }
        
        current_position = np.array([50.0, 50.0, 50.0])
        
        # Get actions and log probabilities
        scaled_actions, log_probs = self.interpreter.interpret_with_log_prob(network_outputs, current_position)
        
        # Check shapes
        self.assertEqual(scaled_actions.shape, current_position.shape)
        self.assertEqual(log_probs.shape, current_position.shape)
        
        # Check that actions are within valid range
        self.assertTrue(np.all(np.abs(scaled_actions) <= self.interpreter.max_position_size))
        
        # Check that log probabilities are valid (negative numbers)
        self.assertTrue(np.all(log_probs <= 0))
        
        # Check that log probabilities correspond to actions
        for i, action in enumerate(scaled_actions):
            action_idx = int(np.sign(action)) + 1  # Convert -1,0,1 to 0,1,2
            expected_log_prob = np.log(network_outputs['action_probs'][0, i, action_idx])
            self.assertAlmostEqual(log_probs[i], expected_log_prob)
            
            # Check that actions are scaled by confidence
            expected_action = (action_idx - 1) * self.interpreter.max_position_size * network_outputs['confidences'][0, i]
            self.assertAlmostEqual(scaled_actions[i], expected_action)
    
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
        
        # Check that actions are within valid range
        self.assertTrue(np.all(np.abs(scaled_actions) <= self.interpreter.max_position_size))
        
        # Check that log probabilities are valid (negative numbers)
        self.assertTrue(np.all(log_probs <= 0))
        
        # Check that log probabilities correspond to actions
        probs = network_outputs['alphas'] / (network_outputs['alphas'] + network_outputs['betas'])
        # Normalize probabilities to match implementation
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        confidences = 1.0 / (np.sum(network_outputs['alphas'], axis=-1) + np.sum(network_outputs['betas'], axis=-1) + 1.0)
        
        for i, action in enumerate(scaled_actions):
            action_idx = int(np.sign(action)) + 1  # Convert -1,0,1 to 0,1,2
            expected_log_prob = np.log(probs[0, i, action_idx])
            self.assertAlmostEqual(log_probs[i], expected_log_prob)
            
            # Check that actions are scaled by confidence
            expected_action = (action_idx - 1) * self.interpreter.max_position_size * confidences[0, i]
            self.assertAlmostEqual(scaled_actions[i], expected_action)

    def test_scale_actions_with_confidence(self):
        """Test scaling of actions with confidence values."""
        # Test single action
        actions = torch.tensor([-1, 0, 1])
        confidences = torch.tensor([0.9, 0.5, 0.7])
        scaled_actions = self.interpreter.scale_actions_with_confidence(actions, confidences)
        
        # Check shape and values
        self.assertEqual(scaled_actions.shape, (3,))
        self.assertAlmostEqual(scaled_actions[0].item(), -9.0)  # -1 * max_position_size * 0.9
        self.assertAlmostEqual(scaled_actions[1].item(), 0.0)   # 0 * max_position_size * 0.5
        self.assertAlmostEqual(scaled_actions[2].item(), 7.0)   # 1 * max_position_size * 0.7
        
        # Test batch of actions
        batch_actions = torch.tensor([[-1, 0, 1], [1, -1, 0]])
        batch_confidences = torch.tensor([[0.9, 0.5, 0.7], [0.8, 0.6, 0.4]])
        batch_scaled = self.interpreter.scale_actions_with_confidence(batch_actions, batch_confidences)
        
        # Check shape and values
        self.assertEqual(batch_scaled.shape, (2, 3))
        self.assertTrue(torch.allclose(batch_scaled[0], torch.tensor([-9.0, 0.0, 7.0])))
        self.assertTrue(torch.allclose(batch_scaled[1], torch.tensor([8.0, -6.0, 0.0])))
    
    def test_evaluate_actions_log_probs(self):
        """Test evaluation of action log probabilities."""
        # Create sample network outputs
        network_outputs = {
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
        
        # Test single action
        actions = torch.tensor([0, 1, 2])  # sell, hold, buy indices
        scaled_actions, log_probs = self.interpreter.evaluate_actions_log_probs(network_outputs, actions)
        
        # Check shapes
        self.assertEqual(scaled_actions.shape, (3,))
        self.assertEqual(log_probs.shape, (3,))
        
        # Check scaled actions
        self.assertAlmostEqual(scaled_actions[0].item(), -9.0)  # sell with 0.9 confidence
        self.assertAlmostEqual(scaled_actions[1].item(), 0.0)   # hold with 0.5 confidence
        self.assertAlmostEqual(scaled_actions[2].item(), 7.0)   # buy with 0.7 confidence
        
        # Check log probabilities
        expected_log_probs = torch.log(torch.tensor([0.8, 0.8, 0.8]))
        self.assertTrue(torch.allclose(log_probs, expected_log_probs))
        
        # Test batch of actions
        batch_actions = torch.tensor([[0, 1, 2], [2, 0, 1]])  # indices for sell, hold, buy
        batch_scaled, batch_log_probs = self.interpreter.evaluate_actions_log_probs(network_outputs, batch_actions)
        
        # Check shapes
        self.assertEqual(batch_scaled.shape, (2, 3))
        self.assertEqual(batch_log_probs.shape, (2, 3))
        
        # Check scaled actions
        self.assertTrue(torch.allclose(batch_scaled[0], torch.tensor([-9.0, 0.0, 7.0])))
        print(f"[DEBUG] batch_scaled[0]: {batch_scaled[0]}")
        print(f"[DEBUG] torch.tensor([-9.0, 0.0, 7.0]): {torch.tensor([-9.0, 0.0, 7.0])}")
        self.assertTrue(torch.allclose(batch_scaled[1], torch.tensor([7.0, -9.0, 0.0])))
        print(f"[DEBUG] batch_scaled[1]: {batch_scaled[1]}")
        print(f"[DEBUG] torch.tensor([7.0, -9.0, 0.0]): {torch.tensor([7.0, -9.0, 0.0])}")
        
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
        actions = torch.tensor([0, 1, 2])  # sell, hold, buy indices
        scaled_actions, log_probs = self.interpreter.evaluate_actions_log_probs(network_outputs, actions)
        
        # Check shapes
        self.assertEqual(scaled_actions.shape, (3,))
        self.assertEqual(log_probs.shape, (3,))
        
        # Check scaled actions (should be scaled by confidence from Beta distribution)
        probs = network_outputs['alphas'] / (network_outputs['alphas'] + network_outputs['betas'])
        confidences = 1.0 / (torch.sum(network_outputs['alphas'], dim=-1) + torch.sum(network_outputs['betas'], dim=-1) + 1.0)
        expected_scaled_actions = torch.tensor([-1, 0, 1]).float() * self.interpreter.max_position_size * confidences[0]
        self.assertTrue(torch.allclose(scaled_actions, expected_scaled_actions))
        
        # Check log probabilities
        expected_log_probs = torch.log(probs[0, torch.arange(3), actions])
        self.assertTrue(torch.allclose(log_probs, expected_log_probs))
        
        # Test batch of actions
        batch_actions = torch.tensor([[0, 1, 2], [2, 0, 1]])  # indices for sell, hold, buy
        batch_scaled, batch_log_probs = self.interpreter.evaluate_actions_log_probs(network_outputs, batch_actions)
        
        # Check shapes
        self.assertEqual(batch_scaled.shape, (2, 3))
        self.assertEqual(batch_log_probs.shape, (2, 3))
        
        # Check scaled actions
        expected_batch_scaled = torch.tensor([[-1, 0, 1], [1, -1, 0]]).float() * self.interpreter.max_position_size * confidences[0]
        self.assertTrue(torch.allclose(batch_scaled, expected_batch_scaled))
        
        # Check log probabilities
        expected_batch_log_probs = torch.log(probs[0, torch.arange(3), batch_actions[0]]).repeat(2, 1)
        print(f"[DEBUG] expected_batch_log_probs: {expected_batch_log_probs}")
        print(f"[DEBUG] batch_log_probs: {batch_log_probs}")
        self.assertTrue(torch.allclose(batch_log_probs, expected_batch_log_probs))

if __name__ == '__main__':
    unittest.main() 