import unittest
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
            max_trade_size=10,
        )
    
    def test_initialization(self):
        """Test that the interpreter initializes correctly."""
        self.assertIsNotNone(self.interpreter)
        self.assertEqual(self.interpreter.n_assets, 3)
        self.assertEqual(self.interpreter.max_trade_size, 10)

    def test_interpret_deterministic(self):
        """Test deterministic interpretation of action probabilities."""
        # Create sample network outputs
        network_outputs = {
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],  # First asset: strongly sell
                [0.1, 0.8, 0.1],  # Second asset: strongly hold
                [0.1, 0.1, 0.8]   # Third asset: strongly buy
            ])
        }
        
        # Interpret actions deterministically
        scaled_actions, action_choices = self.interpreter.interpret(network_outputs, deterministic=True)
        
        # Check shape
        self.assertEqual(scaled_actions.shape, (3,))
        self.assertEqual(action_choices.shape, (3,))
        
        # Check that actions are integers
        self.assertEqual(scaled_actions.dtype, torch.int64)
        
        # Check expected actions (scaled by max_position_size)
        self.assertEqual(scaled_actions[0].item(), -10)  # Sell
        self.assertEqual(scaled_actions[1].item(), 0)    # Hold
        self.assertEqual(scaled_actions[2].item(), 10)   # Buy
        
        # Check action choices
        self.assertEqual(action_choices[0].item(), -1)  # Sell
        self.assertEqual(action_choices[1].item(), 0)   # Hold
        self.assertEqual(action_choices[2].item(), 1)   # Buy
    
    def test_interpret_stochastic(self):
        """Test stochastic interpretation of action probabilities."""
        # Create network outputs with equal distribution
        network_outputs = {
            'action_probs': torch.tensor([
                [0.33, 0.34, 0.33],  # Equal probabilities
                [0.33, 0.34, 0.33],  # Equal probabilities
                [0.33, 0.34, 0.33]   # Equal probabilities
            ])
        }
        
        # Run multiple times to ensure stochastic behavior
        actions_list = []
        for _ in range(10):
            scaled_actions, action_choices = self.interpreter.interpret(network_outputs, deterministic=False)
            actions_list.append((scaled_actions, action_choices))
        
        # Check that we get different actions (not all deterministic)
        unique_actions = set(tuple(actions[0].tolist()) for actions in actions_list)
        self.assertGreater(len(unique_actions), 1)
        
        # Check that all actions are valid
        for scaled_actions, action_choices in actions_list:
            self.assertEqual(scaled_actions.dtype, torch.int64)  # Check that actions are integers
            self.assertTrue(torch.all(torch.isin(scaled_actions, torch.tensor([-10, 0, 10]))))
            self.assertTrue(torch.all(torch.isin(action_choices, torch.tensor([-1, 0, 1]))))
    
    def test_handle_batch_input(self):
        """Test handling of batch inputs."""
        # Create batch of network outputs
        network_outputs = {
            'action_probs': torch.tensor([
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
        
        # Should handle batch input
        scaled_actions, action_choices = self.interpreter.interpret(network_outputs)
        
        # Check that we get valid actions
        self.assertTrue(torch.all(torch.isfinite(scaled_actions)))
        self.assertEqual(scaled_actions.shape, (2, 3))
        self.assertEqual(action_choices.shape, (2, 3))
        self.assertEqual(scaled_actions.dtype, torch.int64)  # Check that actions are integers
        
        # Check expected actions from first batch (scaled by max_position_size)
        self.assertEqual(scaled_actions[0, 0].item(), -10)  # Sell
        self.assertEqual(scaled_actions[0, 1].item(), 0)    # Hold
        self.assertEqual(scaled_actions[0, 2].item(), 10)   # Buy
        
        # Check expected actions from second batch
        self.assertEqual(scaled_actions[1, 0].item(), 10)   # Buy
        self.assertEqual(scaled_actions[1, 1].item(), -10)  # Sell
        self.assertEqual(scaled_actions[1, 2].item(), 0)    # Hold
    
    def test_get_q_values(self):
        """Test Q-value extraction."""
        # Create network outputs
        network_outputs = {
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],  # Sell
                [0.1, 0.8, 0.1],  # Hold
                [0.1, 0.1, 0.8]   # Buy
            ])
        }
        
        # Test actions (scaled by max_position_size)
        actions = torch.tensor([-1, 0, 1])  # sell, hold, buy
        
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
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],  # Sell
                [0.1, 0.8, 0.1],  # Hold
                [0.1, 0.1, 0.8]   # Buy
            ])
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
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],  # First asset: strongly sell
                [0.1, 0.8, 0.1],  # Second asset: strongly hold
                [0.1, 0.1, 0.8]   # Third asset: strongly buy
            ])
        }
        
        # Get actions and log probabilities
        scaled_actions, action_choices, log_probs = self.interpreter.interpret_with_log_prob(network_outputs)
        
        # Check shapes
        self.assertEqual(scaled_actions.shape, (3,))
        self.assertEqual(action_choices.shape, (3,))
        self.assertEqual(log_probs.shape, (3,))
        
        # Check that actions are integers
        self.assertEqual(scaled_actions.dtype, torch.int64)
        
        # Check that actions are valid
        self.assertTrue(torch.all(torch.isin(scaled_actions, torch.tensor([-10, 0, 10]))))
        self.assertTrue(torch.all(torch.isin(action_choices, torch.tensor([-1, 0, 1]))))
        
        # Check that log probabilities are valid (negative numbers)
        self.assertTrue(torch.all(log_probs <= 0))
        
        # Check that log probabilities correspond to actions
        for i, action in enumerate(action_choices):
            action_idx = int(action.item()) + 1  # Convert -1,0,1 to 0,1,2
            expected_log_prob = torch.log(network_outputs['action_probs'][i, action_idx])
            self.assertAlmostEqual(log_probs[i].item(), expected_log_prob.item())
    
    def test_scale_actions(self):
        """Test scaling of actions."""
        # Test single action
        actions = torch.tensor([-1, 0, 1])
        scaled_actions = self.interpreter.scale_actions(actions)
        
        # Check shape and values
        self.assertEqual(scaled_actions.shape, (3,))
        self.assertEqual(scaled_actions.dtype, torch.int64)  # Check that tensor is integer type
        self.assertEqual(scaled_actions[0].item(), -10)  # -1 * max_position_size
        self.assertEqual(scaled_actions[1].item(), 0)    # 0 * max_position_size
        self.assertEqual(scaled_actions[2].item(), 10)   # 1 * max_position_size
        
        # Test batch of actions
        batch_actions = torch.tensor([[-1, 0, 1], [1, -1, 0]])
        batch_scaled = self.interpreter.scale_actions(batch_actions)
        
        # Check shape and values
        self.assertEqual(batch_scaled.shape, (2, 3))
        self.assertEqual(batch_scaled.dtype, torch.int64)  # Check that tensor is integer type
        self.assertTrue(torch.all(batch_scaled[0] == torch.tensor([-10, 0, 10])))
        self.assertTrue(torch.all(batch_scaled[1] == torch.tensor([10, -10, 0])))
    
    def test_evaluate_actions_log_probs(self):
        """Test evaluation of action log probabilities."""
        # Create sample network outputs
        network_outputs = {
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],  # First asset: strongly sell
                [0.1, 0.8, 0.1],  # Second asset: strongly hold
                [0.1, 0.1, 0.8]   # Third asset: strongly buy
            ])
        }
        
        # Test single action
        actions = torch.tensor([-1, 0, 1])  # sell, hold, buy
        log_probs = self.interpreter.evaluate_actions_log_probs(network_outputs, actions)
        
        # Check shape
        self.assertEqual(log_probs.shape, (3,))
        
        # Check log probabilities
        expected_log_probs = torch.log(torch.tensor([0.8, 0.8, 0.8]))
        self.assertTrue(torch.allclose(log_probs, expected_log_probs))
        
    def test_interpret_batch(self):
        """Test interpret method with batch input."""
        network_outputs = {
            'action_probs': torch.tensor([
                [  # First batch
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8]
                ],
                [  # Second batch
                    [0.1, 0.1, 0.8],
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1]
                ]
            ])
        }
        scaled_actions, action_choices = self.interpreter.interpret(network_outputs, deterministic=True)
        self.assertEqual(scaled_actions.shape, (2, 3))
        self.assertEqual(action_choices.shape, (2, 3))
        self.assertTrue(torch.all(torch.isin(scaled_actions, torch.tensor([-10, 0, 10]))))
        self.assertTrue(torch.all(torch.isin(action_choices, torch.tensor([-1, 0, 1]))))

    def test_get_q_values_batch(self):
        """Test get_q_values with batch input."""
        network_outputs = {
            'action_probs': torch.tensor([
                [  # First batch
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8]
                ],
                [  # Second batch
                    [0.1, 0.1, 0.8],
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1]
                ]
            ])
        }
        actions = torch.tensor([[-1, 0, 1], [1, -1, 0]])
        q_values = self.interpreter.get_q_values(network_outputs, actions)
        self.assertEqual(q_values.shape, (2, 3))
        self.assertTrue(torch.allclose(q_values[0], torch.tensor([0.8, 0.8, 0.8]), atol=1e-6))
        self.assertTrue(torch.allclose(q_values[1], torch.tensor([0.8, 0.8, 0.8]), atol=1e-6))

    def test_get_max_q_values_batch(self):
        """Test get_max_q_values with batch input."""
        network_outputs = {
            'action_probs': torch.tensor([
                [  # First batch
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8]
                ],
                [  # Second batch
                    [0.1, 0.1, 0.8],
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1]
                ]
            ])
        }
        max_q_values = self.interpreter.get_max_q_values(network_outputs)
        self.assertEqual(max_q_values.shape, (2, 3))
        self.assertTrue(torch.allclose(max_q_values[0], torch.tensor([0.8, 0.8, 0.8]), atol=1e-6))
        self.assertTrue(torch.allclose(max_q_values[1], torch.tensor([0.8, 0.8, 0.8]), atol=1e-6))

    def test_interpret_with_log_prob_batch(self):
        """Test interpret_with_log_prob with batch input."""
        network_outputs = {
            'action_probs': torch.tensor([
                [  # First batch
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8]
                ],
                [  # Second batch
                    [0.1, 0.1, 0.8],
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1]
                ]
            ])
        }
        scaled_actions, action_choices, log_probs = self.interpreter.interpret_with_log_prob(network_outputs)
        self.assertEqual(scaled_actions.shape, (2, 3))
        self.assertEqual(action_choices.shape, (2, 3))
        self.assertEqual(log_probs.shape, (2, 3))
        self.assertTrue(torch.all(torch.isin(scaled_actions, torch.tensor([-10, 0, 10]))))
        self.assertTrue(torch.all(torch.isin(action_choices, torch.tensor([-1, 0, 1]))))
        self.assertTrue(torch.all(log_probs <= 0))

    def test_scale_actions_batch(self):
        """Test scale_actions with batch input."""
        batch_actions = torch.tensor([[-1, 0, 1], [1, -1, 0]])
        batch_scaled = self.interpreter.scale_actions(batch_actions)
        self.assertEqual(batch_scaled.shape, (2, 3))
        self.assertEqual(batch_scaled.dtype, torch.int64)
        self.assertTrue(torch.all(batch_scaled[0] == torch.tensor([-10, 0, 10])))
        self.assertTrue(torch.all(batch_scaled[1] == torch.tensor([10, -10, 0])))

    def test_evaluate_actions_log_probs_batch(self):
        """Test evaluate_actions_log_probs with batch input."""
        network_outputs = {
            'action_probs': torch.tensor([
                [  # First batch
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8]
                ],
                [  # Second batch
                    [0.1, 0.1, 0.8],
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1]
                ]
            ])
        }
        actions = torch.tensor([[-1, 0, 1], [1, -1, 0]])
        log_probs = self.interpreter.evaluate_actions_log_probs(network_outputs, actions)
        self.assertEqual(log_probs.shape, (2, 3))
        # Check log probabilities for first batch
        expected_log_probs_0 = torch.log(torch.tensor([0.8, 0.8, 0.8]))
        self.assertTrue(torch.allclose(log_probs[0], expected_log_probs_0, atol=1e-6))
        # Check log probabilities for second batch
        expected_log_probs_1 = torch.log(torch.tensor([0.8, 0.8, 0.8]))
        self.assertTrue(torch.allclose(log_probs[1], expected_log_probs_1, atol=1e-6))

    def test_compute_loss(self):
        """Test compute_loss with non-batched (single sample) inputs."""
        current_outputs = {
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]
            ])
        }
        target_outputs = {
            'action_probs': torch.tensor([
                [0.1, 0.1, 0.8],
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1]
            ])
        }
        action_choice = torch.tensor([-1, 0, 1])
        rewards = torch.tensor([1.0, -0.5, 0.5])
        dones = torch.tensor([0, 0, 1])
        gamma = 0.99
        loss = self.interpreter.compute_loss(current_outputs, target_outputs, action_choice, rewards, dones, gamma)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_compute_loss_batch(self):
        """Test compute_loss with batched inputs."""
        current_outputs = {
            'action_probs': torch.tensor([
                [  # First batch
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8]
                ],
                [  # Second batch
                    [0.1, 0.1, 0.8],
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1]
                ]
            ])
        }
        target_outputs = {
            'action_probs': torch.tensor([
                [  # First batch
                    [0.1, 0.1, 0.8],
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1]
                ],
                [  # Second batch
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8]
                ]
            ])
        }
        action_choice = torch.tensor([[-1, 0, 1], [1, -1, 0]])
        rewards = torch.tensor([1.0, -0.5])
        dones = torch.tensor([0, 1])
        gamma = 0.99
        loss = self.interpreter.compute_loss(current_outputs, target_outputs, action_choice, rewards, dones, gamma)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

if __name__ == '__main__':
    unittest.main() 