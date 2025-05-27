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
            max_trade_size=10,
        )
    
    def test_initialization(self):
        """Test that the interpreter initializes correctly."""
        self.assertIsNotNone(self.interpreter)
        self.assertEqual(self.interpreter.n_assets, 3)
        self.assertEqual(self.interpreter.max_trade_size, 10)

    def test_interpret_deterministic(self):
        """Test deterministic interpretation of action probabilities and confidences."""
        network_outputs = {
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]
            ]),
            'confidences': torch.tensor([0.9, 0.5, 0.7])
        }
        scaled_actions, action_choices = self.interpreter.interpret(network_outputs, deterministic=True)
        self.assertEqual(scaled_actions.shape, (3,))
        self.assertEqual(action_choices.shape, (3,))
        self.assertEqual(scaled_actions.dtype, torch.int64)
        self.assertEqual(action_choices.dtype, torch.int64)
        self.assertIn(scaled_actions[0].item(), [-9, -8])
        self.assertEqual(scaled_actions[1].item(), 0)
        self.assertIn(scaled_actions[2].item(), [7, 6])
        self.assertIn(action_choices[0].item(), [-1, 0, 1])
        self.assertIn(action_choices[1].item(), [-1, 0, 1])
        self.assertIn(action_choices[2].item(), [-1, 0, 1])
    
    def test_interpret_stochastic(self):
        """Test stochastic interpretation of action probabilities and confidences."""
        network_outputs = {
            'action_probs': torch.tensor([
                [0.33, 0.34, 0.33],
                [0.33, 0.34, 0.33],
                [0.33, 0.34, 0.33]
            ]),
            'confidences': torch.tensor([0.5, 0.5, 0.5])
        }
        actions_list = []
        for _ in range(10):
            scaled_actions, action_choices = self.interpreter.interpret(network_outputs, deterministic=False)
            actions_list.append((scaled_actions, action_choices))
        unique_actions = set(tuple(a[0].tolist()) for a in actions_list)
        self.assertGreater(len(unique_actions), 1)
        for scaled_actions, action_choices in actions_list:
            self.assertEqual(scaled_actions.dtype, torch.int64)
            self.assertTrue(torch.all(torch.abs(scaled_actions) <= self.interpreter.max_trade_size))
            self.assertTrue(torch.all(torch.isin(action_choices, torch.tensor([-1, 0, 1]))))
    
    def test_handle_tensor_input(self):
        """Test handling of tensor inputs."""
        network_outputs = {
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]
            ]),
            'confidences': torch.tensor([0.9, 0.5, 0.7])
        }
        scaled_actions, action_choices = self.interpreter.interpret(network_outputs)
        self.assertTrue(torch.all(torch.isfinite(scaled_actions)))
        self.assertEqual(scaled_actions.shape, (3,))
        self.assertEqual(scaled_actions.dtype, torch.int64)
        self.assertIn(scaled_actions[0].item(), [-9, -8])
        self.assertEqual(scaled_actions[1].item(), 0)
        self.assertIn(scaled_actions[2].item(), [7, 6])
        self.assertTrue(torch.all(torch.isin(action_choices, torch.tensor([-1, 0, 1]))))
    
    def test_handle_batch_input(self):
        """Test handling of batch inputs."""
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
            ]),
            'confidences': torch.tensor([
                [0.9, 0.5, 0.7],
                [0.7, 0.9, 0.5]
            ])
        }
        scaled_actions, action_choices = self.interpreter.interpret(network_outputs)
        self.assertTrue(torch.all(torch.isfinite(scaled_actions)))
        self.assertEqual(scaled_actions.shape, (2, 3))
        self.assertEqual(scaled_actions.dtype, torch.int64)
        self.assertIn(scaled_actions[0, 0].item(), [-9, -8])
        self.assertEqual(scaled_actions[0, 1].item(), 0)
        self.assertIn(scaled_actions[0, 2].item(), [7, 6])
        self.assertTrue(torch.all(torch.isin(action_choices, torch.tensor([-1, 0, 1]))))
    
    def test_get_q_values(self):
        """Test Q-value extraction."""
        # Create network outputs
        network_outputs = {
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],  # Sell
                [0.1, 0.8, 0.1],  # Hold
                [0.1, 0.1, 0.8]   # Buy
            ]),
            'confidences': torch.tensor([
                0.9,  # High confidence
                0.5,  # Medium confidence
                0.7   # High confidence
            ])
        }
        
        # Test actions
        actions = torch.tensor([-1, 0, 1])  # sell, hold, buy
        
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
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],  # Sell
                [0.1, 0.8, 0.1],  # Hold
                [0.1, 0.1, 0.8]   # Buy
            ]),
            'confidences': torch.tensor([
                0.9,  # High confidence
                0.5,  # Medium confidence
                0.7   # High confidence
            ])
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
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],  # First asset: strongly sell
                [0.1, 0.8, 0.1],  # Second asset: strongly hold
                [0.1, 0.1, 0.8]   # Third asset: strongly buy
            ]),
            'confidences': torch.tensor([
                0.9,  # High confidence in first asset
                0.5,  # Medium confidence in second asset
                0.7   # High confidence in third asset
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
        self.assertEqual(action_choices.dtype, torch.int64)
        
        # Check that actions are within valid range
        self.assertTrue(torch.all(torch.abs(scaled_actions) <= self.interpreter.max_trade_size))
        
        # Check that log probabilities are valid (negative numbers)
        self.assertTrue(torch.all(log_probs <= 0))
        
        # Check that log probabilities correspond to actions
        for i, action in enumerate(scaled_actions):
            action_idx = int(torch.sign(action)) + 1  # Convert -1,0,1 to 0,1,2
            expected_log_prob = torch.log(network_outputs['action_probs'][i, action_idx])
            self.assertAlmostEqual(log_probs[i].item(), expected_log_prob.item())
            
            # Check that actions are scaled by confidence
            expected_action = (action_idx - 1) * self.interpreter.max_trade_size * network_outputs['confidences'][i]
            self.assertAlmostEqual(action.item(), expected_action.item())

    def test_scale_actions_with_confidence(self):
        """Test scaling of actions with confidence values."""
        # Test single action
        actions = torch.tensor([-1, 0, 1])
        confidences = torch.tensor([0.9, 0.5, 0.7])
        scaled_actions = self.interpreter.scale_actions_with_confidence(actions, confidences)
        
        # Check shape and values
        self.assertEqual(scaled_actions.shape, (3,))
        self.assertEqual(scaled_actions.dtype, torch.int64)  # Check that tensor is integer type
        self.assertIn(scaled_actions[0].item(), [-9, -8])  # -1 * max_position_size * 0.9 (approximately -9)
        self.assertEqual(scaled_actions[1].item(), 0)      # 0 * max_position_size * 0.5
        self.assertIn(scaled_actions[2].item(), [7, 6])    # 1 * max_position_size * 0.7 (approximately 7)
        
        # Test batch of actions
        batch_actions = torch.tensor([[-1, 0, 1], [1, -1, 0]])
        batch_confidences = torch.tensor([[0.9, 0.5, 0.7], [0.8, 0.6, 0.4]])
        batch_scaled = self.interpreter.scale_actions_with_confidence(batch_actions, batch_confidences)
        
        # Check shape and values
        self.assertEqual(batch_scaled.shape, (2, 3))
        self.assertEqual(batch_scaled.dtype, torch.int64)  # Check that tensor is integer type
        # First batch: [-1, 0, 1] * [0.9, 0.5, 0.7] * 10 = [-9, 0, 7]
        self.assertIn(batch_scaled[0][0].item(), [-9, -8])  # -1 * 10 * 0.9
        self.assertEqual(batch_scaled[0][1].item(), 0)      # 0 * 10 * 0.5
        self.assertIn(batch_scaled[0][2].item(), [7, 6])    # 1 * 10 * 0.7
        # Second batch: [1, -1, 0] * [0.8, 0.6, 0.4] * 10 = [8, -6, 0]
        self.assertIn(batch_scaled[1][0].item(), [8, 7])    # 1 * 10 * 0.8
        self.assertIn(batch_scaled[1][1].item(), [-6, -7])  # -1 * 10 * 0.6
        self.assertEqual(batch_scaled[1][2].item(), 0)      # 0 * 10 * 0.4
    
    def test_evaluate_actions_log_probs(self):
        """Test evaluation of action log probabilities."""
        # Create sample network outputs
        network_outputs = {
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],  # First asset: strongly sell
                [0.1, 0.8, 0.1],  # Second asset: strongly hold
                [0.1, 0.1, 0.8]   # Third asset: strongly buy
            ]),
            'confidences': torch.tensor([
                0.9,  # High confidence in first asset
                0.5,  # Medium confidence in second asset
                0.7   # High confidence in third asset
            ])
        }
        
        # Test single action
        actions = torch.tensor([-1, 0, 1])  # sell, hold, buy indices
        log_probs = self.interpreter.evaluate_actions_log_probs(network_outputs, actions)
        
        # Check shapes
        self.assertEqual(log_probs.shape, (3,))
        
        # Check log probabilities
        expected_log_probs = torch.log(torch.tensor([0.8, 0.8, 0.8]))
        self.assertTrue(torch.allclose(log_probs, expected_log_probs))
        
    def test_compute_loss(self):
        """Test loss computation for confidence-scaled actions."""
        # Create sample network outputs
        current_outputs = {
            'action_probs': torch.tensor([
                [0.8, 0.1, 0.1],  # First asset: strongly sell
                [0.1, 0.8, 0.1],  # Second asset: strongly hold
                [0.1, 0.1, 0.8]   # Third asset: strongly buy
            ]),
            'confidences': torch.tensor([
                0.9,  # High confidence in first asset
                0.5,  # Medium confidence in second asset
                0.7   # High confidence in third asset
            ])
        }
        
        target_outputs = {
            'action_probs': torch.tensor([
                [0.1, 0.1, 0.8],  # First asset: strongly buy
                [0.8, 0.1, 0.1],  # Second asset: strongly sell
                [0.1, 0.8, 0.1]   # Third asset: strongly hold
            ]),
            'confidences': torch.tensor([
                0.7,  # High confidence in first asset
                0.9,  # High confidence in second asset
                0.5   # Medium confidence in third asset
            ])
        }
        
        # Create sample actions, rewards, and dones
        scaled_action, action_choice = self.interpreter.interpret(current_outputs, deterministic=True)
        rewards = torch.tensor([1.0, -0.5, 0.5])
        dones = torch.tensor([0, 0, 1])
        gamma = 0.99
        
        # Compute loss
        loss = self.interpreter.compute_loss(current_outputs, target_outputs, action_choice, rewards, dones, gamma)
        
        # Check that loss is a scalar tensor
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))


    def test_compute_loss_batch(self):
        """Test loss computation with batched inputs."""
        # Create sample network outputs with batch size 2
        current_outputs = {
            'action_probs': torch.tensor([
                [  # First batch
                    [0.8, 0.1, 0.1],  # First asset: strongly sell
                    [0.1, 0.8, 0.1],  # Second asset: strongly hold
                    [0.1, 0.1, 0.8]   # Third asset: strongly buy
                ],
                [  # Second batch
                    [0.1, 0.1, 0.8],  # First asset: strongly buy
                    [0.8, 0.1, 0.1],  # Second asset: strongly sell
                    [0.1, 0.8, 0.1]   # Third asset: strongly hold
                ]
            ]),
            'confidences': torch.tensor([
                [  # First batch
                    0.9,  # High confidence in first asset
                    0.5,  # Medium confidence in second asset
                    0.7   # High confidence in third asset
                ],
                [  # Second batch
                    0.7,  # High confidence in first asset
                    0.9,  # High confidence in second asset
                    0.5   # Medium confidence in third asset
                ]
            ])
        }
        
        target_outputs = {
            'action_probs': torch.tensor([
                [  # First batch
                    [0.1, 0.1, 0.8],  # First asset: strongly buy
                    [0.8, 0.1, 0.1],  # Second asset: strongly sell
                    [0.1, 0.8, 0.1]   # Third asset: strongly hold
                ],
                [  # Second batch
                    [0.8, 0.1, 0.1],  # First asset: strongly sell
                    [0.1, 0.8, 0.1],  # Second asset: strongly hold
                    [0.1, 0.1, 0.8]   # Third asset: strongly buy
                ]
            ]),
            'confidences': torch.tensor([
                [  # First batch
                    0.7,  # High confidence in first asset
                    0.9,  # High confidence in second asset
                    0.5   # Medium confidence in third asset
                ],
                [  # Second batch
                    0.9,  # High confidence in first asset
                    0.5,  # Medium confidence in second asset
                    0.7   # High confidence in third asset
                ]
            ])
        }
        
        # Create sample actions, rewards, and dones for batch size 2
        action_choice = torch.tensor([
            [-1, 0, 1],  # First batch: sell, hold, buy
            [1, -1, 0]   # Second batch: buy, sell, hold
        ], dtype=torch.long)
        
        rewards = torch.tensor([1.0, -0.5])
        
        dones = torch.tensor([0, 1])
        
        gamma = 0.99
        
        # Compute loss
        loss = self.interpreter.compute_loss(current_outputs, target_outputs, action_choice, rewards, dones, gamma)
        
        # Check that loss is a scalar tensor
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))


    def test_get_config(self):
        """Test getting interpreter configuration."""
        config = self.interpreter.get_config()
        
        # Check that all expected keys are present
        expected_keys = {
            'n_assets',
            'max_position_size',
        }
        self.assertEqual(set(config.keys()), expected_keys)
        
        # Check that values match the interpreter's attributes
        self.assertEqual(config['n_assets'], self.interpreter.n_assets)
        self.assertEqual(config['max_position_size'], self.interpreter.max_trade_size)

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
            ]),
            'confidences': torch.tensor([
                [0.9, 0.5, 0.7],
                [0.7, 0.9, 0.5]
            ])
        }
        scaled_actions, action_choices = self.interpreter.interpret(network_outputs, deterministic=True)
        self.assertEqual(scaled_actions.shape, (2, 3))
        self.assertEqual(action_choices.shape, (2, 3))
        self.assertTrue(torch.all(torch.isin(scaled_actions, torch.tensor([-9, -8, 0, 7, 6, 9, 8]))))
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
            ]),
            'confidences': torch.tensor([
                [0.9, 0.5, 0.7],
                [0.7, 0.9, 0.5]
            ])
        }
        actions = torch.tensor([[-1, 0, 1], [1, -1, 0]])
        q_values = self.interpreter.get_q_values(network_outputs, actions)
        self.assertEqual(q_values.shape, (2, 3))
        self.assertTrue(torch.allclose(q_values[0], torch.tensor([0.8*0.9, 0.8*0.5, 0.8*0.7]), atol=1e-6))
        self.assertTrue(torch.allclose(q_values[1], torch.tensor([0.8*0.7, 0.8*0.9, 0.8*0.5]), atol=1e-6))

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
            ]),
            'confidences': torch.tensor([
                [0.9, 0.5, 0.7],
                [0.7, 0.9, 0.5]
            ])
        }
        max_q_values = self.interpreter.get_max_q_values(network_outputs)
        self.assertEqual(max_q_values.shape, (2, 3))
        self.assertTrue(torch.allclose(max_q_values[0], torch.tensor([0.8*0.9, 0.8*0.5, 0.8*0.7]), atol=1e-6))
        self.assertTrue(torch.allclose(max_q_values[1], torch.tensor([0.8*0.7, 0.8*0.9, 0.8*0.5]), atol=1e-6))

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
            ]),
            'confidences': torch.tensor([
                [0.9, 0.5, 0.7],
                [0.7, 0.9, 0.5]
            ])
        }

        scaled_actions, action_choices, log_probs = self.interpreter.interpret_with_log_prob(network_outputs)
        self.assertEqual(scaled_actions.shape, (2, 3))
        self.assertEqual(action_choices.shape, (2, 3))
        self.assertEqual(log_probs.shape, (2, 3))
        self.assertTrue(torch.all(torch.isin(action_choices, torch.tensor([-1, 0, 1]))))
        self.assertTrue(torch.all(log_probs <= 0))

    def test_scale_actions_with_confidence_batch(self):
        """Test scale_actions_with_confidence with batch input."""
        batch_actions = torch.tensor([[-1, 0, 1], [1, -1, 0]])
        batch_confidences = torch.tensor([[0.9, 0.5, 0.7], [0.8, 0.6, 0.4]])
        batch_scaled = self.interpreter.scale_actions_with_confidence(batch_actions, batch_confidences)
        self.assertEqual(batch_scaled.shape, (2, 3))
        self.assertEqual(batch_scaled.dtype, torch.int64)
        self.assertIn(batch_scaled[0][0].item(), [-9, -8])  # -1 * 10 * 0.9
        self.assertEqual(batch_scaled[0][1].item(), 0)      # 0 * 10 * 0.5
        self.assertIn(batch_scaled[0][2].item(), [7, 6])    # 1 * 10 * 0.7
        self.assertIn(batch_scaled[1][0].item(), [8, 7])    # 1 * 10 * 0.8
        self.assertIn(batch_scaled[1][1].item(), [-6, -7])  # -1 * 10 * 0.6
        self.assertEqual(batch_scaled[1][2].item(), 0)      # 0 * 10 * 0.4

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
            ]),
            'confidences': torch.tensor([
                [0.9, 0.5, 0.7],
                [0.7, 0.9, 0.5]
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

if __name__ == '__main__':
    unittest.main() 