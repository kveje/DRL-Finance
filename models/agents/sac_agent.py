import numpy as np
import torch
from tensordict import TensorDict
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union, Any
from torchrl.data import ReplayBuffer, LazyTensorStorage
import random
import os

from utils.logger import Logger
logger = Logger.get_logger()

from models.agents.base_agent import BaseAgent
from models.networks.unified_network import UnifiedNetwork
from models.action_interpreters.base_action_interpreter import BaseActionInterpreter
from models.agents.temperature_manager import TemperatureManager

class SACAgent(BaseAgent):
    """
    Soft Actor-Critic (SAC) agent implementation.
    Uses unified networks for policy and Q-value prediction, and works with any action interpreter.
    """
    def __init__(
        self,
        network_config: Dict[str, Any],
        interpreter: BaseActionInterpreter,
        temperature_manager: TemperatureManager,
        update_frequency: int,
        device: str = "cuda",
        memory_size: int = 100000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.0003,
        alpha_lr: float = 0.0003,
        target_entropy: Optional[float] = None,
        automatic_entropy_tuning: bool = True,
        use_bayesian: bool = False,
        **kwargs
    ):
        """
        Initialize the SAC agent.
        
        Args:
            network_config: Configuration for the unified network
            interpreter: Action interpreter instance
            temperature_manager: Temperature manager instance
            update_frequency: Update frequency for the temperature manager
            device: Device to run the agent on
            memory_size: Size of the replay memory
            batch_size: Batch size for training
            gamma: Discount factor
            tau: Target smoothing coefficient for soft updates
            alpha: Initial entropy temperature
            target_entropy: Target entropy for automatic tuning
            learning_rate: Default learning rate
            actor_lr: Learning rate for the actor
            critic_lr: Learning rate for the critics
            alpha_lr: Learning rate for the temperature
            automatic_entropy_tuning: Whether to use automatic entropy tuning
            **kwargs: Additional arguments
        """
        super().__init__(network_config, interpreter, temperature_manager, update_frequency, device)
        
        # SAC specific parameters
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.target_entropy = target_entropy
        self.alpha = alpha
        self.use_bayesian = use_bayesian

        # Initialize replay memory
        self.memory_device = device
        self.memory = ReplayBuffer(
            storage=LazyTensorStorage(max_size=memory_size, device=self.memory_device),
            batch_size=batch_size,
        )

        # Create two Q-networks (critics)
        self.critic_1 = UnifiedNetwork(network_config, device=device)
        self.critic_2 = UnifiedNetwork(network_config, device=device)
        self.critic_target_1 = UnifiedNetwork(network_config, device=device)
        self.critic_target_2 = UnifiedNetwork(network_config, device=device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.actor_lr)

        # Entropy temperature (alpha)
        if self.automatic_entropy_tuning:
            if self.target_entropy is None:
                # Default target entropy: -|A| (number of assets)
                self.target_entropy = -float(network_config.get("n_assets", 1))
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None

        # Training step counter
        self.steps = 0

        # Last values
        self.last_critic_1_loss = 0.0
        self.last_critic_2_loss = 0.0
        self.last_actor_loss = 0.0
        self.last_alpha_loss = 0.0
        self.last_critic_1_grad_norm = 0.0
        self.last_critic_2_grad_norm = 0.0
        self.last_actor_grad_norm = 0.0

    def get_intended_action(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = True,
        sample: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select an action using the policy network (actor).
        Args:
            observations: Dictionary of observation numpy arrays
            deterministic: Whether to use deterministic action selection
            sample: Whether to sample from the distribution
        Returns:
            Tuple of (scaled_action, action_choices)
        """
        # Convert observations to tensors
        obs_tensors = {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            for k, v in observations.items()
        }

        # If we are using bayesian, we need to sample from the distribution
        use_sampling = self.use_bayesian and sample
        with torch.no_grad():
            # Get the network outputs
            network_outputs = self.network(obs_tensors, 
                                           use_sampling=use_sampling, 
                                           temperature=self.temperature_manager.get_all_temperatures())
            
        # Move the network outputs to the memory device if they are not already there
        network_outputs = {k: v.to(self.memory_device) for k, v in network_outputs.items()}

        # Interpret the network outputs
        if deterministic:
            scaled_action, action_choice = self.interpreter.interpret(network_outputs, 
                                                                      deterministic=True)
        else:
            scaled_action, action_choice = self.interpreter.interpret(network_outputs, 
                                                                      deterministic=False)
        
        # Convert to numpy and move to cpu (for environment)
        scaled_action = scaled_action.cpu().numpy()
        action_choice = action_choice.cpu().numpy()

        return scaled_action, action_choice

    def update(self, batch: Optional[TensorDict] = None) -> Dict[str, float]:
        """
        Update the networks using a batch of experience.
        Args:
            batch: Optional batch of experience (if None, sample from memory)
        Returns:
            Dictionary of training metrics
        """
        if batch is None:
            batch = self.get_batch()

        # Extract batch data
        obs = batch['observations']
        action_choices = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_observations']
        dones = batch['dones']

        # Move to device
        if self.memory_device != self.device:
            obs = {k: v.to(self.device) for k, v in obs.items()}
            action_choices = action_choices.to(self.device)
            rewards = rewards.to(self.device)
            next_obs = {k: v.to(self.device) for k, v in next_obs.items()}
            dones = dones.to(self.device)

        # --- Critic update ---
        # Sample next actions and log probs from the current policy
        with torch.no_grad():
            next_policy_outputs = self.network(next_obs, use_sampling=True, temperature=self.temperature_manager.get_all_temperatures())
            _, next_action_choices, next_log_probs = self.interpreter.interpret_with_log_prob(next_policy_outputs)
            
            # Compute target Q-values from target critics
            target_q1 = self.interpreter.get_q_values(self.critic_target_1(next_obs), next_action_choices)
            target_q2 = self.interpreter.get_q_values(self.critic_target_2(next_obs), next_action_choices)
            min_target_q = torch.min(target_q1, target_q2)
            
            # Add entropy term
            target_q = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * self.gamma * (min_target_q - self.alpha * next_log_probs)
        
        # Current Q estimates
        current_q1 = self.interpreter.get_q_values(self.critic_1(obs), action_choices)
        current_q2 = self.interpreter.get_q_values(self.critic_2(obs), action_choices)
        
        # Critic losses (MSE)
        critic_1_loss = torch.nn.functional.mse_loss(current_q1, target_q)
        critic_2_loss = torch.nn.functional.mse_loss(current_q2, target_q)
        
        # Optimize critics
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()

        critic_1_loss.backward(retain_graph=True)
        critic_2_loss.backward()

        critic_1_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
        critic_2_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)

        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # --- Actor update ---
        policy_outputs = self.network(obs, use_sampling=True, temperature=self.temperature_manager.get_all_temperatures())
        _, new_action_choices, log_probs = self.interpreter.interpret_with_log_prob(policy_outputs)

        q1_new = self.interpreter.get_q_values(self.critic_1(obs), new_action_choices)
        q2_new = self.interpreter.get_q_values(self.critic_2(obs), new_action_choices)
        min_q_new = torch.min(q1_new, q2_new)
        
        # Actor loss: maximize Q + entropy
        actor_loss = (self.alpha * log_probs - min_q_new).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # --- Alpha (entropy temperature) update ---
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)
        
        # --- Soft update of target networks ---
        with torch.no_grad():
            for (target_param1, param1), (target_param2, param2) in zip(
                zip(self.critic_target_1.parameters(), self.critic_1.parameters()),
                zip(self.critic_target_2.parameters(), self.critic_2.parameters())
            ):
                target_param1.data.mul_(1 - self.tau).add_(param1.data, alpha=self.tau)
                target_param2.data.mul_(1 - self.tau).add_(param2.data, alpha=self.tau)
        
        self.steps += 1

        def safe_item(tensor):
            return tensor.detach().cpu().item() if hasattr(tensor, 'item') else float(tensor)

        # Store last values
        self.last_critic_1_loss = safe_item(critic_1_loss)
        self.last_critic_2_loss = safe_item(critic_2_loss)
        self.last_actor_loss = safe_item(actor_loss)
        self.last_alpha_loss = safe_item(alpha_loss)
        self.last_critic_1_grad_norm = safe_item(critic_1_grad_norm)
        self.last_critic_2_grad_norm = safe_item(critic_2_grad_norm)
        self.last_actor_grad_norm = safe_item(actor_grad_norm)

        metrics = {
            "critic_1_loss": self.last_critic_1_loss,
            "critic_2_loss": self.last_critic_2_loss,
            "actor_loss": self.last_actor_loss,
            "alpha": self.alpha,
            "alpha_loss": self.last_alpha_loss,
            "critic_1_grad_norm": self.last_critic_1_grad_norm,
            "critic_2_grad_norm": self.last_critic_2_grad_norm,
            "actor_grad_norm": self.last_actor_grad_norm,
        }

        if self.use_bayesian:
            metrics.update(self.temperature_manager.get_all_temperatures_printerfriendly())

        return metrics

    def save(self, path: str) -> None:
        """
        Save the agent's networks and optimizer state.
        Args:
            path: Base path to save the models
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.network.state_dict(), os.path.join(path, 'network.pth'))
        torch.save(self.critic_1.state_dict(), os.path.join(path, 'critic_1.pth'))
        torch.save(self.critic_2.state_dict(), os.path.join(path, 'critic_2.pth'))
        torch.save(self.critic_target_1.state_dict(), os.path.join(path, 'critic_target_1.pth'))
        torch.save(self.critic_target_2.state_dict(), os.path.join(path, 'critic_target_2.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pth'))
        torch.save(self.critic_1_optimizer.state_dict(), os.path.join(path, 'critic_1_optimizer.pth'))
        torch.save(self.critic_2_optimizer.state_dict(), os.path.join(path, 'critic_2_optimizer.pth'))
        if self.automatic_entropy_tuning:
            torch.save(self.alpha_optimizer.state_dict(), os.path.join(path, 'alpha_optimizer.pth'))
            torch.save(self.log_alpha, os.path.join(path, 'log_alpha.pth'))
        agent_state = {
            'alpha': self.alpha,
            'steps': self.steps,
        }
        torch.save(agent_state, os.path.join(path, 'agent_state.pth'))

    def load(self, path: str) -> None:
        """
        Load the agent's networks and optimizer state.
        Args:
            path: Base path to load the models from
        """
        self.network.load_state_dict(torch.load(os.path.join(path, 'network.pth'), weights_only=False))
        self.critic_1.load_state_dict(torch.load(os.path.join(path, 'critic_1.pth'), weights_only=False))
        self.critic_2.load_state_dict(torch.load(os.path.join(path, 'critic_2.pth'), weights_only=False))
        self.critic_target_1.load_state_dict(torch.load(os.path.join(path, 'critic_target_1.pth'), weights_only=False))
        self.critic_target_2.load_state_dict(torch.load(os.path.join(path, 'critic_target_2.pth'), weights_only=False))
        self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pth'), weights_only=False))
        self.critic_1_optimizer.load_state_dict(torch.load(os.path.join(path, 'critic_1_optimizer.pth'), weights_only=False))
        self.critic_2_optimizer.load_state_dict(torch.load(os.path.join(path, 'critic_2_optimizer.pth'), weights_only=False))
        if self.automatic_entropy_tuning:
            self.alpha_optimizer.load_state_dict(torch.load(os.path.join(path, 'alpha_optimizer.pth'), weights_only=False))
            self.log_alpha = torch.load(os.path.join(path, 'log_alpha.pth'), weights_only=False)
            self.alpha = self.log_alpha.exp().item()
        agent_state = torch.load(os.path.join(path, 'agent_state.pth'), weights_only=False)
        self.alpha = agent_state['alpha']
        self.steps = agent_state['steps']

    def get_model_name(self) -> str:
        """Get the name of the model."""
        return "sac"

    def get_model(self) -> nn.Module:
        """Get the policy (actor) network."""
        return self.network

    def train(self, mode: bool = True) -> None:
        """Set the model to training mode."""
        self.network.train(mode)
        self.critic_1.train(mode)
        self.critic_2.train(mode)
        self.critic_target_1.train(mode)
        self.critic_target_2.train(mode)

    def eval(self) -> None:
        """Set the model to evaluation mode."""
        self.train(mode=False)

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the agent."""
        return {
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "alpha_lr": self.alpha_lr,
            "gamma": self.gamma,
            "tau": self.tau,
            "alpha": self.alpha,
            "automatic_entropy_tuning": self.automatic_entropy_tuning,
            "memory_size": self.memory_size,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "network_config": self.network_config
        }

    def add_to_memory(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        action_choice: np.ndarray,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool
    ) -> None:
        """
        Add a transition to the replay memory.
        Args:
            observation: Current observation dictionary
            action: Scaled action taken (from interpret())
            action_choice: Unscaled action choice for learning
            reward: Reward received
            next_observation: Next observation dictionary
            done: Whether the episode is done
        """
        obs_tensors = {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in observation.items()
        }
        next_obs_tensors = {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in next_observation.items()
        }
        transition = TensorDict({
            'observations': obs_tensors,
            'actions': torch.as_tensor(action_choice, dtype=torch.long),
            'rewards': torch.as_tensor(reward, dtype=torch.float32),
            'next_observations': next_obs_tensors,
            'dones': torch.as_tensor(done, dtype=torch.float32)
        }, device=self.memory_device)
        self.memory.add(transition)

    def get_batch(self) -> TensorDict:
        """
        Sample a batch of transitions from the replay memory.
        Returns:
            TensorDict containing batched tensors for:
            - observations: Dict of observation tensors
            - actions: Action indices tensor
            - rewards: Reward tensor
            - next_observations: Dict of next observation tensors
            - dones: Done flag tensor
        """
        if len(self.memory) < self.batch_size:
            raise ValueError(f"Not enough samples in memory. Need {self.batch_size}, have {len(self.memory)}")
        return self.memory.sample()

    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information for visualization.
        Returns:
            Dictionary containing agent information:
            - alpha: Current entropy temperature
            - loss: Last actor loss
        """
        info = {
            'alpha': self.alpha,
            'critic_1_loss': self.last_critic_1_loss,
            'critic_2_loss': self.last_critic_2_loss,
            'actor_loss': self.last_actor_loss,
            'alpha_loss': self.last_alpha_loss,
            'critic_1_grad_norm': self.last_critic_1_grad_norm,
            'critic_2_grad_norm': self.last_critic_2_grad_norm,
            'actor_grad_norm': self.last_actor_grad_norm,
        }
        if self.use_bayesian:
            info.update(self.temperature_manager.get_all_temperatures_printerfriendly())
        return info

    def _get_agent_specific_state(self) -> Dict[str, Any]:
        """
        Get SAC-specific state for checkpointing.
        Returns:
            Dictionary containing SAC-specific state
        """
        state = {}
        state['critic_1_state_dict'] = self.critic_1.state_dict()
        state['critic_2_state_dict'] = self.critic_2.state_dict()
        state['critic_target_1_state_dict'] = self.critic_target_1.state_dict()
        state['critic_target_2_state_dict'] = self.critic_target_2.state_dict()
        state['critic_1_optimizer_state_dict'] = self.critic_1_optimizer.state_dict()
        state['critic_2_optimizer_state_dict'] = self.critic_2_optimizer.state_dict()

        return state

    def _load_agent_specific_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load SAC-specific state from checkpoint.
        Args:
            state_dict: Dictionary containing the agent's state
        """
        if 'critic_1_state_dict' in state_dict:
            self.critic_1.load_state_dict(state_dict['critic_1_state_dict'])
        if 'critic_2_state_dict' in state_dict:
            self.critic_2.load_state_dict(state_dict['critic_2_state_dict'])
        if 'critic_target_1_state_dict' in state_dict:
            self.critic_target_1.load_state_dict(state_dict['critic_target_1_state_dict'])
        if 'critic_target_2_state_dict' in state_dict:
            self.critic_target_2.load_state_dict(state_dict['critic_target_2_state_dict'])
        if 'critic_1_optimizer_state_dict' in state_dict:
            self.critic_1_optimizer.load_state_dict(state_dict['critic_1_optimizer_state_dict'])
        if 'critic_2_optimizer_state_dict' in state_dict:
            self.critic_2_optimizer.load_state_dict(state_dict['critic_2_optimizer_state_dict'])

    def sufficient_memory(self) -> bool:
        """Check if the agent has enough memory to perform an update."""
        return len(self.memory) >= self.batch_size
