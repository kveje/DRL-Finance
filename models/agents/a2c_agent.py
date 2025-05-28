import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union, Any
import os

from tensordict import TensorDict

from utils.logger import Logger
logger = Logger.get_logger()

from models.agents.base_agent import BaseAgent
from models.action_interpreters.base_action_interpreter import BaseActionInterpreter
from models.networks.unified_network import UnifiedNetwork
from models.agents.temperature_manager import TemperatureManager

class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic (A2C) agent implementation.
    Uses a unified network for policy and value prediction.
    """
    
    def __init__(
        self,
        network_config: Dict[str, Any],
        interpreter: BaseActionInterpreter,
        temperature_manager: TemperatureManager,
        update_frequency: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.0007,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        use_bayesian: bool = False,
        **kwargs
    ) -> None:
        """
        Initialize the A2C agent.
        
        Args:
            network_config: Configuration for the unified network
            interpreter: Action interpreter instance
            device: Device to run the agent on
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            entropy_coef: Coefficient for entropy loss
            value_coef: Coefficient for value loss
            max_grad_norm: Maximum gradient norm for clipping
            **kwargs: Additional arguments
        """
        super().__init__(network_config, interpreter, temperature_manager, update_frequency, device)
        self.use_bayesian = use_bayesian
        
        # Hyperparameters
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.learning_rate = learning_rate

        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Storage for rollout data
        self.memory_device = "cpu"
        self.rollout: List[TensorDict] = []
        self.max_rollout_size = update_frequency
        
        # Steps counter
        self.steps = 0
        
        # Track metrics for visualization
        self.last_loss = 0.0
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy_loss = 0.0
        self.last_total_loss = 0.0
        self.last_mean_advantage = 0.0
        self.last_mean_return = 0.0
    

    def get_intended_action(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = True,
        sample: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the intended action using the policy network.
        
        Args:
            observations: Dictionary of observation tensors
            current_position: Current position of the agent
            deterministic: Whether to select actions deterministically
            
        Returns:
            Tuple of (scaled_action, action_choice) where:
            - scaled_action is the actual action to execute
            - action_choice is the unscaled action for learning
        """
        # Convert observations to tensors
        obs_tensors = {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            for k, v in observations.items()
        }

        use_sampling = self.use_bayesian and sample
        # Get network outputs
        with torch.no_grad():
            network_outputs = self.network(obs_tensors, use_sampling=use_sampling, temperature = self.temperature_manager.get_all_temperatures())

            # Move the network outputs to memory device
            network_outputs = {k: v.to(self.memory_device) for k, v in network_outputs.items()}
            
            # Sample from policy
            scaled_action, action_choice, log_probs = self.interpreter.interpret_with_log_prob(network_outputs)

        # Store value and log probs for training
        self.last_log_probs = log_probs
        self.last_value = network_outputs["value"]

        # Convert to numpy
        scaled_action = scaled_action.numpy()
        action_choice = action_choice.numpy()
            
        return scaled_action, action_choice
    
    def add_to_rollout(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        action_choice: np.ndarray,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool
    ) -> None:
        """
        Add a transition to the current rollout.
        
        Args:
            observation: Current state observation
            action: Scaled action taken
            action_choice: Unscaled action choice for learning
            reward: Reward received
            next_observation: Next state observation
            done: Whether the episode ended
        """
        # Convert observations to tensors
        obs_tensors = {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            for k, v in observation.items()
        }

        next_obs_tensors = {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            for k, v in next_observation.items()
        }
        
        # Store transition data
        transition = TensorDict({
            "obs": obs_tensors,
            "next_obs": next_obs_tensors,
            "action_choices": torch.as_tensor(action_choice, dtype=torch.long),
            "log_probs": self.last_log_probs,
            "rewards": torch.tensor(reward, dtype=torch.float32),
            "dones": torch.tensor(done, dtype=torch.float32),
            "values": self.last_value
        }, device=self.memory_device)

        self.rollout.append(transition)

    def _compute_gae(self, rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, next_value: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) for the rollout.
        
        Args:
            rewards: Tensor of rewards
            dones: Tensor of done flags
            values: Tensor of current values
            dones: Tensor of done flags
            next_values: Tensor of next values
            
        Returns:
            Tuple of (advantages, returns)
        """
        T = len(rewards)
        
        # Bootstrap value if provided
        if next_value is None:
            next_value = torch.zeros(size=(1,), device=self.device)

        if next_value.dim() == 2:
            next_value = next_value.squeeze(-1)

        # Append next value for bootstrapping
        values_with_next = torch.cat([values, next_value])

        # Compute TD errors
        deltas = rewards + self.gamma * values_with_next[1:] * (1 - dones) - values_with_next[:-1]

        # Compute GAE
        advantages = torch.zeros_like(deltas)
        gae = 0

        for t in reversed(range(T)):
            gae = deltas[t] + self.gamma * self.gae_lambda * gae * (1 - dones[t])
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns
        
    
    def update(self, external_rollout: Optional[List[TensorDict]] = None) -> Dict[str, float]:
        """
        Update the agent's network using collected rollout data.
        
        Args:
            rollout: Dictionary containing rollout data (Default: None - uses internal rollout buffer)
            
        Returns:
            Dictionary of training metrics
        """
        if external_rollout is not None:
            rollout = external_rollout
        else:
            # Use internal rollout buffer
            if len(self.rollout) == 0:
                return {"policy_loss": 0, "value_loss": 0, "entropy_loss": 0, "total_loss": 0}
            
            rollout = self.rollout

        # Stack list of TensorDicts into a single TensorDict
        rollout = TensorDict.stack(rollout, dim = 0).to(self.device)

        # Get data from rollout
        obs = rollout["obs"]
        next_obs = rollout["next_obs"]
        action_choices = rollout["action_choices"]
        old_log_probs = rollout["log_probs"]
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        values = rollout["values"]

        # Compute bootstrap value for final state if episode is not done
        next_value = None
        if len(dones) > 0 and not dones[-1]:
            # Get value of final state for bootstrapping
            final_obs = {k: v[-1] for k, v in obs.items()}
            with torch.no_grad():
                final_outputs = self.network(final_obs, use_sampling=self.use_bayesian, temperature = self.temperature_manager.get_all_temperatures())
                next_value = final_outputs["value"]


        # Handle dimensions of values and next_value
        if values.dim() == 2:
            values = values.squeeze(-1)
        if next_value is not None:
            if next_value.dim() == 2:
                next_value = next_value.squeeze(-1)
                next_value = next_value.to(self.device)
        if rewards.dim() == 2:
            rewards = rewards.squeeze(-1)
        if dones.dim() == 2:
            dones = dones.squeeze(-1)
        if action_choices.dim() == 2:
            action_choices = action_choices.squeeze(-1)


        # Calculate returns and advantages
        advantages, returns = self._compute_gae(rewards, dones, values, next_value)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        
        # Get new outputs for current policy
        network_outputs = self.network(obs, use_sampling=self.use_bayesian, temperature = self.temperature_manager.get_all_temperatures())

        # Get new logprobs and values
        new_log_probs = self.interpreter.evaluate_actions_log_probs(network_outputs, action_choices)
        new_values = network_outputs["value"]

        # Compute losses
        value_loss = self.value_coef * ((returns - new_values) ** 2).mean()
        policy_loss = -(advantages * new_log_probs.sum(dim=-1)).mean()
        
        # Entropy loss (encourage exploration)
        entropy = torch.mean(-new_log_probs.sum(dim=-1))
        entropy_loss = -self.entropy_coef * entropy.mean()
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # Store loss values for visualization
        self.last_policy_loss = policy_loss.detach().cpu().item()
        self.last_value_loss = value_loss.detach().cpu().item()
        self.last_entropy_loss = entropy_loss.detach().cpu().item()
        self.last_total_loss = total_loss.detach().cpu().item()
        self.last_mean_advantage = advantages.mean().detach().cpu().item()
        self.last_mean_return = returns.mean().detach().cpu().item()
        
        # Clear rollout buffers
        if external_rollout is None:
            self.rollout = []
        
        # Increment steps
        self.steps += 1
        
        metrics = {
            "policy_loss": self.last_policy_loss,
            "value_loss": self.last_value_loss,
            "entropy_loss": self.last_entropy_loss,
            "total_loss": self.last_total_loss,
            "mean_advantage": self.last_mean_advantage,
            "mean_return": self.last_mean_return
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
        
        # Save network state
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps
        }, os.path.join(path, 'agent_state.pth'))

    def load(self, path: str) -> None:
        """
        Load the agent's networks and optimizer state.
        
        Args:
            path: Base path to load the models from
        """
        checkpoint = torch.load(os.path.join(path, 'agent_state.pth'), weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']

    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information for visualization.
        
        Returns:
            Dictionary containing agent information:
            - policy_loss: Last policy loss
            - value_loss: Last value loss
            - entropy_loss: Last entropy loss
            - total_loss: Last total loss
        """
        info = {
            "policy_loss": getattr(self, 'last_policy_loss', 0),
            "value_loss": getattr(self, 'last_value_loss', 0),
            "entropy_loss": getattr(self, 'last_entropy_loss', 0),
            "total_loss": getattr(self, 'last_total_loss', 0),
            "mean_advantage": getattr(self, 'mean_advantage', 0),
            "mean_return": getattr(self, 'mean_return', 0)
        }
        if self.use_bayesian:
            info.update(self.temperature_manager.get_all_temperatures_printerfriendly())

        return info

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the agent."""
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "device": str(self.device),
            "network_config": self.network_config,
            "gae_lambda": self.gae_lambda
        }

    def _get_agent_specific_state(self) -> Dict[str, Any]:
        """
        Get A2C-specific state for checkpointing.
        
        Returns:
            Dictionary containing A2C-specific state
        """
        state = {}
        state['last_value'] = self.last_value.detach().cpu() if hasattr(self, 'last_value') else None
        state['last_log_probs'] = self.last_log_probs.detach().cpu() if hasattr(self, 'last_log_probs') else None
        state['last_policy_loss'] = getattr(self, 'last_policy_loss', 0.0)
        state['last_value_loss'] = getattr(self, 'last_value_loss', 0.0)
        state['last_entropy_loss'] = getattr(self, 'last_entropy_loss', 0.0)
        state['last_total_loss'] = getattr(self, 'last_total_loss', 0.0)

        return state
    
    def _load_agent_specific_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load A2C-specific state from checkpoint.
        
        Args:
            state_dict: Dictionary containing the agent's state
        """
        if 'last_value' in state_dict:
            self.last_value = state_dict['last_value'].to(self.device)
        if 'last_log_probs' in state_dict:
            self.last_log_probs = state_dict['last_log_probs'].to(self.device)
        self.last_policy_loss = state_dict.get('last_policy_loss', 0.0)
        self.last_value_loss = state_dict.get('last_value_loss', 0.0)
        self.last_entropy_loss = state_dict.get('last_entropy_loss', 0.0)
        self.last_total_loss = state_dict.get('last_total_loss', 0.0)

    def get_model_name(self) -> str:
        return "a2c" 