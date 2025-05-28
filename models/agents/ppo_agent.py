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
from models.agents.temperature_manager import TemperatureManager

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent implementation.
    Uses clipped surrogate objective to stabilize training.
    """
    
    def __init__(
        self,
        network_config: Dict[str, Any],
        interpreter: BaseActionInterpreter,
        temperature_manager: TemperatureManager,
        update_frequency: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        use_bayesian: bool = False,
        **kwargs
    ):
        """
        Initialize the PPO agent.
        
        Args:
            network_config: Configuration for the unified network
            interpreter: Action interpreter instance
            temperature_manager: Temperature manager instance
            update_frequency: Update frequency for the temperature manager
            device: Device to run the agent on
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            gae_lambda: Lambda for Generalized Advantage Estimation
            clip_ratio: Clip ratio for PPO update
            entropy_coef: Coefficient for entropy loss
            value_coef: Coefficient for value loss
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO epochs for each update
            batch_size: Mini-batch size for PPO update
            use_bayesian: Whether to use Bayesian inference
            **kwargs: Additional arguments
        """
        super().__init__(network_config, interpreter, temperature_manager, update_frequency, device)
        self.use_bayesian = use_bayesian
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Storage for rollout data
        self.memory_device = "cpu"
        self.rollout: List[TensorDict] = []
        self.max_rollout_size = update_frequency
        
        # Steps counter
        self.steps = 0
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Last values
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
            deterministic: Whether to select actions deterministically
            sample: Whether to sample from the distribution (for Bayesian)
        
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
            network_outputs = self.network(obs_tensors, use_sampling=use_sampling, temperature=self.temperature_manager.get_all_temperatures())
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
            next_value: Value of the next state for bootstrapping
        
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
        Update the agent's network using collected rollout data with PPO objective.
        
        Args:
            external_rollout: Optional external rollout data (otherwise uses internal buffer)
        
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
        rollout = TensorDict.stack(rollout, dim=0).to(self.device)
       
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
            final_obs = {k: v[-1] for k, v in obs.items()}
            with torch.no_grad():
                final_outputs = self.network(final_obs, use_sampling=self.use_bayesian, temperature=self.temperature_manager.get_all_temperatures())
                next_value = final_outputs["value"]
        
        # Handle dimensions of values and next_value
        if values.dim() == 2:
            values = values.squeeze(-1)
        if next_value is not None and next_value.dim() == 2:
            next_value = next_value.squeeze(-1)
            next_value = next_value.to(self.device)
        if rewards.dim() == 2:
            rewards = rewards.squeeze(-1)
        if dones.dim() == 2:
            dones = dones.squeeze(-1)
        
        # Calculate returns and advantages
        advantages, returns = self._compute_gae(rewards, dones, values, next_value)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # PPO update: multiple epochs, mini-batch
        batch_size = self.batch_size
        total_steps = len(advantages)
        idxs = np.arange(total_steps)
        policy_loss_epoch = 0
        value_loss_epoch = 0
        entropy_loss_epoch = 0
        total_loss_epoch = 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, total_steps, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]
                mb_obs = {k: v[mb_idx] for k, v in obs.items()}
                mb_action_choices = action_choices[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_values = values[mb_idx]

                # Get new outputs for current policy
                network_outputs = self.network(mb_obs, use_sampling=self.use_bayesian, temperature=self.temperature_manager.get_all_temperatures())
                
                # Get new logprobs and values
                new_log_probs = self.interpreter.evaluate_actions_log_probs(network_outputs, mb_action_choices)
                new_values = network_outputs["value"]
                
                # PPO clipped surrogate objective
                ratio = (new_log_probs.sum(dim=-1) - mb_old_log_probs.sum(dim=-1)).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = self.value_coef * ((mb_returns - new_values) ** 2).mean()
                
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
                
                # Accumulate losses for reporting
                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                total_loss_epoch += total_loss.item()

        n_updates = self.ppo_epochs * (total_steps // batch_size + int(total_steps % batch_size != 0))
        
        # Store loss values for visualization
        self.last_policy_loss = policy_loss_epoch / n_updates
        self.last_value_loss = value_loss_epoch / n_updates
        self.last_entropy_loss = entropy_loss_epoch / n_updates
        self.last_total_loss = total_loss_epoch / n_updates
        self.last_mean_advantage = advantages.mean().item()
        self.last_mean_return = returns.mean().item() 

        # Detach all values

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
            "total_loss": getattr(self, 'last_total_loss', 0)
        }

        if self.use_bayesian:
            info.update(self.temperature_manager.get_all_temperatures_printerfriendly())

        return info

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the agent.
        
        Returns:
            Dictionary containing agent configuration
        """
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "clip_ratio": self.clip_ratio,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "ppo_epochs": self.ppo_epochs,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "network_config": self.network_config,
            "gae_lambda": self.gae_lambda
        }

    def _get_agent_specific_state(self) -> Dict[str, Any]:
        """
        Get PPO-specific state for checkpointing.
        
        Returns:
            Dictionary containing PPO-specific state
        """
        return {
            'last_value': self.last_value,
            'last_log_probs': self.last_log_probs,
            'last_policy_loss': self.last_policy_loss,
            'last_value_loss': self.last_value_loss,
            'last_entropy_loss': self.last_entropy_loss,
            'last_total_loss': self.last_total_loss,
            'last_mean_advantage': self.last_mean_advantage,
            'last_mean_return': self.last_mean_return
        }

    def _load_agent_specific_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load PPO-specific state from checkpoint.
        
        Args:
            state_dict: Dictionary containing the agent's state
        """
        if 'last_value' in state_dict and state_dict['last_value'] is not None:
            self.last_value = state_dict['last_value'].to(self.device)
        if 'last_log_probs' in state_dict and state_dict['last_log_probs'] is not None:
            self.last_log_probs = state_dict['last_log_probs'].to(self.device)
        self.last_policy_loss = state_dict.get('last_policy_loss', 0.0)
        self.last_value_loss = state_dict.get('last_value_loss', 0.0)
        self.last_entropy_loss = state_dict.get('last_entropy_loss', 0.0)
        self.last_total_loss = state_dict.get('last_total_loss', 0.0)

    def get_model_name(self) -> str:
        """Get the name of the model."""
        return "ppo"