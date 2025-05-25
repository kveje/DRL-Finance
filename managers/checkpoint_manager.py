"""Manages model checkpoints and experiment state"""

import os
import json
import time
from typing import Any, Dict, Optional, Tuple
import torch
from models.agents.base_agent import BaseAgent

class CheckpointManager:
    """
    Manages model checkpoints, state saving/loading, and experiment resumption.
    Handles saving and loading of agent state, metrics, and experiment configuration.
    """
    
    def __init__(
        self,
        experiment_dir: str,
        agent: BaseAgent,
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        best_metric: str = "eval_reward",
        logger: Optional[Any] = None
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            experiment_dir: Directory to save checkpoints
            agent: Agent instance to manage checkpoints for
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Whether to only save the best model
            best_metric: Metric to use for determining the best model
            logger: Logger instance for logging checkpoint operations
        """
        self.experiment_dir = experiment_dir
        self.agent = agent
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.best_metric = best_metric
        self.logger = logger
        
        # Create checkpoint directory
        self.checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Track best model performance
        self.best_metric_value = float('-inf')
        self.best_checkpoint_path = None
        
        # Track checkpoint history
        self.checkpoint_history = []
    
    def save_checkpoint(
        self,
        episode: int,
        metrics: Dict[str, float],
        force_save: bool = False
    ) -> Optional[str]:
        """
        Save a model checkpoint.
        
        Args:
            episode: Current episode number
            metrics: Current metrics dictionary
            force_save: Whether to force saving regardless of best metric
            
        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        # Check if we should save based on best metric
        if self.save_best_only and not force_save:
            current_metric = metrics.get(self.best_metric)
            if current_metric is None or current_metric <= self.best_metric_value:
                return None
        
        # Create checkpoint filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_ep{episode}_{timestamp}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Get agent state
        agent_state = self.agent.get_state_dict()
        
        # Create checkpoint dictionary
        checkpoint = {
            'episode': episode,
            'timestamp': timestamp,
            'metrics': metrics,
            'agent_state': agent_state
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update best model if needed
        if not self.save_best_only or metrics.get(self.best_metric, float('-inf')) > self.best_metric_value:
            self.best_metric_value = metrics.get(self.best_metric, float('-inf'))
            self.best_checkpoint_path = checkpoint_path
        
        # Update checkpoint history
        self.checkpoint_history.append(checkpoint_path)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        if self.logger:
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.agent.device)
        
        # Load agent state
        self.agent.load_state_dict(checkpoint['agent_state'])
        
        if self.logger:
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint if exists, None otherwise
        """
        if not self.checkpoint_history:
            return None
        return self.checkpoint_history[-1]
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get the path to the best checkpoint.
        
        Returns:
            Path to best checkpoint if exists, None otherwise
        """
        return self.best_checkpoint_path
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max_checkpoints."""
        while len(self.checkpoint_history) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_history.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                if self.logger:
                    self.logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    def save_experiment_state(self, state: Dict[str, Any]) -> None:
        """
        Save the current experiment state.
        
        Args:
            state: Dictionary containing experiment state
        """
        state_path = os.path.join(self.experiment_dir, "experiment_state.json")
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4)
        
        if self.logger:
            self.logger.info(f"Saved experiment state to {state_path}")
    
    def load_experiment_state(self) -> Optional[Dict[str, Any]]:
        """
        Load the experiment state.
        
        Returns:
            Dictionary containing experiment state if exists, None otherwise
        """
        state_path = os.path.join(self.experiment_dir, "experiment_state.json")
        if not os.path.exists(state_path):
            return None
        
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        if self.logger:
            self.logger.info(f"Loaded experiment state from {state_path}")
        
        return state 