"""Manages experiment configuration and state"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from utils.logger import Logger

class ConfigManager:
    """
    Manages experiment configuration, including saving and loading of:
    - Experiment configuration
    - Environment configuration
    - Agent configuration
    - Interpreter configuration
    - Data processing configuration
    """
    
    def __init__(
        self,
        experiment_dir: Path,
        logger: Optional[Logger] = None
    ):
        """
        Initialize the configuration manager.
        
        Args:
            experiment_dir: Directory to save configurations
            logger: Logger instance for logging configuration operations
        """
        self.experiment_dir = experiment_dir
        self.logger = logger
        
        # Create config directory
        self.config_dir = experiment_dir / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize configuration storage
        self.configs = {
            "experiment": {},
            "environment": {},
            "agent": {},
            "interpreter": {},
            "data": {}
        }
    
    def save_experiment_config(
        self,
        experiment_name: str,
        max_train_time: Optional[int],
        eval_interval: int,
        save_interval: int,
        save_metric_interval: int,
        early_stopping_patience: int,
        early_stopping_threshold: float,
        early_stopping_metric: str,
        render_train: bool,
        render_eval: bool,
        agent_type: str,
        train_env_params: Dict[str, Any],
        gcs_bucket: str
    ) -> None:
        """Save experiment configuration."""
        config = {
            "experiment_name": experiment_name,
            "max_train_time": max_train_time,
            "eval_interval": eval_interval,
            "save_interval": save_interval,
            "save_metric_interval": save_metric_interval,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_threshold": early_stopping_threshold,
            "early_stopping_metric": early_stopping_metric,
            "render_train": render_train,
            "render_eval": render_eval,
            "agent_type": agent_type,
            "train_env_params": train_env_params,
            "gcs_bucket": gcs_bucket
        }
        
        self.configs["experiment"] = config
        self._save_config("experiment_config.json", config)
    
    def save_environment_config(
        self,
        env_params: Dict[str, Any],
        friction_params: Dict[str, Any],
        constraint_params: Dict[str, Any],
        reward_params: Dict[str, Any],
        processor_configs: Dict[str, Any]
    ) -> None:
        """Save environment configuration."""
        config = {
            "env_params": env_params,
            "friction_params": friction_params,
            "constraint_params": constraint_params,
            "reward_params": reward_params,
            "processor_configs": processor_configs
        }
        
        self.configs["environment"] = config
        self._save_config("environment_config.json", config)
    
    def save_agent_config(self, agent_config: Dict[str, Any]) -> None:
        """Save agent configuration."""
        self.configs["agent"] = agent_config
        self._save_config("agent_config.json", agent_config)
    
    def save_interpreter_config(self, interpreter_config: Dict[str, Any]) -> None:
        """Save interpreter configuration."""
        self.configs["interpreter"] = interpreter_config
        self._save_config("interpreter_config.json", interpreter_config)
    
    def save_data_config(
        self,
        normalization_params: Dict[str, Any],
        processor_params: Dict[str, Any]
    ) -> None:
        """Save data processing configuration."""
        config = {
            "normalization_params": normalization_params,
            "processor_params": processor_params
        }
        
        self.configs["data"] = config
        self._save_config("data_config.json", config)
    
    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific configuration.
        
        Args:
            config_name: Name of the configuration file (without .json extension)
            
        Returns:
            Dictionary containing the configuration if exists, None otherwise
        """
        config_path = self.config_dir / f"{config_name}.json"
        
        # Check for older config path for backward compatibility
        if not config_path.exists():
            old_config_path = self.experiment_dir / f"{config_name}.json"
            if old_config_path.exists():
                config_path = old_config_path
        
        if not config_path.exists():
            return None
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update internal storage
        config_type = config_name.replace("_config", "")
        if config_type in self.configs:
            self.configs[config_type] = config
        
        if self.logger:
            self.logger.info(f"Loaded configuration from {config_path}")
        
        return config
    
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configurations.
        
        Returns:
            Dictionary containing all configurations
        """
        config_files = [
            "experiment_config.json",
            "environment_config.json",
            "agent_config.json",
            "interpreter_config.json",
            "data_config.json"
        ]
        
        for config_file in config_files:
            config_name = config_file.replace(".json", "")
            self.load_config(config_name)
        
        return self.configs
    
    def get_config(self, config_type: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific configuration from memory.
        
        Args:
            config_type: Type of configuration to get
            
        Returns:
            Dictionary containing the configuration if exists, None otherwise
        """
        return self.configs.get(config_type)
    
    def _save_config(self, filename: str, config: Dict[str, Any]) -> None:
        """Save a configuration to disk."""
        config_path = self.config_dir / filename
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        if self.logger:
            self.logger.info(f"Saved configuration to {config_path}")
    
    def save_experiment_state(self, state: Dict[str, Any]) -> None:
        """
        Save the current experiment state.
        
        Args:
            state: Dictionary containing experiment state
        """
        state_path = self.experiment_dir / "experiment_state.json"
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
        state_path = self.experiment_dir / "experiment_state.json"
        if not state_path.exists():
            return None
        
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        if self.logger:
            self.logger.info(f"Loaded experiment state from {state_path}")
        
        return state 