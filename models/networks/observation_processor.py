from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

class ObservationProcessor:
    """Processes raw observations into a standardized format."""
    
    def __init__(
        self,
        observation_config: Dict[str, Any],
        device: str = "cuda"
    ):
        """
        Initialize the observation processor.
        
        Args:
            observation_config: Configuration for processing different observation components
            device: Device to process tensors on
        """
        self.config = observation_config
        self.device = device
        self.processors = {}
        self._setup_processors()
        
    def _setup_processors(self):
        """Set up processors for each observation component."""
        for obs_type, config in self.config.items():
            if obs_type == "price":
                self.processors[obs_type] = self._setup_price_processor(config)
            elif obs_type == "cash":
                self.processors[obs_type] = self._setup_cash_processor(config)
            elif obs_type == "position":
                self.processors[obs_type] = self._setup_position_processor(config)
            elif obs_type == "tech":
                self.processors[obs_type] = self._setup_tech_processor(config)
            elif obs_type == "ohlcv":
                self.processors[obs_type] = self._setup_ohlcv_processor(config)
            else:
                raise ValueError(f"Unsupported observation type: {obs_type}")
    
    def _setup_price_processor(self, config: Dict[str, Any]) -> nn.Module:
        """Set up processor for price data."""
        return nn.Sequential(
            nn.Unflatten(1, (1, config["window_size"])),
            nn.Conv1d(1, config["hidden_dim"], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(config["hidden_dim"], config["hidden_dim"], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        ).to(self.device)
    
    def _setup_cash_processor(self, config: Dict[str, Any]) -> nn.Module:
        """Set up processor for cash data."""
        return nn.Sequential(
            nn.Linear(config["input_dim"], config["hidden_dim"]),
            nn.ReLU()
        ).to(self.device)
    
    def _setup_position_processor(self, config: Dict[str, Any]) -> nn.Module:
        """Set up processor for position data."""
        return nn.Sequential(
            nn.Linear(config["input_dim"], config["hidden_dim"]),
            nn.ReLU()
        ).to(self.device)
    
    def _setup_tech_processor(self, config: Dict[str, Any]) -> nn.Module:
        """Set up processor for technical indicators."""
        return nn.Sequential(
            nn.Linear(config["input_dim"], config["hidden_dim"]),
            nn.ReLU()
        ).to(self.device)

    def _setup_ohlcv_processor(self, config: Dict[str, Any]) -> nn.Module:
        """Set up processor for OHLCV data."""
        return nn.Sequential(
            nn.Linear(config["input_dim"], config["hidden_dim"]),
            nn.ReLU()
        ).to(self.device)
    
    def process(self, observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Process the observation dictionary into a standardized format.
        
        Args:
            observation: Dictionary of observation components
            
        Returns:
            Dictionary of processed tensors
        """
        processed = {}
        
        for obs_type, obs_data in observation.items():
            if obs_type not in self.processors:
                continue
                
            # Convert numpy array to tensor
            obs_tensor = torch.FloatTensor(obs_data).to(self.device)
            
            # Add batch dimension if needed
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            elif len(obs_tensor.shape) == 2:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # Process the observation component
            processed[obs_type] = self.processors[obs_type](obs_tensor)
        
        return processed 