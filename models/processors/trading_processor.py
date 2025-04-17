from typing import Dict, Any
import numpy as np
import torch

from logging import getLogger
logger = getLogger()

from .base_processor import BaseObservationProcessor, ObservationFormat
from environments.trading_env import TradingEnv

class TradingObservationProcessor(BaseObservationProcessor):
    """
    Processor for trading environment observations.
    Handles the conversion from the trading environment's dictionary observation to various tensor formats.
    """
    
    def __init__(
        self,
        env: TradingEnv,  # TradingEnv instance
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the trading observation processor.
        
        Args:
            env: TradingEnv instance
            device: Device to place tensors on
        """
        super().__init__(device)
        
        # Get dimensions from environment
        self.n_assets = env.n_assets
        self.ohlcv_dim = env.ohlcv_dim
        self.tech_dim = env.tech_dim
        self.window_size = env.window_size
        
        # Calculate dimensions for different formats
        self.market_data_dim = self.n_assets * (self.ohlcv_dim + self.tech_dim) * self.window_size
        self.positions_dim = self.n_assets
        self.portfolio_info_dim = env.portfolio_info_dim
        self.total_flat_dim = self.market_data_dim + self.positions_dim + self.portfolio_info_dim
    
    def process(
        self,
        observation: Dict[str, np.ndarray],
        format: ObservationFormat = ObservationFormat.FLAT
    ) -> torch.Tensor:
        """
        Process the observation into the specified tensor format.
        
        Args:
            observation: Dictionary observation from the trading environment
            format: Desired output format
            
        Returns:
            Processed tensor in the specified format
        """
        if format == ObservationFormat.FLAT:
            return self._process_flat(observation)
        elif format == ObservationFormat.STACKED:
            return self._process_stacked(observation)
        else:
            raise ValueError(f"Unsupported observation format: {format}")
    
    def _process_flat(self, observation: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Process observation into a flat tensor.
        Shape: (market_data_dim + positions_dim + portfolio_info_dim,)
        """
        # Flatten market data
        market_data = observation['data'].flatten()
        
        # Combine all components
        flat_obs = np.concatenate([
            market_data,
            observation['positions'],
            observation['portfolio_info']
        ])
        
        return torch.FloatTensor(flat_obs).to(self.device)
    
    def _process_stacked(self, observation: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Process observation into a stacked tensor format.
        Shape: (n_assets, ohlcv_dim + tech_dim, window_size + 1)
        The +1 in window_size is for the current positions and portfolio info.
        """
        # Get market data
        market_data = observation['data']
        
        # Create a new tensor with an extra time step
        stacked_data = torch.zeros(
            (self.n_assets, self.ohlcv_dim + self.tech_dim, self.window_size + 1),
            device=self.device
        )
        
        # Copy market data
        stacked_data[:, :, :self.window_size] = torch.FloatTensor(market_data)
        
        # Add positions and portfolio info as the last time step
        stacked_data[:, 0, -1] = torch.FloatTensor(observation['positions'])
        for i, info in enumerate(observation['portfolio_info']):
            stacked_data[:, i + 1, -1] = torch.FloatTensor(info)
            if i == self.ohlcv_dim + self.tech_dim:
                logger.info(f"Could not add all portfolio info to the stacked data. Only added {i} out of {self.portfolio_info_dim}.")
                break
        
        return stacked_data
    
    def get_observation_dim(self, format: ObservationFormat) -> int:
        """
        Get the dimension of the processed observation for the specified format.
        
        Args:
            format: Observation format
            
        Returns:
            Dimension of the processed observation
        """
        if format == ObservationFormat.FLAT:
            return self.total_flat_dim
        elif format == ObservationFormat.STACKED:
            return (self.n_assets, self.ohlcv_dim + self.tech_dim, self.window_size + 1)
        else:
            raise ValueError(f"Unsupported observation format: {format}") 