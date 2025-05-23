from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from gym import spaces
from .base_processor import BaseProcessor
from .processor_factory import ProcessorFactory

class CompositeProcessor(BaseProcessor):
    """
    A processor that combines multiple processors into a single observation.
    Each processor's output is stored in a dictionary with its name as the key.
    """
    
    def __init__(
        self,
        processor_configs: List[Dict[str, Any]]
    ):
        """
        Initialize the composite processor.
        
        Args:
            processor_configs: List of dictionaries containing processor configurations.
                Each dict should have:
                - 'type': str - The type of processor ('price', 'ohlcv', etc.)
                - 'data_name': str - The name of the data to use for this processor's input
                - 'kwargs': dict - Additional arguments for the processor
        """
        self.processors: Dict[str, BaseProcessor] = {}
        self.data_names: Dict[str, str] = {}

        # Initialize each processor
        for config in processor_configs:
            processor_type = config['type']
            data_name = config['data_name']
            kwargs = config.get('kwargs', {})
            
            # Create processor using factory
            processor = ProcessorFactory.create_processor(processor_type, **kwargs)
            
            # Store processor and its output name
            self.processors[processor_type] = processor
            self.data_names[processor_type] = data_name
    
    def process(self, data: Dict[str, Any], current_step: int) -> Dict[str, np.ndarray]:
        """
        Process the data using all registered processors.
        
        Args:
            data: Dictionary containing the data for each processor.
                Keys should match the processor names.
            current_step: Current step in the environment
                
        Returns:
            Dict[str, np.ndarray]: Dictionary containing processed data from each processor
        """
        processed_data = {}
        
        for name, processor in self.processors.items():
            data_name = self.data_names[name]
            # Process the data using the appropriate processor
            # Some processors need current_step, others don't
            if name in ['price', 'ohlcv', 'tech', 'vix']:
                processed_data[name] = processor.process(data[data_name], current_step)
            else:
                processed_data[name] = processor.process(data[data_name])
            
        return processed_data
    
    def get_observation_space(self) -> Dict[str, spaces.Space]:
        """
        Get the observation space for all processors.
        
        Returns:
            Dict containing the observation space for each processor
        """
        observation_space = {}
        
        for name, processor in self.processors.items():
            # Get the space specification from the processor
            space_spec = processor.get_observation_space()
            
            # Convert the space specification to a gym Space object
            for key, spec in space_spec.items():
                if isinstance(spec, dict):
                    # Create a Box space from the specification
                    observation_space[key] = spaces.Box(
                        low=spec['low'],
                        high=spec['high'],
                        shape=spec['shape'],
                        dtype=spec['dtype']
                    )
                else:
                    # If it's already a Space object, use it directly
                    observation_space[key] = spec
            
        return observation_space
    
    def get_input_dim(self) -> Dict[str, Tuple[int, ...]]:
        """
        Get the input dim for each processor.
        
        Returns:
            Dict[str, Tuple[int, ...]]: Input dimension for each processor
        """
        input_dim = {}
        for name, processor in self.processors.items():
            input_dim[name] = processor.get_input_dim()
            
        return input_dim