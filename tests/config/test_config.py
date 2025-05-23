from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class NetworkConfig:
    """Configuration for neural network."""
    type: str  # 'mlp', 'lstm', 'transformer'
    input_dim: int
    output_dim: int
    hidden_dims: List[int]
    activation: str = "relu"
    dropout: float = 0.0
    device: str = "cuda"
    additional_params: Dict[str, Any] = None

@dataclass
class ProcessorConfig:
    """Configuration for observation processor."""
    type: str  # 'trading', 'price', etc.
    additional_params: Dict[str, Any] = None

@dataclass
class InterpreterConfig:
    """Configuration for action interpreter."""
    type: str  # 'allocation', 'directional', 'discrete'
    additional_params: Dict[str, Any] = None

@dataclass
class TestConfig:
    """Complete test configuration."""
    network: NetworkConfig
    processor: ProcessorConfig
    interpreter: InterpreterConfig
    test_name: str
    description: Optional[str] = None

def create_test_configs() -> List[TestConfig]:
    """
    Create a list of test configurations to run.
    Add your test configurations here.
    """
    configs = []
    
    # Example: MLP with price processor and allocation interpreter
    configs.append(TestConfig(
        network=NetworkConfig(
            type='mlp',
            input_dim=50,  # 5 assets * 10 window_size
            output_dim=5,
            hidden_dims=[64, 64]
        ),
        processor=ProcessorConfig(
            type='price',
            additional_params={
                'price_col': 'close',
                'window_size': 10
            }
        ),
        interpreter=InterpreterConfig(
            type='allocation'
        ),
        test_name='mlp_price_allocation',
        description='MLP network with price processor and allocation interpreter'
    ))
    
    # Example: LSTM with trading processor and directional interpreter
    configs.append(TestConfig(
        network=NetworkConfig(
            type='lstm',
            input_dim=10,
            output_dim=3,
            hidden_dims=[64, 64],
            additional_params={'sequence_length': 10}
        ),
        processor=ProcessorConfig(
            type='trading'
        ),
        interpreter=InterpreterConfig(
            type='directional'
        ),
        test_name='lstm_trading_directional',
        description='LSTM network with trading processor and directional interpreter'
    ))
    
    # Add more test configurations as needed
    
    return configs 