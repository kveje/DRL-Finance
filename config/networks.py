"""Network configurations for different architectures."""
from typing import Optional
from copy import deepcopy

# Base configuration with default values
BASE_CONFIG = {
    "n_assets": 3,
    "window_size": 20,
    "processors": {
        "ohlcv": {
            "enabled": True,
            "hidden_dim": 256,
            "asset_embedding_dim": 32
        },
        "price": {
            "enabled": True,
            "hidden_dim": 128,
            "asset_embedding_dim": 16
        },
        "technical": {
            "enabled": True,
            "tech_dim": 20,  # SHOULD BE MODIFIED BY THE ACTUAL NUMBER OF TECHNICAL INDICATORS!!!
            "hidden_dim": 64
        },
        "position": {
            "enabled": True,
            "hidden_dim": 32
        },
        "cash": {
            "enabled": True,
            "input_dim": 2,  # [cash_balance, portfolio_value]
            "hidden_dim": 16
        },
        "affordability": {
            "enabled": True,
            "hidden_dim": 32
        },
        "current_price": {
            "enabled": True,
            "hidden_dim": 32
        }
    },
    "backbone": {
        "type": "mlp",
        "hidden_dims": [128, 128],
        "dropout": 0.0,
        "use_layer_norm": True
    },
    "heads": {
        "discrete": {
            "enabled": False,
            "type": "parametric",
            "hidden_dim": 128,
            "num_actions": 3,
            "sampling_strategy": "thompson" # ["thompson", "entropy"]
        },
        "value": {
            "enabled": False,
            "type": "parametric",
            "hidden_dim": 32,
            "sampling_strategy": "thompson" # ["thompson", "ucb", "optimistic"]
        },
        "confidence": {
            "enabled": False,
            "type": "parametric",
            "hidden_dim": 64,
            "sampling_strategy": "thompson" # ["thompson", "entropy"]
        }
    }
}

def get_network_config(
    n_assets: int,
    window_size: int,
    price_type: str = "price",
    head_type: str = "parametric",
    include_discrete: bool = True,
    include_confidence: bool = False,
    include_value: bool = False,
    technical_dim: Optional[int] = None,
    discrete_sampling_strategy: Optional[str] = None, # ["thompson", "entropy"]
    confidence_sampling_strategy: Optional[str] = None, # ["thompson", "entropy"]
    value_sampling_strategy: Optional[str] = None, # ["thompson", "ucb", "optimistic"]
) -> dict:
    """
    Generate a network configuration based on specified parameters.
    
    Args:
        n_assets: Number of assets
        window_size: Window size
        price_type: Type of price ('price', 'ohlcv', 'both')
        head_type: Type of head ('parametric' or 'bayesian')
        include_discrete: Whether to include discrete head
        include_confidence: Whether to include confidence head
        include_value: Whether to include value head
        technical_dim: Number of technical indicators (if not provided, not used)
    Returns:
        dict: Network configuration
    """
    config = deepcopy(BASE_CONFIG)

    # Update network config with environment-specific parameters
    config["n_assets"] = n_assets
    config["window_size"] = window_size

    # Adjust based on price type
    if price_type == "price":
        config["processors"]["ohlcv"]["enabled"] = False
    elif price_type == "ohlcv":
        config["processors"]["price"]["enabled"] = False
    elif price_type == "both":
        pass

    # Adjust based on technical dimension
    if technical_dim is not None:
        config["processors"]["technical"]["tech_dim"] = technical_dim
    else:
        config["processors"]["technical"]["enabled"] = False

    # Configure heads
    if include_discrete:
        config["heads"]["discrete"]["enabled"] = True
        if head_type == "bayesian":
            config["heads"]["discrete"]["type"] = "bayesian"
        if discrete_sampling_strategy is not None:
            config["heads"]["discrete"]["sampling_strategy"] = discrete_sampling_strategy
    
    # Add confidence head if requested
    if include_confidence:
        config["heads"]["confidence"]["enabled"] = True
        if head_type == "bayesian":
            config["heads"]["confidence"]["type"] = "bayesian"
        if confidence_sampling_strategy is not None:
            config["heads"]["confidence"]["sampling_strategy"] = confidence_sampling_strategy
    
    # Add value head if requested
    if include_value:
        config["heads"]["value"]["enabled"] = True
        if head_type == "bayesian":
            config["heads"]["value"]["type"] = "bayesian"
        if value_sampling_strategy is not None:
            config["heads"]["value"]["sampling_strategy"] = value_sampling_strategy

    return config
