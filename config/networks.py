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
            "hidden_dim": 128,
            "asset_embedding_dim": 32
        },
        "price": {
            "enabled": True,
            "hidden_dim": 64,
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
            "hidden_dim": 64
        },
        "current_price": {
            "enabled": True,
            "hidden_dim": 64
        }
    },
    "backbone": {
        "type": "mlp",
        "hidden_dims": [256, 128],
        "dropout": 0.0,
        "use_layer_norm": True
    },
    "heads": {
        "discrete": {
            "enabled": False,
            "type": "parametric",
            "hidden_dim": 128,
            "num_actions": 3
        },
        "value": {
            "enabled": False,
            "type": "parametric",
            "hidden_dim": 64
        },
        "confidence": {
            "enabled": False,
            "type": "parametric",
            "hidden_dim": 128
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

    # Configure hidden dims for backbone
    embedded_dim = sum(processor["hidden_dim"] for processor in config["processors"].values() if processor["enabled"])

    if embedded_dim > 512:
        config["backbone"]["hidden_dims"] = [256, 128]
    elif embedded_dim > 256:
        config["backbone"]["hidden_dims"] = [128, 64]
    else:
        config["backbone"]["hidden_dims"] = [64, 32]

    # Configure heads
    if include_discrete:
        config["heads"]["discrete"]["enabled"] = True
        if head_type == "bayesian":
            config["heads"]["discrete"]["type"] = "bayesian"
    
    # Add confidence head if requested
    if include_confidence:
        config["heads"]["confidence"]["enabled"] = True
        if head_type == "bayesian":
            config["heads"]["confidence"]["type"] = "bayesian"
    
    # Add value head if requested
    if include_value:
        config["heads"]["value"]["enabled"] = True
        if head_type == "bayesian":
            config["heads"]["value"]["type"] = "bayesian"

    # Configure head dims
    if embedded_dim > 512:
        config["heads"]["discrete"]["hidden_dim"] = 128
        config["heads"]["confidence"]["hidden_dim"] = 64
        config["heads"]["value"]["hidden_dim"] = 32
    elif embedded_dim > 256:
        config["heads"]["discrete"]["hidden_dim"] = 64
        config["heads"]["confidence"]["hidden_dim"] = 32
        config["heads"]["value"]["hidden_dim"] = 16
    
    return config
