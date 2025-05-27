"""Default configurations for action interpreters."""

# Default configuration for discrete action interpreter
DISCRETE_INTERPRETER_CONFIG = {
    "n_assets": 0,  # Will be updated based on environment
    "max_trade_size": 10,
}

# Default configuration for confidence scaled interpreter
CONFIDENCE_SCALED_INTERPRETER_CONFIG = {
    "n_assets": 0,  # Will be updated based on environment
    "max_trade_size": 10,
}
 
def get_interpreter_config(
    type: str,
    n_assets: int,
) -> dict:
    """
    Get the interpreter config based on the type and number of assets.
    
    Args:
        type: The type of interpreter to use.
        n_assets: The number of assets in the environment.

    Returns:
        The interpreter config.
    """
    if type == "discrete":
        config = DISCRETE_INTERPRETER_CONFIG.copy()
    elif type == "confidence_scaled":
        config = CONFIDENCE_SCALED_INTERPRETER_CONFIG.copy()
    else:
        raise ValueError(f"Invalid interpreter type: {type}")
    
    config["n_assets"] = n_assets

    return config

