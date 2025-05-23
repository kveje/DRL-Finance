"""Default configurations for action interpreters."""

# Default configuration for discrete action interpreter
DISCRETE_INTERPRETER_CONFIG = {
    "n_assets": 3,  # Will be updated based on environment
    "max_position_size": 1.0,
    "temperature": 1.0,
    "temperature_decay": 0.995,
    "min_temperature": 0.1,
    "action_mapping": "standard"  # or "directional" for directional actions
}

# Default configuration for confidence scaled interpreter
CONFIDENCE_SCALED_INTERPRETER_CONFIG = {
    "n_assets": 3,  # Will be updated based on environment
    "max_position_size": 1.0,
    "temperature": 1.0,
    "temperature_decay": 0.995,
    "min_temperature": 0.1,
    "confidence_threshold": 0.7,
    "scaling_factor": 0.5
}
