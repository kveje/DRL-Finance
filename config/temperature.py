from typing import Dict, Any, Optional
from copy import deepcopy

BASE_TEMPERATURE_HEAD_CONFIG = {
    "discrete": {
        "initial_temp": 2.0,
        "final_temp": 1.2,
        "decay_rate": 2.5,
        "decay_fraction": 0.8
    },
    "confidence": {
        "initial_temp": 1.8,
        "final_temp": 1.1,
        "decay_rate": 2.5,
        "decay_fraction": 0.75
    },
    "value": {
        "initial_temp": 1.5,
        "final_temp": 0.9,
        "decay_rate": 2.0,
        "decay_fraction": 0.7
    }
}

BASE_TEMPERATURE_CONFIG = {
    "update_frequency": 1,
    "head_configs": {},
    "total_env_steps": 340000, # Roughly 200 episodes
    "warmup_steps": 10000,
    "training_step": 0,
    "global_step": 0,
}

def get_temperature_config(network_config: Dict[str, Dict[str, Any]], update_frequency: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    config = deepcopy(BASE_TEMPERATURE_CONFIG)

    for head_type, head_config in network_config["heads"].items():
        if head_config["enabled"]:
            config["head_configs"][head_type] = BASE_TEMPERATURE_HEAD_CONFIG[head_type]
    
    if update_frequency is not None:
        config["update_frequency"] = update_frequency

    return config

def get_default_temperature_config() -> Dict[str, Dict[str, float]]:
    config = deepcopy(BASE_TEMPERATURE_CONFIG)
    config["head_configs"] = deepcopy(BASE_TEMPERATURE_HEAD_CONFIG)
    return config
