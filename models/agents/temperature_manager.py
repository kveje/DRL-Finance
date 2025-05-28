import numpy as np
from typing import Dict, Union
from config.temperature import BASE_TEMPERATURE_HEAD_CONFIG
class TemperatureManager:
    """
    Multi-head temperature manager.
    
    Manages temperature schedules for:
    - Discrete action head (Beta distribution)
    - Confidence head (Beta distribution)  
    - Value head (Normal distribution)
    """
    def __init__(
        self,
        update_frequency: int,
        head_configs: Dict[str, Dict[str, float]], # {"discrete": {...}, "confidence": {...}, "value": {...}}
        total_env_steps: int = 340000,
        warmup_steps: int = 10000,
        training_step: int = 0,
        global_step: int = 0,
        ):
        """
        Initialize temperature manager.
        
        Args:
            active_heads: List of heads to use ["discrete", "confidence", "value"]
            update_frequency: How often to the temperature manager is updated
            total_env_steps: Total environment steps for training
            warmup_steps: Steps before temperature decay starts
            training_step: Current training step
            global_step: Current global step
        """
        self.total_env_steps = total_env_steps
        self.warmup_steps = warmup_steps
        self.training_step = training_step
        self.global_step = global_step
        
        # Auto-detect update frequency if not provided
        self.update_frequency = update_frequency
        self.total_updates = (self.total_env_steps - self.warmup_steps) // self.update_frequency
        
        # Initialize head configurations
        self.head_configs: Dict[str, Dict[str, float]] = head_configs

        # Track which heads are active
        self.active_heads = {head_type: True if head_type in self.head_configs.keys() else False for head_type in ["discrete", "confidence", "value"]}
        active_head_list = list(head_type for head_type, active in self.active_heads.items() if active)

        # Calculate decay steps for each head
        self.decay_steps = {
            head_type: int(self.head_configs[head_type].get("decay_fraction", 0.8) * self.total_updates)
            for head_type in active_head_list
        }
    
    def _calculate_temperature(self, head_type: str) -> float:
        """Calculate current temperature for a specific head."""
        if not self.active_heads.get(head_type, False):
            return 1.0  # Default temperature for unused heads
            
        config = self.head_configs[head_type]
        
        # During warmup
        if self.global_step < self.warmup_steps:
            return config.get("initial_temp", 1.0) * config.get("warmup_multiplier", 1.0)
        
        # After decay period
        decay_steps = self.decay_steps[head_type]
        
        if self.training_step >= decay_steps:
            return config.get("final_temp", 1.0)
        
        # During decay
        progress = self.training_step / decay_steps
        temp_range = config.get("initial_temp", 1.0) - config.get("final_temp", 1.0)
        
        return config.get("final_temp", 1.0) + temp_range * np.exp(-config.get("decay_rate", 1.0) * progress)
    
    def get_discrete_temperature(self) -> float:
        """Get temperature for discrete action head (Beta distribution)."""
        return self._calculate_temperature("discrete")
    
    def get_confidence_temperature(self) -> float:
        """Get temperature for confidence head (Beta distribution)."""
        return self._calculate_temperature("confidence")
    
    def get_value_temperature(self) -> float:
        """Get temperature for value head (Normal distribution)."""
        return self._calculate_temperature("value")
    
    def get_all_temperatures(self) -> Dict[str, float]:
        """Get all temperatures in a single call."""
        return {
            "discrete": self.get_discrete_temperature(),
            "confidence": self.get_confidence_temperature(), 
            "value": self.get_value_temperature()
        }
    
    def get_all_temperatures_printerfriendly(self) -> Dict[str, float]:
        """Get all temperatures in a single call with printerfriendly format."""
        return {
            "discrete_temperature": self.get_discrete_temperature(),
            "confidence_temperature": self.get_confidence_temperature(), 
            "value_temperature": self.get_value_temperature()
        }
    
    def _should_update_temperature(self, global_step: int) -> bool:
        """Check if temperature should be updated this step."""
        return (global_step % self.update_frequency == 0 and 
                global_step >= self.warmup_steps)
    
    def step(self):
        """Call after each environment step to advance temperature schedule."""
        self.global_step += 1
        if self._should_update_temperature(self.global_step):
            self.training_step += 1
    
    def get_progress_info(self) -> Dict:
        """Get detailed information about current training progress."""
        warmup_updates = self.warmup_steps // self.update_frequency
        progress = self.training_step / self.total_updates if self.total_updates > 0 else 0
        
        info = {
            "training_step": self.training_step,
            "total_updates": self.total_updates,
            "progress": progress,
            "warmup_complete": self.training_step >= warmup_updates,
            "temperatures": self.get_all_temperatures(),
            "active_heads": [key for key, value in self.active_heads.items() if value]
        }
        return info
    
    def reset(self):
        """Reset temperature manager to initial state."""
        self.training_step = 0
        self.global_step = 0

    def get_config(self) -> Dict[str, float]:
        return {
            "update_frequency": self.update_frequency,
            "head_configs": self.head_configs,
            "total_env_steps": self.total_env_steps,
            "warmup_steps": self.warmup_steps,
            "training_step": self.training_step,
            "global_step": self.global_step
        }

    def get_update_frequency(self) -> int:
        return self.update_frequency
    
    def get_step_info(self) -> Dict:
        return {
            "training_step": self.training_step,
            "global_step": self.global_step
        }

    def set_step_info(self, training_step: int, global_step: int):
        self.training_step = training_step
        self.global_step = global_step
