import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agents.temperature_manager import TemperatureManager
from config.temperature import get_default_temperature_config

def collect_temperature_curves(total_steps: int = 340000, warmup_steps: int = 10000):
    """Collect temperature curves for a specific configuration from step 0 to total_steps, including before warmup ends."""
    config = get_default_temperature_config()
    manager = TemperatureManager(
        head_config=config,
        total_env_steps=total_steps,
        warmup_steps=warmup_steps,
        update_frequency=1000
    )
    steps = []
    discrete_temps = []
    confidence_temps = []
    value_temps = []
    for step in range(total_steps):
        temps = manager.get_all_temperatures()
        discrete_temps.append(temps["discrete"])
        confidence_temps.append(temps["confidence"])
        value_temps.append(temps["value"])
        steps.append(step)
        manager.step()
    return np.array(steps), np.array(discrete_temps), np.array(confidence_temps), np.array(value_temps)

def main():
    total_steps = 340000
    warmup_steps = 10000
    steps, discrete_temps, confidence_temps, value_temps = collect_temperature_curves(
        total_steps, warmup_steps)
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, discrete_temps, label="Discrete", linewidth=2)
    plt.plot(steps, confidence_temps, label="Confidence", linewidth=2)
    plt.plot(steps, value_temps, label="Value", linewidth=2)
    plt.axvline(x=warmup_steps, color='r', linestyle='--', label='Warmup End')
    plt.xlabel("Environment Steps")
    plt.ylabel("Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig("temperature_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main() 