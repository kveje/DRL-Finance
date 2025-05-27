import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import deque
import random
import os

from utils.logger import Logger
logger = Logger.get_logger()

from models.agents.base_agent import BaseAgent
from models.action_interpreters.base_action_interpreter import BaseActionInterpreter
from models.networks.unified_network import UnifiedNetwork
from models.agents.temperature_manager import TemperatureManager


class SACAgent(BaseAgent):
    def __init__(self, network_config: Dict[str, Any], temperature_manager: TemperatureManager, interpreter: BaseActionInterpreter, update_frequency: int, device: str = "cuda" if torch.cuda.is_available() else "cpu", **kwargs):
        super().__init__(network_config, interpreter, temperature_manager, update_frequency, device, **kwargs)
