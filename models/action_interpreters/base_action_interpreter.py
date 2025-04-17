"""Base class for all action heads. Action heads are used to convert the output of the model into an action."""

from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np

from environments.base_env import BaseTradingEnv

class BaseActionInterpreter(ABC):
    """Base class for all action interpreters"""
    def __init__(self, env: BaseTradingEnv):
        self.action_space = env.action_space

    @abstractmethod
    def interpret(self, model_output: np.ndarray) -> np.ndarray:
        """Interpret the model output into an action
        
        Args:
            model_output (np.ndarray): The output of the model

        Returns:
            np.ndarray: The action
        """
        pass

