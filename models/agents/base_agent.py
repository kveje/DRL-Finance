"""Base class for all agents"""

from abc import ABC, abstractmethod
from environments.base_env import BaseTradingEnv

class BaseAgent(ABC):
    """Base class for all agents"""
    def __init__(self, env: BaseTradingEnv):
        self.env = env

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_model_name(self):
        pass
