from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import gym
from gym import spaces


class BaseTradingEnv(gym.Env, ABC):
    """
    Base class for all trading environments.
    Implements the core Gym interface and defines the abstract methods that must be implemented
    by any concrete trading environment.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Define action and observation spaces
        self.action_space = None  # To be defined by concrete implementations
        self.observation_space = None  # To be defined by concrete implementations

        # Environment state
        self.current_step = 0
        self.done = False
        self.info = {}

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
            np.ndarray: Initial observation
        """
        pass

    @abstractmethod
    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: The action to take

        Returns:
            Tuple containing:
                - observation (np.ndarray): The observation after taking the action
                - reward (float): The reward obtained after taking the action
                - done (bool): Whether the episode has ended
                - info (dict): Additional information about the step
        """
        pass

    @abstractmethod
    def render(self, mode: str = "human") -> None:
        """
        Render the environment to the screen.

        Args:
            mode: The mode to render in ('human' or 'rgb_array')
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Clean up resources.
        """
        pass

    @abstractmethod
    def _set_seed(self, seed: Optional[int] = None) -> list:
        """
        Set the seed for this environment's random number generator(s).

        Args:
            seed: The seed value

        Returns:
            list: The list of seeds used in this environment's RNG
        """
        pass

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation of the environment.
        To be implemented by concrete environments.

        Returns:
            np.ndarray: The current observation
        """
        raise NotImplementedError

    def _calculate_reward(self, action: Union[int, np.ndarray]) -> float:
        """
        Calculate the reward for the given action.
        To be implemented by concrete environments.

        Args:
            action: The action taken

        Returns:
            float: The calculated reward
        """
        raise NotImplementedError

    def _is_done(self) -> bool:
        """
        Check if the episode is done.
        To be implemented by concrete environments.

        Returns:
            bool: Whether the episode has ended
        """
        raise NotImplementedError

    def _update_info(self) -> Dict[str, Any]:
        """
        Update the info dictionary with current environment state.
        To be implemented by concrete environments.

        Returns:
            Dict[str, Any]: Updated info dictionary
        """
        raise NotImplementedError
