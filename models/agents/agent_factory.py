"""Factory for creating different types of agents."""

from typing import Dict, Any, Type
from .base_agent import BaseAgent
from .dqn_agent import DQNAgent
from .ppo_agent import PPOAgent
from .a2c_agent import A2CAgent
from .sac_agent import SACAgent
from ..action_interpreters.interpreter_factory import InterpreterFactory
from .temperature_manager import TemperatureManager

class AgentFactory:
    """Factory class for creating different types of agents."""
    
    _agents: Dict[str, Type[BaseAgent]] = {
        'dqn': DQNAgent,
        'ppo': PPOAgent,
        'a2c': A2CAgent,
        'sac': SACAgent
    }
    
    @classmethod
    def create_agent(
        cls,
        agent_type: str,
        network_config: Dict[str, Any],
        temperature_config: Dict[str, Dict[str, float]],
        update_frequency: int,
        interpreter_type: str,
        interpreter_config: Dict[str, Any],
        device: str = "cuda",
        **kwargs
    ) -> BaseAgent:
        """
        Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            network_config: Configuration for the unified network
            temperature_config: Configuration for the temperature manager
            update_frequency: Update frequency for the temperature manager
            interpreter_type: Type of action interpreter to use
            interpreter_config: Configuration for the action interpreter
            device: Device to run the agent on
            **kwargs: Additional arguments for the agent
            
        Returns:
            An instance of the specified agent type
            
        Raises:
            ValueError: If the agent type is not supported
        """
        if agent_type not in cls._agents:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        
        # Create the interpreter with environment and configuration
        interpreter = InterpreterFactory.create_interpreter(
            interpreter_type=interpreter_type,
            **interpreter_config  # Pass all interpreter configuration
        )

        temperature_manager = TemperatureManager(
            **temperature_config # Pass all temperature configuration
        )
        
        # Create and return the agent
        agent_class = cls._agents[agent_type]
        return agent_class(
            network_config=network_config,
            interpreter=interpreter,
            temperature_manager=temperature_manager,
            update_frequency=update_frequency,
            device=device,
            **kwargs
        )
    
    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a new agent type.
        
        Args:
            name: Name of the agent type
            agent_class: Agent class to register
        """
        cls._agents[name] = agent_class 