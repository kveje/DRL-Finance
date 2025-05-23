"""Factory for creating different types of agents."""

from typing import Dict, Any, Type
from .base_agent import BaseAgent
from .dqn_agent import DQNAgent
from .ddpg_agent import DDPGAgent
from .ppo_agent import PPOAgent
from .a2c_agent import A2CAgent
from ..action_interpreters.interpreter_factory import InterpreterFactory

class AgentFactory:
    """Factory class for creating different types of agents."""
    
    _agents: Dict[str, Type[BaseAgent]] = {
        'dqn': DQNAgent,
        'ddpg': DDPGAgent,
        'ppo': PPOAgent,
        'a2c': A2CAgent
    }
    
    @classmethod
    def create_agent(
        cls,
        agent_type: str,
        env: Any,
        network_config: Dict[str, Any],
        interpreter_type: str,
        interpreter_config: Dict[str, Any],
        device: str = "cuda",
        **kwargs
    ) -> BaseAgent:
        """
        Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            env: Trading environment instance
            network_config: Configuration for the unified network
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
        
        # Create the interpreter
        interpreter = InterpreterFactory.create_interpreter(
            interpreter_type=interpreter_type,
            env=env,
            **interpreter_config
        )
        
        # Create and return the agent
        agent_class = cls._agents[agent_type]
        return agent_class(
            env=env,
            network_config=network_config,
            interpreter=interpreter,
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