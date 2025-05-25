from typing import Dict, Type

from .base_action_interpreter import BaseActionInterpreter
from .discrete_interpreter import DiscreteInterpreter
from .confidence_scaled_interpreter import ConfidenceScaledInterpreter

class InterpreterFactory:
    """Factory class for creating different types of action interpreters."""
    
    _interpreters: Dict[str, Type[BaseActionInterpreter]] = {
        'discrete': DiscreteInterpreter,
        'confidence_scaled': ConfidenceScaledInterpreter
    }
    
    @classmethod
    def create_interpreter(
        cls,
        interpreter_type: str,
        **kwargs
    ) -> BaseActionInterpreter:
        """
        Create an interpreter instance based on the specified type.
        
        Args:
            interpreter_type: Type of interpreter to create
            **kwargs: Additional arguments specific to the interpreter type
            
        Returns:
            Instance of the specified interpreter type
        """
        if interpreter_type not in cls._interpreters:
            raise ValueError(f"Unsupported interpreter type: {interpreter_type}")
            
        interpreter_class = cls._interpreters[interpreter_type]
        return interpreter_class(interpreter_type=interpreter_type, **kwargs)
    
    @classmethod
    def register_interpreter(cls, name: str, interpreter_class: Type[BaseActionInterpreter]) -> None:
        """
        Register a new interpreter type.
        
        Args:
            name: Name of the interpreter type
            interpreter_class: Interpreter class to register
        """
        cls._interpreters[name] = interpreter_class 