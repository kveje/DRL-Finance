from typing import Dict, Type
from .base_processor import BaseProcessor
from .price_processor import PriceProcessor
from .ohlcv_processor import OHLCVProcessor
from .tech_processor import TechProcessor
from .position_processor import PositionProcessor
from .vix_processor import VIXProcessor
from .cash_processor import CashProcessor
from .affordability_processor import AffordabilityProcessor
from .current_price_processor import CurrentPriceProcessor


class ProcessorFactory:
    """Factory class for creating different types of observation processors for the trading environment."""
    
    _processors: Dict[str, Type[BaseProcessor]] = {
        'price': PriceProcessor,
        'ohlcv': OHLCVProcessor,
        'tech': TechProcessor,
        'position': PositionProcessor,
        'vix': VIXProcessor,
        'cash': CashProcessor,
        'affordability': AffordabilityProcessor,
        'current_price': CurrentPriceProcessor,
    }
    
    @classmethod
    def create_processor(
        cls,
        processor_type: str,
        **kwargs
    ) -> BaseProcessor:
        """
        Create a processor instance based on the specified type.
        
        Args:
            processor_type: Type of processor to create
            **kwargs: Additional arguments specific to the processor type
            
        Returns:
            Instance of the specified processor type
        """
        if processor_type not in cls._processors:
            raise ValueError(f"Unsupported processor type: {processor_type}")
            
        processor_class = cls._processors[processor_type]
        return processor_class(**kwargs)
    
    @classmethod
    def register_processor(cls, name: str, processor_class: Type[BaseProcessor]) -> None:
        """
        Register a new processor type.
        
        Args:
            name: Name of the processor type
            processor_class: Processor class to register
        """
        cls._processors[name] = processor_class 