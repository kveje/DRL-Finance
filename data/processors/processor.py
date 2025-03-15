"""Base processor class for FinRL framework."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseProcessor(ABC):
    """Abstract base class for all data processors.

    All data processor implementations should inherit from this class
    and implement the required methods.
    """

    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process the input data.

        Args:
            data: Input DataFrame to process.
            **kwargs: Additional keyword arguments.

        Returns:
            Processed DataFrame.
        """
        pass
