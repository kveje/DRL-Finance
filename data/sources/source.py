"""Base data source class for FinRL framework."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


class BaseSource(ABC):
    """Abstract base class for all data sources.

    All data source implementations should inherit from this class
    and implement the required methods.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize data source.

        Args:
            cache_dir: Directory to store cached data.
        """
        self.cache_dir = cache_dir

    @abstractmethod
    def download_data(
        self,
        ticker_list: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        time_interval: str,
        **kwargs
    ) -> pd.DataFrame:
        """Download data for specified tickers and date range.

        Args:
            ticker_list: List of ticker symbols to download.
            start_date: Start date for data (YYYY-MM-DD format or datetime).
            end_date: End date for data (YYYY-MM-DD format or datetime).
            time_interval: Time interval for data, e.g., '1d', '1h'.
            **kwargs: Additional source-specific parameters.

        Returns:
            DataFrame containing the downloaded data.
        """
        pass

    @abstractmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the downloaded data.

        Args:
            data: Raw data DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        pass
