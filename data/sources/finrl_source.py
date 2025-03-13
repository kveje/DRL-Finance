"""Base data source class for FinRL framework."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


class FinRLSource(ABC):
    """Abstract base class for all data sources.
    
    All data source implementations should inherit from this class
    and implement the required methods.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize data source.
        
        Args:
            api_key: Optional API key for authenticated data sources.
        """
        self.api_key = api_key
    
    @abstractmethod
    def download_data(
        self,
        ticker_list: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        time_interval: str,
        proxy: Optional[Dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Download data for specified tickers and date range.
        
        Args:
            ticker_list: List of ticker symbols to download.
            start_date: Start date for data (YYYY-MM-DD format or datetime).
            end_date: End date for data (YYYY-MM-DD format or datetime).
            time_interval: Time interval for data, e.g., '1d', '1h'.
            proxy: Optional proxy configuration.
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
    
    @abstractmethod
    def get_trading_days(
        self, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        market: str = "NYSE"
    ) -> List[datetime]:
        """Get list of trading days for a specific market.
        
        Args:
            start_date: Start date for range.
            end_date: End date for range.
            market: Market name (e.g., "NYSE", "NASDAQ").
            
        Returns:
            List of trading days as datetime objects.
        """
        pass
    
    def is_valid_ticker(self, ticker: str) -> bool:
        """Check if a ticker symbol is valid.
        
        Args:
            ticker: Ticker symbol to check.
            
        Returns:
            Boolean indicating if ticker is valid.
        """
        try:
            data = self.download_data(
                ticker_list=[ticker],
                start_date=(datetime.now().date().replace(day=1)),
                end_date=datetime.now().date(),
                time_interval="1d"
            )
            return not data.empty
        except Exception:
            return False