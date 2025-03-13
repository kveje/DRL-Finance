"""FinRL library data source implementation."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

import exchange_calendars as xcals
import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.data_processor import DataProcessor

from data.sources.finrl_source import FinRLSource

logger = logging.getLogger(__name__)


class FinRLData(FinRLSource):
    """FinRL library data source implementation.
    
    This class uses FinRL's built-in data processing capabilities to download,
    clean, and preprocess financial data.
    """
    
    def __init__(
        self,
        data_source: str = "yahoofinance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        """Initialize FinRL data source.
        
        Args:
            data_source: The data source to use ('yahoofinance', 'alpaca', 'wrds', 'ccxt', 'joinquant', 'quantconnect').
            api_key: Optional API key for authenticated data sources.
            api_secret: Optional API secret for authenticated data sources.
        """
        super().__init__(api_key=api_key)
        self.data_source = data_source
        self.api_secret = api_secret
        
        # Initialize FinRL's DataProcessor
        self.processor = DataProcessor(data_source=data_source)
        
        # Set API keys if needed
        if api_key and api_secret and data_source == "alpaca":
            self.processor.alpaca = self.processor._get_alpaca(api_key, api_secret)
    
    def download_data(
        self,
        ticker_list: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        time_interval: str = "1d",
        proxy: Optional[Dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Download data using FinRL's data processor.
        
        Args:
            ticker_list: List of ticker symbols to download.
            start_date: Start date for data (YYYY-MM-DD format or datetime).
            end_date: End date for data (YYYY-MM-DD format or datetime).
            time_interval: Time interval for data, default "1d".
            proxy: Not used by FinRL processor, kept for interface consistency.
            **kwargs: Additional parameters passed to the FinRL processor.
            
        Returns:
            DataFrame containing price data for all tickers.
        """
        try:
            # Convert datetime objects to string format if needed
            start_str = start_date.strftime("%Y-%m-%d") if isinstance(start_date, datetime) else start_date
            end_str = end_date.strftime("%Y-%m-%d") if isinstance(end_date, datetime) else end_date
            
            # Map time_interval to FinRL format if needed
            interval_mapping = {
                "1d": "1D",
                "1h": "1H",
                "5m": "5Min",
                "1m": "1Min",
                "1D": "1D",
                "1H": "1H"
            }
            finrl_interval = interval_mapping.get(time_interval, time_interval)
            
            # Download data using FinRL's processor
            data = self.processor.download_data(
                ticker_list=ticker_list,
                start_date=start_str,
                end_date=end_str,
                time_interval=finrl_interval,
                **kwargs
            )
            
            # Make sure we have consistent column names
            if "time" in data.columns and "date" not in data.columns:
                data.rename(columns={"time": "date"}, inplace=True)
                
            logger.info(f"Downloaded data for {len(ticker_list)} tickers from {start_str} to {end_str}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data from FinRL: {e}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data using FinRL's data processor.
        
        Args:
            data: Raw data DataFrame.
            
        Returns:
            Cleaned DataFrame with consistent formatting.
        """
        if data.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return data
        
        try:
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Process data through FinRL's preprocessing steps
            df = self.processor.clean_data(df)
            
            # Ensure date column is datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            
            # Sort by date
            if "date" in df.columns:
                df.sort_values("date", inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            # Fall back to basic cleaning if FinRL processor fails
            try:
                # Make a copy to avoid modifying the original
                df = data.copy()
                
                # Ensure date column is datetime
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                
                # Handle missing values
                df.fillna(method="ffill", inplace=True)  # Forward fill
                df.fillna(method="bfill", inplace=True)  # Backward fill any remaining NAs
                
                # Drop rows with missing values that couldn't be filled
                df.dropna(inplace=True)
                
                # Sort by date
                if "date" in df.columns:
                    df.sort_values("date", inplace=True)
                
                return df
            except Exception as fallback_e:
                logger.error(f"Fallback cleaning also failed: {fallback_e}")
                raise
    
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
            market: Market name, default "NYSE".
            
        Returns:
            List of trading days as datetime objects.
        """
        try:
            # Convert string dates to datetime objects if needed
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Get calendar
            calendar = xcals.get_calendar(market)
            
            # Get trading days
            trading_days = calendar.sessions_in_range(
                pd.Timestamp(start_date),
                pd.Timestamp(end_date)
            )
            
            return [day.to_pydatetime() for day in trading_days]
            
        except Exception as e:
            logger.error(f"Error retrieving trading days: {e}")
            raise
    
    def add_technical_indicators(
        self, 
        data: pd.DataFrame, 
        tech_indicator_list: List[str] = None
    ) -> pd.DataFrame:
        """Add technical indicators to the data using FinRL's FeatureEngineer.
        
        Args:
            data: Price data DataFrame.
            tech_indicator_list: List of technical indicators to add.
                If None, uses default list from FinRL.
                
        Returns:
            DataFrame with added technical indicators.
        """
        try:
            # Use FinRL's FeatureEngineer to add technical indicators
            feature_engineer = FeatureEngineer(
                use_technical_indicator=True,
                tech_indicator_list=tech_indicator_list
            )
            
            processed_data = feature_engineer.preprocess_data(data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            raise
    
    def add_turbulence(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add turbulence index to the data.
        
        Args:
            data: Price data DataFrame.
                
        Returns:
            DataFrame with added turbulence index.
        """
        try:
            # Use FinRL's processor to add turbulence
            df = self.processor.add_turbulence(data)
            return df
            
        except Exception as e:
            logger.error(f"Error adding turbulence: {e}")
            raise
    
    def add_vix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add VIX (volatility index) to the data.
        
        Args:
            data: Price data DataFrame.
                
        Returns:
            DataFrame with added VIX index.
        """
        try:
            # Use FinRL's processor to add VIX
            df = self.processor.add_vix(data)
            return df
            
        except Exception as e:
            logger.error(f"Error adding VIX: {e}")
            raise
    
    def split_data(self, data: pd.DataFrame, split: str = "train") -> pd.DataFrame:
        """Split data into train/validation/test sets.
        
        Args:
            data: DataFrame containing price data.
            split: Which split to return ("train", "validation", or "test").
                
        Returns:
            DataFrame containing the requested split.
        """
        try:
            # Use FinRL's data_split function
            train, validation, test = data_split(data, split_ratio=[0.7, 0.15, 0.15])
            
            if split.lower() == "train":
                return train
            elif split.lower() == "validation":
                return validation
            elif split.lower() == "test":
                return test
            else:
                raise ValueError(f"Invalid split type: {split}. Must be 'train', 'validation', or 'test'")
                
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise