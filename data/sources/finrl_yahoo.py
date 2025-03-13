"""Yahoo Finance data source implementation."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

import exchange_calendars as xcals
import pandas as pd
import yfinance as yf

from data.sources.finrl_source import FinRLSource

logger = logging.getLogger(__name__)


class FinRLYahoo(FinRLSource):
    """Yahoo Finance data source implementation.
    
    This class uses the yfinance library to download historical market data
    from Yahoo Finance.
    """
    
    def __init__(self):
        """Initialize Yahoo Finance data source."""
        super().__init__(api_key=None)  # Yahoo Finance doesn't require an API key
    
    def download_data(
        self,
        ticker_list: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        time_interval: str = "1d",
        proxy: Optional[Dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Download data from Yahoo Finance.
        
        Args:
            ticker_list: List of ticker symbols to download.
            start_date: Start date for data (YYYY-MM-DD format or datetime).
            end_date: End date for data (YYYY-MM-DD format or datetime).
            time_interval: Time interval for data, default "1d" (daily).
                Options: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d",
                "1wk", "1mo", "3mo".
            proxy: Optional proxy configuration.
            **kwargs: Additional parameters for yfinance download function.
            
        Returns:
            DataFrame containing price data for all tickers.
        """
        try:
            # Convert string dates to datetime objects if needed
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Format for yfinance
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Download data
            data = yf.download(
                tickers=ticker_list,
                start=start_str,
                end=end_str,
                interval=time_interval,
                proxy=proxy,
                group_by="ticker",  # Group by ticker for multi-ticker download
                auto_adjust=True,   # Adjust OHLC automatically
                prepost=False,      # No pre/post market data
                threads=True,       # Use multithreading
                **kwargs
            )
            
            # If only one ticker, the format is different
            if len(ticker_list) == 1:
                ticker = ticker_list[0]
                data.columns = [f"{ticker}_{col}" for col in data.columns]
            else:
                # Flatten multi-level columns 
                data.columns = ["_".join(col).strip() for col in data.columns.values]
            
            # Reset index to have date as a column
            data.reset_index(inplace=True)
            data.rename(columns={"index": "date", "Date": "date"}, inplace=True)
            
            logger.info(f"Downloaded data for {len(ticker_list)} tickers from {start_str} to {end_str}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data from Yahoo Finance: {e}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess Yahoo Finance data.
        
        Args:
            data: Raw data DataFrame from Yahoo Finance.
            
        Returns:
            Cleaned DataFrame with consistent formatting.
        """
        if data.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return data
        
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
        df.sort_values("date", inplace=True)
        
        return df
    
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