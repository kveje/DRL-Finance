"""Alpaca data source implementation."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import alpaca_trade_api as tradeapi
import exchange_calendars as xcals
import pandas as pd

from data.sources.finrl_source import FinRLSource

logger = logging.getLogger(__name__)


class FinRLAlpaca(FinRLSource):
    """Alpaca API data source implementation.
    
    This class uses the Alpaca Trade API to download market data.
    Requires API key and secret from Alpaca.
    """
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        """Initialize Alpaca data source.
        
        Args:
            api_key: Alpaca API key.
            api_secret: Alpaca API secret.
            base_url: Alpaca API URL, default is paper trading URL.
        """
        super().__init__(api_key=api_key)
        self.api_secret = api_secret
        self.base_url = base_url
        self._api = tradeapi.REST(api_key, api_secret, base_url, api_version="v2")
    
    def download_data(
        self,
        ticker_list: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        time_interval: str = "1Day",
        proxy: Optional[Dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Download data from Alpaca.
        
        Args:
            ticker_list: List of ticker symbols to download.
            start_date: Start date for data (YYYY-MM-DD format or datetime).
            end_date: End date for data (YYYY-MM-DD format or datetime).
            time_interval: Time interval for data, default "1Day".
                Options: "1Min", "5Min", "15Min", "1Hour", "1Day".
            proxy: Not used for Alpaca, kept for interface consistency.
            **kwargs: Additional parameters for Alpaca API.
            
        Returns:
            DataFrame containing price data for all tickers.
        """
        try:
            # Convert string dates to datetime objects if needed
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Adjust time interval for Alpaca API format
            interval_mapping = {
                "1d": "1Day",
                "1h": "1Hour",
                "15m": "15Min",
                "5m": "5Min",
                "1m": "1Min"
            }
            
            alpaca_interval = interval_mapping.get(time_interval, time_interval)
            
            # Format dates for Alpaca
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")  # Add a day to include end_date
            
            # Initialize empty dataframe for all data
            all_data = []
            
            # Download data for each ticker
            for ticker in ticker_list:
                try:
                    # Get daily bars
                    bars = self._api.get_bars(
                        ticker,
                        alpaca_interval,
                        start=start_str,
                        end=end_str,
                        adjustment='raw',
                        **kwargs
                    ).df
                    
                    if not bars.empty:
                        # Add ticker column and rename columns
                        bars["ticker"] = ticker
                        bars.rename(columns={
                            "open": f"{ticker}_open",
                            "high": f"{ticker}_high",
                            "low": f"{ticker}_low",
                            "close": f"{ticker}_close",
                            "volume": f"{ticker}_volume"
                        }, inplace=True)
                        
                        all_data.append(bars)
                    
                except Exception as e:
                    logger.warning(f"Error downloading data for {ticker}: {e}")
            
            if not all_data:
                logger.warning("No data downloaded from Alpaca")
                return pd.DataFrame()
            
            # Combine all ticker data
            if len(all_data) > 1:
                # Merge all dataframes on timestamp
                merged_data = pd.concat(all_data)
                merged_data.reset_index(inplace=True)
                merged_data.rename(columns={"timestamp": "date"}, inplace=True)
            else:
                merged_data = all_data[0].reset_index()
                merged_data.rename(columns={"timestamp": "date"}, inplace=True)
            
            logger.info(f"Downloaded data for {len(ticker_list)} tickers from {start_str} to {end_str}")
            return merged_data
            
        except Exception as e:
            logger.error(f"Error downloading data from Alpaca: {e}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess Alpaca data.
        
        Args:
            data: Raw data DataFrame from Alpaca.
            
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
            
            # Format for Alpaca API
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Get calendar
            calendar = self._api.get_calendar(
                start=start_str,
                end=end_str
            )
            
            # Extract dates from calendar
            trading_days = [day.date.to_pydatetime() for day in calendar]
            
            return trading_days
            
        except Exception as e:
            logger.error(f"Error retrieving trading days from Alpaca: {e}")
            
            # Fallback to exchange_calendars if Alpaca API fails
            try:
                calendar = xcals.get_calendar(market)
                trading_days = calendar.sessions_in_range(
                    pd.Timestamp(start_date),
                    pd.Timestamp(end_date)
                )
                return [day.to_pydatetime() for day in trading_days]
            except Exception as e_fallback:
                logger.error(f"Error in fallback calendar retrieval: {e_fallback}")
                raise