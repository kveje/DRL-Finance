"""Market synchronization utilities for cross-market analysis."""

import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, Union

import exchange_calendars as xcals
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketSynchronizer:
    """Market synchronization class for cross-market analysis.
    
    This class provides utilities to synchronize data from different markets
    with different trading hours and holidays.
    """
    
    def __init__(self):
        """Initialize market synchronizer."""
        # Cache for market calendars
        self._calendars = {}
    
    def get_market_calendar(self, market_name: str) -> xcals.ExchangeCalendar:
        """Get calendar for a specific market.
        
        Args:
            market_name: Name of the market (e.g., "NYSE", "LSE").
            
        Returns:
            Exchange calendar object.
        """
        if market_name not in self._calendars:
            try:
                self._calendars[market_name] = xcals.get_calendar(market_name)
            except Exception as e:
                logger.error(f"Error getting calendar for {market_name}: {e}")
                raise ValueError(f"Invalid market name: {market_name}")
        
        return self._calendars[market_name]
    
    def get_trading_days(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        market_name: str = "NYSE"
    ) -> List[datetime]:
        """Get list of trading days for a specific market.
        
        Args:
            start_date: Start date for range.
            end_date: End date for range.
            market_name: Market name.
            
        Returns:
            List of trading days as datetime objects.
        """
        # Convert string dates to datetime
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        
        # Get calendar
        calendar = self.get_market_calendar(market_name)
        
        # Get trading days
        trading_days = calendar.sessions_in_range(
            pd.Timestamp(start_date),
            pd.Timestamp(end_date)
        )
        
        return [day.to_pydatetime() for day in trading_days]
    
    def is_trading_day(
        self,
        date: Union[str, datetime],
        market_name: str = "NYSE"
    ) -> bool:
        """Check if a date is a trading day for a specific market.
        
        Args:
            date: Date to check.
            market_name: Market name.
            
        Returns:
            Boolean indicating if date is a trading day.
        """
        # Convert string date to datetime
        if isinstance(date, str):
            date = pd.Timestamp(date)
        
        # Get calendar
        calendar = self.get_market_calendar(market_name)
        
        # Check if date is a trading day
        return calendar.is_session(pd.Timestamp(date))
    
    def get_next_trading_day(
        self,
        date: Union[str, datetime],
        market_name: str = "NYSE"
    ) -> datetime:
        """Get the next trading day after a given date.
        
        Args:
            date: Starting date.
            market_name: Market name.
            
        Returns:
            Next trading day as datetime.
        """
        # Convert string date to datetime
        if isinstance(date, str):
            date = pd.Timestamp(date)
        
        # Get calendar
        calendar = self.get_market_calendar(market_name)
        
        # Get next trading day
        next_day = calendar.next_session(pd.Timestamp(date))
        
        return next_day.to_pydatetime()
    
    def get_previous_trading_day(
        self,
        date: Union[str, datetime],
        market_name: str = "NYSE"
    ) -> datetime:
        """Get the previous trading day before a given date.
        
        Args:
            date: Starting date.
            market_name: Market name.
            
        Returns:
            Previous trading day as datetime.
        """
        # Convert string date to datetime
        if isinstance(date, str):
            date = pd.Timestamp(date)
        
        # Get calendar
        calendar = self.get_market_calendar(market_name)
        
        # Get previous trading day
        prev_day = calendar.previous_session(pd.Timestamp(date))
        
        return prev_day.to_pydatetime()
    
    def align_market_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        time_column: str = "date",
        method: str = "outer",
        fill_method: str = "ffill"
    ) -> Dict[str, pd.DataFrame]:
        """Align data from different markets to a common timeline.
        
        Args:
            data_dict: Dictionary mapping market names to DataFrames.
            time_column: Name of the time/date column.
            method: Join method ("inner" or "outer").
            fill_method: Method for filling missing values.
            
        Returns:
            Dictionary with aligned DataFrames.
        """
        if not data_dict:
            return {}
        
        # Convert all time columns to datetime
        for market, df in data_dict.items():
            if time_column in df.columns:
                df[time_column] = pd.to_datetime(df[time_column])
        
        # Get all unique dates
        all_dates = set()
        for df in data_dict.values():
            if time_column in df.columns:
                all_dates.update(df[time_column].tolist())
        
        all_dates = sorted(all_dates)
        
        # Align data based on method
        aligned_data = {}
        
        if method == "inner":
            # Get common dates across all datasets
            common_dates = set(all_dates)
            for df in data_dict.values():
                if time_column in df.columns:
                    common_dates &= set(df[time_column].tolist())
            
            common_dates = sorted(common_dates)
            
            # Filter data to common dates
            for market, df in data_dict.items():
                aligned_df = df[df[time_column].isin(common_dates)].copy()
                aligned_df.sort_values(by=time_column, inplace=True)
                aligned_data[market] = aligned_df
        
        else:  # "outer" join
            # Create a reference DataFrame with all dates
            ref_df = pd.DataFrame({time_column: all_dates})
            
            # Merge each dataset with reference
            for market, df in data_dict.items():
                # Merge with reference dates
                merged_df = pd.merge(
                    ref_df,
                    df,
                    on=time_column,
                    how="left"
                )
                
                # Fill missing values
                if fill_method == "ffill":
                    merged_df.fillna(method="ffill", inplace=True)
                elif fill_method == "bfill":
                    merged_df.fillna(method="bfill", inplace=True)
                elif fill_method == "zero":
                    merged_df.fillna(0, inplace=True)
                
                aligned_data[market] = merged_df
        
        return aligned_data
    
    def convert_timezone(
        self,
        df: pd.DataFrame,
        time_column: str = "date",
        from_tz: str = "America/New_York",
        to_tz: str = "UTC"
    ) -> pd.DataFrame:
        """Convert timezone of datetime column in DataFrame.
        
        Args:
            df: DataFrame to convert.
            time_column: Name of the time/date column.
            from_tz: Source timezone.
            to_tz: Target timezone.
            
        Returns:
            DataFrame with converted timezone.
        """
        if time_column not in df.columns:
            logger.warning(f"Time column {time_column} not found in DataFrame")
            return df
        
        # Ensure datetime column
        df[time_column] = pd.to_datetime(df[time_column])
        
        # Convert timezone
        if df[time_column].dt.tz is None:
            # Set source timezone if not set
            df[time_column] = df[time_column].dt.tz_localize(from_tz)
        
        # Convert to target timezone
        df[time_column] = df[time_column].dt.tz_convert(to_tz)
        
        return df
    
    def merge_intraday_data(
        self,
        df_list: List[pd.DataFrame],
        time_column: str = "date",
        market_column: str = "market",
        how: str = "outer"
    ) -> pd.DataFrame:
        """Merge intraday data from multiple markets.
        
        Args:
            df_list: List of DataFrames with intraday data.
            time_column: Name of the time/date column.
            market_column: Name of the column to store market name.
            how: Merge method ("inner" or "outer").
            
        Returns:
            Merged DataFrame.
        """
        if not df_list:
            return pd.DataFrame()
        
        # Ensure all DataFrames have the time column
        for i, df in enumerate(df_list):
            if time_column not in df.columns:
                logger.warning(f"DataFrame {i} missing time column {time_column}")
                return pd.DataFrame()
            
            # Convert to datetime
            df[time_column] = pd.to_datetime(df[time_column])
        
        # Start with first DataFrame
        merged_df = df_list[0].copy()
        
        # Add market column if not exists
        if market_column not in merged_df.columns:
            merged_df[market_column] = f"market_0"
        
        # Merge with remaining DataFrames
        for i, df in enumerate(df_list[1:], 1):
            # Add market column if not exists
            df_copy = df.copy()
            if market_column not in df_copy.columns:
                df_copy[market_column] = f"market_{i}"
            
            # Merge on time column
            merged_df = pd.merge(
                merged_df,
                df_copy,
                on=time_column,
                how=how,
                suffixes=(f"", f"_{i}")
            )
        
        return merged_df