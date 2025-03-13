"""Market data container for financial time series."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketData:
    """Market data container class.

    This class stores and provides access to financial time series data
    with functionality for slicing, filtering, and transformation.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        ticker_list: List[str],
        time_column: str = "date",
        price_column_pattern: str = "_Close",
        metadata: Optional[Dict] = None,
    ):
        """Initialize market data container.

        Args:
            data: DataFrame with financial time series data.
            ticker_list: List of tickers in the data.
            time_column: Name of the time/date column.
            price_column_pattern: Pattern to identify price columns.
            metadata: Optional additional metadata about the market.
        """
        self.data = data.copy()
        self.ticker_list = ticker_list.copy()
        self.time_column = time_column
        self.price_column_pattern = price_column_pattern
        self.metadata = metadata or {}

        # Ensure date column is datetime type
        if self.time_column in self.data.columns:
            self.data[self.time_column] = pd.to_datetime(self.data[self.time_column])

        # Identify price columns for each ticker
        self.price_columns = {}
        for ticker in self.ticker_list:
            price_col = next(
                (
                    col
                    for col in self.data.columns
                    if ticker in col and self.price_column_pattern in col
                ),
                None,
            )
            if price_col:
                self.price_columns[ticker] = price_col
            else:
                logger.warning(f"No price column found for ticker {ticker}")

        # Sort data by time
        if self.time_column in self.data.columns:
            self.data.sort_values(by=self.time_column, inplace=True)

    @property
    def dates(self) -> pd.Series:
        """Get the unique dates in the dataset.

        Returns:
            Series of unique dates.
        """
        if self.time_column in self.data.columns:
            return self.data[self.time_column].unique()
        return pd.Series(dtype="datetime64[ns]")

    @property
    def start_date(self) -> Optional[datetime]:
        """Get the start date of the dataset.

        Returns:
            Start date as datetime or None if no data.
        """
        if self.time_column in self.data.columns and not self.data.empty:
            return self.data[self.time_column].min()
        return None

    @property
    def end_date(self) -> Optional[datetime]:
        """Get the end date of the dataset.

        Returns:
            End date as datetime or None if no data.
        """
        if self.time_column in self.data.columns and not self.data.empty:
            return self.data[self.time_column].max()
        return None

    def get_price_matrix(self) -> pd.DataFrame:
        """Get a matrix of close prices for all tickers.

        Returns:
            DataFrame with dates as index and tickers as columns.
        """
        price_df = pd.DataFrame(index=self.data[self.time_column])

        for ticker, price_col in self.price_columns.items():
            price_df[ticker] = self.data[price_col]

        # Set date as index
        print(price_df.head())
        print("Setting index")
        price_df.index = self.data[self.time_column]
        print(price_df.head())

        return price_df

    def get_returns(self, window: int = 1) -> pd.DataFrame:
        """Calculate returns for all tickers over specified window.

        Args:
            window: Number of periods for return calculation.

        Returns:
            DataFrame with returns for each ticker.
        """
        price_df = self.get_price_matrix()

        # Calculate returns
        if window == 1:
            return price_df.pct_change().dropna()
        else:
            return (price_df / price_df.shift(window) - 1).dropna()

    def get_ticker_data(self, ticker: str) -> pd.DataFrame:
        """Get all data for a specific ticker.

        Args:
            ticker: Ticker symbol.

        Returns:
            DataFrame with all columns for the specified ticker.
        """
        ticker_cols = [col for col in self.data.columns if ticker in col]
        ticker_cols.append(self.time_column)  # Include time column

        return self.data[ticker_cols].copy()

    def slice_by_date(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> "MarketData":
        """Slice market data by date range.

        Args:
            start_date: Start date for slice.
            end_date: End date for slice.

        Returns:
            New MarketData object with sliced data.
        """
        # Convert dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Apply date filters
        mask = pd.Series(True, index=self.data.index)

        if start_date:
            mask &= self.data[self.time_column] >= start_date

        if end_date:
            mask &= self.data[self.time_column] <= end_date

        sliced_data = self.data[mask].copy()

        # Create new MarketData object
        return MarketData(
            data=sliced_data,
            ticker_list=self.ticker_list,
            time_column=self.time_column,
            price_column_pattern=self.price_column_pattern,
            metadata=self.metadata,
        )

    def slice_by_tickers(self, tickers: List[str]) -> "MarketData":
        """Slice market data to include only specified tickers.

        Args:
            tickers: List of tickers to include.

        Returns:
            New MarketData object with data for specified tickers.
        """
        # Validate tickers
        valid_tickers = [ticker for ticker in tickers if ticker in self.ticker_list]

        if not valid_tickers:
            logger.warning("No valid tickers specified")
            return self

        # Get columns for specified tickers + time column
        ticker_cols = []
        for ticker in valid_tickers:
            ticker_cols.extend([col for col in self.data.columns if ticker in col])

        ticker_cols.append(self.time_column)  # Include time column
        ticker_cols = list(set(ticker_cols))  # Remove duplicates

        sliced_data = self.data[ticker_cols].copy()

        # Create new MarketData object
        return MarketData(
            data=sliced_data,
            ticker_list=valid_tickers,
            time_column=self.time_column,
            price_column_pattern=self.price_column_pattern,
            metadata=self.metadata,
        )

    def add_technical_indicators(
        self, indicators: List[str], feature_engineer
    ) -> "MarketData":
        """Add technical indicators to the market data.

        Args:
            indicators: List of indicators to add.
            feature_engineer: FeatureEngineer instance to use.

        Returns:
            New MarketData object with added indicators.
        """
        # Use feature engineer to add indicators
        enhanced_data = feature_engineer.preprocess(
            df=self.data, ticker_list=self.ticker_list, technical_indicators=indicators
        )

        # Create new MarketData object
        return MarketData(
            data=enhanced_data,
            ticker_list=self.ticker_list,
            time_column=self.time_column,
            price_column_pattern=self.price_column_pattern,
            metadata=self.metadata,
        )

    def to_numpy(self, columns: Optional[List[str]] = None) -> np.ndarray:
        """Convert specified columns to numpy array.

        Args:
            columns: List of columns to convert. If None, uses all columns except time.

        Returns:
            Numpy array with data.
        """
        if columns is None:
            # Use all columns except time column
            columns = [col for col in self.data.columns if col != self.time_column]

        # Filter to only include columns that exist
        valid_columns = [col for col in columns if col in self.data.columns]

        # Convert to numpy array
        return self.data[valid_columns].values

    def split_train_test(
        self, train_ratio: float = 0.8
    ) -> Tuple["MarketData", "MarketData"]:
        """Split data into training and testing sets.

        Args:
            train_ratio: Ratio of data to use for training.

        Returns:
            Tuple of (train_data, test_data) as MarketData objects.
        """
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")

        # Sort data by time
        sorted_data = self.data.sort_values(by=self.time_column)

        # Calculate split point
        split_idx = int(len(sorted_data) * train_ratio)

        # Split data
        train_data = sorted_data.iloc[:split_idx].copy()
        test_data = sorted_data.iloc[split_idx:].copy()

        # Create MarketData objects
        train_market_data = MarketData(
            data=train_data,
            ticker_list=self.ticker_list,
            time_column=self.time_column,
            price_column_pattern=self.price_column_pattern,
            metadata=self.metadata,
        )

        test_market_data = MarketData(
            data=test_data,
            ticker_list=self.ticker_list,
            time_column=self.time_column,
            price_column_pattern=self.price_column_pattern,
            metadata=self.metadata,
        )

        return train_market_data, test_market_data
