"""Yahoo Finance data source implementation."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

from data.sources.source import BaseSource

logger = logging.getLogger(__name__)


class YahooSource(BaseSource):
    """Yahoo Finance data source implementation.

    This class uses the yfinance library to download historical market data
    from Yahoo Finance.
    """

    def __init__(self):
        """Initialize Yahoo Finance data source."""
        super().__init__()  # Yahoo Finance doesn't require an API key

    def download_data(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        time_interval: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        """Download data from Yahoo Finance.

        Args:
            tickers: List of ticker symbols to download.
            start_date: Start date for data (YYYY-MM-DD format or datetime).
            end_date: End date for data (YYYY-MM-DD format or datetime).
            time_interval: Time interval for data, default "1d" (daily).
                Options: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d",
                "1wk", "1mo", "3mo".
            **kwargs: Additional parameters for yfinance download function.

        Returns:
            DataFrame containing price data for all tickers with standardized format.

        Raises:
            ValueError: If tickers list is empty, dates are invalid, or end_date is before start_date.
        """
        # Validate inputs
        if not tickers:
            raise ValueError("Ticker list cannot be empty")

        # Parse and validate dates
        try:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")

        # Check if end_date is before start_date
        if end_date < start_date:
            raise ValueError("End date cannot be before start date")

        # Format for yfinance
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        try:
            # Download data
            data = yf.download(
                tickers=tickers,
                start=start_str,
                end=end_str,
                interval=time_interval,
                group_by="ticker",  # Group by ticker for multi-ticker download
                auto_adjust=True,  # Adjust OHLC automatically
                prepost=False,  # No pre/post market data
                threads=True,  # Use multithreading
                multi_level_index=False,  # Flatten multi-level columns
                **kwargs,
            )

            logger.info(
                f"Downloaded data for {len(tickers)} tickers from {start_str} to {end_str}"
            )

            # Process data based on number of tickers
            if len(tickers) == 1:
                return self._process_single_ticker_data(data, tickers[0])
            else:
                return self._process_multiple_ticker_data(data, tickers)

        except Exception as e:
            logger.error(f"Error downloading data from Yahoo Finance: {e}")
            raise

    def _process_single_ticker_data(
        self, data: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """Process data for a single ticker.

        Args:
            data: Raw DataFrame from yfinance
            ticker: Ticker symbol

        Returns:
            Processed DataFrame
        """
        df = data.copy()

        # Reset index to make date a column
        df.reset_index(inplace=True)

        # Modify column names for consistency
        df.columns = [col.lower() for col in df.columns]

        # Add ticker column
        df["ticker"] = ticker

        # Rename some columns to our standard names
        column_map = {"adj close": "close", "index": "date", "date": "date"}
        df.rename(columns=column_map, inplace=True)

        # Ensure all expected columns are present with proper names
        expected_columns = ["date", "open", "high", "low", "close", "volume", "ticker"]
        for col in expected_columns:
            if col not in df.columns:
                raise ValueError(f"Expected column {col} not found in data")

        return df[expected_columns]  # Return only the expected columns

    def _process_multiple_ticker_data(
        self, data: pd.DataFrame, tickers: List[str]
    ) -> pd.DataFrame:
        """Process data for multiple tickers.

        Args:
            data: Raw DataFrame from yfinance with multi-level columns
            tickers: List of ticker symbols

        Returns:
            Processed DataFrame with all tickers combined, one row per date per ticker
        """
        # For multi-ticker data, yfinance returns a DataFrame with multi-level columns
        # where level 0 is the ticker and level 1 is the price type (Open, High, etc.)

        # First check if we have data
        if data.empty:
            raise ValueError("No data returned from Yahoo Finance")

        # Create an empty list to store individual DataFrames
        all_dfs = []

        # Copy the date index for future use
        dates = data.index.copy()

        # Process each ticker
        for ticker in tickers:
            if ticker in data.columns.levels[0]:
                # Extract this ticker's data
                ticker_data = data[ticker].copy()

                # Reset index to make date a column
                ticker_data.reset_index(inplace=True)

                # Rename columns to lowercase
                ticker_data.columns = [col.lower() for col in ticker_data.columns]

                # Rename some columns to our standard names
                column_map = {"adj close": "close", "index": "date", "date": "date"}
                ticker_data.rename(columns=column_map, inplace=True)

                # Add ticker column
                ticker_data["ticker"] = ticker

                # Select only required columns
                expected_columns = [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "ticker",
                ]

                # Check if all required columns exist
                missing_columns = [
                    col for col in expected_columns if col not in ticker_data.columns
                ]
                if missing_columns:
                    logger.warning(f"Missing columns for {ticker}: {missing_columns}")
                    # Create missing columns with NaN values
                    for col in missing_columns:
                        if col != "ticker":  # We already added ticker
                            ticker_data[col] = float("nan")

                ticker_data = ticker_data[expected_columns]
                all_dfs.append(ticker_data)
            else:
                logger.warning(f"No data found for ticker {ticker}")

        if not all_dfs:
            raise ValueError("No valid data found for any tickers")

        # Combine all dataframes
        result = pd.concat(all_dfs, ignore_index=True)

        # Sort by date and ticker for consistency
        result.sort_values(["date", "ticker"], inplace=True)

        return result

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
        df.ffill(inplace=True)  # Forward fill
        df.bfill(inplace=True)  # Backward fill any remaining NAs

        # Drop rows with missing values that couldn't be filled
        df.dropna(inplace=True)

        # Sort by date and ticker
        if "ticker" in df.columns:
            df.sort_values(["date", "ticker"], inplace=True)
        else:
            df.sort_values("date", inplace=True)

        return df
