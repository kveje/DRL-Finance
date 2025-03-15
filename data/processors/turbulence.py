"""Turbulence processor for financial data."""

import logging
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TurbulenceProcessor:
    """Processor to calculate and add turbulence index."""

    def process(
        self,
        data: pd.DataFrame,
        window: int = 252,
        ticker_column: str = "ticker",
        **kwargs,
    ) -> pd.DataFrame:
        """Add turbulence index to the input DataFrame.

        Args:
            data: DataFrame to add turbulence to
            window: Rolling window for covariance calculation
            ticker_column: Name of the column containing ticker symbols
            **kwargs: Additional arguments

        Returns:
            DataFrame with added turbulence column
        """
        # Make a copy to avoid modifying the input
        result = data.copy()

        # Get list of unique dates and tickers
        unique_dates = sorted(result["date"].unique())
        unique_tickers = result[ticker_column].unique()

        if len(unique_tickers) < 2:
            raise ValueError("Turbulence calculation requires at least 2 tickers")

        # Create a pivot table of returns
        # First, sort by ticker and date
        result = result.sort_values([ticker_column, "date"])

        # Calculate daily returns
        result["daily_return"] = result.groupby(ticker_column)["close"].pct_change(
            fill_method=None
        )

        # Create a pivot table: date x ticker
        returns_pivot = result.pivot(
            index="date", columns=ticker_column, values="daily_return"
        )

        # Initialize turbulence series
        turbulence_index = pd.Series(index=unique_dates)

        # Calculate turbulence for each date
        for i in range(window, len(unique_dates)):
            current_date = unique_dates[i]

            # Get historical window of returns
            historical_returns = returns_pivot.iloc[i - window : i]

            # Skip if we have NaN values
            if historical_returns.isnull().values.any():
                continue

            # Calculate covariance matrix
            cov_matrix = historical_returns.cov()

            # Get current day's returns
            current_returns = returns_pivot.loc[current_date]

            # Calculate Mahalanobis distance (turbulence)
            # d = (x - mu)' Sigma^(-1) (x - mu)
            # where x is current returns, mu is historical mean, Sigma is covariance
            try:
                # Inverse of covariance matrix
                cov_inv = np.linalg.inv(cov_matrix.values)

                # Mean returns
                mean_returns = historical_returns.mean()

                # Demean current returns
                demean_returns = current_returns - mean_returns

                # Calculate Mahalanobis distance
                turbulence = np.sqrt(
                    np.dot(np.dot(demean_returns, cov_inv), demean_returns)
                )

                turbulence_index[current_date] = turbulence
            except np.linalg.LinAlgError:
                logger.warning(
                    f"LinAlgError: Singular matrix for date {current_date}, skipping"
                )
                continue

        # Convert turbulence index to DataFrame
        turbulence_df = pd.DataFrame(
            {"date": turbulence_index.index, "turbulence": turbulence_index.values}
        )

        # Merge with original data
        result = pd.merge(data, turbulence_df, on="date", how="left")

        return result
