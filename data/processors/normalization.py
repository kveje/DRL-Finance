"""Data normalization module for financial data."""

# Initialize the logger
from utils.logger import Logger
logger = Logger.get_logger()


from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from data.processors.processor import BaseProcessor
from config.data import NORMALIZATION_PARAMS


class NormalizeProcessor(BaseProcessor):
    """Data normalization class for financial data.

    This class provides methods to normalize financial time series data
    using various normalization techniques: z-score, rolling window, and percentage change.
    """

    VALID_METHODS = ["zscore", "rolling", "percentage"]
    COLUMNS_TO_IGNORE = ["date", "day", "ticker", "day_index", "time_index", "timestamp"]
    STARTS_WITH_COLUMNS_TO_IGNORE = ["linreg_"]

    def __init__(self, method: str = NORMALIZATION_PARAMS["method"]):
        """Initialize normalization processor.

        Args:
            method: Normalization method to use.
                - 'zscore': Standardization (mean=0, std=1)
                - 'rolling': Rolling window normalization
                - 'percentage': Percentage change normalization
        """
        super().__init__()
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid normalization method: {method}. Valid methods: {self.VALID_METHODS}"
            )
        self.method = method
        self.stats = {}  # Storage for statistics if needed for inversion

    def process(
        self,
        data: pd.DataFrame,
        columns: List[str],
        window: int = 20,
        group_by: Optional[str] = None,
        handle_outliers: bool = True,
        fill_value: Union[float, str, None] = 0,
        **kwargs,
    ) -> pd.DataFrame:
        """Normalize the input data.

        Args:
            data: Input DataFrame to normalize.
            columns: List of columns to normalize.
            window: Window size for rolling normalization (default: 20).
            group_by: Column to group by before normalizing (e.g., 'ticker').
            handle_outliers: Whether to replace inf/-inf values (default: True).
            fill_value: How to handle NA/inf values after normalization:
                       - If numeric value: Fill with that value
                       - If 'ffill': Forward fill (use last valid value)
                       - If 'bfill': Backward fill
                       - If None: Leave NA values as is
            **kwargs: Additional keyword arguments.

        Returns:
            Normalized DataFrame.
        """
        # Make a copy to avoid modifying the original data
        result = data.copy()

        # Cast results to float (but not date, ticker, day, etc.)
        result[result.select_dtypes(include=[np.number]).columns] = result.select_dtypes(include=[np.number]).astype(float)

        # Drop columns to ignore
        columns = [col for col in columns if col not in self.COLUMNS_TO_IGNORE]
        columns = [col for col in columns if not any(col.startswith(prefix) for prefix in self.STARTS_WITH_COLUMNS_TO_IGNORE)]

        # Process data with the specified method
        if self.method == "zscore":
            result = self._zscore_normalize(result, columns, group_by)
        elif self.method == "rolling":
            result = self._rolling_normalize(result, columns, window, group_by)
        elif self.method == "percentage":
            result = self._percentage_normalize(result, columns, group_by)
        else:
            raise ValueError(f"Invalid normalization method: {self.method}")

        # Handle outliers (inf, -inf values) that may result from division operations
        if handle_outliers:
            for col in columns:
                if col in result.columns:
                    # Replace inf and -inf with NaN temporarily
                    result[col] = result[col].replace([np.inf, -np.inf], np.nan)

        # Handle NA values based on fill_value parameter
        if fill_value is not None:
            if fill_value == 'ffill':
                result = result.ffill()
            elif fill_value == 'bfill':
                result = result.bfill()
            else:
                # Assume fill_value is a numeric value
                result = result.fillna(fill_value)

        return result

    def process_by_groups(
        self,
        data: pd.DataFrame,
        feature_groups: Dict[str, List[str]],
        group_by: Optional[str] = "ticker",
        handle_outliers: bool = True,
        fill_value: Union[float, str, None] = 0,
    ) -> pd.DataFrame:
        """Normalize the input data by feature groups.

        Args:
            data: Input DataFrame to normalize.
            feature_groups: Dictionary mapping feature categories to column lists.
                        For example: {"price": ["open", "high", "low", "close"],
                                    "volume": ["volume"],
                                    "momentum": ["rsi", "rsi_signal"]}
            group_by: Column to group by before normalizing (default: "ticker").
            handle_outliers: Whether to replace inf/-inf values.
            fill_value: How to handle NA/inf values after normalization.

        Returns:
            Normalized DataFrame.
        """
        result = data.copy()

        # Process each feature group separately
        for category_name, feature_columns in feature_groups.items():
            logger.info(f"Normalizing {category_name} features: {feature_columns}")

            # Skip if any columns are missing
            if not all(col in data.columns for col in feature_columns):
                missing = [col for col in feature_columns if col not in data.columns]
                logger.warning(f"Skipping missing columns: {missing}")
                continue

            # Apply normalization to this feature group
            result = self.process(
                data=result, 
                columns=feature_columns, 
                group_by=group_by,
                handle_outliers=handle_outliers,
                fill_value=fill_value
            )

        return result

    def _zscore_normalize(
        self, data: pd.DataFrame, columns: List[str], group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """Perform z-score normalization on the input data.

        Standardizes features to have zero mean and unit variance.

        Args:
            data: Input DataFrame to normalize.
            columns: List of columns to normalize.
            group_by: Column to group by before normalizing (e.g., 'ticker').

        Returns:
            Normalized DataFrame.
        """

        # First, convert all columns to normalize to float dtype
        result = data.copy()
        for col in columns:
            result[col] = result[col].astype(float)

        if group_by:
            # Apply normalization separately for each group
            groups = result[group_by].unique()
            self.stats["zscore"] = {}

            for group in groups:
                group_mask = result[group_by] == group
                group_data = result.loc[group_mask, columns]

                # Calculate mean and std for each column
                means = group_data.mean()
                stds = group_data.std()

                # Replace zero std with 1 to avoid division by zero
                stds = stds.replace(0, 1)
                
                # Replace NaN std with 1 to avoid invalid operations
                stds = stds.fillna(1)

                # Store stats for potential future use
                self.stats["zscore"][group] = {"mean": means, "std": stds}

                # Apply normalization
                for col in columns:
                    # Calculate normalized values
                    normalized_values = (group_data[col] - means[col]) / stds[col]
                    # Assign to result DataFrame
                    result.loc[group_mask, col] = normalized_values
        else:
            # Apply normalization to the entire dataset
            means = result[columns].mean()
            stds = result[columns].std()

            # Replace zero std with 1 to avoid division by zero
            stds = stds.replace(0, 1)
            
            # Replace NaN std with 1 to avoid invalid operations
            stds = stds.fillna(1)

            # Store stats for potential future use
            self.stats["zscore"] = {"mean": means, "std": stds}

            # Apply normalization
            for col in columns:
                result[col] = (result[col] - means[col]) / stds[col]

        return result

    def _rolling_normalize(
        self,
        data: pd.DataFrame,
        columns: List[str],
        window: int = 20,
        group_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """Perform rolling window normalization on the input data.

        Normalizes data using a rolling window mean and standard deviation.
        This adapts to changing market regimes and reduces the impact of old data.

        Args:
            data: Input DataFrame to normalize.
            columns: List of columns to normalize.
            window: Size of rolling window (default: 20).
            group_by: Column to group by before normalizing (e.g., 'ticker').

        Returns:
            Normalized DataFrame.
        """
        result = data.copy()

        if group_by:
            # Apply rolling normalization separately for each group
            groups = result[group_by].unique()

            for group in groups:
                group_mask = result[group_by] == group

                # Ensure data is sorted (typically by date for time series)
                if "date" in result.columns:
                    group_data = result.loc[group_mask].sort_values("date")
                else:
                    group_data = result.loc[group_mask]

                group_indices = group_data.index

                for col in columns:
                    # Calculate rolling mean and std
                    rolling_mean = (
                        group_data[col].rolling(window=window, min_periods=1).mean()
                    )
                    rolling_std = (
                        group_data[col].rolling(window=window, min_periods=1).std()
                    )

                    # Replace zero or NaN in std with 1 to avoid division by zero
                    rolling_std = rolling_std.replace(0, 1).fillna(1)

                    # Apply normalization
                    result.loc[group_indices, col] = (
                        group_data[col] - rolling_mean
                    ) / rolling_std
        else:
            # Ensure data is sorted (typically by date for time series)
            if "date" in result.columns:
                result = result.sort_values("date")

            for col in columns:
                # Calculate rolling mean and std
                rolling_mean = result[col].rolling(window=window, min_periods=1).mean()
                rolling_std = result[col].rolling(window=window, min_periods=1).std()

                # Replace zero or NaN in std with 1 to avoid division by zero
                rolling_std = rolling_std.replace(0, 1).fillna(1)

                # Apply normalization
                result[col] = (result[col] - rolling_mean) / rolling_std

        return result

    def _percentage_normalize(
        self, data: pd.DataFrame, columns: List[str], group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """Perform percentage change normalization on the input data.

        Converts data to percentage changes, which makes different assets
        directly comparable regardless of their price levels.

        Args:
            data: Input DataFrame to normalize.
            columns: List of columns to normalize.
            group_by: Column to group by before normalizing (e.g., 'ticker').

        Returns:
            Normalized DataFrame.
        """
        result = data.copy()

        if group_by:
            # Apply percentage change separately for each group
            groups = result[group_by].unique()

            for group in groups:
                group_mask = result[group_by] == group

                # Ensure data is sorted (typically by date for time series)
                if "date" in result.columns:
                    group_data = result.loc[group_mask].sort_values("date")
                else:
                    group_data = result.loc[group_mask]

                group_indices = group_data.index

                for col in columns:
                    # Calculate percentage change
                    pct_change = group_data[col].pct_change(fill_method=None)

                    # Fill NaN values (first row) with zeros
                    pct_change = pct_change.fillna(0)

                    # Apply transformation
                    result.loc[group_indices, col] = pct_change
        else:
            # Ensure data is sorted (typically by date for time series)
            if "date" in result.columns:
                result = result.sort_values("date")

            for col in columns:
                # Calculate percentage change
                result[col] = result[col].pct_change(fill_method=None).fillna(0)

        return result

    def inverse_transform(
        self,
        normalized_data: pd.DataFrame,
        original_data: pd.DataFrame,
        columns: List[str],
        group_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """Inverse transform normalized data back to original scale.

        Note: This is only fully implemented for z-score normalization.
        For rolling and percentage change, this is more complex and might
        not be exact for all use cases.

        Args:
            normalized_data: The normalized DataFrame to inverse transform.
            original_data: The original DataFrame used for normalization.
            columns: List of columns to inverse transform.
            group_by: Column used for grouping during normalization.

        Returns:
            DataFrame with values returned to original scale.
        """
        result = normalized_data.copy()

        if self.method == "zscore" and "zscore" in self.stats:
            if group_by:
                for group, stats in self.stats["zscore"].items():
                    group_mask = result[group_by] == group
                    means = stats["mean"]
                    stds = stats["std"]

                    for col in columns:
                        if col in means and col in stds:
                            result.loc[group_mask, col] = (
                                normalized_data.loc[group_mask, col] * stds[col]
                                + means[col]
                            )
            else:
                means = self.stats["zscore"]["mean"]
                stds = self.stats["zscore"]["std"]

                for col in columns:
                    if col in means and col in stds:
                        result[col] = normalized_data[col] * stds[col] + means[col]

        elif self.method == "percentage":
            logger.warning(
                "Inverse transformation for percentage change is approximate"
            )
            # This requires the first values and cumulative product calculation
            # Implementation would depend on specific requirements

        elif self.method == "rolling":
            logger.warning(
                "Inverse transformation for rolling normalization is not fully supported"
            )
            # This would require storing all rolling statistics

        return result
