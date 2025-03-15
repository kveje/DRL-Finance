"""Data manager for financial reinforcement learning.

This module provides a central data management system that handles downloading,
processing, and preparing financial data for reinforcement learning.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Set
import hashlib
import json

import pandas as pd
import numpy as np

from data.sources.source import BaseSource
from data.sources.yahoo import YahooSource
from data.processors.technical_indicator import TechnicalIndicatorProcessor
from data.processors.turbulence import TurbulenceProcessor
from data.processors.vix import VIXProcessor
from data.processors.processor import BaseProcessor
from data.utility.feature_groups import create_feature_groups
from data.processors.normalization import NormalizeProcessor

from config.data import NORMALIZATION_PARAMS

logger = logging.getLogger(__name__)


class DataManager:
    """Central manager for financial data used in reinforcement learning.

    The DataManager handles:
    1. Downloading data from various sources
    2. Processing with multiple processors (technical indicators, VIX, turbulence, etc.)
    3. Data splitting for training, validation, and testing
    4. Data augmentation to prevent overfitting
    5. Data caching to improve performance
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        cache_dir: str = "data/processed",
        save_raw_data: bool = True,
        use_cache: bool = True,
    ):
        """Initialize the data manager.

        Args:
            data_dir: Directory to store raw data files
            cache_dir: Directory to store processed data files
            use_cache: Whether to use cached data when available
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.save_raw_data = save_raw_data
        self.use_cache = use_cache

        self._current_cache_base = None

        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize default sources
        self.sources = {"yahoo": YahooSource()}

        # Initialize default processors
        self.processors = {
            "technical_indicator": TechnicalIndicatorProcessor(),
            "turbulence": TurbulenceProcessor(),
            "vix": VIXProcessor(),
        }

        # Store loaded datasets
        self.datasets = {}

        logger.info("Data manager initialized")
        logger.info(f"  Data directory: {data_dir}")
        logger.info(f"  Cache directory: {cache_dir}")
        logger.info(f"  Use cache: {use_cache}")
        logger.info(f"  Save raw data: {save_raw_data}")
        logger.info(f"  Available sources: {list(self.sources.keys())}")
        logger.info(f"  Available processors: {list(self.processors.keys())}")

    def add_source(self, name: str, source: BaseSource) -> None:
        """Add a new data source.

        Args:
            name: Name to identify the source
            source: Data source instance
        """
        self.sources[name] = source
        logger.info(f"Added data source: {name}")

    def add_processor(self, name: str, processor: Any) -> None:
        """Add a new data processor.

        Args:
            name: Name to identify the processor
            processor: Data processor instance
        """
        self.processors[name] = processor
        logger.info(f"Added data processor: {name}")

    def download_data(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        source: str = "yahoo",
        time_interval: str = "1d",
        force_download: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Download data from the specified source.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            source: Name of the data source to use
            time_interval: Time interval for the data
            force_download: Whether to force download even if cached data exists
            **kwargs: Additional arguments to pass to the data source

        Returns:
            DataFrame containing the downloaded data

        Raises:
            ValueError: If the specified source doesn't exist
        """
        if source not in self.sources:
            raise ValueError(
                f"Source {source} not found. Available sources: {list(self.sources.keys())}"
            )

        # Check if cached data exists
        raw_file = self._get_cache_filename(
            tickers, start_date, end_date, source, time_interval, "raw"
        )
        file_path = os.path.join(self.data_dir, raw_file)

        if self.save_raw_data and not force_download and os.path.exists(file_path):
            logger.info(f"Loading saved raw data from {file_path}")
            return pd.read_csv(file_path, parse_dates=["date"])

        # Download data using the specified source
        logger.info(
            f"Downloading data for {len(tickers)} tickers using {source} source"
        )
        df = self.sources[source].download_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
            **kwargs,
        )

        # Save to cache
        logger.info(f"Saving raw data to file path: {file_path}")
        df.to_csv(file_path, index=False)

        return df

    def process_data(
        self,
        data: pd.DataFrame,
        processors: List[str] = None,
        processor_params: Dict[str, Dict[str, Any]] = None,
        save_cache: bool = True,
    ) -> pd.DataFrame:
        """Process data using specified processors.

        Args:
            data: DataFrame containing the data to process
            processors: List of processor names to apply. If None, applies all processors.
            processor_params: Parameters for each processor, keyed by processor name

        Returns:
            DataFrame with processed data

        Raises:
            ValueError: If a specified processor doesn't exist
        """
        if processors is None:
            processors = list(self.processors.keys())

        processor_params = processor_params or {}
        result = data.copy()

        for processor_name in processors:
            if processor_name not in self.processors:
                raise ValueError(
                    f"Processor {processor_name} not found. Available processors: {list(self.processors.keys())}"
                )

            # Get processor-specific parameters
            params = processor_params.get(processor_name, {})

            # Apply processor
            logger.info(f"Applying processor: {processor_name}")
            processor: BaseProcessor = self.processors[processor_name]

            # Handle different processor types
            if processor_name == "technical_indicator":
                indicators = params.pop("indicators", None)
                indicator_params = params.pop("params", None)
                result = processor.process(
                    result, indicators=indicators, params=indicator_params, **params
                )
            elif processor_name == "turbulence" and len(result["ticker"].unique()) < 2:
                logger.warning(
                    "Skipping turbulence processor due to single ticker data"
                )
            else:
                # Generic processor interface
                result = processor.process(result, **params)

        return result

    def normalize_data(
        self,
        data: pd.DataFrame,
        method: str = NORMALIZATION_PARAMS["method"],
        feature_groups: Optional[Dict[str, List[str]]] = None,
        group_by: Optional[str] = "ticker",
        exclude_columns: Optional[Set[str]] = None,
        save_cache: bool = True,
    ) -> pd.DataFrame:
        """Normalize data using the specified method and feature groups.

        Args:
            data: DataFrame to normalize
            method: Normalization method ('zscore', 'rolling', or 'percentage')
            feature_groups: Dictionary mapping feature categories to column lists
                        If None, will auto-generate using create_feature_groups
            group_by: Column to group by before normalizing (default: "ticker")
            exclude_columns: Set of columns to exclude from normalization
            save_cache: Whether to save the normalized data to cache

        Returns:
            Normalized DataFrame
        """
        # Columns that should never be normalized
        default_exclude = {"date", "ticker", "symbol"}
        if exclude_columns:
            exclude_columns = exclude_columns.union(default_exclude)
        else:
            exclude_columns = default_exclude

        # Create feature groups if not provided
        if feature_groups is None:
            logger.info("Auto-generating feature groups from columns")
            feature_groups = create_feature_groups(data.columns.tolist())

            # Remove excluded columns from feature groups
            for group, columns in feature_groups.items():
                feature_groups[group] = [
                    col for col in columns if col not in exclude_columns
                ]

        # Log the feature groups
        for group, columns in feature_groups.items():
            logger.info(f"Normalizing {group} features: {columns}")

        # Create a normalizer with the specified method
        normalizer = NormalizeProcessor(method=method)

        # Initialize result DataFrame
        result = data.copy()

        # Process each feature group separately
        for group_name, columns in feature_groups.items():
            if not columns:  # Skip empty groups
                continue

            logger.info(f"Applying {method} normalization to {group_name} features")
            result = normalizer.process(data=result, columns=columns, group_by=group_by)

        # Generate cache path if requested
        if save_cache and hasattr(self, "cache_dir"):
            # Generate a base cache filename
            if hasattr(self, "_current_cache_base") and self._current_cache_base:
                # Use existing base if preparing data
                base_filename = self._current_cache_base
            else:
                # Create a generic name based on data shape
                tickers_str = "-".join(sorted(data["ticker"].unique()))
                min_date = data["date"].min().strftime("%Y-%m-%d")
                max_date = data["date"].max().strftime("%Y-%m-%d")
                base_filename = f"norm_{tickers_str}_{min_date}_{max_date}"

            # Add normalization method to filename
            norm_filename = f"{base_filename}_norm_{method}"

            # Append file extension
            norm_filename = f"{norm_filename}.csv"

            # Full path
            cache_path = os.path.join(self.cache_dir, norm_filename)

            # Save to cache
            logger.info(f"Saving normalized data to cache: {cache_path}")
            result.to_csv(cache_path, index=False)

        return result

    def prepare_data(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        source: str = "yahoo",
        time_interval: str = "1d",
        processors: List[str] = None,
        processor_params: Dict[str, Dict[str, Any]] = None,
        normalize: bool = True,
        normalize_method: str = NORMALIZATION_PARAMS["method"],
        force_download: bool = False,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Download and process data in one step.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            source: Name of the data source to use
            time_interval: Time interval for the data
            processors: List of processor names to apply
            processor_params: Parameters for each processor
            normalize: Whether to normalize the data
            normalize_method: Method to use for normalization ('zscore', 'rolling', or 'percentage')
            force_download: Whether to force download even if cached data exists
            use_cache: Whether to use cached processed data

        Returns:
            DataFrame with downloaded and processed data
        """
        # Store base cache filename for potential use by normalize_data
        cache_file = self._get_cache_filename(
            tickers,
            start_date,
            end_date,
            source,
            time_interval,
            "processed",
            processors=processors,
            params=processor_params,
        )

        # Store without file extension
        self._current_cache_base = os.path.splitext(cache_file)[0]

        # Check if normalized cached data exists first
        if normalize and use_cache and not force_download:
            norm_cache_file = f"{self._current_cache_base}_norm_{normalize_method}.csv"
            norm_cache_path = os.path.join(self.cache_dir, norm_cache_file)

            if os.path.exists(norm_cache_path):
                logger.info(f"Loading cached normalized data from {norm_cache_path}")
                return pd.read_csv(norm_cache_path, parse_dates=["date"])

        # Check for regular processed cache
        cache_path = os.path.join(self.cache_dir, cache_file)

        if use_cache and not force_download and os.path.exists(cache_path):
            logger.info(f"Loading cached processed data from {cache_path}")
            data = pd.read_csv(cache_path, parse_dates=["date"])

            # Normalize if requested and no normalized cache exists
            if normalize:
                logger.info(f"Normalizing cached data with method: {normalize_method}")
                return self.normalize_data(
                    data=data, method=normalize_method, save_cache=True
                )
            return data

        # Download data
        data = self.download_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            source=source,
            time_interval=time_interval,
            force_download=force_download,
        )

        # Process data
        if processors:
            data = self.process_data(data, processors, processor_params)

            # Save processed data to cache
            logger.info(f"Saving processed data to cache: {cache_path}")
            data.to_csv(cache_path, index=False)

        # Normalize if requested
        if normalize:
            logger.info(f"Normalizing data with method: {normalize_method}")
            return self.normalize_data(
                data=data, method=normalize_method, save_cache=True
            )

        return data

    def split_data(
        self,
        data: pd.DataFrame,
        train_start_date: Union[str, datetime],
        train_end_date: Union[str, datetime],
        test_start_date: Union[str, datetime],
        test_end_date: Union[str, datetime],
        trade_start_date: Union[str, datetime],
        trade_end_date: Union[str, datetime],
        date_column: str = "date",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into training, validation, and test sets.

        Args:
            data: DataFrame to split
            train_start_date: Start date for training data
            train_end_date: End date for training data
            test_start_date: Start date for test data
            test_end_date: End date for test data
            trade_start_date: Start date for trading data
            trade_end_date: End date for trading data
            date_column: Name of the date column in the DataFrame (default: "date")

        Returns:
            Tuple of (train_data, test_data, trade_data)

        Raises:
            ValueError: If the split dates are invalid, out of range, or overlapping
        """
        # Ensure data is sorted by date
        data = data.sort_values(by=date_column).copy()

        # Convert string dates to datetime objects if necessary
        if isinstance(train_start_date, str):
            train_start_date = datetime.strptime(train_start_date, "%Y-%m-%d")
        if isinstance(train_end_date, str):
            train_end_date = datetime.strptime(train_end_date, "%Y-%m-%d")
        if isinstance(test_start_date, str):
            test_start_date = datetime.strptime(test_start_date, "%Y-%m-%d")
        if isinstance(test_end_date, str):
            test_end_date = datetime.strptime(test_end_date, "%Y-%m-%d")
        if isinstance(trade_start_date, str):
            trade_start_date = datetime.strptime(trade_start_date, "%Y-%m-%d")
        if isinstance(trade_end_date, str):
            trade_end_date = datetime.strptime(trade_end_date, "%Y-%m-%d")

        # Check if date column exists
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")

        # Convert the date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            try:
                data[date_column] = pd.to_datetime(data[date_column])
            except Exception as e:
                raise ValueError(f"Failed to convert {date_column} to datetime: {e}")

        # Get min and max dates from the data
        min_date = data[date_column].min()
        max_date = data[date_column].max()

        # Check if provided dates are within the data range
        if train_start_date < min_date:
            raise ValueError(
                f"Training start date {train_start_date} is before the earliest available date {min_date}"
            )
        if trade_end_date > max_date:
            raise ValueError(
                f"Trade end date {trade_end_date} is after the latest available date {max_date}"
            )

        # Validate date order
        if train_start_date >= train_end_date:
            raise ValueError("Training start date must be before end date")
        if test_start_date >= test_end_date:
            raise ValueError("Test start date must be before end date")
        if trade_start_date >= trade_end_date:
            raise ValueError("Trade start date must be before end date")

        # Check for overlapping dates
        if train_end_date >= test_start_date:
            raise ValueError("Training and test dates cannot overlap")
        if test_end_date >= trade_start_date:
            raise ValueError("Test and trade dates cannot overlap")

        # Validate chronological order of splits
        if not (
            train_start_date
            < train_end_date
            < test_start_date
            < test_end_date
            < trade_start_date
            < trade_end_date
        ):
            raise ValueError(
                "Dates must be in chronological order: train -> test -> trade"
            )

        # Split the data
        train_data = data[
            (data[date_column] >= train_start_date)
            & (data[date_column] <= train_end_date)
        ]
        test_data = data[
            (data[date_column] >= test_start_date)
            & (data[date_column] <= test_end_date)
        ]
        trade_data = data[
            (data[date_column] >= trade_start_date)
            & (data[date_column] <= trade_end_date)
        ]

        # Check for empty datasets
        if len(train_data) == 0:
            raise ValueError("Training data is empty. Check your date ranges.")
        if len(test_data) == 0:
            raise ValueError("Test data is empty. Check your date ranges.")
        if len(trade_data) == 0:
            raise ValueError("Trade data is empty. Check your date ranges.")

        # Log the split sizes
        logger.info(
            f"Data split: Training: {len(train_data)} rows, Test: {len(test_data)} rows, Trading: {len(trade_data)} rows"
        )

        return train_data, test_data, trade_data

    def _get_cache_filename(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        source: str,
        time_interval: str,
        data_type: str,
        processors: List[str] = None,
        params: Dict[str, Dict[str, Any]] = None,
    ) -> str:
        """Generate a cache filename based on data parameters.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            source: Name of the data source
            time_interval: Time interval for the data
            data_type: Type of data ('raw' or 'processed')
            processors: List of processors applied (for processed data)
            params: Parameters used for processing

        Returns:
            Cache filename
        """
        # Convert dates to strings if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%d")

        # Sort tickers for consistent filenames
        tickers_str = "-".join(sorted(tickers))

        # Base filename
        filename = f"{source}_{tickers_str}_{start_date}_{end_date}_{time_interval}_{data_type}"

        # Add processor info for processed data
        if data_type == "processed" and processors:
            # Sort processors for consistent filenames
            processors_str = "-".join(sorted(processors))
            filename = f"{filename}_{processors_str}"

        # Add hash of params if present
        if params:
            # Convert params to a stable string representation using JSON
            # Sort keys for consistency
            params_str = json.dumps(params, sort_keys=True)

            # Use a deterministic hashing algorithm (md5, sha1, sha256, etc.)
            hash_obj = hashlib.md5(params_str.encode())
            params_hash = hash_obj.hexdigest()[:8]  # Take first 8 chars of hash

            filename = f"{filename}_{params_hash}"

        return f"{filename}.csv"
