"""Data manager for financial reinforcement learning.

This module provides a central data management system that handles downloading,
processing, and preparing financial data for reinforcement learning.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Set
import hashlib
import json

import pandas as pd
import numpy as np

# Initialize the logger
from utils.logger import Logger
logger = Logger.get_logger()

# Internal imports
from data.sources.source import BaseSource
from data.sources.yahoo import YahooSource
from data.processors.technical_indicator import TechnicalIndicatorProcessor
from data.processors.turbulence import TurbulenceProcessor
from data.processors.vix import VIXProcessor
from data.processors.processor import BaseProcessor
from data.utility.feature_groups import create_feature_groups
from data.processors.normalization import NormalizeProcessor
from data.processors.index_adder import add_day_index

from config.data import NORMALIZATION_PARAMS

# Import visualization class (add this import)
try:
    from visualization.data_visualization import DataVisualization
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False
    logger.warning("DataVisualization class not found. Visualization features will be disabled.")


class DataManager:
    """Central manager for financial data used in reinforcement learning.

    The DataManager handles:
    1. Downloading data from various sources
    2. Processing with multiple processors (technical indicators, VIX, turbulence, etc.)
    3. Data splitting for training, validation, and testing
    4. Data augmentation to prevent overfitting
    5. Data saving and loading to improve performance
    """

    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        processed_data_dir: str = "data/processed",
        normalized_data_dir: str = "data/normalized",
        save_raw_data: bool = True,
        save_processed_data: bool = True,
        save_normalized_data: bool = True,
        use_saved_data: bool = True,
    ):
        """Initialize the data manager.

        Args:
            raw_data_dir: Directory to store raw data files
            processed_data_dir: Directory to store processed data files
            normalized_data_dir: Directory to store normalized data files
            save_raw_data: Whether to save raw data when downloaded
            save_processed_data: Whether to save processed data
            save_normalized_data: Whether to save normalized data
            use_saved_data: Whether to use previously saved data when available
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.normalized_data_dir = normalized_data_dir
        self.save_raw_data = save_raw_data
        self.save_processed_data = save_processed_data
        self.save_normalized_data = save_normalized_data
        self.use_saved_data = use_saved_data

        self._current_file_base = None

        # Create directories if they don't exist
        os.makedirs(raw_data_dir, exist_ok=True)
        os.makedirs(processed_data_dir, exist_ok=True)
        os.makedirs(normalized_data_dir, exist_ok=True)

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
        logger.info(f"  Raw data directory: {raw_data_dir}")
        logger.info(f"  Processed data directory: {processed_data_dir}")
        logger.info(f"  Normalized data directory: {normalized_data_dir}")
        logger.info(f"  Use saved data: {use_saved_data}")
        logger.info(f"  Save raw data: {save_raw_data}")
        logger.info(f"  Save processed data: {save_processed_data}")
        logger.info(f"  Save normalized data: {save_normalized_data}")
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

    def set_current_file_base(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        source: str = "yahoo",
        time_interval: str = "1d",
        processors: List[str] = None,
        params: Dict[str, Dict[str, Any]] = None,
    ) -> str:
        """Set the current file base name for consistent file naming.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            source: Name of the data source
            time_interval: Time interval for the data
            processors: List of processors applied
            params: Parameters used for processing
            
        Returns:
            The generated base filename
        """
        self._current_file_base = self._get_base_filename(
            tickers, start_date, end_date, source, time_interval, processors, params
        )
        logger.info(f"Set current file base to: {self._current_file_base}")
        return self._current_file_base

    def download_data(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        source: str = "yahoo",
        time_interval: str = "1d",
        force_download: bool = False,
        add_day_index_bool: bool = True,
        save_data: bool = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Download data from the specified source.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            source: Name of the data source to use
            time_interval: Time interval for the data
            force_download: Whether to force download even if saved data exists
            add_day_index_bool: Whether to add day index column
            save_data: Whether to save raw data. If None, uses self.save_raw_data
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

        # Use instance default if not specified
        if save_data is None:
            save_data = self.save_raw_data

        # Set the current file base if not already set
        if not hasattr(self, "_current_file_base") or self._current_file_base is None:
            self.set_current_file_base(tickers, start_date, end_date, source, time_interval)

        # Check if saved data exists
        raw_file = f"{self._current_file_base}_raw.csv"
        file_path = os.path.join(self.raw_data_dir, raw_file)

        if self.use_saved_data and not force_download and os.path.exists(file_path):
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

        if add_day_index_bool:
            df = add_day_index(df, "date")

        # Save raw data if requested
        if save_data:
            logger.info(f"Saving raw data to file path: {file_path}")
            df.to_csv(file_path, index=False)

        return df

    def process_data(
        self,
        data: pd.DataFrame,
        processors: Optional[List[str]] = None,
        processor_params: Optional[Dict[str, Dict[str, Any]]] = None,
        save_data: bool = None,
        add_day_index_bool: bool = True,
    ) -> pd.DataFrame:
        """Process data using specified processors.

        Args:
            data: DataFrame containing the data to process
            processors: List of processor names to apply. If None, applies all processors.
            processor_params: Parameters for each processor, keyed by processor name
            save_data: Whether to save processed data. If None, uses self.save_processed_data
            add_day_index_bool: Whether to add day index column

        Returns:
            DataFrame with processed data

        Raises:
            ValueError: If a specified processor doesn't exist
        """
        # Use instance default if not specified
        if save_data is None:
            save_data = self.save_processed_data

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
                result = processor.process(result, params)
            elif processor_name == "turbulence" and len(result["ticker"].unique()) < 2:
                logger.warning(
                    "Skipping turbulence processor due to single ticker data"
                )
            else:
                # Generic processor interface
                result = processor.process(result, **params)

        if add_day_index_bool:
            result = add_day_index(result, "date")

        # Save processed data if requested
        if save_data:
            # If _current_file_base is not set, set it based on the data
            if not hasattr(self, "_current_file_base") or self._current_file_base is None:
                tickers = sorted(result["ticker"].unique())
                min_date = result["date"].min()
                max_date = result["date"].max()
                self.set_current_file_base(
                    tickers, 
                    min_date, 
                    max_date, 
                    "auto", 
                    "unknown", 
                    processors
                )
                
            # Generate a file name based on base file name
            proc_filename = f"{self._current_file_base}_processed.csv"
            proc_path = os.path.join(self.processed_data_dir, proc_filename)
            
            logger.info(f"Saving processed data to: {proc_path}")
            result.to_csv(proc_path, index=False)

        return result

    def normalize_data(
        self,
        data: pd.DataFrame,
        method: str = NORMALIZATION_PARAMS["method"],
        feature_groups: Optional[Dict[str, List[str]]] = None,
        group_by: Optional[str] = "ticker",
        exclude_columns: Optional[Set[str]] = None,
        save_data: bool = None,
        add_day_index_bool: bool = True,
        handle_outliers: bool = True,
        fill_value: Union[float, str, None] = 0,
    ) -> pd.DataFrame:
        """Normalize data using the specified method and feature groups.

        Args:
            data: DataFrame to normalize
            method: Normalization method ('zscore', 'rolling', or 'percentage')
            feature_groups: Dictionary mapping feature categories to column lists
                        If None, will auto-generate using create_feature_groups
            group_by: Column to group by before normalizing (default: "ticker")
            exclude_columns: Set of columns to exclude from normalization
            save_data: Whether to save the normalized data. If None, uses self.save_normalized_data
            add_day_index_bool: Whether to add day index column
            handle_outliers: Whether to replace inf/-inf values
            fill_value: How to handle NA/inf values after normalization

        Returns:
            Normalized DataFrame
        """
        # Use instance default if not specified
        if save_data is None:
            save_data = self.save_normalized_data

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

        # Create a normalizer with the specified method
        normalizer = NormalizeProcessor(method=method)

        # Initialize result DataFrame
        result = data.copy()

        # Process each feature group separately
        for group_name, columns in feature_groups.items():
            if not columns:  # Skip empty groups
                continue

            logger.info(f"Applying {method} normalization to {group_name} features: {columns}")
            result = normalizer.process(
                data=result, 
                columns=columns, 
                group_by=group_by,
                handle_outliers=handle_outliers,
                fill_value=fill_value
            )

        # Generate normalized data path if requested to save
        if save_data:
            # If _current_file_base is not set, set it based on the data
            if not hasattr(self, "_current_file_base") or self._current_file_base is None:
                tickers = sorted(result["ticker"].unique())
                min_date = result["date"].min()
                max_date = result["date"].max()
                self.set_current_file_base(tickers, min_date, max_date, "auto", "unknown")
            
            # Add normalization method to filename
            norm_filename = f"{self._current_file_base}_normalized_{method}.csv"

            # Full path
            norm_path = os.path.join(self.normalized_data_dir, norm_filename)

            # Save to file
            logger.info(f"Saving normalized data to: {norm_path}")
            result.to_csv(norm_path, index=False)

        if add_day_index_bool:
            result = add_day_index(result, "date")

        return result
    
    def simple_normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data using the specified method and feature groups.

        Args:
            data: DataFrame to normalize

        Returns:
            Normalized DataFrame
        """
        # Copy data
        result = data.copy()

        # Exclude columns
        exclude_columns = {"date", "ticker", "symbol", "day", "day_index", "time_index", "timestamp"}

        # OHLCV data should be percentage, grouped by ticker
        ohlcv_columns = ["open", "high", "low", "close", "volume"]
        for col in ohlcv_columns:
            result[col] = result.groupby("ticker")[col].pct_change().fillna(0)

        # Technical indicators should be zscore (not grouped by ticker)
        non_ohlcv_columns = set(col for col in result.columns if col not in ohlcv_columns)
        technical_columns = list(non_ohlcv_columns - exclude_columns)
        
        for col in technical_columns:
            if col.startswith("linreg"):
                result[col] = result[col].fillna(0)
            else:
                mean_val = result[col].mean()
                std_val = result[col].std()
                result[col] = (result[col] - mean_val) / std_val
                result[col] = result[col].fillna(0)

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
        use_saved_data: bool = None,
        add_day_index_bool: bool = True,
        handle_outliers: bool = True,
        fill_value: Union[float, str, None] = 0,
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
            force_download: Whether to force download even if saved data exists
            use_saved_data: Whether to use previously saved data. If None, uses self.use_saved_data
            add_day_index_bool: Whether to add day index column
            handle_outliers: Whether to replace inf/-inf values in normalization
            fill_value: How to handle NA/inf values in normalization

        Returns:
            DataFrame with downloaded and processed data
        """
        # Use instance default if not specified
        if use_saved_data is None:
            use_saved_data = self.use_saved_data

        # Set the current file base before any operations
        self.set_current_file_base(
            tickers,
            start_date,
            end_date,
            source,
            time_interval,
            processors=processors,
            params=processor_params,
        )

        # Check if normalized saved data exists first
        if normalize and use_saved_data and not force_download:
            norm_filename = f"{self._current_file_base}_normalized_{normalize_method}.csv"
            norm_path = os.path.join(self.normalized_data_dir, norm_filename)

            if os.path.exists(norm_path):
                logger.info(f"Loading saved normalized data from {norm_path}")
                data = pd.read_csv(norm_path, parse_dates=["date"])

                if add_day_index_bool:
                    data = add_day_index(data, "date")

                return data

        # Check for processed data file
        proc_filename = f"{self._current_file_base}_processed.csv"
        proc_path = os.path.join(self.processed_data_dir, proc_filename)

        if use_saved_data and not force_download and os.path.exists(proc_path):
            logger.info(f"Loading saved processed data from {proc_path}")
            data = pd.read_csv(proc_path, parse_dates=["date"])

            # Normalize if requested and no normalized data exists
            if normalize:
                logger.info(f"Normalizing saved data with method: {normalize_method}")
                data = self.normalize_data(
                    data=data, 
                    method=normalize_method, 
                    save_data=self.save_normalized_data,
                    handle_outliers=handle_outliers,
                    fill_value=fill_value
                )

            if add_day_index_bool:
                data = add_day_index(data, "date")

            return data

        # Download data
        data = self.download_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            source=source,
            time_interval=time_interval,
            force_download=force_download,
            add_day_index_bool=add_day_index_bool,
        )

        # Process data
        if processors:
            data = self.process_data(
                data, 
                processors, 
                processor_params,
                save_data=self.save_processed_data
            )

        # Normalize if requested
        if normalize:
            logger.info(f"Normalizing data with method: {normalize_method}")
            return self.normalize_data(
                data=data, 
                method=normalize_method, 
                save_data=self.save_normalized_data,
                handle_outliers=handle_outliers,
                fill_value=fill_value
            )

        return data

    def split_data(
        self,
        data: pd.DataFrame,
        train_start_date: Union[str, datetime],
        train_end_date: Union[str, datetime],
        test_start_date: Union[str, datetime] = None,
        test_end_date: Union[str, datetime] = None,
        trade_start_date: Union[str, datetime] = None,
        trade_end_date: Union[str, datetime] = None,
        date_column: str = "date",
        strict_chronological: bool = False,  # Set to False for more flexibility
        reset_day_index: bool = True  # New parameter to control day index reset
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Split data into training, validation, and test sets.

        Args:
            data: DataFrame to split
            train_start_date: Start date for training data
            train_end_date: End date for training data
            test_start_date: Start date for test data (validation)
            test_end_date: End date for test data (validation)
            trade_start_date: Start date for trading data (optional)
            trade_end_date: End date for trading data (optional)
            date_column: Name of the date column in the DataFrame (default: "date")
            strict_chronological: If True, enforces strict chronological ordering without overlaps
            reset_day_index: If True, resets the day index to start from 1 for each split

        Returns:
            If trade dates are provided: Tuple of (train_data, test_data, trade_data)
            Otherwise: Tuple of (train_data, test_data)

        Raises:
            ValueError: If the split dates are invalid or out of range
        """
        # Ensure data is sorted by date
        data = data.sort_values(by=date_column).copy()

        # Convert string dates to datetime objects if necessary
        if isinstance(train_start_date, str):
            train_start_date = datetime.strptime(train_start_date, "%Y-%m-%d")
        if isinstance(train_end_date, str):
            train_end_date = datetime.strptime(train_end_date, "%Y-%m-%d")
        
        # Check if we're doing a two-way split (train/test only)
        two_way_split = (trade_start_date is None and trade_end_date is None)
        
        # Convert test dates if provided
        if test_start_date is not None:
            if isinstance(test_start_date, str):
                test_start_date = datetime.strptime(test_start_date, "%Y-%m-%d")
        if test_end_date is not None:
            if isinstance(test_end_date, str):
                test_end_date = datetime.strptime(test_end_date, "%Y-%m-%d")
        
        # Convert trade dates if provided for three-way split
        if not two_way_split:
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
            logger.warning(
                f"Training start date {train_start_date} is before the earliest available date {min_date}. "
                f"Using {min_date} instead."
            )
            train_start_date = min_date
            
        # Check end dates against max available date
        end_dates_to_check = [('Training', train_end_date)]
        if test_end_date is not None:
            end_dates_to_check.append(('Test', test_end_date))
        if not two_way_split:
            end_dates_to_check.append(('Trade', trade_end_date))
            
        for label, end_date in end_dates_to_check:
            if end_date > max_date:
                logger.warning(
                    f"{label} end date {end_date} is after the latest available date {max_date}. "
                    f"Using {max_date} instead."
                )
                if label == 'Training':
                    train_end_date = max_date
                elif label == 'Test':
                    test_end_date = max_date
                else:  # Trade
                    trade_end_date = max_date

        # Validate date order for individual periods
        if train_start_date >= train_end_date:
            raise ValueError("Training start date must be before end date")
            
        if test_start_date is not None and test_end_date is not None:
            if test_start_date >= test_end_date:
                raise ValueError("Test start date must be before end date")
                
        if not two_way_split:
            if trade_start_date >= trade_end_date:
                raise ValueError("Trade start date must be before end date")

        # If strict chronological ordering is required, check for overlapping dates
        if strict_chronological:
            if test_start_date is not None and test_end_date is not None:
                if train_end_date >= test_start_date:
                    raise ValueError("Training and test dates cannot overlap in strict chronological mode")
                    
            if not two_way_split:
                if test_end_date >= trade_start_date:
                    raise ValueError("Test and trade dates cannot overlap in strict chronological mode")
                
                # Validate full chronological order
                if not (
                    train_start_date < train_end_date < test_start_date < test_end_date < trade_start_date < trade_end_date
                ):
                    raise ValueError(
                        "In strict chronological mode, dates must be in order: train -> test -> trade"
                    )

        # Split the data
        train_data = data[
            (data[date_column] >= train_start_date)
            & (data[date_column] <= train_end_date)
        ].copy()
        
        # Process test data if test dates are provided
        if test_start_date is not None and test_end_date is not None:
            test_data = data[
                (data[date_column] >= test_start_date)
                & (data[date_column] <= test_end_date)
            ].copy()
            
            # Check for empty datasets
            if len(test_data) == 0:
                logger.warning("Test data is empty. Check your date ranges.")
                # Use the last 20% of training data as a fallback
                train_len = len(train_data)
                split_idx = int(train_len * 0.8)
                train_data, test_data = train_data.iloc[:split_idx].copy(), train_data.iloc[split_idx:].copy()
                logger.info(f"Using last 20% of training data as test data: {len(test_data)} rows")
        else:
            # If no test dates provided, use last 20% of training data as test data
            train_len = len(train_data)
            split_idx = int(train_len * 0.8)
            train_data, test_data = train_data.iloc[:split_idx].copy(), train_data.iloc[split_idx:].copy()
            logger.info(f"No test dates provided. Using last 20% of training data: {len(test_data)} rows")
        
        # Check for empty training data
        if len(train_data) == 0:
            raise ValueError("Training data is empty. Check your date ranges.")
        
        # Reset day indices if requested
        if reset_day_index and 'day' in train_data.columns:
            logger.info("Resetting day indices for each split to start from 1")
            
            # Reset day indices for train_data
            train_data = self._reset_day_index(train_data)
            
            # Reset day indices for test_data
            test_data = self._reset_day_index(test_data)
            
        # Handle three-way split if trade dates are provided
        if not two_way_split:
            trade_data = data[
                (data[date_column] >= trade_start_date)
                & (data[date_column] <= trade_end_date)
            ].copy()
            
            # Check for empty trade data
            if len(trade_data) == 0:
                logger.warning("Trade data is empty. Check your date ranges.")
                # Return two-way split as fallback
                logger.info(f"Data split: Training: {len(train_data)} rows, Test: {len(test_data)} rows")
                return train_data, test_data
            
            # Reset day indices for trade_data if requested
            if reset_day_index and 'day' in trade_data.columns:
                trade_data = self._reset_day_index(trade_data)
                
            # Return three-way split
            logger.info(
                f"Data split: Training: {len(train_data)} rows, Test: {len(test_data)} rows, Trading: {len(trade_data)} rows"
            )
            return train_data, test_data, trade_data
        
        # Return two-way split
        logger.info(f"Data split: Training: {len(train_data)} rows, Test: {len(test_data)} rows")
        return train_data, test_data

    def _reset_day_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reset the day index in a DataFrame to start from 1.
        
        Args:
            df: DataFrame with a 'day' column
            
        Returns:
            DataFrame with reset day index
        """
        if 'day' not in df.columns:
            logger.warning("No 'day' column found to reset")
            return df
            
        # Get unique sorted day values
        unique_days = sorted(df['day'].unique())
        
        # Create mapping from old to new day values
        day_mapping = {old_day: new_day for new_day, old_day in enumerate(unique_days, 1)}
        
        # Apply mapping to create new day values
        df['day'] = df['day'].map(day_mapping)
        
        logger.info(f"Reset day index from {min(unique_days)}-{max(unique_days)} to 1-{len(unique_days)}")
        return df

    def _get_base_filename(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        source: str,
        time_interval: str,
        processors: List[str] = None,
        params: Dict[str, Dict[str, Any]] = None,
    ) -> str:
        """Generate a base filename for data without extension.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            source: Name of the data source
            time_interval: Time interval for the data
            processors: List of processors applied
            params: Parameters used for processing

        Returns:
            Base filename without extension
        """
        # Convert dates to strings if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%d")

        # Sort tickers for consistent filenames
        tickers_str = "-".join(sorted(tickers))

        # Base filename
        filename = f"{source}_{tickers_str}_{start_date}_{end_date}_{time_interval}"

        # Add processor info if processors are specified
        if processors:
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

        return filename

    def _get_filename(
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
        """Generate a filename with extension for specific data type.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            source: Name of the data source
            time_interval: Time interval for the data
            data_type: Type of data ('raw', 'processed', or 'normalized')
            processors: List of processors applied
            params: Parameters used for processing

        Returns:
            Filename with extension
        """
        # Get base filename
        base_filename = self._get_base_filename(
            tickers, start_date, end_date, source, time_interval, processors, params
        )
        
        # Add data type and extension
        return f"{base_filename}_{data_type}.csv"

    def visualize_data(
        self,
        raw_data: pd.DataFrame = None,
        processed_data: pd.DataFrame = None,
        normalized_data: pd.DataFrame = None,
        columns: List[str] = None,
        time_series_columns: List[str] = None,
        tickers: List[str] = None,
        save_dir: str = "visualizations",
        output_prefix: str = ""
    ) -> None:
        """Visualize the data using the DataVisualization class.
        
        This is a convenience method to generate visualizations for the data.
        It will create distribution plots, time series plots, correlation matrices,
        and statistical summaries.
        
        Args:
            raw_data: Raw data DataFrame. If None, will try to load from file.
            processed_data: Processed data DataFrame. If None, will try to load from file.
            normalized_data: Normalized data DataFrame. If None, will try to load from file.
            columns: Specific columns to visualize. If None, uses all numeric columns.
            time_series_columns: Columns to use for time series visualization.
            tickers: List of tickers to include. If None, includes all.
            save_dir: Directory to save visualizations.
            output_prefix: Prefix for output filenames.
            
        Returns:
            None
        """
        if not _HAS_VISUALIZATION:
            logger.error("DataVisualization class not available. Cannot visualize data.")
            return
            
        # Create visualization object
        visualizer = DataVisualization(save_dir=save_dir)
        
        # Generate timestamp for consistent filenames if no prefix provided
        if not output_prefix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"analysis_{timestamp}"
            
        # Run all visualizations
        logger.info("Generating visualizations...")
        visualizer.visualize_all(
            raw_data=raw_data,
            processed_data=processed_data,
            normalized_data=normalized_data,
            columns=columns,
            time_series_columns=time_series_columns,
            tickers=tickers,
            output_prefix=output_prefix
        )
        
        logger.info(f"All visualizations completed and saved to {save_dir}")
