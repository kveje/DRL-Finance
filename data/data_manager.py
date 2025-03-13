"""Data management module for FinRL framework."""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import yaml

from data.market.market_data import MarketData
from data.market.universe import Universe
from data.processors.feature_engineering import FeatureEngineer
from data.processors.normalization import Normalizer
from data.processors.universe_selection import UniverseSelector
from data.sources.finrl_source import FinRLSource

logger = logging.getLogger(__name__)


class DataManager:
    """Unified data management class for FinRL.
    
    This class coordinates data sources, processing, and storage for
    financial reinforcement learning applications.
    """
    
    def __init__(
        self,
        data_source: FinRLSource,
        data_dir: str = "data_storage",
        config_path: Optional[str] = None
    ):
        """Initialize data manager.
        
        Args:
            data_source: Data source implementation.
            data_dir: Directory for data storage.
            config_path: Path to configuration file.
        """
        self.data_source = data_source
        self.data_dir = data_dir
        self.config = self._load_config(config_path)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize processors
        self.feature_engineer = FeatureEngineer()
        self.normalizer = Normalizer(method=self.config.get("normalization_method", "minmax"))
        self.universe_selector = UniverseSelector(
            selection_method=self.config.get("universe_selection", "market_cap")
        )
        
        # Cache for loaded data
        self._data_cache: Dict[str, MarketData] = {}
        self._universe_cache: Dict[str, Universe] = {}
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file.
            
        Returns:
            Dictionary with configuration.
        """
        default_config = {
            "cache_data": True,
            "normalization_method": "minmax",
            "universe_selection": "market_cap",
            "default_features": [
                "macd", "rsi_14", "cci_30", "dx_30", "close_20_sma", "close_50_sma"
            ],
            "default_market": "NYSE"
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as file:
                    loaded_config = yaml.safe_load(file)
                return {**default_config, **loaded_config}
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        return default_config
    
    def download_data(
        self,
        ticker_list: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        time_interval: str = "1d",
        include_indicators: bool = True,
        save_to_disk: bool = True,
        dataset_name: Optional[str] = None
    ) -> MarketData:
        """Download and process market data.
        
        Args:
            ticker_list: List of ticker symbols to download.
            start_date: Start date for data.
            end_date: End date for data.
            time_interval: Time interval for data.
            include_indicators: Whether to include technical indicators.
            save_to_disk: Whether to save data to disk.
            dataset_name: Name for the dataset.
            
        Returns:
            MarketData object with downloaded data.
        """
        logger.info(f"Downloading data for {len(ticker_list)} tickers from {start_date} to {end_date}")
        
        # Download raw data
        df = self.data_source.download_data(
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval
        )
        
        # Clean data
        df = self.data_source.clean_data(df)
        
        # Create market data object
        market_data = MarketData(
            data=df,
            ticker_list=ticker_list,
            metadata={
                "start_date": start_date,
                "end_date": end_date,
                "time_interval": time_interval
            }
        )
        
        # Add technical indicators if requested
        if include_indicators:
            indicators = self.config.get("default_features", [])
            market_data = market_data.add_technical_indicators(
                indicators=indicators,
                feature_engineer=self.feature_engineer
            )
        
        # Generate dataset name if not provided
        if dataset_name is None:
            date_str = datetime.now().strftime("%Y%m%d")
            dataset_name = f"market_data_{date_str}"
        
        # Save to disk if requested
        if save_to_disk:
            self._save_market_data(market_data, dataset_name)
        
        # Store in cache
        if self.config.get("cache_data", True):
            self._data_cache[dataset_name] = market_data
        
        return market_data
    
    def _save_market_data(self, market_data: MarketData, dataset_name: str) -> None:
        """Save market data to disk.
        
        Args:
            market_data: MarketData object to save.
            dataset_name: Name for the dataset.
        """
        # Create dataset directory
        dataset_dir = os.path.join(self.data_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save data
        data_path = os.path.join(dataset_dir, "data.csv")
        market_data.data.to_csv(data_path, index=False)
        
        # Save metadata
        metadata = {
            "tickers": market_data.ticker_list,
            "time_column": market_data.time_column,
            "price_column_pattern": market_data.price_column_pattern,
            "metadata": market_data.metadata
        }
        
        metadata_path = os.path.join(dataset_dir, "metadata.yaml")
        with open(metadata_path, 'w') as file:
            yaml.dump(metadata, file)
        
        logger.info(f"Saved market data to {dataset_dir}")
    
    def load_market_data(self, dataset_name: str) -> Optional[MarketData]:
        """Load market data from disk.
        
        Args:
            dataset_name: Name of the dataset to load.
            
        Returns:
            MarketData object or None if not found.
        """
        # Check cache first
        if dataset_name in self._data_cache:
            return self._data_cache[dataset_name]
        
        # Check if dataset exists on disk
        dataset_dir = os.path.join(self.data_dir, dataset_name)
        data_path = os.path.join(dataset_dir, "data.csv")
        metadata_path = os.path.join(dataset_dir, "metadata.yaml")
        
        if not os.path.exists(data_path) or not os.path.exists(metadata_path):
            logger.warning(f"Dataset {dataset_name} not found")
            return None
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Load metadata
            with open(metadata_path, 'r') as file:
                metadata = yaml.safe_load(file)
            
            # Create MarketData object
            market_data = MarketData(
                data=df,
                ticker_list=metadata["tickers"],
                time_column=metadata["time_column"],
                price_column_pattern=metadata["price_column_pattern"],
                metadata=metadata["metadata"]
            )
            
            # Store in cache
            if self.config.get("cache_data", True):
                self._data_cache[dataset_name] = market_data
            
            logger.info(f"Loaded market data from {dataset_dir}")
            return market_data
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            return None
    
    def create_universe(
        self,
        ticker_list: Optional[List[str]] = None,
        market_data: Optional[MarketData] = None,
        selection_method: str = "market_cap",
        n_assets: int = 30,
        name: str = "default",
        save_to_disk: bool = True
    ) -> Universe:
        """Create an asset universe.
        
        Args:
            ticker_list: List of ticker symbols for universe.
            market_data: Optional MarketData object for selection-based universe.
            selection_method: Method for universe selection.
            n_assets: Number of assets for selection-based universe.
            name: Name for the universe.
            save_to_disk: Whether to save universe to disk.
            
        Returns:
            Universe object.
        """
        # Create universe
        if ticker_list is not None:
            # Create from explicit list
            universe = Universe(tickers=ticker_list, name=name)
            logger.info(f"Created universe {name} with {len(ticker_list)} tickers")
            
        elif market_data is not None:
            # Create using selection method
            universe = Universe.from_market_data(
                market_data=market_data,
                selector_method=selection_method,
                n_assets=n_assets,
                name=name
            )
            logger.info(f"Created universe {name} with {universe.size} tickers using {selection_method}")
            
        else:
            logger.error("Must provide either ticker_list or market_data")
            return Universe(tickers=[], name=name)
        
        # Save to disk if requested
        if save_to_disk:
            universe_dir = os.path.join(self.data_dir, "universes")
            os.makedirs(universe_dir, exist_ok=True)
            
            filepath = os.path.join(universe_dir, f"{name}.csv")
            universe.save_to_file(filepath)
        
        # Store in cache
        self._universe_cache[name] = universe
        
        return universe
    
    def load_universe(self, name: str) -> Optional[Universe]:
        """Load an asset universe.
        
        Args:
            name: Name of the universe to load.
            
        Returns:
            Universe object or None if not found.
        """
        # Check cache first
        if name in self._universe_cache:
            return self._universe_cache[name]
        
        # Check if universe exists on disk
        universe_dir = os.path.join(self.data_dir, "universes")
        filepath = os.path.join(universe_dir, f"{name}.csv")
        
        if not os.path.exists(filepath):
            logger.warning(f"Universe {name} not found")
            return None
        
        try:
            # Load universe
            universe = Universe.load_from_file(filepath, name)
            
            # Store in cache
            self._universe_cache[name] = universe
            
            return universe
            
        except Exception as e:
            logger.error(f"Error loading universe {name}: {e}")
            return None
    
    def prepare_training_data(
        self,
        market_data: MarketData,
        universe: Optional[Universe] = None,
        normalize: bool = True,
        train_ratio: float = 0.8,
        include_indicators: bool = True
    ) -> Tuple[MarketData, MarketData]:
        """Prepare data for training and testing.
        
        Args:
            market_data: MarketData object.
            universe: Optional Universe to filter assets.
            normalize: Whether to normalize data.
            train_ratio: Ratio of data for training.
            include_indicators: Whether to include technical indicators.
            
        Returns:
            Tuple of (train_data, test_data) as MarketData objects.
        """
        # Filter to universe if provided
        if universe is not None:
            filtered_data = market_data.slice_by_tickers(universe.tickers)
        else:
            filtered_data = market_data
        
        # Add technical indicators if requested
        if include_indicators and not any(col for col in filtered_data.data.columns if 'rsi' in col.lower()):
            indicators = self.config.get("default_features", [])
            filtered_data = filtered_data.add_technical_indicators(
                indicators=indicators,
                feature_engineer=self.feature_engineer
            )
        
        # Split into training and testing sets
        train_data, test_data = filtered_data.split_train_test(train_ratio=train_ratio)
        
        # Normalize data if requested
        if normalize:
            # Fit normalizer on training data
            train_data.data = self.normalizer.fit_transform(train_data.data)
            
            # Transform test data using same normalization
            test_data.data = self.normalizer.transform(test_data.data)
        
        logger.info(f"Prepared training data with {len(train_data.data)} training samples and {len(test_data.data)} testing samples")
        return train_data, test_data
    
    def get_trading_dates(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        market: str = "NYSE"
    ) -> List[datetime]:
        """Get trading days for a specific market.
        
        Args:
            start_date: Start date.
            end_date: End date.
            market: Market name.
            
        Returns:
            List of trading days.
        """
        return self.data_source.get_trading_days(
            start_date=start_date,
            end_date=end_date,
            market=market
        )