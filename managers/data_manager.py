import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime

class ExperimentDataManager:
    """
    Manages data storage and retrieval for experiments.
    Handles saving and loading of raw and normalized data specific to each experiment.
    """
    
    def __init__(
        self,
        experiment_dir: Union[str, Path],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the experiment data manager.
        
        Args:
            experiment_dir: Path to the experiment directory
            logger: Optional logger instance
        """
        self.experiment_dir = Path(experiment_dir)
        self.data_dir = self.experiment_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Create subdirectories for different data types
        self.raw_dir = self.data_dir / "raw"
        self.normalized_dir = self.data_dir / "normalized"
        
        for directory in [self.raw_dir, self.normalized_dir]:
            directory.mkdir(exist_ok=True)
    
    def save_data(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        data_type: str = "normalized"
    ) -> None:
        """
        Save training and validation data for the experiment.
        
        Args:
            train_data: Training data DataFrame
            val_data: Validation data DataFrame
            data_type: Type of data ('raw' or 'normalized')
        """
        if data_type not in ["raw", "normalized"]:
            raise ValueError("data_type must be one of: 'raw', 'normalized'")
        
        # Select appropriate directory
        save_dir = getattr(self, f"{data_type}_dir")
        
        # Save data as CSV
        train_data.to_csv(save_dir / "train_data.csv", index=True)
        val_data.to_csv(save_dir / "val_data.csv", index=True)
        
        # Add basic metadata
        metadata = {
            "train_shape": train_data.shape,
            "val_shape": val_data.shape,
            "train_columns": train_data.columns.tolist(),
            "val_columns": val_data.columns.tolist(),
            "train_tickers": train_data["ticker"].unique().tolist(),
            "val_tickers": val_data["ticker"].unique().tolist(),
            "train_date_range": [
                train_data["date"].min().strftime("%Y-%m-%d"),
                train_data["date"].max().strftime("%Y-%m-%d")
            ],
            "val_date_range": [
                val_data["date"].min().strftime("%Y-%m-%d"),
                val_data["date"].max().strftime("%Y-%m-%d")
            ],
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(save_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.logger.info(f"Saved {data_type} data to {save_dir}")
    
    def load_data(
        self,
        data_type: str = "normalized"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load training and validation data for the experiment.
        
        Args:
            data_type: Type of data to load ('raw' or 'normalized')
            
        Returns:
            Tuple of (train_data, val_data, metadata)
        """
        if data_type not in ["raw", "normalized"]:
            raise ValueError("data_type must be one of: 'raw', 'normalized'")
        
        # Select appropriate directory
        load_dir = getattr(self, f"{data_type}_dir")
        
        # Check if data exists
        train_path = load_dir / "train_data.csv"
        val_path = load_dir / "val_data.csv"
        metadata_path = load_dir / "metadata.json"
        
        if not all(path.exists() for path in [train_path, val_path, metadata_path]):
            raise FileNotFoundError(f"Required {data_type} data files not found in {load_dir}")
        
        # Load data
        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.logger.info(f"Loaded {data_type} data from {load_dir}")
        
        return train_data, val_data, metadata
    
    def save_data_info(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        columns_mapping: Dict,
        processor_params: Optional[Dict] = None,
        normalization_params: Optional[Dict] = None
    ) -> None:
        """
        Save comprehensive information about the experiment's data.
        
        Args:
            train_data: Training data DataFrame
            val_data: Validation data DataFrame
            columns_mapping: Dictionary mapping column types to column names
            processor_params: Optional parameters used for data processing
            normalization_params: Optional parameters used for normalization
        """
        # Compute statistics for both datasets
        train_stats = self._compute_data_statistics(train_data)
        val_stats = self._compute_data_statistics(val_data)
        
        # Save statistics
        train_stats.to_csv(self.data_dir / "train_data_statistics.csv")
        val_stats.to_csv(self.data_dir / "val_data_statistics.csv")
        
        # Create comprehensive data info
        data_info = {
            "train_data": {
                "shape": train_data.shape,
                "columns": train_data.columns.tolist(),
                "tickers": train_data["ticker"].unique().tolist(),
                "date_range": [
                    train_data["date"].min().strftime("%Y-%m-%d"),
                    train_data["date"].max().strftime("%Y-%m-%d")
                ]
            },
            "val_data": {
                "shape": val_data.shape,
                "columns": val_data.columns.tolist(),
                "tickers": val_data["ticker"].unique().tolist(),
                "date_range": [
                    val_data["date"].min().strftime("%Y-%m-%d"),
                    val_data["date"].max().strftime("%Y-%m-%d")
                ]
            },
            "columns_mapping": columns_mapping,
            "processor_params": processor_params or {},
            "normalization_params": normalization_params or {},
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save data info
        with open(self.data_dir / "data_info.json", 'w') as f:
            json.dump(data_info, f, indent=4)
        
        self.logger.info(f"Saved comprehensive data information to {self.data_dir}")
    
    def _compute_data_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics for each column in the dataset.
        
        Args:
            data: DataFrame to compute statistics for
            
        Returns:
            DataFrame containing statistics
        """
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Compute basic statistics
        stats = data[numeric_cols].describe().T
        
        # Add additional statistics
        stats['skew'] = data[numeric_cols].skew()
        stats['kurtosis'] = data[numeric_cols].kurtosis()
        stats['missing'] = data[numeric_cols].isnull().sum()
        stats['missing_pct'] = data[numeric_cols].isnull().mean() * 100
        
        return stats
    
    def get_data_info(self) -> Dict:
        """
        Get comprehensive information about the experiment's data.
        
        Returns:
            Dictionary containing data information
        """
        info_path = self.data_dir / "data_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Data info file not found at {info_path}")
        
        with open(info_path, 'r') as f:
            return json.load(f)
    
    def verify_data_integrity(self) -> bool:
        """
        Verify that all required data files exist and are consistent.
        
        Returns:
            True if all data is present and consistent, False otherwise
        """
        try:
            # Check for data info
            if not (self.data_dir / "data_info.json").exists():
                self.logger.error("Data info file missing")
                return False
            
            # Check for statistics files
            if not all((self.data_dir / f"{data_type}_data_statistics.csv").exists() 
                      for data_type in ["train", "val"]):
                self.logger.error("Data statistics files missing")
                return False
            
            # Check for data files in each subdirectory
            for data_type in ["raw", "normalized"]:
                data_dir = getattr(self, f"{data_type}_dir")
                if not all((data_dir / f"{split}_data.csv").exists() 
                          for split in ["train", "val"]):
                    self.logger.error(f"{data_type} data files missing")
                    return False
                if not (data_dir / "metadata.json").exists():
                    self.logger.error(f"{data_type} metadata file missing")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying data integrity: {str(e)}")
            return False 