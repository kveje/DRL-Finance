"""Data normalization module for financial data."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class Normalizer:
    """Data normalization class for financial data.
    
    This class provides methods to normalize financial time series data
    using various normalization techniques.
    """
    
    VALID_METHODS = ["minmax", "zscore", "robust", "decimal"]
    
    def __init__(
        self, 
        method: str = "minmax",
        feature_range: Tuple[float, float] = (0, 1)
    ):
        """Initialize the normalizer.
        
        Args:
            method: Normalization method. Options: "minmax", "zscore", "robust", "decimal".
            feature_range: Feature range for MinMaxScaler.
        """
        if method not in self.VALID_METHODS:
            raise ValueError(f"Method must be one of {self.VALID_METHODS}")
            
        self.method = method
        self.feature_range = feature_range
        self.scalers: Dict[str, Union[MinMaxScaler, StandardScaler]] = {}
        
    def fit_transform(
        self,
        df: pd.DataFrame,
        cols_to_normalize: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fit scalers to data and transform it.
        
        Args:
            df: DataFrame to normalize.
            cols_to_normalize: List of columns to normalize. If None, normalizes all numeric columns
                except date and categorical columns.
                
        Returns:
            Normalized DataFrame.
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for normalization")
            return df
        
        # Copy the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Determine columns to normalize
        if cols_to_normalize is None:
            # Exclude date and categorical columns
            cols_to_normalize = [
                col for col in df.columns 
                if pd.api.types.is_numeric_dtype(df[col]) and col != "date"
            ]
        
        logger.info(f"Normalizing {len(cols_to_normalize)} columns using {self.method} method")
        
        # Apply normalization based on method
        if self.method == "minmax":
            return self._apply_minmax(result_df, cols_to_normalize)
        elif self.method == "zscore":
            return self._apply_zscore(result_df, cols_to_normalize)
        elif self.method == "robust":
            return self._apply_robust(result_df, cols_to_normalize)
        elif self.method == "decimal":
            return self._apply_decimal(result_df, cols_to_normalize)
        else:
            logger.warning(f"Unknown normalization method: {self.method}")
            return result_df
    
    def transform(
        self,
        df: pd.DataFrame,
        cols_to_normalize: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Transform data using previously fit scalers.
        
        Args:
            df: DataFrame to normalize.
            cols_to_normalize: List of columns to normalize. If None, normalizes all columns
                that have saved scalers.
                
        Returns:
            Normalized DataFrame.
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for normalization")
            return df
        
        # Copy the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Determine columns to normalize
        if cols_to_normalize is None:
            cols_to_normalize = list(self.scalers.keys())
        else:
            # Filter to only include columns with saved scalers
            cols_to_normalize = [col for col in cols_to_normalize if col in self.scalers]
        
        logger.info(f"Transforming {len(cols_to_normalize)} columns using {self.method} method")
        
        # Apply transformation for each column
        for col in cols_to_normalize:
            if col in self.scalers and col in df.columns:
                # Reshape for scikit-learn
                col_data = df[col].values.reshape(-1, 1)
                result_df[col] = self.scalers[col].transform(col_data).flatten()
        
        return result_df
    
    def inverse_transform(
        self,
        df: pd.DataFrame,
        cols_to_denormalize: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Inverse transform normalized data back to original scale.
        
        Args:
            df: DataFrame to denormalize.
            cols_to_denormalize: List of columns to denormalize. If None, denormalizes all columns
                that have saved scalers.
                
        Returns:
            Denormalized DataFrame.
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for denormalization")
            return df
        
        # Copy the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Determine columns to denormalize
        if cols_to_denormalize is None:
            cols_to_denormalize = list(self.scalers.keys())
        else:
            # Filter to only include columns with saved scalers
            cols_to_denormalize = [col for col in cols_to_denormalize if col in self.scalers]
        
        logger.info(f"Denormalizing {len(cols_to_denormalize)} columns")
        
        # Apply inverse transformation for each column
        for col in cols_to_denormalize:
            if col in self.scalers and col in df.columns:
                # Reshape for scikit-learn
                col_data = df[col].values.reshape(-1, 1)
                result_df[col] = self.scalers[col].inverse_transform(col_data).flatten()
        
        return result_df
    
    def _apply_minmax(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Apply Min-Max normalization.
        
        Args:
            df: DataFrame to normalize.
            cols: List of columns to normalize.
            
        Returns:
            Normalized DataFrame.
        """
        for col in cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Create scaler for this column
                scaler = MinMaxScaler(feature_range=self.feature_range)
                # Reshape for scikit-learn
                col_data = df[col].values.reshape(-1, 1)
                # Fit and transform
                df[col] = scaler.fit_transform(col_data).flatten()
                # Save scaler
                self.scalers[col] = scaler
        
        return df
    
    def _apply_zscore(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Apply Z-score normalization.
        
        Args:
            df: DataFrame to normalize.
            cols: List of columns to normalize.
            
        Returns:
            Normalized DataFrame.
        """
        for col in cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Create scaler for this column
                scaler = StandardScaler()
                # Reshape for scikit-learn
                col_data = df[col].values.reshape(-1, 1)
                # Fit and transform
                df[col] = scaler.fit_transform(col_data).flatten()
                # Save scaler
                self.scalers[col] = scaler
        
        return df
    
    def _apply_robust(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Apply robust normalization using median and IQR.
        
        Args:
            df: DataFrame to normalize.
            cols: List of columns to normalize.
            
        Returns:
            Normalized DataFrame.
        """
        for col in cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Calculate median and IQR
                median = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                
                if iqr > 0:
                    # Normalize using median and IQR
                    df[col] = (df[col] - median) / iqr
                    
                    # Create a custom scaler for inverse transform
                    # Using StandardScaler for simplicity, but adjusting params
                    scaler = StandardScaler()
                    scaler.mean_ = np.array([median])
                    scaler.scale_ = np.array([iqr])
                    self.scalers[col] = scaler
        
        return df
    
    def _apply_decimal(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Apply decimal scaling normalization.
        
        Args:
            df: DataFrame to normalize.
            cols: List of columns to normalize.
            
        Returns:
            Normalized DataFrame.
        """
        for col in cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Find maximum absolute value
                max_abs = df[col].abs().max()
                
                if max_abs > 0:
                    # Calculate power of 10 for scaling
                    j = int(np.ceil(np.log10(max_abs)))
                    scale_factor = 10 ** j
                    
                    # Scale the data
                    df[col] = df[col] / scale_factor
                    
                    # Create a custom scaler for inverse transform
                    # Using StandardScaler with adjusted params
                    scaler = StandardScaler()
                    scaler.mean_ = np.array([0])
                    scaler.scale_ = np.array([1 / scale_factor])
                    self.scalers[col] = scaler
        
        return df