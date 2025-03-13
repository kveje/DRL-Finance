"""Feature engineering module for financial data."""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from stockstats import StockDataFrame

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Financial feature engineering class.
    
    This class provides methods to calculate technical indicators and other
    features for financial time series data.
    """
    
    def __init__(self, use_stockstats: bool = True):
        """Initialize the feature engineer.
        
        Args:
            use_stockstats: Whether to use stockstats library for technical indicators.
        """
        self.use_stockstats = use_stockstats
    
    def preprocess(
        self, 
        df: pd.DataFrame,
        ticker_list: List[str],
        technical_indicators: List[str] = None
    ) -> pd.DataFrame:
        """Preprocess the data and extract technical indicators.
        
        Args:
            df: DataFrame with financial data.
            ticker_list: List of ticker symbols to process.
            technical_indicators: List of technical indicators to calculate.
                If None, uses a default set of indicators.
                
        Returns:
            DataFrame with added technical indicators.
        """
        # Default technical indicators if none provided
        if technical_indicators is None:
            technical_indicators = [
                "macd", "rsi_14", "cci_30", "dx_30", "close_20_sma", "close_50_sma"
            ]
        
        logger.info(f"Preprocessing data for {len(ticker_list)} tickers")
        
        # Process each ticker
        processed_dfs = []
        for ticker in ticker_list:
            ticker_df = self._process_ticker(df, ticker, technical_indicators)
            if not ticker_df.empty:
                processed_dfs.append(ticker_df)
        
        # Concatenate all processed dataframes
        if processed_dfs:
            processed_df = pd.concat(processed_dfs, axis=1)
            processed_df["date"] = df["date"]  # Add back the date column
            return processed_df
        else:
            logger.warning("No data processed")
            return pd.DataFrame()
    
    def _process_ticker(
        self, 
        df: pd.DataFrame, 
        ticker: str, 
        technical_indicators: List[str]
    ) -> pd.DataFrame:
        """Process a single ticker's data.
        
        Args:
            df: DataFrame with financial data.
            ticker: Ticker symbol to process.
            technical_indicators: List of technical indicators to calculate.
            
        Returns:
            DataFrame with ticker's processed data.
        """
        try:
            # Extract columns for this ticker
            cols = [col for col in df.columns if ticker in col]
            if not cols:
                logger.warning(f"No data found for ticker {ticker}")
                return pd.DataFrame()
            
            ticker_df = df[cols].copy()
            
            # Rename columns to standard names for stockstats
            rename_dict = {}
            for col in ticker_df.columns:
                if 'open' in col.lower():
                    rename_dict[col] = 'open'
                elif 'high' in col.lower():
                    rename_dict[col] = 'high'
                elif 'low' in col.lower():
                    rename_dict[col] = 'low'
                elif 'close' in col.lower():
                    rename_dict[col] = 'close'
                elif 'volume' in col.lower():
                    rename_dict[col] = 'volume'
            
            # Only proceed if we have core price columns
            if len(rename_dict) < 4:  # Need at least OHLC 
                logger.warning(f"Insufficient price data for ticker {ticker}")
                return pd.DataFrame()
            
            temp_df = ticker_df.rename(columns=rename_dict)
            
            # Calculate indicators
            if self.use_stockstats:
                # Use stockstats for technical indicators
                stock_df = StockDataFrame.retype(temp_df)
                
                # Calculate each requested indicator
                for indicator in technical_indicators:
                    # Handle special cases with parameters (e.g., close_20_sma)
                    if '_' in indicator and indicator.startswith(('open', 'high', 'low', 'close', 'volume')):
                        parts = indicator.split('_')
                        if len(parts) >= 3 and parts[2] in ('sma', 'ema'):
                            # Format: close_20_sma
                            field, window, type_ = parts[0], parts[1], parts[2]
                            if type_ == 'sma':
                                stock_df[f'{field}_{window}_sma']
                            elif type_ == 'ema':
                                stock_df[f'{field}_{window}_ema']
                        continue
                    
                    # Handle standard indicators
                    stock_df[indicator]
                
                # Rename columns back with ticker prefix
                result_df = stock_df.copy()
                
                # Rename standard columns back
                inverse_rename = {v: k for k, v in rename_dict.items()}
                result_df.rename(columns=inverse_rename, inplace=True)
                
                # Rename indicator columns
                for col in result_df.columns:
                    if col not in ticker_df.columns:
                        result_df.rename(columns={col: f"{ticker}_{col}"}, inplace=True)
                
                return result_df
            else:
                # Calculate indicators manually
                result_df = ticker_df.copy()
                for indicator in technical_indicators:
                    if indicator.startswith("close_") and indicator.endswith("_sma"):
                        # Simple moving average
                        window = int(indicator.split("_")[1])
                        close_col = next((col for col in ticker_df.columns if 'close' in col.lower()), None)
                        if close_col:
                            result_df[f"{ticker}_{indicator}"] = ticker_df[close_col].rolling(window=window).mean()
                    elif indicator == "rsi_14":
                        # RSI calculation
                        self._add_rsi(result_df, ticker, 14)
                    # Add more manual calculations as needed
                
                return result_df
                
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {e}")
            return pd.DataFrame()
    
    def _add_rsi(self, df: pd.DataFrame, ticker: str, window: int = 14) -> None:
        """Add Relative Strength Index (RSI) to dataframe.
        
        Args:
            df: DataFrame to add RSI to.
            ticker: Ticker symbol.
            window: RSI window period.
        """
        close_col = next((col for col in df.columns if 'close' in col.lower()), None)
        if not close_col:
            return
        
        # Get price diff
        delta = df[close_col].diff()
        
        # Get gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Add to dataframe
        df[f"{ticker}_rsi_{window}"] = rsi