"""Universe selection module for financial data."""

import logging
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UniverseSelector:
    """Universe selection class for financial trading.
    
    This class provides methods to select asset universes based on
    different criteria like market cap, volatility, liquidity, etc.
    """
    
    def __init__(self, selection_method: str = "market_cap"):
        """Initialize the universe selector.
        
        Args:
            selection_method: Method for universe selection.
                Options: "market_cap", "liquidity", "volatility", "momentum", "custom".
        """
        self.selection_method = selection_method
    
    def select_universe(
        self,
        df: pd.DataFrame,
        ticker_list: List[str],
        n_assets: int = 30,
        lookback_period: int = 252,
        selection_data: Optional[Dict] = None,
        custom_selection_func: Optional[Callable] = None
    ) -> List[str]:
        """Select assets universe based on specified criteria.
        
        Args:
            df: DataFrame with financial data.
            ticker_list: List of all tickers to select from.
            n_assets: Number of assets to select.
            lookback_period: Lookback period for calculations.
            selection_data: Optional additional data for selection (e.g., market caps).
            custom_selection_func: Custom selection function for "custom" method.
            
        Returns:
            List of selected ticker symbols.
        """
        logger.info(f"Selecting universe of {n_assets} assets using {self.selection_method} method")
        
        if self.selection_method == "market_cap":
            return self._select_by_market_cap(df, ticker_list, n_assets, selection_data)
        elif self.selection_method == "liquidity":
            return self._select_by_liquidity(df, ticker_list, n_assets, lookback_period)
        elif self.selection_method == "volatility":
            return self._select_by_volatility(df, ticker_list, n_assets, lookback_period)
        elif self.selection_method == "momentum":
            return self._select_by_momentum(df, ticker_list, n_assets, lookback_period)
        elif self.selection_method == "custom" and custom_selection_func is not None:
            return custom_selection_func(df, ticker_list, n_assets)
        else:
            logger.warning(f"Unknown selection method: {self.selection_method}. Using market cap method.")
            return self._select_by_market_cap(df, ticker_list, n_assets, selection_data)
    
    def _select_by_market_cap(
        self,
        df: pd.DataFrame,
        ticker_list: List[str],
        n_assets: int,
        market_cap_data: Optional[Dict] = None
    ) -> List[str]:
        """Select assets based on market capitalization.
        
        Args:
            df: DataFrame with financial data.
            ticker_list: List of all tickers to select from.
            n_assets: Number of assets to select.
            market_cap_data: Dictionary mapping tickers to market cap values.
                If None, estimates market cap using price * volume.
                
        Returns:
            List of selected ticker symbols.
        """
        if market_cap_data:
            # Use provided market cap data
            market_caps = {ticker: market_cap_data.get(ticker, 0) for ticker in ticker_list}
        else:
            # Estimate market cap using price * volume
            market_caps = {}
            for ticker in ticker_list:
                try:
                    price_col = next((col for col in df.columns if ticker in col and 'close' in col.lower()), None)
                    volume_col = next((col for col in df.columns if ticker in col and 'volume' in col.lower()), None)
                    
                    if price_col and volume_col:
                        # Use latest data point
                        latest_price = df[price_col].iloc[-1]
                        latest_volume = df[volume_col].iloc[-1]
                        market_caps[ticker] = latest_price * latest_volume
                    else:
                        market_caps[ticker] = 0
                except Exception as e:
                    logger.warning(f"Error calculating market cap for {ticker}: {e}")
                    market_caps[ticker] = 0
        
        # Sort tickers by market cap and select top n_assets
        selected_tickers = sorted(
            ticker_list, 
            key=lambda ticker: market_caps.get(ticker, 0), 
            reverse=True
        )[:n_assets]
        
        logger.info(f"Selected {len(selected_tickers)} tickers by market cap")
        return selected_tickers
    
    def _select_by_liquidity(
        self,
        df: pd.DataFrame,
        ticker_list: List[str],
        n_assets: int,
        lookback_period: int
    ) -> List[str]:
        """Select assets based on liquidity (trading volume).
        
        Args:
            df: DataFrame with financial data.
            ticker_list: List of all tickers to select from.
            n_assets: Number of assets to select.
            lookback_period: Number of days to look back for average volume.
            
        Returns:
            List of selected ticker symbols.
        """
        # Calculate average daily volume
        avg_volumes = {}
        for ticker in ticker_list:
            try:
                volume_col = next((col for col in df.columns if ticker in col and 'volume' in col.lower()), None)
                
                if volume_col and volume_col in df.columns:
                    # Calculate average daily volume over lookback period
                    recent_data = df[volume_col].tail(lookback_period)
                    avg_volumes[ticker] = recent_data.mean()
                else:
                    avg_volumes[ticker] = 0
            except Exception as e:
                logger.warning(f"Error calculating average volume for {ticker}: {e}")
                avg_volumes[ticker] = 0
        
        # Sort tickers by average volume and select top n_assets
        selected_tickers = sorted(
            ticker_list, 
            key=lambda ticker: avg_volumes.get(ticker, 0), 
            reverse=True
        )[:n_assets]
        
        logger.info(f"Selected {len(selected_tickers)} tickers by liquidity")
        return selected_tickers
    
    def _select_by_volatility(
        self,
        df: pd.DataFrame,
        ticker_list: List[str],
        n_assets: int,
        lookback_period: int,
        high_vol: bool = True
    ) -> List[str]:
        """Select assets based on historical volatility.
        
        Args:
            df: DataFrame with financial data.
            ticker_list: List of all tickers to select from.
            n_assets: Number of assets to select.
            lookback_period: Number of days for volatility calculation.
            high_vol: If True, select highest volatility assets, else lowest.
            
        Returns:
            List of selected ticker symbols.
        """
        # Calculate historical volatility (standard deviation of returns)
        volatilities = {}
        for ticker in ticker_list:
            try:
                price_col = next((col for col in df.columns if ticker in col and 'close' in col.lower()), None)
                
                if price_col and price_col in df.columns:
                    # Calculate daily returns
                    prices = df[price_col].tail(lookback_period + 1)
                    returns = prices.pct_change().dropna()
                    
                    # Calculate annualized volatility
                    daily_vol = returns.std()
                    annualized_vol = daily_vol * np.sqrt(252)  # Annualize assuming 252 trading days
                    
                    volatilities[ticker] = annualized_vol
                else:
                    volatilities[ticker] = 0 if high_vol else float('inf')
            except Exception as e:
                logger.warning(f"Error calculating volatility for {ticker}: {e}")
                volatilities[ticker] = 0 if high_vol else float('inf')
        
        # Sort tickers by volatility
        selected_tickers = sorted(
            ticker_list, 
            key=lambda ticker: volatilities.get(ticker, 0),
            reverse=high_vol  # High to low if high_vol, low to high otherwise
        )[:n_assets]
        
        vol_type = "highest" if high_vol else "lowest"
        logger.info(f"Selected {len(selected_tickers)} tickers with {vol_type} volatility")
        return selected_tickers
    
    def _select_by_momentum(
        self,
        df: pd.DataFrame,
        ticker_list: List[str],
        n_assets: int,
        lookback_period: int
    ) -> List[str]:
        """Select assets based on price momentum.
        
        Args:
            df: DataFrame with financial data.
            ticker_list: List of all tickers to select from.
            n_assets: Number of assets to select.
            lookback_period: Number of days for momentum calculation.
            
        Returns:
            List of selected ticker symbols.
        """
        # Calculate momentum (returns over lookback period)
        momentum = {}
        for ticker in ticker_list:
            try:
                price_col = next((col for col in df.columns if ticker in col and 'close' in col.lower()), None)
                
                if price_col and price_col in df.columns:
                    # Get first and last price in lookback period
                    prices = df[price_col].tail(lookback_period)
                    if len(prices) >= lookback_period:
                        start_price = prices.iloc[0]
                        end_price = prices.iloc[-1]
                        
                        # Calculate total return over period
                        if start_price > 0:
                            momentum[ticker] = (end_price / start_price) - 1
                        else:
                            momentum[ticker] = 0
                    else:
                        momentum[ticker] = 0
                else:
                    momentum[ticker] = 0
            except Exception as e:
                logger.warning(f"Error calculating momentum for {ticker}: {e}")
                momentum[ticker] = 0
        
        # Sort tickers by momentum and select top n_assets
        selected_tickers = sorted(
            ticker_list, 
            key=lambda ticker: momentum.get(ticker, 0), 
            reverse=True
        )[:n_assets]
        
        logger.info(f"Selected {len(selected_tickers)} tickers by momentum")
        return selected_tickers