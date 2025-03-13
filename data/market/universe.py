"""Asset universe management for financial trading."""

import logging
from typing import Dict, List, Optional, Set, Union

import pandas as pd

logger = logging.getLogger(__name__)


class Universe:
    """Asset universe class for financial trading.
    
    This class manages the set of assets that are eligible for trading,
    with support for dynamic universe changes over time.
    """
    
    def __init__(
        self,
        tickers: List[str],
        name: str = "default",
        metadata: Optional[Dict] = None
    ):
        """Initialize asset universe.
        
        Args:
            tickers: List of ticker symbols in the universe.
            name: Name of the universe.
            metadata: Optional additional metadata about the universe.
        """
        self.tickers = list(set(tickers))  # Remove duplicates
        self.name = name
        self.metadata = metadata or {}
        
        # Map for ticker indices
        self._ticker_indices = {ticker: idx for idx, ticker in enumerate(self.tickers)}
        
        # Historical universe snapshots (for dynamic universes)
        self._history: Dict[pd.Timestamp, Set[str]] = {}
    
    @property
    def size(self) -> int:
        """Get the number of assets in the universe.
        
        Returns:
            Number of assets.
        """
        return len(self.tickers)
    
    def contains(self, ticker: str) -> bool:
        """Check if a ticker is in the universe.
        
        Args:
            ticker: Ticker symbol to check.
            
        Returns:
            Boolean indicating if ticker is in universe.
        """
        return ticker in self._ticker_indices
    
    def get_index(self, ticker: str) -> int:
        """Get the index of a ticker in the universe.
        
        Args:
            ticker: Ticker symbol.
            
        Returns:
            Index of ticker or -1 if not found.
        """
        return self._ticker_indices.get(ticker, -1)
    
    def add_ticker(self, ticker: str) -> None:
        """Add a ticker to the universe.
        
        Args:
            ticker: Ticker symbol to add.
        """
        if not self.contains(ticker):
            self.tickers.append(ticker)
            self._ticker_indices[ticker] = len(self.tickers) - 1
            logger.info(f"Added {ticker} to universe {self.name}")
    
    def remove_ticker(self, ticker: str) -> None:
        """Remove a ticker from the universe.
        
        Args:
            ticker: Ticker symbol to remove.
        """
        if self.contains(ticker):
            idx = self._ticker_indices[ticker]
            self.tickers.pop(idx)
            # Update indices for all tickers after this one
            self._ticker_indices = {t: i for i, t in enumerate(self.tickers)}
            logger.info(f"Removed {ticker} from universe {self.name}")
    
    def update_universe(self, tickers: List[str]) -> None:
        """Update the entire universe to a new set of tickers.
        
        Args:
            tickers: New list of ticker symbols.
        """
        # Store old universe for logging
        old_tickers = set(self.tickers)
        new_tickers = set(tickers)
        
        added = new_tickers - old_tickers
        removed = old_tickers - new_tickers
        
        # Update universe
        self.tickers = list(new_tickers)
        self._ticker_indices = {ticker: idx for idx, ticker in enumerate(self.tickers)}
        
        # Log changes
        if added:
            logger.info(f"Added {len(added)} tickers to universe {self.name}: {', '.join(added)}")
        if removed:
            logger.info(f"Removed {len(removed)} tickers from universe {self.name}: {', '.join(removed)}")
    
    def save_snapshot(self, timestamp: pd.Timestamp) -> None:
        """Save a snapshot of the current universe at a specific time.
        
        Args:
            timestamp: Timestamp for the snapshot.
        """
        self._history[timestamp] = set(self.tickers)
    
    def get_universe_at_time(self, timestamp: pd.Timestamp) -> List[str]:
        """Get the universe as it was at a specific time.
        
        Args:
            timestamp: Timestamp to query.
            
        Returns:
            List of tickers in the universe at the specified time.
        """
        # Find the closest timestamp before or equal to the requested time
        valid_times = [t for t in self._history.keys() if t <= timestamp]
        
        if not valid_times:
            # No historical data, return current universe
            return self.tickers
        
        # Get most recent snapshot
        closest_time = max(valid_times)
        return list(self._history[closest_time])
    
    def get_rebalance_dates(self) -> List[pd.Timestamp]:
        """Get dates when the universe was rebalanced.
        
        Returns:
            List of timestamps when the universe changed.
        """
        return sorted(self._history.keys())
    
    @classmethod
    def from_market_data(
        cls,
        market_data,
        selector_method: str = "market_cap",
        n_assets: int = 30,
        name: str = "auto_selected"
    ) -> 'Universe':
        """Create a universe from market data using a selection method.
        
        Args:
            market_data: MarketData object.
            selector_method: Method for universe selection.
            n_assets: Number of assets to include.
            name: Name for the new universe.
            
        Returns:
            New Universe object with selected assets.
        """
        # Import here to avoid circular imports
        from data.processors.universe_selection import UniverseSelector
        
        # Create selector and select universe
        selector = UniverseSelector(selection_method=selector_method)
        selected_tickers = selector.select_universe(
            df=market_data.data,
            ticker_list=market_data.ticker_list,
            n_assets=n_assets
        )
        
        # Create universe
        return cls(
            tickers=selected_tickers,
            name=name,
            metadata={"selection_method": selector_method}
        )
    
    def save_to_file(self, filepath: str) -> None:
        """Save universe to a file.
        
        Args:
            filepath: Path to save file.
        """
        data = {
            "name": self.name,
            "tickers": self.tickers,
            "metadata": self.metadata
        }
        
        # Convert to dataframe and save
        tickers_df = pd.DataFrame({"ticker": self.tickers})
        tickers_df.to_csv(filepath, index=False)
        logger.info(f"Saved universe {self.name} with {len(self.tickers)} tickers to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str, name: Optional[str] = None) -> 'Universe':
        """Load universe from a file.
        
        Args:
            filepath: Path to load file from.
            name: Optional name for the universe (overrides saved name).
            
        Returns:
            Loaded Universe object.
        """
        try:
            # Load tickers from CSV
            tickers_df = pd.read_csv(filepath)
            tickers = tickers_df["ticker"].tolist()
            
            # Use provided name or extract from filepath
            if name is None:
                import os
                name = os.path.splitext(os.path.basename(filepath))[0]
            
            logger.info(f"Loaded universe {name} with {len(tickers)} tickers from {filepath}")
            return cls(tickers=tickers, name=name)
            
        except Exception as e:
            logger.error(f"Error loading universe from {filepath}: {e}")
            # Return empty universe
            return cls(tickers=[], name=name or "error_loading")