import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class MeanReversionStrategy:
    """
    Mean Reversion strategy that buys assets when their price falls below
    their historical mean and sells when they rise above it.
    """
    
    def __init__(self, 
                 assets: List[str], 
                 initial_capital: float = 10000.0,
                 window_size: int = 20,
                 z_threshold: float = 1.0,
                 position_size: float = 0.2,
                 transaction_cost: float = 0.001):
        """
        Initialize the Mean Reversion strategy.
        
        Args:
            assets: List of asset tickers to trade
            initial_capital: Starting capital
            window_size: Window size for calculating moving average and standard deviation
            z_threshold: Z-score threshold for trading signals
            position_size: Percentage of portfolio to allocate to each position
            transaction_cost: Cost of transaction as a fraction of trade value
        """
        self.assets = assets
        self.initial_capital = initial_capital
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        
        self.positions = {asset: 0 for asset in assets}
        self.cash = initial_capital
        self.portfolio_value_history = []
        self.position_history = []
        
        # Price history for mean reversion calculation
        self.price_history = {asset: [] for asset in assets}
    
    def reset(self):
        """Reset the strategy to initial state."""
        self.positions = {asset: 0 for asset in self.assets}
        self.cash = self.initial_capital
        self.portfolio_value_history = []
        self.position_history = []
        self.price_history = {asset: [] for asset in self.assets}
    
    def calculate_z_scores(self) -> Dict[str, float]:
        """
        Calculate z-scores for each asset based on current price and historical mean/std.
        
        Returns:
            Dictionary mapping assets to z-scores
        """
        z_scores = {}
        
        for asset, prices in self.price_history.items():
            if len(prices) >= self.window_size:
                # Calculate mean and standard deviation for the window
                window = prices[-self.window_size:]
                mean = np.mean(window)
                std = np.std(window)
                
                if std > 0:  # Avoid division by zero
                    # Calculate z-score: (current_price - mean) / std
                    current_price = prices[-1]
                    z_score = (current_price - mean) / std
                    z_scores[asset] = z_score
        
        return z_scores
    
    def execute_trades(self, prices: Dict[str, float], z_scores: Dict[str, float]):
        """
        Execute trades based on z-scores.
        
        Args:
            prices: Dictionary mapping assets to their current prices
            z_scores: Dictionary mapping assets to z-scores
        """
        portfolio_value = self.calculate_portfolio_value(prices)
        
        for asset, z_score in z_scores.items():
            if asset not in prices:
                continue
                
            current_price = prices[asset]
            current_shares = self.positions[asset]
            
            # Buy signal: price is below mean (negative z-score)
            if z_score < -self.z_threshold:
                # Only buy if we don't already have a long position
                if current_shares <= 0:
                    # Calculate position size based on portfolio value
                    amount_to_invest = portfolio_value * self.position_size
                    shares_to_buy = int(amount_to_invest / current_price)
                    
                    if shares_to_buy > 0:
                        # Execute purchase
                        cost = shares_to_buy * current_price
                        transaction_fee = cost * self.transaction_cost
                        total_cost = cost + transaction_fee
                        
                        if total_cost <= self.cash:
                            self.positions[asset] = shares_to_buy
                            self.cash -= total_cost
            
            # Sell signal: price is above mean (positive z-score)
            elif z_score > self.z_threshold:
                # Sell if we have a long position
                if current_shares > 0:
                    # Execute sell
                    sell_value = current_shares * current_price
                    transaction_fee = sell_value * self.transaction_cost
                    
                    self.cash += sell_value - transaction_fee
                    self.positions[asset] = 0
    
    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            prices: Dictionary mapping assets to their current prices
            
        Returns:
            Total portfolio value
        """
        portfolio_value = self.cash
        
        for asset, shares in self.positions.items():
            if asset in prices:
                asset_value = shares * prices[asset]
                portfolio_value += asset_value
        
        return portfolio_value
    
    def step(self, 
             current_step: int, 
             prices: Dict[str, float], 
             features: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Execute a single step of the strategy.
        
        Args:
            current_step: Current step in the trading episode
            prices: Dictionary mapping assets to their current prices
            features: Dictionary of additional features per asset (not used in this strategy)
            
        Returns:
            Dict containing current positions, cash, and portfolio value
        """
        # Update price history
        for asset, price in prices.items():
            if asset in self.price_history:
                self.price_history[asset].append(price)
        
        # Execute trades if we have enough history
        if current_step >= self.window_size:
            z_scores = self.calculate_z_scores()
            if z_scores:
                self.execute_trades(prices, z_scores)
        
        # Calculate portfolio value
        portfolio_value = self.calculate_portfolio_value(prices)
        current_positions = {asset: shares for asset, shares in self.positions.items() if shares != 0}
        
        # Record history
        self.portfolio_value_history.append(portfolio_value)
        self.position_history.append(current_positions.copy())
        
        return {
            "positions": current_positions,
            "cash": self.cash,
            "portfolio_value": portfolio_value
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the strategy.
        
        Returns:
            Dictionary of performance metrics
        """
        if len(self.portfolio_value_history) < 2:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0
            }
        
        # Calculate returns
        values = np.array(self.portfolio_value_history)
        returns = np.diff(values) / values[:-1]
        
        # Calculate metrics
        total_return = (values[-1] / values[0]) - 1.0
        sharpe_ratio = 0.0
        if np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        
        # Calculate max drawdown
        peak = values[0]
        max_drawdown = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
