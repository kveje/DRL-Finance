import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class MomentumStrategy:
    """
    Momentum trading strategy that buys assets that have performed well over 
    a specified lookback period and sells those that haven't.
    """
    
    def __init__(self, 
                 assets: List[str], 
                 initial_capital: float = 10000.0,
                 lookback_period: int = 20,
                 top_n: int = 2,
                 rebalance_frequency: int = 5,
                 transaction_cost: float = 0.001):
        """
        Initialize the Momentum strategy.
        
        Args:
            assets: List of asset tickers to trade
            initial_capital: Starting capital
            lookback_period: Number of days to look back for momentum calculation
            top_n: Number of top performing assets to hold
            rebalance_frequency: How often to rebalance the portfolio (in days)
            transaction_cost: Cost of transaction as a fraction of trade value
        """
        self.assets = assets
        self.initial_capital = initial_capital
        self.lookback_period = lookback_period
        self.top_n = min(top_n, len(assets))  # Can't be more than number of assets
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        
        self.positions = {asset: 0 for asset in assets}
        self.cash = initial_capital
        self.portfolio_value_history = []
        self.position_history = []
        
        # Price history for momentum calculation
        self.price_history = {asset: [] for asset in assets}
    
    def reset(self):
        """Reset the strategy to initial state."""
        self.positions = {asset: 0 for asset in self.assets}
        self.cash = self.initial_capital
        self.portfolio_value_history = []
        self.position_history = []
        self.price_history = {asset: [] for asset in self.assets}
    
    def calculate_momentum(self) -> Dict[str, float]:
        """
        Calculate momentum scores for each asset.
        
        Returns:
            Dictionary mapping assets to momentum scores (returns over lookback period)
        """
        momentum_scores = {}
        
        for asset, prices in self.price_history.items():
            if len(prices) >= self.lookback_period:
                # Calculate return over lookback period
                start_price = prices[-self.lookback_period]
                end_price = prices[-1]
                if start_price > 0:  # Avoid division by zero
                    momentum = (end_price - start_price) / start_price
                    momentum_scores[asset] = momentum
        
        return momentum_scores
    
    def rebalance_portfolio(self, prices: Dict[str, float], momentum_scores: Dict[str, float]):
        """
        Rebalance portfolio based on momentum scores.
        
        Args:
            prices: Dictionary mapping assets to their current prices
            momentum_scores: Dictionary mapping assets to momentum scores
        """
        # Sell all current positions
        for asset, shares in self.positions.items():
            if shares > 0 and asset in prices:
                sell_value = shares * prices[asset]
                transaction_fee = sell_value * self.transaction_cost
                self.cash += sell_value - transaction_fee
                self.positions[asset] = 0
        
        # Select top N assets by momentum
        top_assets = sorted(momentum_scores.keys(), 
                           key=lambda x: momentum_scores[x], 
                           reverse=True)[:self.top_n]
        
        # Equally allocate cash to top assets
        if top_assets:
            amount_per_asset = self.cash / len(top_assets) * 0.99  # Keep some cash as buffer
            
            for asset in top_assets:
                if asset in prices and prices[asset] > 0:
                    # Calculate shares to buy
                    shares = int(amount_per_asset / prices[asset])
                    
                    if shares > 0:
                        # Execute purchase
                        cost = shares * prices[asset]
                        transaction_fee = cost * self.transaction_cost
                        total_cost = cost + transaction_fee
                        
                        if total_cost <= self.cash:
                            self.positions[asset] = shares
                            self.cash -= total_cost
    
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
        
        # Rebalance portfolio if we have enough history and it's a rebalance day
        if (current_step >= self.lookback_period and 
            current_step % self.rebalance_frequency == 0):
            
            momentum_scores = self.calculate_momentum()
            
            # Only rebalance if we have scores for at least top_n assets
            if len(momentum_scores) >= self.top_n:
                self.rebalance_portfolio(prices, momentum_scores)
        
        # Calculate portfolio value
        portfolio_value = self.cash
        current_positions = {}
        
        for asset, shares in self.positions.items():
            if asset in prices:
                asset_value = shares * prices[asset]
                portfolio_value += asset_value
                current_positions[asset] = shares
        
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
