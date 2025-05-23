import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class EqualWeightStrategy:
    """
    Equal Weight (1/N) strategy that allocates capital equally across all assets
    and rebalances the portfolio periodically.
    """
    
    def __init__(self, 
                 assets: List[str], 
                 initial_capital: float = 10000.0,
                 rebalance_frequency: int = 20,  # Rebalance every 20 days
                 transaction_cost: float = 0.001):
        """
        Initialize the Equal Weight strategy.
        
        Args:
            assets: List of asset tickers to trade
            initial_capital: Starting capital
            rebalance_frequency: How often to rebalance the portfolio (in days)
            transaction_cost: Cost of transaction as a fraction of trade value
        """
        self.assets = assets
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        
        self.positions = {asset: 0 for asset in assets}
        self.cash = initial_capital
        self.portfolio_value_history = []
        self.position_history = []
    
    def reset(self):
        """Reset the strategy to initial state."""
        self.positions = {asset: 0 for asset in self.assets}
        self.cash = self.initial_capital
        self.portfolio_value_history = []
        self.position_history = []
    
    def rebalance_portfolio(self, prices: Dict[str, float]):
        """
        Rebalance portfolio to equal weights.
        
        Args:
            prices: Dictionary mapping assets to their current prices
        """
        # Calculate current portfolio value
        portfolio_value = self.calculate_portfolio_value(prices)
        
        # First sell all positions to get a clean slate
        for asset, shares in self.positions.items():
            if shares > 0 and asset in prices:
                sell_value = shares * prices[asset]
                transaction_fee = sell_value * self.transaction_cost
                self.cash += sell_value - transaction_fee
                self.positions[asset] = 0
        
        # Count valid assets (those with prices)
        valid_assets = [asset for asset in self.assets if asset in prices and prices[asset] > 0]
        
        if not valid_assets:
            return
        
        # Calculate equal allocation
        amount_per_asset = self.cash / len(valid_assets) * 0.99  # Keep some cash as buffer
        
        # Buy shares according to equal allocation
        for asset in valid_assets:
            current_price = prices[asset]
            if current_price <= 0:
                continue
                
            # Calculate shares to buy
            shares = int(amount_per_asset / current_price)
            
            if shares > 0:
                # Execute purchase
                cost = shares * current_price
                transaction_fee = cost * self.transaction_cost
                total_cost = cost + transaction_fee
                
                if total_cost <= self.cash:
                    self.positions[asset] = shares
                    self.cash -= total_cost
    
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
        # Initial allocation or periodic rebalancing
        if current_step == 0 or current_step % self.rebalance_frequency == 0:
            self.rebalance_portfolio(prices)
        
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