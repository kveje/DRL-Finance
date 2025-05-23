import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class BuyAndHoldStrategy:
    """
    Simple Buy and Hold strategy that purchases assets at the beginning and holds 
    until the end of the trading period.
    """
    
    def __init__(self, 
                 assets: List[str], 
                 initial_capital: float = 100000.0,
                 allocation: Dict[str, float] = None,
                 transaction_cost: float = 0.001):
        """
        Initialize the Buy and Hold strategy.
        
        Args:
            assets: List of asset tickers to trade
            initial_capital: Starting capital
            allocation: Dictionary mapping assets to allocation weights (if None, equal weights)
            transaction_cost: Cost of transaction as a fraction of trade value
        """
        self.assets = assets
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Equal weight by default
        if allocation is None:
            self.allocation = {asset: 1.0 / len(assets) for asset in assets}
        else:
            # Normalize allocation to sum to 1.0
            total = sum(allocation.values())
            self.allocation = {k: v / total for k, v in allocation.items()}
        
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
    
    def initial_allocation(self, prices: Dict[str, float]):
        """
        Allocate initial capital according to the specified allocation weights.
        
        Args:
            prices: Dictionary mapping assets to their current prices
        """
        for asset, weight in self.allocation.items():
            if asset not in prices:
                continue
                
            # Calculate shares to buy
            amount_to_invest = self.cash * weight
            shares = int(amount_to_invest / prices[asset])
            
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
        At step 0, allocate capital according to weights.
        For all other steps, hold the position.
        
        Args:
            current_step: Current step in the trading episode
            prices: Dictionary mapping assets to their current prices
            features: Dictionary of additional features per asset (not used in this strategy)
            
        Returns:
            Dict containing current positions, cash, and portfolio value
        """
        if current_step == 0:
            self.initial_allocation(prices)
        
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
