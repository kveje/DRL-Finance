import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import random

class RandomStrategy:
    """
    Random trading strategy that makes random buy/sell/hold decisions.
    Serves as a baseline comparison for algorithmic strategies.
    """
    
    def __init__(self, 
                 assets: List[str], 
                 initial_capital: float = 10000.0,
                 trade_probability: float = 0.2,
                 position_size: float = 0.1,
                 transaction_cost: float = 0.001,
                 seed: int = None):
        """
        Initialize the Random strategy.
        
        Args:
            assets: List of asset tickers to trade
            initial_capital: Starting capital
            trade_probability: Probability of making a trade on any given day
            position_size: Percentage of portfolio to allocate to each position
            transaction_cost: Cost of transaction as a fraction of trade value
            seed: Random seed for reproducibility
        """
        self.assets = assets
        self.initial_capital = initial_capital
        self.trade_probability = trade_probability
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
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
    
    def make_random_trades(self, prices: Dict[str, float]):
        """
        Make random trading decisions.
        
        Args:
            prices: Dictionary mapping assets to their current prices
        """
        portfolio_value = self.calculate_portfolio_value(prices)
        
        for asset in self.assets:
            if asset not in prices:
                continue
                
            # Randomly decide whether to trade this asset
            if random.random() < self.trade_probability:
                current_price = prices[asset]
                current_shares = self.positions[asset]
                
                # Random choice: 0 = sell, 1 = hold, 2 = buy
                action = random.randint(0, 2)
                
                # Sell action
                if action == 0 and current_shares > 0:
                    # Sell all shares
                    sell_value = current_shares * current_price
                    transaction_fee = sell_value * self.transaction_cost
                    
                    self.cash += sell_value - transaction_fee
                    self.positions[asset] = 0
                
                # Buy action
                elif action == 2:
                    # Calculate position size based on portfolio value
                    amount_to_invest = portfolio_value * self.position_size
                    shares_to_buy = int(amount_to_invest / current_price)
                    
                    if shares_to_buy > 0:
                        # Execute purchase
                        cost = shares_to_buy * current_price
                        transaction_fee = cost * self.transaction_cost
                        total_cost = cost + transaction_fee
                        
                        if total_cost <= self.cash:
                            # Add to existing position if any
                            self.positions[asset] += shares_to_buy
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
        # Make random trades
        self.make_random_trades(prices)
        
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