import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class MomentumStrategy:
    """
    Momentum trading strategy based on relative strength comparison.
    Implements the thesis formulation with proper momentum scoring and position sizing.
    """
    
    def __init__(self, 
                 assets: List[str], 
                 initial_capital: float = 10000.0,
                 lookback_period: int = 20,  # τ in thesis notation
                 top_k: int = 2,  # k in thesis notation (number of top assets to hold)
                 rebalance_frequency: int = 20,  # How often to rebalance (in days)
                 transaction_cost: float = 0.001):
        """
        Initialize the Momentum strategy.
        
        Args:
            assets: List of asset tickers to trade
            initial_capital: Starting capital
            lookback_period: Lookback period τ for momentum calculation
            top_k: Number of top performing assets to hold (k in thesis)
            rebalance_frequency: How often to rebalance the portfolio (in days)
            transaction_cost: Cost of transaction as a fraction of trade value
        """
        self.assets = assets
        self.initial_capital = initial_capital
        self.lookback_period = lookback_period  # τ
        self.top_k = min(top_k, len(assets))  # k (can't be more than number of assets)
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
    
    def initial_allocation(self, prices: Dict[str, float]):
        """
        Allocate initial capital equally across all assets.
        This ensures the strategy starts with positions rather than holding cash.
        
        Args:
            prices: Dictionary mapping assets to their current prices
        """
        # Equal weight allocation across all available assets
        available_assets = [asset for asset in self.assets if asset in prices]
        if not available_assets:
            return
            
        allocation_per_asset = 1.0 / len(available_assets)
        
        for asset in available_assets:
            # Calculate shares to buy
            amount_to_invest = self.cash * allocation_per_asset
            shares = int(amount_to_invest / prices[asset])
            
            # Execute purchase
            cost = shares * prices[asset]
            transaction_fee = cost * self.transaction_cost
            total_cost = cost + transaction_fee
            
            if total_cost <= self.cash:
                self.positions[asset] = shares
                self.cash -= total_cost

    def calculate_momentum_scores(self) -> Dict[str, float]:
        """
        Calculate momentum scores for each asset based on thesis formulation:
        M_i(t, τ) = P_i(t) / P_i(t-τ) - 1
        
        Returns:
            Dictionary mapping assets to momentum scores
        """
        momentum_scores = {}
        
        for asset, prices in self.price_history.items():
            if len(prices) >= self.lookback_period + 1:  # Need τ+1 prices
                # M_i(t, τ) = P_i(t) / P_i(t-τ) - 1
                P_t = prices[-1]  # P_i(t)
                P_t_minus_tau = prices[-(self.lookback_period + 1)]  # P_i(t-τ)
                
                if P_t_minus_tau > 0:  # Avoid division by zero
                    momentum = (P_t / P_t_minus_tau) - 1
                    momentum_scores[asset] = momentum
        
        return momentum_scores
    
    def calculate_target_positions(self, prices: Dict[str, float], momentum_scores: Dict[str, float]) -> Dict[str, int]:
        """
        Calculate target positions based on thesis formulation:
        q_i(t) = floor(V_t / (k * P_i(t))) if i in top-k assets, 0 otherwise
        
        Args:
            prices: Dictionary mapping assets to their current prices
            momentum_scores: Dictionary mapping assets to momentum scores
            
        Returns:
            Dictionary mapping assets to target position sizes
        """
        target_positions = {asset: 0 for asset in self.assets}
        
        # Calculate current portfolio value
        portfolio_value = self.calculate_portfolio_value(prices)
        
        # Select top-k assets by momentum score
        if len(momentum_scores) >= self.top_k:
            top_assets = sorted(momentum_scores.keys(), 
                              key=lambda x: momentum_scores[x], 
                              reverse=True)[:self.top_k]
            
            # Calculate position sizes for top-k assets
            for asset in top_assets:
                if asset in prices and prices[asset] > 0:
                    # q_i(t) = floor(V_t / (k * P_i(t)))
                    target_shares = int(portfolio_value / (self.top_k * prices[asset]))
                    target_positions[asset] = target_shares
        
        return target_positions
    
    def rebalance_to_targets(self, prices: Dict[str, float], target_positions: Dict[str, int]):
        """
        Rebalance portfolio to match target positions.
        
        Args:
            prices: Dictionary mapping assets to their current prices
            target_positions: Dictionary mapping assets to target position sizes
        """
        # First, sell positions that need to be reduced or eliminated
        for asset in self.assets:
            if asset not in prices:
                continue
                
            current_shares = self.positions[asset]
            target_shares = target_positions[asset]
            
            if current_shares > target_shares:
                shares_to_sell = current_shares - target_shares
                if shares_to_sell > 0:
                    # Execute sell
                    sell_value = shares_to_sell * prices[asset]
                    transaction_fee = sell_value * self.transaction_cost
                    
                    self.cash += sell_value - transaction_fee
                    self.positions[asset] -= shares_to_sell
        
        # Then, buy positions that need to be increased
        for asset in self.assets:
            if asset not in prices:
                continue
                
            current_shares = self.positions[asset]
            target_shares = target_positions[asset]
            
            if target_shares > current_shares:
                shares_to_buy = target_shares - current_shares
                if shares_to_buy > 0:
                    # Execute purchase
                    cost = shares_to_buy * prices[asset]
                    transaction_fee = cost * self.transaction_cost
                    total_cost = cost + transaction_fee
                    
                    if total_cost <= self.cash:
                        self.positions[asset] += shares_to_buy
                        self.cash -= total_cost
                    else:
                        # Buy as many as we can afford
                        affordable_shares = int(self.cash / (prices[asset] * (1 + self.transaction_cost)))
                        if affordable_shares > 0:
                            cost = affordable_shares * prices[asset]
                            transaction_fee = cost * self.transaction_cost
                            total_cost = cost + transaction_fee
                            
                            self.positions[asset] += affordable_shares
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
        # Initial allocation on first step
        if current_step == 0:
            self.initial_allocation(prices)
        
        # Update price history
        for asset, price in prices.items():
            if asset in self.price_history:
                self.price_history[asset].append(price)
        
        # Rebalance portfolio if we have enough history and it's a rebalance day
        if (current_step >= self.lookback_period and 
            current_step % self.rebalance_frequency == 0):
            
            momentum_scores = self.calculate_momentum_scores()
            
            # Only rebalance if we have scores for at least top_k assets
            if len(momentum_scores) >= self.top_k:
                target_positions = self.calculate_target_positions(prices, momentum_scores)
                self.rebalance_to_targets(prices, target_positions)
        
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
