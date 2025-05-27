import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class MeanReversionStrategy:
    """
    Mean Reversion strategy based on normalized price deviations (z-scores).
    Implements the formulation from thesis with proper buy/sell thresholds and position sizing.
    """
    
    def __init__(self, 
                 assets: List[str], 
                 initial_capital: float = 10000.0,
                 window_size: int = 20,
                 theta_buy: float = 0.5,  # Buy threshold (z-score above which we buy)
                 theta_sell: float = -0.5,  # Sell threshold (z-score below which we sell)
                 alpha: float = 0.2,  # Position sizing parameter
                 z_max: float = 2.0,  # Maximum z-score for position sizing
                 transaction_cost: float = 0.001):
        """
        Initialize the Mean Reversion strategy.
        
        Args:
            assets: List of asset tickers to trade
            initial_capital: Starting capital
            window_size: Window size for calculating moving average and standard deviation
            theta_buy: Z-score threshold above which we buy (positive value)
            theta_sell: Z-score threshold below which we sell (negative value)
            alpha: Position sizing parameter (fraction of portfolio value)
            z_max: Maximum z-score for position sizing calculations
            transaction_cost: Cost of transaction as a fraction of trade value
        """
        self.assets = assets
        self.initial_capital = initial_capital
        self.window_size = window_size
        self.theta_buy = theta_buy
        self.theta_sell = theta_sell
        self.alpha = alpha
        self.z_max = z_max
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
    
    def calculate_z_scores(self) -> Dict[str, float]:
        """
        Calculate z-scores for each asset based on thesis formulation:
        z_i(t) = -(P_i(t) - μ_i(t)) / σ_i(t)
        
        Returns:
            Dictionary mapping assets to z-scores
        """
        z_scores = {}
        
        for asset, prices in self.price_history.items():
            if len(prices) >= self.window_size:
                # Calculate mean and standard deviation for the window
                window = prices[-self.window_size:]
                mu_i = np.mean(window)  # μ_i(t)
                sigma_i = np.std(window)  # σ_i(t)
                
                if sigma_i > 0:  # Avoid division by zero
                    # Calculate z-score with negative sign as per thesis
                    current_price = prices[-1]  # P_i(t)
                    z_score = -(current_price - mu_i) / sigma_i
                    z_scores[asset] = z_score
        
        return z_scores
    
    def calculate_position_change(self, asset: str, z_score: float, portfolio_value: float, current_price: float) -> int:
        """
        Calculate position change based on thesis formulation.
        
        Args:
            asset: Asset ticker
            z_score: Current z-score for the asset
            portfolio_value: Current portfolio value
            current_price: Current asset price
            
        Returns:
            Change in position (positive for buy, negative for sell)
        """
        current_shares = self.positions[asset]
        
        # Buy signal: z_i(t) > θ_buy
        if z_score > self.theta_buy:
            # Δq_i(t) = floor(α * V_t * min(z_i(t) - θ_buy, z_max) / P_i(t))
            signal_strength = min(z_score - self.theta_buy, self.z_max)
            shares_to_buy = int((self.alpha * portfolio_value * signal_strength) / current_price)
            return shares_to_buy
        
        # Sell signal: z_i(t) < θ_sell
        elif z_score < self.theta_sell:
            # Δq_i(t) = -min(q_i(t-1), floor(α * V_t * min(θ_sell - z_i(t), z_max) / P_i(t)))
            signal_strength = min(self.theta_sell - z_score, self.z_max)
            shares_to_sell = int((self.alpha * portfolio_value * signal_strength) / current_price)
            # Can't sell more than we own
            shares_to_sell = min(current_shares, shares_to_sell)
            return -shares_to_sell
        
        # No signal: z_i(t) between θ_sell and θ_buy
        else:
            return 0

    def execute_trades(self, prices: Dict[str, float], z_scores: Dict[str, float]):
        """
        Execute trades based on z-scores using thesis formulation.
        
        Args:
            prices: Dictionary mapping assets to their current prices
            z_scores: Dictionary mapping assets to z-scores
        """
        portfolio_value = self.calculate_portfolio_value(prices)
        
        for asset, z_score in z_scores.items():
            if asset not in prices:
                continue
                
            current_price = prices[asset]
            position_change = self.calculate_position_change(asset, z_score, portfolio_value, current_price)
            
            if position_change > 0:  # Buy
                # Execute purchase
                cost = position_change * current_price
                transaction_fee = cost * self.transaction_cost
                total_cost = cost + transaction_fee
                
                if total_cost <= self.cash:
                    self.positions[asset] += position_change
                    self.cash -= total_cost
            
            elif position_change < 0:  # Sell
                shares_to_sell = -position_change
                # Execute sell
                sell_value = shares_to_sell * current_price
                transaction_fee = sell_value * self.transaction_cost
                
                self.cash += sell_value - transaction_fee
                self.positions[asset] -= shares_to_sell

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
        
        # Execute trades if we have enough price history for any asset
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
