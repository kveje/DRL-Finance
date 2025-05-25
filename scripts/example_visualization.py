import numpy as np
from managers.visualization_manager import VisualizationManager
from datetime import datetime, timedelta

def generate_sample_data(n_steps=1000):
    """Generate synthetic trading data for visualization testing."""
    
    # Generate time series data
    dates = list(range(n_steps))
    
    # Generate portfolio values with some realistic patterns
    base_value = 100000
    trend = np.linspace(0, 0.5, n_steps)  # Upward trend
    noise = np.random.normal(0, 0.02, n_steps)  # Random noise
    volatility = np.sin(np.linspace(0, 8*np.pi, n_steps)) * 0.1  # Cyclical pattern
    portfolio_values = base_value * (1 + trend + noise + volatility)
    
    # Generate positions for 3 assets
    positions = np.zeros((n_steps, 3))
    for i in range(3):
        positions[:, i] = np.random.normal(0.3, 0.1, n_steps)  # Random positions around 0.3
    
    # Generate trading actions
    actions = np.zeros((n_steps, 3))
    for i in range(3):
        actions[:, i] = np.random.normal(0, 0.1, n_steps)  # Random actions
    
    # Generate returns
    returns = np.random.normal(0.001, 0.02, n_steps-1)  # Daily returns (one less than portfolio values)
    
    # Generate rewards and reward components
    rewards = returns * 100  # Scaled returns as rewards
    
    # Generate different reward components
    reward_components = {
        'returns': returns * 80,  # Main returns component
        'sharpe': np.random.normal(0.1, 0.05, n_steps-1),  # Sharpe-based component
        'constraint_violation': -np.abs(actions[:-1, :].sum(axis=1)) * 10,  # Penalty for large positions
        'zero_action': -np.random.choice([0, 1], size=n_steps-1, p=[0.7, 0.3]) * 5,  # Penalty for no action
        'log_returns': np.log1p(returns) * 50  # Log returns component
    }
    
    # Generate cash balance
    cash = np.random.normal(20000, 5000, n_steps)  # Cash balance around $20,000
    
    # Generate trade history
    trade_history = []
    for step in range(0, n_steps, 50):  # Create a trade every 50 steps
        trade_history.append({
            'step': step,
            'portfolio_value': portfolio_values[step],
            'action': np.random.choice(['buy', 'sell']),
            'asset': np.random.choice(['Asset1', 'Asset2', 'Asset3']),
            'quantity': np.random.normal(100, 20)
        })
    
    # Calculate some metrics
    metrics = {
        'sharpe_ratio': 1.5,
        'sortino_ratio': 2.1,
        'calmar_ratio': 1.8,
        'max_drawdown': 0.15,
        'volatility': 0.18,
        'win_rate': 0.65,
        'avg_trade_return': 0.02
    }
    
    return {
        'portfolio_values': portfolio_values,
        'positions': positions,
        'actions': actions,
        'returns': returns,
        'rewards': rewards,
        'reward_components': reward_components,  # Add reward components
        'cash': cash,
        'metrics': metrics,
        'trade_history': trade_history
    }

def main():
    # Create visualization manager
    viz_manager = VisualizationManager()
    
    # Generate sample data
    asset_names = ['Asset1', 'Asset2', 'Asset3']
    backtest_data = generate_sample_data()
    
    # Create and save visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"example_visualization_{timestamp}.png"
    title = "Example Trading Strategy Visualization with Reward Components"
    
    viz_manager.create_and_save_backtest_visualization(
        asset_names=asset_names,
        backtest_data=backtest_data,
        filename=filename,
        title=title
    )
    
    print(f"Visualization saved as: {filename}")

if __name__ == "__main__":
    main() 