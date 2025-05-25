# DRL-Finance

A comprehensive deep reinforcement learning framework for financial trading applications. This project implements DRL algorithms for algorithmic trading with realistic market conditions, advanced data processing, and comprehensive experiment management.

## 🚀 Features

- **Advanced RL Algorithms**: DQN, DDPG, PPO, A2C with Bayesian neural network support
- **Realistic Trading Environment**: Built on Gymnasium with market frictions, constraints, and multiple reward functions
- **Comprehensive Data Pipeline**: Yahoo Finance integration with technical indicators, VIX, and turbulence data
- **Experiment Management**: Complete lifecycle management with checkpointing, metrics tracking, and visualization
- **Flexible Action Interpreters**: Discrete and confidence-scaled action spaces
- **Advanced Visualization**: Trading performance, data analysis, and comprehensive backtesting reports
- **Modular Architecture**: Easily extensible components for agents, rewards, constraints, and data sources

## 📁 Project Structure

```
DRL-Finance/
├── config/                     # Configuration files and parameters
│   ├── data.py                 # Data processing configurations
│   ├── env.py                  # Environment parameters
│   ├── models.py               # Model hyperparameters
│   ├── networks.py             # Neural network architectures
│   ├── tickers.py              # Stock universe definitions
│   └── interpreter.py          # Action interpreter configs
├── data/                       # Data management and processing
│   ├── sources/                # Data providers (Yahoo Finance)
│   ├── processors/             # Technical indicators, VIX, turbulence
│   ├── raw/                    # Raw downloaded data
│   ├── processed/              # Processed data with indicators
│   └── normalized/             # Normalized data for training
├── environments/               # Trading environments
│   ├── constraints/            # Position limits and trading constraints
│   ├── market_friction/        # Slippage and commission models
│   ├── rewards/                # Multiple reward function implementations
│   ├── processors/             # Environment-specific processors
│   └── trading_env.py          # Main trading environment
├── models/                     # RL models and components
│   ├── agents/                 # RL algorithm implementations
│   ├── action_interpreters/    # Action space handlers
│   ├── networks/               # Neural network architectures
│   └── baseline/               # Baseline strategies
├── managers/                   # Experiment and system management
│   ├── experiment_manager.py   # Complete experiment lifecycle
│   ├── metrics_manager.py      # Performance metrics tracking
│   ├── checkpoint_manager.py   # Model checkpointing
│   ├── visualization_manager.py # Visualization generation
│   ├── backtest_manager.py     # Backtesting framework
│   ├── config_manager.py       # Configuration management
│   └── data_manager.py         # Experiment data management
├── scripts/                    # Experiment execution scripts
│   ├── setup_experiment.py     # Experiment setup and data preparation
│   ├── start_experiment.py     # Training execution
│   ├── continue_experiment.py  # Resume training from checkpoint
│   ├── baseline.py             # Baseline strategy evaluation
│   └── visualize_data.py       # Data visualization utilities
├── utils/                      # Utility functions and logging
├── visualization/              # Visualization tools
├── tests/                      # Unit and integration tests
└── experiments/                # Experiment outputs and results
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/kveje/DRL-Finance.git
cd DRL-Finance

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- **Core**: `numpy`, `pandas`, `torch`, `gymnasium`
- **Financial Data**: `yfinance`, `finrl`
- **Visualization**: `matplotlib`, `seaborn`

## 🎯 Quick Start

### 1. Set Up an Experiment

```bash
# Basic experiment setup with DQN on DOW 30 stocks
python scripts/setup_experiment.py \
    --experiment-name "my_trading_experiment" \
    --agent-type dqn \
    --assets AAPL MSFT GOOGL \
    --train-start-date "2020-01-01" \
    --train-end-date "2022-12-31" \
    --val-start-date "2023-01-01" \
    --val-end-date "2023-12-31"

# Advanced setup with technical indicators and VIX data
python scripts/setup_experiment.py \
    --experiment-name "advanced_experiment" \
    --agent-type a2c \
    --indicator-type advanced \
    --use-vix \
    --use-turbulence \
    --visualize-data \
    --interpreter-type confidence_scaled
```

### 2. Start Training

```bash
# Start training the experiment
python scripts/start_experiment.py \
    --experiment-name "my_trading_experiment" \
    --n-episodes 10000 \
    --max-train-time 86400  # 24 hours
```

### 3. Continue Training

```bash
# Resume training from the latest checkpoint
python scripts/continue_experiment.py \
    --experiment-name "my_trading_experiment" \
    --episodes 5000
```

## 📊 Available Configurations

### RL Agents
- **DQN**: Deep Q-Network with experience replay and target networks
- **DDPG**: Deep Deterministic Policy Gradient for continuous actions
- **PPO**: Proximal Policy Optimization with clipped objectives
- **A2C**: Advantage Actor-Critic with entropy regularization

### Action Interpreters
- **Discrete**: Traditional discrete action spaces (buy/sell/hold)
- **Confidence Scaled**: Continuous actions scaled by confidence levels

### Stock Universes
- **DOW 30**: Dow Jones Industrial Average components
- **NASDAQ 100**: NASDAQ-100 index components  
- **S&P 500**: S&P 500 index components
- **Custom**: Specify your own list of tickers

### Technical Indicators
- **No Indicators**: Price and volume data only
- **Simple**: Basic indicators (SMA, EMA, RSI)
- **Advanced**: Extended set including MACD, Bollinger Bands, Stochastic
- **Full**: Complete technical analysis suite

### Reward Functions
- **Returns**: Simple return-based rewards
- **Sharpe**: Sharpe ratio optimization
- **Log Returns**: Log-transformed returns
- **Constraint Violation**: Penalties for constraint violations
- **Zero Action**: Penalties for inaction

## 🔧 Advanced Usage

### Custom Agent Development

```python
from models.agents.base_agent import BaseAgent
from models.networks.dqn_network import DQNNetwork

class CustomAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.network = DQNNetwork(state_dim, action_dim)
    
    def act(self, state, training=True):
        # Implement custom action selection
        pass
    
    def update(self, batch):
        # Implement custom learning update
        pass
```

### Custom Reward Function

```python
from environments.rewards.base_reward import BaseReward

class CustomReward(BaseReward):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def calculate_reward(self, prev_portfolio_value, curr_portfolio_value, 
                        action, positions, **kwargs):
        # Implement custom reward logic
        return reward
```

### Programmatic Experiment Setup

```python
from managers.experiment_manager import ExperimentManager
from models.agents.agent_factory import AgentFactory
from environments.trading_env import TradingEnv
from data.data_manager import DataManager

# Initialize data manager
data_manager = DataManager()

# Download and process data
raw_data = data_manager.download_data(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2023-12-31",
    source="yahoo"
)

processed_data = data_manager.process_data(
    data=raw_data,
    processors=["technical_indicator"],
    processor_params={"technical_indicator": {"indicators": ["sma", "rsi"]}}
)

# Create environments
train_env = TradingEnv(
    processed_data=train_data,
    raw_data=raw_train_data,
    columns=columns,
    env_params={"initial_balance": 100000, "window_size": 10},
    friction_params={"commission": {"commission_rate": 0.001}},
    reward_params=("sharpe", {"lookback_window": 30})
)

# Create agent
agent = AgentFactory.create_agent(
    agent_type="dqn",
    env=train_env,
    learning_rate=0.001,
    gamma=0.99,
    epsilon_decay=10000
)

# Create and run experiment
experiment = ExperimentManager(
    experiment_name="programmatic_experiment",
    train_env=train_env,
    val_env=val_env,
    agent=agent,
    eval_interval=10,
    save_interval=50
)

experiment.train(n_episodes=1000)
```

## 📈 Experiment Management

### Directory Structure
Each experiment creates a structured directory:
```
experiments/my_experiment/
├── config/                 # Experiment configuration files
├── checkpoints/            # Model checkpoints
├── metrics/                # Training and evaluation metrics
├── visualizations/         # Generated plots and charts
├── backtest/              # Backtesting results
├── logs/                  # Training logs
└── data/                  # Experiment-specific data
```

### Metrics Tracking
- **Training Metrics**: Episode rewards, losses, learning curves
- **Evaluation Metrics**: Sharpe ratio, total return, maximum drawdown
- **Trading Metrics**: Win rate, average trade return, portfolio volatility
- **Risk Metrics**: VaR, CVaR, Sortino ratio, Calmar ratio

### Visualization
- **Portfolio Performance**: Value over time, drawdown analysis
- **Trading Activity**: Position changes, action distributions
- **Data Analysis**: Price movements, indicator correlations
- **Comparison Charts**: Train vs validation performance

## 🧪 Testing and Validation

```bash
# Run unit tests
python -m tests.run_all_tests
```

## 📚 Documentation

The thesis will be available in the `docs/` directory once completed.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [FinRL](https://github.com/AI4Finance-Foundation/FinRL) for financial data utilities
- Uses [Gymnasium](https://gymnasium.farama.org/) for the RL environment interface
- Inspired by modern algorithmic trading research and practices

---

**Note**: This framework is for research and educational purposes!