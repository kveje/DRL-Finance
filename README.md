# DRL-Finance

A deep reinforcement learning framework for financial trading applications. This project implements various RL algorithms for trading in financial markets with realistic constraints, market frictions, and advanced reward functions.

## Features

- **Modular Trading Environment**: Built on OpenAI Gym/Gymnasium with realistic market simulations
- **Multiple RL Algorithms**: Implementations of DQN, Directional DQN, PPO, and A2C agents
- **Realistic Market Conditions**: Slippage, commission fees, and position constraints
- **Comprehensive Experiment Management**: Training, validation, visualization, and metrics tracking
- **Backtesting Framework**: Evaluate strategies on historical data
- **Advanced Visualization**: Trading activity, portfolio performance, and data insights

## Project Structure

```
DRL-Finance/
├── config/             # Configuration files and parameters
├── data/               # Data management, sources and processing
│   ├── processors/     # Data preprocessing components
│   └── sources/        # Data providers (Yahoo Finance, etc.)
├── environments/       # Trading environments
│   ├── constraints/    # Trading constraints (position limits, etc.)
│   ├── market_friction/# Market frictions (slippage, commission)
│   └── rewards/        # Reward functions (returns-based, Sharpe ratio)
├── models/             # RL models and components
│   ├── agents/         # RL algorithm implementations
│   ├── action_interpreters/ # Action space handlers
│   └── networks/       # Neural network architectures
├── scripts/            # Experiment and training scripts
├── tests/              # Unit and integration tests
├── utils/              # Utility functions and logging
└── visualization/      # Visualization tools for trading and data
```

## Installation

```bash
# Clone the repository
git clone https://github.com/kveje/DRL-Finance.git
cd DRL-Finance

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Setting Up an Experiment

```python
from models.experiment_manager import ExperimentManager
from models.agents.dqn_agent import DQNAgent
from environments.trading_env import TradingEnv
from models.backtesting import Backtester

# Create environments
train_env = TradingEnv(
    processed_data=train_data,
    raw_data=train_raw_data,
    columns=data_columns,
    env_params={"initial_balance": 100000, "window_size": 10},
    friction_params={"commission": {"commission_rate": 0.001}},
    reward_params=("returns_based", {"scale": 1.0})
)

val_env = TradingEnv(
    # Similar configuration as train_env but with validation data
)

# Create agent
agent = DQNAgent(
    state_dim=train_env.observation_space,
    action_dim=train_env.action_space,
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=10000
)

# Create backtester
backtester = Backtester(val_env)

# Create experiment manager
experiment = ExperimentManager(
    experiment_name="dqn_trading_experiment",
    train_env=train_env,
    val_env=val_env,
    agent=agent,
    backtester=backtester,
    eval_interval=10,
    save_interval=5
)

# Run training
experiment.train(n_episodes=1000)
```

### Running Scripts

```bash
# Setup a new experiment
python scripts/setup_experiment.py --config config/dqn_experiment.json

# Start training
python scripts/start_experiment.py --name dqn_trading_experiment

# Continue training from a checkpoint
python scripts/continue_experiment.py --name dqn_trading_experiment --episodes 500
```

## Extending the Framework

### Adding a New RL Agent

Create a new agent class in `models/agents/` that inherits from `BaseAgent` and implements the required methods. See the existing agents for examples.

### Adding a New Reward Function

Implement a new reward function in `environments/rewards/` and register it in the `RewardManager`.

### Custom Data Sources

Add new data sources in `data/sources/` and implement the appropriate preprocessing in `data/processors/`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
