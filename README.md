# DRL-Finance

This is a repository for my masters thesis about deep reinforcement learning in finance.

## Dependencies

Install dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Project Structure

financial_rl/
├── config/ ✅
│ ├── env.py # Environment configuration
│ ├── data.py # Data processing configuration
│ ├── models.py # Model architecture configuration
│ ├── path.py # Path configuration
│ └── tickers.py # Ticker symbols configuration
├── data/ ✅
│ ├── __init__.py
│ ├── data_manager.py # Unified data management
│ ├── sources/ # Data source implementations
│ │ ├── __init__.py
│ │ ├── base_source.py # Abstract base class
│ │ ├── yahoo_finance.py
│ │ ├── alpha_vantage.py
│ │ └── csv_source.py
│ ├── processors/ # Data processing pipeline
│ │ ├── __init__.py
│ │ ├── feature_engineering.py
│ │ ├── normalization.py
│ │ └── universe_selection.py
│ └── market/ # Market specific logic
│ ├── __init__.py
│ ├── market_data.py # Market data container
│ ├── universe.py # Asset universe management
│ └── synchronization.py # Cross-market time syncing
├── environments/ ✅
│ ├── __init__.py
│ ├── base_env.py # Abstract gym.Env base
│ ├── trading_env.py # Main trading environment
│ ├── market_friction/ # Market friction models
│ │ ├── __init__.py
│ │ ├── slippage.py
│ │ ├── commission.py
│ │ └── market_impact.py
│ ├── constraints/ # Trading constraints
│ │ ├── __init__.py
│ │ ├── position_limits.py
│ │ ├── risk_limits.py
│ │ └── regulatory_limits.py
│ └── rewards/ # Reward functions
│ ├── __init__.py
│ ├── returns_based.py
│ ├── sharpe_based.py
│ └── risk_adjusted.py
├── models/ 🔄
│ ├── __init__.py
│ ├── networks/ # Neural network architectures
│ │ ├── __init__.py
│ │ ├── mlp.py
│ │ ├── lstm.py
│ │ └── transformer.py
│ ├── agents/ # RL agents
│ │ ├── __init__.py
│ │ ├── base_agent.py
│ │ ├── dqn.py
│ │ ├── ppo.py
│ │ └── sac.py
│ └── risk/ # Risk-sensitive components
│ ├── __init__.py
│ ├── bayesian_layer.py
│ ├── distributional_rl.py
│ └── risk_measures.py
├── training/ 🔄
│ ├── __init__.py
│ ├── trainer.py # Main training loop
│ ├── hyperopt/ # Hyperparameter optimization
│ │ ├── __init__.py
│ │ ├── wandb_sweep.py # W&B integration
│ │ ├── optuna_optimizer.py # Alternative to W&B
│ │ └── param_space.py # Parameter space definitions
│ └── callbacks/ # Training callbacks
│ ├── __init__.py
│ ├── checkpoint.py
│ ├── early_stopping.py
│ └── logging.py
├── evaluation/ 🔄
│ ├── __init__.py
│ ├── backtest.py # Backtesting framework
│ ├── metrics/ # Performance metrics
│ │ ├── __init__.py
│ │ ├── returns.py
│ │ ├── risk.py
│ │ └── trading.py # Trading-specific metrics
│ └── visualization/ # Results visualization
│ ├── __init__.py
│ ├── performance_plots.py
│ ├── trade_analysis.py
│ └── risk_analysis.py
├── experiments/ ⏳
│ ├── __init__.py
│ ├── experiment.py # Experiment base class
│ ├── ablation.py # Ablation study framework
│ ├── hyperparameter_sweep.py # Sweep configuration
│ └── baseline/ # Baseline strategies
│ ├── __init__.py
│ ├── buy_and_hold.py
│ ├── momentum.py
│ └── mean_reversion.py
├── utils/ ✅
│ ├── __init__.py
│ ├── config.py # Configuration utilities
│ ├── logger.py # Logging setup
│ ├── reproducibility.py # Seed and version control
│ └── profiling.py # Performance profiling
├── notebooks/ 🔄
│ ├── data_exploration.ipynb
│ ├── model_analysis.ipynb
│ └── results_visualization.ipynb
├── scripts/ 🔄
│ ├── download_data.py
│ ├── preprocess_data.py
│ └── run_experiment.py
├── tests/ 🔄
│ ├── __init__.py
│ ├── test_env.py
│ ├── test_models.py
│ └── test_data.py
├── docker/ ⏳
│ ├── Dockerfile
│ └── docker-compose.yml
├── requirements.txt ✅
├── setup.py ⏳
├── .env.example ⏳
├── .gitignore ✅
├── README.md ✅
└── main.py 🔄

Legend:
✅ Done - Implemented and functional
🔄 In Progress - Partially implemented
⏳ Not Begun - Empty or not started
