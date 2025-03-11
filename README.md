# DRL-Finance

This is a repository for my masters thesis about deep reinforcement learning in finance.

## Dependencies

Install dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Project Structure

financial_rl/
├── config/
│ ├── default.yaml # Default configuration
│ ├── markets/ # Market-specific configs
│ │ ├── sp500.yaml
│ │ ├── nasdaq.yaml
│ │ ├── eu_markets.yaml
│ │ └── asia_markets.yaml
│ ├── models/ # Model architectures
│ │ ├── dqn.yaml
│ │ ├── ppo.yaml
│ │ └── bayesian_ppo.yaml
│ └── experiments/ # Experiment configs
│ ├── baseline.yaml
│ ├── ablation_study.yaml
│ └── hyperparameter_sweep.yaml
├── data/
│ ├── **init**.py
│ ├── data_manager.py # Unified data management
│ ├── sources/ # Data source implementations
│ │ ├── **init**.py
│ │ ├── base_source.py # Abstract base class
│ │ ├── yahoo_finance.py
│ │ ├── alpha_vantage.py
│ │ └── csv_source.py
│ ├── processors/ # Data processing pipeline
│ │ ├── **init**.py
│ │ ├── feature_engineering.py
│ │ ├── normalization.py
│ │ └── universe_selection.py
│ └── market/ # Market specific logic
│ ├── **init**.py
│ ├── market_data.py # Market data container
│ ├── universe.py # Asset universe management
│ └── synchronization.py # Cross-market time syncing
├── environments/
│ ├── **init**.py
│ ├── base_env.py # Abstract gym.Env base
│ ├── trading_env.py # Main trading environment
│ ├── market_friction/ # Market friction models
│ │ ├── **init**.py
│ │ ├── slippage.py
│ │ ├── commission.py
│ │ └── market_impact.py
│ ├── constraints/ # Trading constraints
│ │ ├── **init**.py
│ │ ├── position_limits.py
│ │ ├── risk_limits.py
│ │ └── regulatory_limits.py
│ └── rewards/ # Reward functions
│ ├── **init**.py
│ ├── returns_based.py
│ ├── sharpe_based.py
│ └── risk_adjusted.py
├── models/
│ ├── **init**.py
│ ├── networks/ # Neural network architectures
│ │ ├── **init**.py
│ │ ├── mlp.py
│ │ ├── lstm.py
│ │ └── transformer.py
│ ├── agents/ # RL agents
│ │ ├── **init**.py
│ │ ├── base_agent.py
│ │ ├── dqn.py
│ │ ├── ppo.py
│ │ └── sac.py
│ └── risk/ # Risk-sensitive components
│ ├── **init**.py
│ ├── bayesian_layer.py
│ ├── distributional_rl.py
│ └── risk_measures.py
├── training/
│ ├── **init**.py
│ ├── trainer.py # Main training loop
│ ├── hyperopt/ # Hyperparameter optimization
│ │ ├── **init**.py
│ │ ├── wandb_sweep.py # W&B integration
│ │ ├── optuna_optimizer.py # Alternative to W&B
│ │ └── param_space.py # Parameter space definitions
│ └── callbacks/ # Training callbacks
│ ├── **init**.py
│ ├── checkpoint.py
│ ├── early_stopping.py
│ └── logging.py
├── evaluation/
│ ├── **init**.py
│ ├── backtest.py # Backtesting framework
│ ├── metrics/ # Performance metrics
│ │ ├── **init**.py
│ │ ├── returns.py
│ │ ├── risk.py
│ │ └── trading.py # Trading-specific metrics
│ └── visualization/ # Results visualization
│ ├── **init**.py
│ ├── performance_plots.py
│ ├── trade_analysis.py
│ └── risk_analysis.py
├── experiments/
│ ├── **init**.py
│ ├── experiment.py # Experiment base class
│ ├── ablation.py # Ablation study framework
│ ├── hyperparameter_sweep.py # Sweep configuration
│ └── baseline/ # Baseline strategies
│ ├── **init**.py
│ ├── buy_and_hold.py
│ ├── momentum.py
│ └── mean_reversion.py
├── utils/
│ ├── **init**.py
│ ├── config.py # Configuration utilities
│ ├── logger.py # Logging setup
│ ├── reproducibility.py # Seed and version control
│ └── profiling.py # Performance profiling
├── notebooks/ # Analysis notebooks
│ ├── data_exploration.ipynb
│ ├── model_analysis.ipynb
│ └── results_visualization.ipynb
├── scripts/ # Utility scripts
│ ├── download_data.py
│ ├── preprocess_data.py
│ └── run_experiment.py
├── tests/ # Unit and integration tests
│ ├── **init**.py
│ ├── test_env.py
│ ├── test_models.py
│ └── test_data.py
├── docker/ # Docker configuration
│ ├── Dockerfile
│ └── docker-compose.yml
├── requirements.txt # Package dependencies
├── setup.py # Project installation
├── .env.example # Example environment variables
├── .gitignore
├── README.md
└── main.py # Entry point
