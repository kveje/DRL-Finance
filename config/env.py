"""Environment configuration."""

import numpy as np
ENV_PARAMS = {
    "initial_capital": 100000,
    "transaction_fee_percent": 0.001,
    "window_size": 10,
}

MARKET_FRIC_PARAMS = {
    'slippage': {
        'slippage_mean': 0.0,
        'slippage_std': 0.001
    },
    'commission': {
        'commission_rate': 0.001
    }
}

CONSTRAINT_PARAMS = {
    'position_limits': {
        'min': -1000,
        'max': 1000
    },
    'cash_limit': {
        'min': 0,
        'max': np.inf
    }
}

REWARD_PARAMS = {
    "returns_based": ("returns_based", {
        "scale": 1.0
    }),
    "sharpe_based": ("sharpe_based", {
        "annual_risk_free_rate": 0.01,
        "annualization_factor": 252,
        "window_size": 20,
        "min_history_size": 10,
        "scale": 1.0
    }),
}


