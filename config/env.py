"""Environment configuration."""

import numpy as np
ENV_PARAMS = {
    "initial_capital": 100000,
    "window_size": 3,
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
        'min': 0,
        'max': 100
    },
    'cash_limit': {
        'min': 0,
        'max': 100000 * 2
    }
}

REWARD_PARAMS = {
    "returns_based": ("returns_based", {
        "scale": 10.0
    }),
    "sharpe_based": ("sharpe_based", {
        "annual_risk_free_rate": 0.02,
        "annualization_factor": 252,
        "window_size": 20,
        "min_history_size": 10,
        "scale": 1.0
    }),
}