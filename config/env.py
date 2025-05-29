"""Environment configuration."""

import numpy as np
from copy import deepcopy
from typing import Dict, List, Optional

ENV_PARAMS = {
    "initial_capital": 100000,
    "window_size": 20,
}
 
MARKET_FRIC_PARAMS = {
    'slippage': {
        'slippage_mean': 0.0001,  # 0.01% mean slippage
        'slippage_std': 0.001     # 0.1% standard deviation
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

# Available reward types:
REWARD_PARAMS = {
    "returns": {
        "scale": 10
    },
    "log_returns": {
        "scale": 10
    },
    "sharpe": {
        "annual_risk_free_rate": 0.01,
        "annualization_factor": 252,
        "window_size": 20,
        "min_history_size": 2,
        "scale": 0.01
    },
    "constraint_violation": {
        "scale": 0.01
    },
    "zero_action": {
        "scale": 0.01,
        "window_size": 10,
        "min_consecutive_days": 5
    },
    "projected_log_returns": {
        "projection_period": 20,
        "scale": 5
    },
    "projected_returns": {
        "projection_period": 20,
        "scale": 5
    },
    "projected_sharpe": {
        "projection_period": 20,
        "scale": 0.01
    },
    "projected_max_drawdown": {
        "projection_period": 20,
        "scale": 0.5
    },
    "max_drawdown": {
        "window_size": 20,
        "scale": 0.5
    }
}

def get_reward_config(reward_config_str: str, projection_period: Optional[int] = 10) -> Dict:
    """Get the reward config based on the reward config string."""
    config = {}
    if projection_period is not None:
        if reward_config_str == "returns":
            config["projected_returns"] = REWARD_PARAMS["projected_returns"]
            config["projected_returns"]["projection_period"] = projection_period
        elif reward_config_str == "log_returns":
            config["projected_log_returns"] = REWARD_PARAMS["projected_log_returns"]
            config["projected_log_returns"]["projection_period"] = projection_period
        elif reward_config_str == "risk_adjusted":
            config["projected_sharpe"] = REWARD_PARAMS["projected_sharpe"]
            config["projected_max_drawdown"] = REWARD_PARAMS["projected_max_drawdown"]
            config["projected_sharpe"]["projection_period"] = projection_period
            config["projected_max_drawdown"]["projection_period"] = projection_period
    else:
        if reward_config_str == "returns":
            config["returns"] = REWARD_PARAMS["returns"]
        elif reward_config_str == "log_returns":
            config["log_returns"] = REWARD_PARAMS["log_returns"]
        elif reward_config_str == "risk_adjusted":
            config["sharpe"] = REWARD_PARAMS["sharpe"]

    # Add constraint violation and zero action
    config["constraint_violation"] = REWARD_PARAMS["constraint_violation"]
    config["zero_action"] = REWARD_PARAMS["zero_action"]

    return config

# Default configurations for all processors
PROCESSOR_CONFIGS = {
    'price': {
        'type': 'price',
        'kwargs': {
            'window_size': ENV_PARAMS['window_size'],
            'asset_list': None  # Will be set dynamically
        }
    },
    'cash': {
        'type': 'cash',
        'kwargs': {
            'cash_limit': CONSTRAINT_PARAMS['cash_limit']['max']
        }
    },
    'position': {
        'type': 'position',
        'kwargs': {
            'position_limits': CONSTRAINT_PARAMS['position_limits'],
            'asset_list': None  # Will be set dynamically
        }
    },
    'ohlcv': {
        'type': 'ohlcv',
        'kwargs': {
            'window_size': ENV_PARAMS['window_size'],
            'ohlcv_cols': ['open', 'high', 'low', 'close', 'volume'],
            'asset_list': None  # Will be set dynamically
        }
    },
    'tech': {
        'type': 'tech',
        'kwargs': {
            'tech_cols': None, # Will be set dynamically
            'asset_list': None  # Will be set dynamically
        }
    },
    'affordability': {
        'type': 'affordability',
        'kwargs': {
            'n_assets': None,  # Will be set dynamically
            'min_cash_limit': CONSTRAINT_PARAMS['cash_limit']['min'],
            'max_trade_size': 10,
            'price_col': 'close',
            'transaction_cost': MARKET_FRIC_PARAMS['commission']['commission_rate'],
            'slippage_mean': MARKET_FRIC_PARAMS['slippage']['slippage_mean']
        }
    },
    'current_price': {
        'type': 'current_price',
        'kwargs': {
            'n_assets': None,  # Will be set dynamically
            'min_cash_limit': CONSTRAINT_PARAMS['cash_limit']['min'],
            'price_col': 'close',
            'transaction_cost': MARKET_FRIC_PARAMS['commission']['commission_rate'],
            'slippage_mean': MARKET_FRIC_PARAMS['slippage']['slippage_mean']
        }
    }
}

def get_processor_config(price_type: str, n_assets: int, asset_list: List[str], tech_cols: Optional[List[str]] = None, price_col: str = "close") -> List[Dict]:
    """
    Get the processor config based on the price type, number of assets, asset list, technical indicators, and price column.

    Args:
        price_type (str): The type of price to use.
        n_assets (int): The number of assets to use.
        asset_list (List[str]): The list of assets to use.
        tech_cols (Optional[List[str]]): The technical indicators to use.
        price_col (str): The price column to use.

    Returns:
        List[Dict]: The processor config.
    """
    dict_config = deepcopy(PROCESSOR_CONFIGS)

    # Update asset_list and n_assets for each config
    for config in dict_config.values():
        if "kwargs" in config and "asset_list" in config["kwargs"]:
            config["kwargs"]["asset_list"] = asset_list
        if "kwargs" in config and "n_assets" in config["kwargs"]:
            config["kwargs"]["n_assets"] = n_assets
        if "kwargs" in config and "price_col" in config["kwargs"]:
            config["kwargs"]["price_col"] = price_col

    # Remove price, ohlcv, and tech configs
    config = list(config for config in dict_config.values() if config['type'] not in ["price", "ohlcv", "tech"])

    if price_type == "price":
        config.append(dict_config['price'])
    elif price_type == "ohlcv":
        config.append(dict_config['ohlcv'])
    elif price_type == "both":
        config.append(dict_config['price'])
        config.append(dict_config['ohlcv'])

    if tech_cols is not None:
        dict_config['tech']['kwargs']['tech_cols'] = tech_cols
        config.append(dict_config['tech'])

    return config
