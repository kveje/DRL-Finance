# Simple Network Configurations (Price, Cash, Position inputs)
SIMPLE_DISCRETE_PARAMETRIC_CONFIG = {
    "n_assets": 3,
    "window_size": 20,
    "processors": {
        "price": {
            "enabled": True,
            "hidden_dim": 64
        },
        "position": {
            "enabled": True,
            "hidden_dim": 16
        },
        "cash": {
            "enabled": True,
            "input_dim": 2,  # [cash_balance, portfolio_value]
            "hidden_dim": 16
        }
    },
    "backbone": {
        "type": "mlp",
        "hidden_dims": [96, 64],  # Total input dim = 64 (price) + 16 (position) + 16 (cash) = 96
        "dropout": 0.0,
        "use_layer_norm": True
    },
    "heads": {
        "discrete": {
            "enabled": True,
            "type": "parametric",
            "hidden_dim": 64,
            "num_actions": 3
        }
    }
}

SIMPLE_DISCRETE_BAYESIAN_CONFIG = {
    "n_assets": 3,
    "window_size": 20,
    "processors": {
        "price": {
            "enabled": True,
            "hidden_dim": 64
        },
        "position": {
            "enabled": True,
            "hidden_dim": 16
        },
        "cash": {
            "enabled": True,
            "input_dim": 2,  # [cash_balance, portfolio_value]
            "hidden_dim": 16
        }
    },
    "backbone": {
        "type": "mlp",
        "hidden_dims": [96, 64],  # Total input dim = 64 (price) + 16 (position) + 16 (cash) = 96
        "dropout": 0.0,
        "use_layer_norm": True
    },
    "heads": {
        "discrete": {
            "enabled": True,
            "type": "bayesian",
            "hidden_dim": 64,
            "num_actions": 3,
            "bayesian_config": {
                "num_samples": 10,
                "prior_scale": 1.0
            }
        }
    }
}

# Advanced Network Configurations (OHLCV, Technical, Cash, Position inputs)
ADVANCED_DISCRETE_PARAMETRIC_CONFIG = {
    "n_assets": 3,
    "window_size": 20,
    "processors": {
        "ohlcv": {
            "enabled": True,
            "hidden_dim": 128,
            "n_heads": 4
        },
        "technical": {
            "enabled": True,
            "tech_dim": 20,
            "hidden_dim": 128
        },
        "position": {
            "enabled": True,
            "hidden_dim": 16
        },
        "cash": {
            "enabled": True,
            "input_dim": 2,  # [cash_balance, portfolio_value]
            "hidden_dim": 16
        }
    },
    "backbone": {
        "type": "mlp",
        "hidden_dims": [288, 128],  # Total input dim = 128 (ohlcv) + 128 (technical) + 16 (position) + 16 (cash) = 288
        "dropout": 0.0,
        "use_layer_norm": True
    },
    "heads": {
        "discrete": {
            "enabled": True,
            "type": "parametric",
            "hidden_dim": 128,
            "num_actions": 3
        }
    }
}

ADVANCED_DISCRETE_BAYESIAN_CONFIG = {
    "n_assets": 3,
    "window_size": 20,
    "processors": {
        "ohlcv": {
            "enabled": True,
            "hidden_dim": 128,
            "n_heads": 4
        },
        "technical": {
            "enabled": True,
            "tech_dim": 20,
            "hidden_dim": 128
        },
        "position": {
            "enabled": True,
            "hidden_dim": 16
        },
        "cash": {
            "enabled": True,
            "input_dim": 2,  # [cash_balance, portfolio_value]
            "hidden_dim": 16
        }
    },
    "backbone": {
        "type": "mlp",
        "hidden_dims": [288, 128],  # Total input dim = 128 (ohlcv) + 128 (technical) + 16 (position) + 16 (cash) = 288
        "dropout": 0.0,
        "use_layer_norm": True
    },
    "heads": {
        "discrete": {
            "enabled": True,
            "type": "bayesian",
            "hidden_dim": 128,
            "num_actions": 3,
            "bayesian_config": {
                "num_samples": 10,
                "prior_scale": 1.0
            }
        }
    }
}

# Advanced Network Configurations with Confidence
ADVANCED_CONFIDENCE_PARAMETRIC_CONFIG = {
    "n_assets": 3,
    "window_size": 20,
    "processors": {
        "ohlcv": {
            "enabled": True,
            "hidden_dim": 128,
            "n_heads": 4
        },
        "technical": {
            "enabled": True,
            "tech_dim": 20,
            "hidden_dim": 128
        },
        "position": {
            "enabled": True,
            "hidden_dim": 16
        },
        "cash": {
            "enabled": True,
            "input_dim": 2,  # [cash_balance, portfolio_value]
            "hidden_dim": 16
        }
    },
    "backbone": {
        "type": "mlp",
        "hidden_dims": [288, 128],  # Total input dim = 128 (ohlcv) + 128 (technical) + 16 (position) + 16 (cash) = 288
        "dropout": 0.0,
        "use_layer_norm": True
    },
    "heads": {
        "discrete": {
            "enabled": True,
            "type": "parametric",
            "hidden_dim": 128,
            "num_actions": 3
        },
        "confidence": {
            "enabled": True,
            "type": "parametric",
            "hidden_dim": 128
        }
    }
}

ADVANCED_CONFIDENCE_BAYESIAN_CONFIG = {
    "n_assets": 3,
    "window_size": 20,
    "processors": {
        "ohlcv": {
            "enabled": True,
            "hidden_dim": 128,
            "n_heads": 4
        },
        "technical": {
            "enabled": True,
            "tech_dim": 20,
            "hidden_dim": 128
        },
        "position": {
            "enabled": True,
            "hidden_dim": 16
        },
        "cash": {
            "enabled": True,
            "input_dim": 2,  # [cash_balance, portfolio_value]
            "hidden_dim": 16
        }
    },
    "backbone": {
        "type": "mlp",
        "hidden_dims": [288, 128],  # Total input dim = 128 (ohlcv) + 128 (technical) + 16 (position) + 16 (cash) = 288
        "dropout": 0.0,
        "use_layer_norm": True
    },
    "heads": {
        "discrete": {
            "enabled": True,
            "type": "bayesian",
            "hidden_dim": 128,
            "num_actions": 3,
            "bayesian_config": {
                "num_samples": 10,
                "prior_scale": 1.0
            }
        },
        "confidence": {
            "enabled": True,
            "type": "bayesian",
            "hidden_dim": 128,
            "bayesian_config": {
                "num_samples": 10,
                "prior_scale": 1.0
            }
        }
    }
}

# Advanced Network Configurations with Value Heads
ADVANCED_VALUE_PARAMETRIC_CONFIG = {
    "n_assets": 3,
    "window_size": 20,
    "processors": {
        "ohlcv": {
            "enabled": True,
            "hidden_dim": 128,
            "n_heads": 4
        },
        "technical": {
            "enabled": True,
            "tech_dim": 20,
            "hidden_dim": 128
        },
        "position": {
            "enabled": True,
            "hidden_dim": 16
        },
        "cash": {
            "enabled": True,
            "input_dim": 2,  # [cash_balance, portfolio_value]
            "hidden_dim": 16
        }
    },
    "backbone": {
        "type": "mlp",
        "hidden_dims": [288, 128],  # Total input dim = 128 (ohlcv) + 128 (technical) + 16 (position) + 16 (cash) = 288
        "dropout": 0.0,
        "use_layer_norm": True
    },
    "heads": {
        "discrete": {
            "enabled": True,
            "type": "parametric",
            "hidden_dim": 128,
            "num_actions": 3
        },
        "value": {
            "enabled": True,
            "type": "parametric",
            "hidden_dim": 128
        }
    }
}

ADVANCED_VALUE_BAYESIAN_CONFIG = {
    "n_assets": 3,
    "window_size": 20,
    "processors": {
        "ohlcv": {
            "enabled": True,
            "hidden_dim": 128,
            "n_heads": 4
        },
        "technical": {
            "enabled": True,
            "tech_dim": 20,
            "hidden_dim": 128
        },
        "position": {
            "enabled": True,
            "hidden_dim": 16
        },
        "cash": {
            "enabled": True,
            "input_dim": 2,  # [cash_balance, portfolio_value]
            "hidden_dim": 16
        }
    },
    "backbone": {
        "type": "mlp",
        "hidden_dims": [288, 128],  # Total input dim = 128 (ohlcv) + 128 (technical) + 16 (position) + 16 (cash) = 288
        "dropout": 0.0,
        "use_layer_norm": True
    },
    "heads": {
        "discrete": {
            "enabled": True,
            "type": "bayesian",
            "hidden_dim": 128,
            "num_actions": 3,
            "bayesian_config": {
                "num_samples": 10,
                "prior_scale": 1.0
            }
        },
        "value": {
            "enabled": True,
            "type": "bayesian",
            "hidden_dim": 128,
            "bayesian_config": {
                "num_samples": 10,
                "prior_scale": 1.0
            }
        }
    }
}

# Advanced Network Configurations with both Value and Confidence Heads
ADVANCED_FULL_PARAMETRIC_CONFIG = {
    "n_assets": 3,
    "window_size": 20,
    "processors": {
        "ohlcv": {
            "enabled": True,
            "hidden_dim": 128,
            "n_heads": 4
        },
        "technical": {
            "enabled": True,
            "tech_dim": 20,
            "hidden_dim": 128
        },
        "position": {
            "enabled": True,
            "hidden_dim": 16
        },
        "cash": {
            "enabled": True,
            "input_dim": 2,  # [cash_balance, portfolio_value]
            "hidden_dim": 16
        }
    },
    "backbone": {
        "type": "mlp",
        "hidden_dims": [288, 128],  # Total input dim = 128 (ohlcv) + 128 (technical) + 16 (position) + 16 (cash) = 288
        "dropout": 0.0,
        "use_layer_norm": True
    },
    "heads": {
        "discrete": {
            "enabled": True,
            "type": "parametric",
            "hidden_dim": 128,
            "num_actions": 3
        },
        "value": {
            "enabled": True,
            "type": "parametric",
            "hidden_dim": 128
        },
        "confidence": {
            "enabled": True,
            "type": "parametric",
            "hidden_dim": 128
        }
    }
}

ADVANCED_FULL_BAYESIAN_CONFIG = {
    "n_assets": 3,
    "window_size": 20,
    "processors": {
        "ohlcv": {
            "enabled": True,
            "hidden_dim": 128,
            "n_heads": 4
        },
        "technical": {
            "enabled": True,
            "tech_dim": 20,
            "hidden_dim": 128
        },
        "position": {
            "enabled": True,
            "hidden_dim": 16
        },
        "cash": {
            "enabled": True,
            "input_dim": 2,  # [cash_balance, portfolio_value]
            "hidden_dim": 16
        }
    },
    "backbone": {
        "type": "mlp",
        "hidden_dims": [288, 128],  # Total input dim = 128 (ohlcv) + 128 (technical) + 16 (position) + 16 (cash) = 288
        "dropout": 0.0,
        "use_layer_norm": True
    },
    "heads": {
        "discrete": {
            "enabled": True,
            "type": "bayesian",
            "hidden_dim": 128,
            "num_actions": 3,
            "bayesian_config": {
                "num_samples": 10,
                "prior_scale": 1.0
            }
        },
        "value": {
            "enabled": True,
            "type": "bayesian",
            "hidden_dim": 128,
            "bayesian_config": {
                "num_samples": 10,
                "prior_scale": 1.0
            }
        },
        "confidence": {
            "enabled": True,
            "type": "bayesian",
            "hidden_dim": 128,
            "bayesian_config": {
                "num_samples": 10,
                "prior_scale": 1.0
            }
        }
    }
}