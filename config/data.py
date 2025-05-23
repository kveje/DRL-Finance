"""Data configuration."""

from __future__ import annotations


# Dates
TRAIN_START_DATE = "2016-04-01"
TRAIN_END_DATE = "2022-03-31"

TEST_START_DATE = "2022-04-01"
TEST_END_DATE = "2023-03-31"

TRADE_START_DATE = "2023-04-01"
TRADE_END_DATE = "2025-03-27"


# Indicator Parameters
INDICATOR_PARAMS = {
    "sma": {"windows": [5, 10, 20, 50]},
    "ema": {"windows": [5, 10, 20, 50]},
    "rsi": {"window": 14},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"window": 20, "num_std": 2},
    "atr": {"window": 14},
    "obv": {},
    "adx": {"window": 14},
    "cci": {"window": 20},
    "stoch": {"k_window": 14, "d_window": 3},
    "mfi": {"window": 14},
    "roc": {"window": 12},
    "williams_r": {"window": 14},
    "vwap": {"reset_period": "daily"},
    "ichimoku": {
        "tenkan_window": 9,
        "kijun_window": 26,
        "senkou_span_b_window": 52,
        "displacement": 26,
    },
    "keltner": {"ema_window": 20, "atr_window": 10, "multiplier": 2},
    "linreg": {"window": 10},
}

# Processor Parameters
PROCESSOR_PARAMS = {
    "vix": {},
    "turbulence": {"window": 252},
    "technical_indicator": INDICATOR_PARAMS,
}

# Normalization Features
EXCLUDE_COLUMNS = {"date", "ticker", "symbol", "day"}
PRICE_COLUMNS = {"open", "high", "low", "close"}
VOLUME_COLUMNS = {"volume"}
MARKET_INDICATORS = {"vix", "turbulence"}
TECHNICAL_INDICATOR_GROUPS = {
    "trend": [
        "sma",
        "ema",
        "macd",
        "vwap",
        "ichimoku",
        "keltner",
        "tenkan_sen",
        "kijun_sen",
        "senkou_span_a",
        "senkou_span_b",
        "chikou_span",
    ],
    "momentum": ["rsi", "stoch", "cci", "williams_r", "roc", "mfi"],
    "volatility": ["bb", "atr", "adx", "pdi", "ndi"],
    "volume_indicators": ["obv"],
}

# Normalization Parameters
NORMALIZATION_PARAMS = {"method": "percentage"}
