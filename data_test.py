import pandas as pd

from data import DataManager

from config.data import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)
from config.data import INDICATOR_PARAMS, PROCESSOR_PARAMS, NORMALIZATION_PARAMS
from config.path import (
    DATA_PATH,
    TRAINED_MODEL_PATH,
    LOG_PATH,
    RESULTS_PATH,
    CACHE_PATH,
)

# Initialize the DataManager
data_manager = DataManager(
    data_dir=DATA_PATH,
    cache_dir=CACHE_PATH,
    use_cache=True,
)

# Load and process the data
data = data_manager.prepare_data(
    tickers=["AAPL", "MSFT"],
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    source="yahoo",
    time_interval="1d",
    processors=list(PROCESSOR_PARAMS.keys()),
    processor_params=PROCESSOR_PARAMS,
    normalize=False,
    normalize_method=NORMALIZATION_PARAMS["method"],
    force_download=False,
)

print(data.tail())
