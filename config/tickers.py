from finrl import config_tickers as cfg_tickers

DOW_30_TICKER = [i for i in cfg_tickers.DOW_30_TICKER if i != "DOW"]
NASDAQ_100_TICKER = cfg_tickers.NAS_100_TICKER
SP_500_TICKER = cfg_tickers.SP_500_TICKER