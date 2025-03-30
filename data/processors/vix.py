# Initialize the logger
from utils.logger import Logger
logger = Logger.get_logger()


import pandas as pd

from data.sources.source import BaseSource
from data.sources.yahoo import YahooSource


class VIXProcessor:
    """Processor to add VIX data."""

    def __init__(self, vix_source: BaseSource = None):
        """Initialize VIX processor.

        Args:
            vix_source: Data source for VIX data
        """
        self.vix_source = vix_source or YahooSource()

    def process(
        self,
        data: pd.DataFrame,
        ticker: str = "^VIX",
        merge_how: str = "left",
        **kwargs
    ) -> pd.DataFrame:
        """Add VIX data to the input DataFrame.

        Args:
            data: DataFrame to add VIX data to
            ticker: VIX ticker symbol
            merge_how: How to merge VIX data ('left', 'inner', etc.)
            **kwargs: Additional arguments to pass to the data source

        Returns:
            DataFrame with added VIX column
        """
        # Extract date range from input data
        start_date = data["date"].min()
        end_date = data["date"].max()

        # Download VIX data
        vix_data = self.vix_source.download_data(
            tickers=[ticker], start_date=start_date, end_date=end_date, **kwargs
        )

        # Keep only date and close columns
        vix_data = vix_data[["date", "close"]].rename(columns={"close": "vix"})

        # Merge with input data
        result = pd.merge(data, vix_data, on="date", how=merge_how)

        return result
