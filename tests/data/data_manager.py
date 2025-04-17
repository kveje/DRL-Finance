"""Unit tests for the data manager class."""

import unittest
from typing import Dict, List
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the function to test
from data.data_manager import DataManager

class TestDataManager(unittest.TestCase):
    """Test cases for the DataManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.data_manager = DataManager(save_raw_data=False, use_cache=False)

    def test_download_data(self):
        """Test the download_data method."""
        # Test with a single ticker
        data = self.data_manager.download_data(tickers=["AAPL"], start_date="2020-01-01", end_date="2020-01-31")
        self.assertEqual(len(data), 20)

        # Test with multiple tickers
        data = self.data_manager.download_data(tickers=["AAPL", "GOOG"], start_date="2020-01-01", end_date="2020-01-31")
        self.assertEqual(len(data), 40)
        # Test whether both tickers are present
        self.assertTrue("AAPL" in data["ticker"].unique())
        self.assertTrue("GOOG" in data["ticker"].unique())

    def test_process_data(self):
        """Test the process_data method."""
        # Test with a single ticker
        data = self.data_manager.download_data(tickers=["AAPL"], start_date="2020-01-01", end_date="2020-01-31")
        processed_data = self.data_manager.process_data(data, processors=["technical_indicator"])
        # Test number of rows is the same
        self.assertEqual(len(data), len(processed_data))

        # Test processor params
        processors = ["technical_indicator"]
        params = {"technical_indicator": {"sma": {"windows": [5, 12]}}}

        processed_data = self.data_manager.process_data(data, processors=processors, processor_params=params)
        self.assertEqual(len(data), len(processed_data))

        print(processed_data.tail())

        # Test that the processed data has the new columns
        self.assertTrue("sma_5" in processed_data.columns)
        self.assertTrue("sma_12" in processed_data.columns)
        
        # Test that the processed data has not generated other new columns
        self.assertEqual("ema_5" not in processed_data.columns, True)

        



if __name__ == "__main__":
    unittest.main()

