import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.sources.yahoo import YahooSource
from data.sources.source import BaseSource


class TestYahooFinanceDataSource(unittest.TestCase):
    """Test cases for the Yahoo Finance data source implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.tickers = ["AAPL", "MSFT"]
        self.start_date = "2020-01-01"
        self.end_date = "2020-01-31"
        self.data_source = YahooSource()

    def tearDown(self):
        """Tear down test fixtures."""
        pass

    @patch("yfinance.download")
    def test_download_data_structure(self, mock_download):
        """Test that download_data returns properly structured DataFrame."""
        # Create mock data that mimics yfinance's return format
        mock_data = pd.DataFrame(
            {
                "Open": [150.0, 151.0, 152.0],
                "High": [155.0, 156.0, 157.0],
                "Low": [149.0, 150.0, 151.0],
                "Close": [153.0, 154.0, 155.0],
                "Adj Close": [153.0, 154.0, 155.0],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range(start="2020-01-01", periods=3),
        )

        # Set up the mock to return our fake data
        mock_download.return_value = mock_data

        # Call the method being tested
        result = self.data_source.download_data(
            tickers=["AAPL"], start_date=self.start_date, end_date=self.end_date
        )

        # Assert that yfinance.download was called (with correct parameters, but not checking specific values)
        mock_download.assert_called_once()
        # Get the call arguments
        args, kwargs = mock_download.call_args
        # Check essential parameters without being too strict on additional parameters
        self.assertEqual(kwargs["tickers"], ["AAPL"])
        self.assertEqual(kwargs["start"], self.start_date)
        self.assertEqual(kwargs["end"], self.end_date)
        self.assertEqual(kwargs["group_by"], "ticker")
        self.assertEqual(kwargs["auto_adjust"], True)

        # Verify returned DataFrame structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)  # Should have 3 rows

        # Check that the result has the expected columns
        expected_columns = ["date", "open", "high", "low", "close", "volume", "ticker"]
        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Verify ticker column is set correctly
        self.assertTrue((result["ticker"] == "AAPL").all())

    @patch("yfinance.download")
    def test_fetch_multiple_tickers(self, mock_download):
        """Test fetching data for multiple tickers."""
        # Create mock data that mimics yfinance's return format for multiple tickers
        dates = pd.date_range(start="2020-01-01", periods=3)

        # Multi-level columns for multiple tickers
        mock_data = pd.DataFrame(
            {
                ("AAPL", "Open"): [150.0, 151.0, 152.0],
                ("AAPL", "High"): [155.0, 156.0, 157.0],
                ("AAPL", "Low"): [149.0, 150.0, 151.0],
                ("AAPL", "Close"): [153.0, 154.0, 155.0],
                ("AAPL", "Adj Close"): [153.0, 154.0, 155.0],
                ("AAPL", "Volume"): [1000000, 1100000, 1200000],
                ("MSFT", "Open"): [200.0, 201.0, 202.0],
                ("MSFT", "High"): [205.0, 206.0, 207.0],
                ("MSFT", "Low"): [199.0, 200.0, 201.0],
                ("MSFT", "Close"): [203.0, 204.0, 205.0],
                ("MSFT", "Adj Close"): [203.0, 204.0, 205.0],
                ("MSFT", "Volume"): [2000000, 2100000, 2200000],
            },
            index=dates,
        )

        # Set up the mock to return our fake data
        mock_download.return_value = mock_data

        # Call the method being tested
        result = self.data_source.download_data(
            tickers=self.tickers, start_date=self.start_date, end_date=self.end_date
        )

        # Verify returned DataFrame structure
        self.assertIsInstance(result, pd.DataFrame)
        # Should have 6 rows (3 dates × 2 tickers)
        self.assertEqual(len(result), 6)

        # Check unique tickers
        unique_tickers = result["ticker"].unique()
        self.assertEqual(len(unique_tickers), 2)
        self.assertIn("AAPL", unique_tickers)
        self.assertIn("MSFT", unique_tickers)

        # Count rows per ticker
        apple_rows = result[result["ticker"] == "AAPL"]
        microsoft_rows = result[result["ticker"] == "MSFT"]
        self.assertEqual(len(apple_rows), 3)
        self.assertEqual(len(microsoft_rows), 3)

    @patch("yfinance.download")
    def test_error_handling(self, mock_download):
        """Test error handling when yfinance fails."""
        # Make the mock raise an exception
        mock_download.side_effect = Exception("API Error")

        # Verify that the method handles the error appropriately
        with self.assertRaises(Exception):
            self.data_source.download_data(
                tickers=self.tickers, start_date=self.start_date, end_date=self.end_date
            )

    def test_empty_ticker_list(self):
        """Test behavior with empty ticker list."""
        with self.assertRaises(ValueError):
            self.data_source.download_data(
                tickers=[], start_date=self.start_date, end_date=self.end_date
            )

    def test_invalid_dates(self):
        """Test behavior with invalid date formats."""
        with self.assertRaises(ValueError):
            self.data_source.download_data(
                tickers=self.tickers, start_date="invalid-date", end_date=self.end_date
            )

    def test_end_before_start(self):
        """Test behavior when end_date is before start_date."""
        with self.assertRaises(ValueError):
            self.data_source.download_data(
                tickers=self.tickers, start_date="2020-01-31", end_date="2020-01-01"
            )


if __name__ == "__main__":
    unittest.main()
