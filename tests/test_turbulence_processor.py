"""Unit tests for the TurbulenceProcessor class."""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.processors.turbulence import TurbulenceProcessor


class TestTurbulenceProcessor(unittest.TestCase):
    """Test cases for TurbulenceProcessor."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = TurbulenceProcessor()

        # Create synthetic data for testing
        # Generate dates for the past 300 days
        base_date = datetime(2023, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(300)]

        # Create 3 tickers with synthetic prices
        tickers = ["AAPL", "MSFT", "GOOGL"]

        # Generate random walk prices for each ticker
        np.random.seed(42)  # For reproducibility
        data = []

        for ticker in tickers:
            # Start with a base price and add random movements
            price = 100.0
            for date in dates:
                # Add random movement (-1% to +1%)
                change = np.random.uniform(-0.01, 0.01)
                price *= 1 + change

                # Add record to data
                data.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "open": price * 0.99,
                        "high": price * 1.01,
                        "low": price * 0.98,
                        "close": price,
                        "volume": np.random.randint(1000000, 5000000),
                    }
                )

        # Create DataFrame
        self.test_data = pd.DataFrame(data)

        # Convert date to datetime if not already
        self.test_data["date"] = pd.to_datetime(self.test_data["date"])

    def test_process_normal_case(self):
        """Test normal processing with valid data."""
        # Process data with default window
        result = self.processor.process(self.test_data)

        # Verify result has expected structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("turbulence", result.columns)

        # First window days should have NaN turbulence
        window_size = 252
        unique_dates = sorted(self.test_data["date"].unique())

        # Check that dates after window have turbulence values
        for date in unique_dates[window_size + 1 :]:
            turbulence_values = result[result["date"] == date]["turbulence"].values
            self.assertTrue(all(~np.isnan(turbulence_values)))

    def test_process_small_window(self):
        """Test with a smaller window size."""
        small_window = 20
        result = self.processor.process(self.test_data, window=small_window)

        # Verify more values are calculated with smaller window
        non_nan_count = result["turbulence"].notna().sum()
        # Each unique date should have a turbulence value after the window
        expected_non_nan_count = (
            len(self.test_data["date"].unique()) - (small_window + 1)
        ) * 3  # 3 tickers
        self.assertEqual(non_nan_count, expected_non_nan_count)

    def test_single_ticker_error(self):
        """Test that an error is raised with a single ticker."""
        # Create single ticker data
        single_ticker_data = self.test_data[self.test_data["ticker"] == "AAPL"].copy()

        # Process should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.processor.process(single_ticker_data)

        self.assertIn("requires at least 2 tickers", str(context.exception))

    def test_custom_ticker_column(self):
        """Test with a custom ticker column name."""
        # Rename ticker column
        renamed_data = self.test_data.rename(columns={"ticker": "symbol"})

        # Process with custom ticker column name
        result = self.processor.process(renamed_data, ticker_column="symbol")

        # Verify result has turbulence column
        self.assertIn("turbulence", result.columns)

    def test_singular_matrix_handling(self):
        """Test handling of singular covariance matrix."""
        # Create data that will produce a singular matrix
        # Two tickers with perfectly correlated returns
        dates = pd.date_range(start="2023-01-01", periods=300)

        # Create perfectly correlated data for two tickers
        data = []
        for date in dates:
            price = 100.0 * (1 + 0.001 * (date.day % 10))
            for ticker in ["AAPL", "MSFT"]:
                data.append({"date": date, "ticker": ticker, "close": price})

        singular_data = pd.DataFrame(data)

        # Mock the logger to capture warning
        with patch("logging.Logger.warning") as mock_warning:
            # Process should continue without error, but log warnings
            result = self.processor.process(singular_data, window=5)

            # Verify warning was logged
            self.assertTrue(mock_warning.called)
            self.assertTrue(
                any(
                    "LinAlgError" in str(args[0])
                    for args, _ in mock_warning.call_args_list
                )
            )

    def test_missing_values_handling(self):
        """Test handling of missing values in the data."""
        # Introduce NaN values in some places
        data_with_nans = self.test_data.copy()
        random_indices = np.random.choice(len(data_with_nans), 20, replace=False)
        data_with_nans.loc[random_indices, "close"] = np.nan

        # Process with NaN values
        result = self.processor.process(data_with_nans)

        # Verify result has turbulence column
        self.assertIn("turbulence", result.columns)

        # Check that result has same number of rows as input
        self.assertEqual(len(result), len(data_with_nans))

    def test_data_unchanged(self):
        """Test that the original data is not modified."""
        original_data = self.test_data.copy()
        _ = self.processor.process(self.test_data)

        # Verify original data is unchanged
        pd.testing.assert_frame_equal(self.test_data, original_data)

    def test_turbulence_values(self):
        """Test the actual values of the turbulence index."""
        # Create a simple dataset with known covariance
        dates = pd.date_range(start="2023-01-01", periods=30)

        # Create data with specific patterns to validate turbulence calculation
        np.random.seed(42)
        data = []

        # First ticker with low volatility
        price_a = 100.0
        # Second ticker with high volatility
        price_b = 100.0

        for date in dates:
            # Low volatility ticker
            change_a = np.random.normal(0, 0.005)
            price_a *= 1 + change_a

            # High volatility ticker
            change_b = np.random.normal(0, 0.02)
            price_b *= 1 + change_b

            # Add to data
            data.append({"date": date, "ticker": "A", "close": price_a})
            data.append({"date": date, "ticker": "B", "close": price_b})

        test_data = pd.DataFrame(data)

        # Add extreme outlier on a specific date to test turbulence spike
        outlier_date = dates[25]
        test_data.loc[
            (test_data["date"] == outlier_date) & (test_data["ticker"] == "B"), "close"
        ] *= 1.2

        # Process with a small window
        result = self.processor.process(test_data, window=10)

        # Get the turbulence values
        turbulence_values = result[result["date"] == outlier_date][
            "turbulence"
        ].unique()

        # Verify there's only one unique turbulence value for this date
        self.assertEqual(len(turbulence_values), 1)

        # The outlier date should have higher turbulence than the average
        avg_turbulence = result["turbulence"].mean()
        self.assertGreater(turbulence_values[0], avg_turbulence)


if __name__ == "__main__":
    unittest.main()
