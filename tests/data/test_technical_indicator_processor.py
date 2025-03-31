import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.processors.technical_indicator import TechnicalIndicatorProcessor


class TestTechnicalIndicatorProcessor(unittest.TestCase):
    """Test cases for the Technical Indicator Processor."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = TechnicalIndicatorProcessor()

        # Create a simple DataFrame for testing with clean, predictable data
        # We'll use linear price movements to make it easier to verify calculations
        dates = pd.date_range(start="2020-01-01", periods=200)

        # Create predictable price data:
        # - Linear uptrend for close prices from 100 to 150
        # - High prices 2% above close
        # - Low prices 2% below close
        # - Open prices alternating 1% above and below previous close
        close_prices = np.linspace(100, 150, 200)
        high_prices = close_prices * 1.02
        low_prices = close_prices * 0.98

        # Create alternating open prices
        open_prices = np.zeros(200)
        open_prices[0] = 100  # First day
        for i in range(1, 200):
            if i % 2 == 0:
                open_prices[i] = close_prices[i - 1] * 1.01  # 1% above previous close
            else:
                open_prices[i] = close_prices[i - 1] * 0.99  # 1% below previous close

        # Create increasing volume
        volumes = np.linspace(1000000, 2000000, 200).astype(int)

        # Create test data for two tickers
        data1 = pd.DataFrame(
            {
                "date": dates,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volumes,
                "ticker": "AAPL",
            }
        )

        data2 = pd.DataFrame(
            {
                "date": dates,
                "open": open_prices * 0.5,  # Half the price for the second ticker
                "high": high_prices * 0.5,
                "low": low_prices * 0.5,
                "close": close_prices * 0.5,
                "volume": volumes * 1.5,  # 50% more volume
                "ticker": "MSFT",
            }
        )

        self.test_data = pd.concat([data1, data2], ignore_index=True)

    def tearDown(self):
        """Tear down test fixtures."""
        pass

    def test_process_all_indicators(self):
        """Test processing all indicators at once."""
        params = {
            "sma": {"windows": [5, 10, 20]},
            "ema": {"windows": [20]},
            "rsi": {"window": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bb": {"window": 20, "num_std": 2},
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
        }
        result = self.processor.process(
            self.test_data, indicators=None, ticker_column="ticker", params=params
        )

        # Check that we have more columns than original data
        self.assertGreater(len(result.columns), len(self.test_data.columns))

        # Check that all tickers are preserved
        self.assertEqual(
            set(result["ticker"].unique()), set(self.test_data["ticker"].unique())
        )

        # Check that the number of rows is preserved
        self.assertEqual(len(result), len(self.test_data))

        # Check for all indicator columns
        self.assertIn("sma_5", result.columns)
        self.assertIn("sma_10", result.columns)
        self.assertIn("sma_20", result.columns)
        self.assertIn("ema_20", result.columns)
        self.assertIn("rsi_14", result.columns)
        self.assertIn("macd_line", result.columns)
        self.assertIn("macd_signal", result.columns)
        self.assertIn("macd_hist", result.columns)
        self.assertIn("bb_middle_20", result.columns)
        self.assertIn("bb_upper_20", result.columns)
        self.assertIn("bb_lower_20", result.columns)
        self.assertIn("bb_pct_b_20", result.columns)
        self.assertIn("atr_14", result.columns)
        self.assertIn("obv", result.columns)
        self.assertIn("adx_14", result.columns)
        self.assertIn("pdi_14", result.columns)
        self.assertIn("ndi_14", result.columns)
        self.assertIn("cci_20", result.columns)
        self.assertIn("stoch_k_14", result.columns)
        self.assertIn("stoch_d_14_3", result.columns)
        self.assertIn("mfi_14", result.columns)
        self.assertIn("roc_12", result.columns)
        self.assertIn("williams_r_14", result.columns)
        self.assertIn("vwap", result.columns)
        self.assertIn("tenkan_sen", result.columns)
        self.assertIn("kijun_sen", result.columns)
        self.assertIn("senkou_span_a", result.columns)
        self.assertIn("senkou_span_b", result.columns)
        self.assertIn("chikou_span", result.columns)
        self.assertIn("keltner_middle", result.columns)
        self.assertIn("keltner_upper", result.columns)
        self.assertIn("keltner_lower", result.columns)

    def test_process_specific_indicators(self):
        """Test processing only specific indicators."""
        indicators = ["sma", "rsi"]
        result = self.processor.process(self.test_data, indicators=indicators)

        # Check for SMA columns
        self.assertIn("sma_5", result.columns)
        self.assertIn("sma_10", result.columns)

        # Check for RSI column
        self.assertIn("rsi_14", result.columns)

        # Make sure MACD wasn't calculated
        self.assertNotIn("ema_20", result.columns)
        self.assertNotIn("macd_line", result.columns)
        self.assertNotIn("macd_signal", result.columns)
        self.assertNotIn("macd_hist", result.columns)
        self.assertNotIn("bb_middle_20", result.columns)
        self.assertNotIn("bb_upper_20", result.columns)
        self.assertNotIn("bb_lower_20", result.columns)
        self.assertNotIn("bb_pct_b_20", result.columns)
        self.assertNotIn("atr_14", result.columns)
        self.assertNotIn("obv", result.columns)
        self.assertNotIn("adx_14", result.columns)
        self.assertNotIn("pdi_14", result.columns)
        self.assertNotIn("ndi_14", result.columns)
        self.assertNotIn("cci_20", result.columns)
        self.assertNotIn("stoch_k_14", result.columns)
        self.assertNotIn("stoch_d_14_3", result.columns)
        self.assertNotIn("mfi_14", result.columns)
        self.assertNotIn("roc_12", result.columns)
        self.assertNotIn("williams_r_14", result.columns)
        self.assertNotIn("vwap", result.columns)
        self.assertNotIn("tenkan_sen", result.columns)
        self.assertNotIn("kijun_sen", result.columns)
        self.assertNotIn("senkou_span_a", result.columns)
        self.assertNotIn("senkou_span_b", result.columns)
        self.assertNotIn("chikou_span", result.columns)
        self.assertNotIn("keltner_middle", result.columns)
        self.assertNotIn("keltner_upper", result.columns)
        self.assertNotIn("keltner_lower", result.columns)

    def test_sma(self):
        """Test SMA calculation specifically."""
        # Only test on AAPL data for simplicity
        aapl_data = self.test_data[self.test_data["ticker"] == "AAPL"].copy()

        # Test with window of 5
        result = self.processor.calc_sma(aapl_data, windows=[5])

        # Verify SMA calculation for a specific point
        # SMA_5 at index 10 should be average of close prices from index 6-10
        expected_sma = aapl_data["close"].iloc[6:11].mean()
        calculated_sma = result["sma_5"].iloc[10]
        self.assertAlmostEqual(calculated_sma, expected_sma, places=6)

        # First (window-1) values should be NaN
        self.assertTrue(result["sma_5"].iloc[:4].isna().all())

        # Rest should be calculated
        self.assertTrue(result["sma_5"].iloc[4:].notna().all())

    def test_rsi(self):
        """Test RSI calculation specifically."""
        # Only test on AAPL data for simplicity
        apple_data = self.test_data[self.test_data["ticker"] == "AAPL"].copy()

        # Test with window of 14
        result = self.processor.calc_rsi(apple_data, window=14)

        # Since we're using linear uptrend data, RSI should be high (>70)
        # after having enough data points
        self.assertTrue((result["rsi_14"].iloc[20:] > 70).all())

        # First (window) values should be NaN
        print(result["rsi_14"].iloc[:13])
        self.assertTrue(result["rsi_14"].iloc[:13].isna().all())

        # Rest should be calculated
        self.assertTrue(result["rsi_14"].iloc[14:].notna().all())

    def test_macd(self):
        """Test MACD calculation specifically."""
        # Only test on AAPL data for simplicity
        apple_data = self.test_data[self.test_data["ticker"] == "AAPL"].copy()

        # Calculate MACD
        result = self.processor.calc_macd(apple_data)

        # Check that all MACD columns are created
        self.assertIn("macd_line", result.columns)
        self.assertIn("macd_signal", result.columns)
        self.assertIn("macd_hist", result.columns)

        # For our uptrend test data, MACD line should be positive after enough data points
        self.assertTrue((result["macd_line"].iloc[30:] > 0).all())

        # Histogram should be difference of line and signal
        for i in range(30, 40):
            self.assertAlmostEqual(
                result["macd_hist"].iloc[i],
                result["macd_line"].iloc[i] - result["macd_signal"].iloc[i],
                places=6,
            )

    def test_bollinger(self):
        """Test Bollinger Bands calculation specifically."""
        # Only test on AAPL data for simplicity
        apple_data = self.test_data[self.test_data["ticker"] == "AAPL"].copy()
        # Sort by date to make it easier to verify calculations
        aapl_data = apple_data.sort_values("date").reset_index(drop=True)

        # Calculate Bollinger Bands
        result = self.processor.calc_bollinger_bands(apple_data)

        # Check that all Bollinger Band columns are created
        self.assertIn("bb_middle_20", result.columns)
        self.assertIn("bb_upper_20", result.columns)
        self.assertIn("bb_lower_20", result.columns)
        self.assertIn("bb_pct_b_20", result.columns)

        # Middle band should be 20-day SMA
        for i in range(25, 35):
            expected_middle = apple_data["close"].iloc[i - 19 : i + 1].mean()
            self.assertAlmostEqual(
                result["bb_middle_20"].iloc[i], expected_middle, places=6
            )

        # Upper band should be middle + 2*std
        for i in range(25, 35):
            std = apple_data["close"].iloc[i - 19 : i + 1].std()
            expected_upper = result["bb_middle_20"].iloc[i] + 2 * std
            self.assertAlmostEqual(
                result["bb_upper_20"].iloc[i], expected_upper, places=6
            )

        # Lower band should be middle - 2*std
        for i in range(25, 35):
            std = apple_data["close"].iloc[i - 19 : i + 1].std()
            expected_lower = result["bb_middle_20"].iloc[i] - 2 * std
            self.assertAlmostEqual(
                result["bb_lower_20"].iloc[i], expected_lower, places=6
            )

        # %B should be between 0 and 1 for most points
        valid_pct_b = result["bb_pct_b_20"].iloc[25:35]
        self.assertTrue((valid_pct_b >= 0).all() and (valid_pct_b <= 1).all())

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Empty DataFrame
        with self.assertRaises(ValueError):
            self.processor.process(pd.DataFrame())

        # Missing required columns
        invalid_data = pd.DataFrame(
            {
                "date": pd.date_range(start="2020-01-01", periods=10),
                "close": np.linspace(100, 110, 10),
                "ticker": "AAPL",
            }
        )
        with self.assertRaises(ValueError):
            self.processor.process(invalid_data)

        # Invalid indicator name
        result = self.processor.process(
            self.test_data, indicators=["invalid_indicator", "sma"]
        )
        # Should still calculate SMA but ignore invalid indicator
        self.assertIn("sma_5", result.columns)


if __name__ == "__main__":
    unittest.main()
