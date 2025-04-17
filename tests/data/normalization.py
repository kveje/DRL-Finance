"""Unit tests for the normalization class."""

import unittest
from typing import Dict, List
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the function to test
from data.processors.normalization import NormalizeProcessor


class TestNormalizeProcessor(unittest.TestCase):
    """Test cases for the NormalizeProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.dates = pd.date_range(start="2020-01-01", periods=100)
        self.tickers = ["AAPL"] * 100

    def test_zscore_normalize(self):
        """Test the zscore normalization method."""
        close = np.random.randn(100)
        data = pd.DataFrame({
            "date": self.dates,
            "ticker": self.tickers,
            "close": close
        })
        processor = NormalizeProcessor(method="zscore")
        normalized_data = processor.process(data, columns=["close"])
        self.assertAlmostEqual(normalized_data["close"].mean(), 0, places=2)
        self.assertAlmostEqual(normalized_data["close"].std(), 1, places=2)

    def test_percentage_normalize(self):
        """Test the percentage normalization method."""
        close = [100 * (1 + 0.01) ** i for i in range(100)]
        data = pd.DataFrame({
            "date": self.dates,
            "ticker": self.tickers,
            "close": close
        })
        processor = NormalizeProcessor(method="percentage")
        normalized_data = processor.process(data, columns=["close"])
        self.assertAlmostEqual(normalized_data["close"].mean(), 0.01, places=2)

if __name__ == "__main__":
    unittest.main()

