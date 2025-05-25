"""Tests for environment configuration."""
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.env import get_processor_config, PROCESSOR_CONFIGS

class TestProcessorConfig(unittest.TestCase):
    """Test cases for processor configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.asset_list = ["AAPL", "GOOGL", "MSFT"]
        self.n_assets = len(self.asset_list)
        self.tech_cols = ["RSI", "MACD", "Bollinger_Bands"]

    def test_price_type_config(self):
        """Test different price type configurations."""
        # Test price type
        config = get_processor_config(
            price_type="price",
            n_assets=self.n_assets,
            asset_list=self.asset_list
        )
        self.assertTrue(any(p["type"] == "price" for p in config))
        self.assertFalse(any(p["type"] == "ohlcv" for p in config))

        # Test OHLCV type
        config = get_processor_config(
            price_type="ohlcv",
            n_assets=self.n_assets,
            asset_list=self.asset_list
        )
        self.assertFalse(any(p["type"] == "price" for p in config))
        self.assertTrue(any(p["type"] == "ohlcv" for p in config))

        # Test both types
        config = get_processor_config(
            price_type="both",
            n_assets=self.n_assets,
            asset_list=self.asset_list
        )
        self.assertTrue(any(p["type"] == "price" for p in config))
        self.assertTrue(any(p["type"] == "ohlcv" for p in config))

    def test_technical_indicators(self):
        """Test technical indicators configuration."""
        # Test without technical indicators
        config = get_processor_config(
            price_type="price",
            n_assets=self.n_assets,
            asset_list=self.asset_list
        )
        self.assertFalse(any(p["type"] == "tech" for p in config))

        # Test with technical indicators
        config = get_processor_config(
            price_type="price",
            n_assets=self.n_assets,
            asset_list=self.asset_list,
            tech_cols=self.tech_cols
        )
        tech_config = next(p for p in config if p["type"] == "tech")
        self.assertEqual(tech_config["kwargs"]["tech_cols"], self.tech_cols)

    def test_asset_configuration(self):
        """Test asset list and number of assets configuration."""
        config = get_processor_config(
            price_type="price",
            n_assets=self.n_assets,
            asset_list=self.asset_list
        )

        # Check position processor
        position_config = next(p for p in config if p["type"] == "position")
        self.assertEqual(position_config["kwargs"]["asset_list"], self.asset_list)

        # Check affordability processor
        affordability_config = next(p for p in config if p["type"] == "affordability")
        self.assertEqual(affordability_config["kwargs"]["n_assets"], self.n_assets)

    def test_price_column(self):
        """Test price column configuration."""
        custom_price_col = "adj_close"
        config = get_processor_config(
            price_type="price",
            n_assets=self.n_assets,
            asset_list=self.asset_list,
            price_col=custom_price_col
        )

        # Check affordability processor
        affordability_config = next(p for p in config if p["type"] == "affordability")
        self.assertEqual(affordability_config["kwargs"]["price_col"], custom_price_col)

        # Check current price processor
        current_price_config = next(p for p in config if p["type"] == "current_price")
        self.assertEqual(current_price_config["kwargs"]["price_col"], custom_price_col)

    def test_required_processors(self):
        """Test that required processors are always present."""
        config = get_processor_config(
            price_type="price",
            n_assets=self.n_assets,
            asset_list=self.asset_list
        )

        # Check that cash processor is always present
        self.assertTrue(any(p["type"] == "cash" for p in config))

        # Check that position processor is always present
        self.assertTrue(any(p["type"] == "position" for p in config))

        # Check that affordability processor is always present
        self.assertTrue(any(p["type"] == "affordability" for p in config))

        # Check that current price processor is always present
        self.assertTrue(any(p["type"] == "current_price" for p in config))

if __name__ == "__main__":
    unittest.main() 