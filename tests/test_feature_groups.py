"""Unit tests for the feature groups creation utility."""

import unittest
from typing import Dict, List
import sys
import os

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the function to test
from data.utility.feature_groups import create_feature_groups


class TestFeatureGroups(unittest.TestCase):
    """Test cases for the create_feature_groups function."""

    def setUp(self):
        """Set up test fixtures."""
        # Example columns for testing
        self.all_columns = [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "sma_5",
            "sma_10",
            "sma_20",
            "sma_50",
            "ema_5",
            "ema_10",
            "ema_20",
            "ema_50",
            "rsi_14",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "bb_middle_20",
            "bb_upper_20",
            "bb_lower_20",
            "bb_pct_b_20",
            "atr_14",
            "adx_14",
            "pdi_14",
            "ndi_14",
            "cci_20",
            "stoch_k_14",
            "stoch_d_14_3",
            "obv",
            "mfi_14",
            "roc_12",
            "williams_r_14",
            "vwap",
            "tenkan_sen",
            "kijun_sen",
            "senkou_span_a",
            "senkou_span_b",
            "chikou_span",
            "keltner_middle",
            "keltner_upper",
            "keltner_lower",
            "vix",
            "turbulence",
        ]

        # Define expected group membership for specific columns
        self.expected_groups = {
            "price": ["open", "high", "low", "close"],
            "volume": ["volume"],
            "market": ["vix", "turbulence"],
            "trend": [
                "sma_5",
                "sma_10",
                "sma_20",
                "sma_50",
                "ema_5",
                "ema_10",
                "ema_20",
                "ema_50",
                "macd_line",
                "macd_signal",
                "macd_hist",
                "vwap",
                "tenkan_sen",
                "kijun_sen",
                "senkou_span_a",
                "senkou_span_b",
                "chikou_span",
                "keltner_middle",
                "keltner_upper",
                "keltner_lower",
            ],
            "momentum": [
                "rsi_14",
                "cci_20",
                "stoch_k_14",
                "stoch_d_14_3",
                "mfi_14",
                "roc_12",
                "williams_r_14",
            ],
            "volatility": [
                "bb_middle_20",
                "bb_upper_20",
                "bb_lower_20",
                "bb_pct_b_20",
                "atr_14",
                "adx_14",
                "pdi_14",
                "ndi_14",
            ],
            "volume_indicators": ["obv"],
        }

    def test_basic_grouping(self):
        """Test that columns are grouped correctly."""
        feature_groups = create_feature_groups(self.all_columns)

        # Check that all expected groups are present
        for group in self.expected_groups:
            self.assertIn(group, feature_groups, f"Expected group '{group}' is missing")

        # Check that excluded columns are not in any group
        excluded_columns = ["date", "ticker"]
        for col in excluded_columns:
            for group_cols in feature_groups.values():
                self.assertNotIn(
                    col, group_cols, f"Excluded column '{col}' found in groups"
                )

        # Check for 'other' group - there shouldn't be one if all columns are grouped correctly
        self.assertNotIn(
            "other",
            feature_groups,
            "Found 'other' group when all columns should be categorized",
        )

    def test_specific_column_placement(self):
        """Test that specific columns go to their correct groups."""
        feature_groups = create_feature_groups(self.all_columns)

        # Test sample columns from each expected group
        test_columns = {
            "price": "close",
            "volume": "volume",
            "market": "vix",
            "trend": "sma_10",
            "momentum": "rsi_14",
            "volatility": "atr_14",
            "volume_indicators": "obv",
        }

        for group, sample_col in test_columns.items():
            self.assertIn(
                sample_col,
                feature_groups[group],
                f"Column '{sample_col}' not found in expected group '{group}'",
            )

    def test_directional_indicators_grouped_with_adx(self):
        """Test that PDI and NDI are grouped with ADX in volatility."""
        feature_groups = create_feature_groups(self.all_columns)

        # All three should be in the volatility group
        self.assertIn("adx_14", feature_groups["volatility"])
        self.assertIn("pdi_14", feature_groups["volatility"])
        self.assertIn("ndi_14", feature_groups["volatility"])

    def test_ichimoku_components_grouped_together(self):
        """Test that all Ichimoku components are grouped together in trend."""
        feature_groups = create_feature_groups(self.all_columns)

        ichimoku_components = [
            "tenkan_sen",
            "kijun_sen",
            "senkou_span_a",
            "senkou_span_b",
            "chikou_span",
        ]

        for component in ichimoku_components:
            self.assertIn(
                component,
                feature_groups["trend"],
                f"Ichimoku component '{component}' not in trend group",
            )

    def test_partial_column_list(self):
        """Test with a partial list of columns."""
        # A subset of columns
        partial_columns = [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi_14",
            "vix",
        ]

        feature_groups = create_feature_groups(partial_columns)

        # Check expected groups with this subset
        self.assertEqual(len(feature_groups["price"]), 4)
        self.assertEqual(len(feature_groups["volume"]), 1)
        self.assertEqual(len(feature_groups["market"]), 1)
        self.assertEqual(len(feature_groups["momentum"]), 1)

        # Trend and volatility should not be present or should be empty
        self.assertTrue(
            "trend" not in feature_groups or len(feature_groups["trend"]) == 0
        )
        self.assertTrue(
            "volatility" not in feature_groups or len(feature_groups["volatility"]) == 0
        )

    def test_custom_indicators(self):
        """Test with custom indicator names that might not be in predefined lists."""
        # Add some custom columns
        custom_columns = self.all_columns + ["custom_indicator", "my_strategy_signal"]

        feature_groups = create_feature_groups(custom_columns)

        # These should end up in "other"
        self.assertIn("other", feature_groups)
        self.assertIn("custom_indicator", feature_groups["other"])
        self.assertIn("my_strategy_signal", feature_groups["other"])

    def test_group_count_matches_column_count(self):
        """Test that the total count of columns in groups matches input (minus excluded)."""
        feature_groups = create_feature_groups(self.all_columns)

        # Count total columns in all groups
        total_in_groups = sum(len(cols) for cols in feature_groups.values())

        # Count excluded columns
        excluded_count = sum(1 for col in self.all_columns if col in ["date", "ticker"])

        # Total should match
        self.assertEqual(total_in_groups, len(self.all_columns) - excluded_count)


if __name__ == "__main__":
    unittest.main()
