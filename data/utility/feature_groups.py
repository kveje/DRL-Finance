"""Utility function to create feature groups for normalization."""

# Initialize the logger
from utils.logger import Logger
logger = Logger.get_logger()


from typing import Dict, List, Set, Optional

from config.data import (
    EXCLUDE_COLUMNS,
    PRICE_COLUMNS,
    VOLUME_COLUMNS,
    MARKET_INDICATORS,
    TECHNICAL_INDICATOR_GROUPS,
)


def create_feature_groups(
    columns: List[str], include_pattern_matching: bool = True
) -> Dict[str, List[str]]:
    """Create feature groups from column names for normalization.

    Organizes financial data columns into logical groups based on their types.
    This ensures that similar features are normalized together.

    Args:
        columns: List of column names from the DataFrame
        include_pattern_matching: Whether to use pattern matching for
                                  technical indicators with parameters

    Returns:
        Dictionary mapping feature categories to lists of column names
    """
    feature_groups = {}

    # Core price columns
    feature_groups["price"] = [col for col in columns if col.lower() in PRICE_COLUMNS]

    # Volume-related columns
    feature_groups["volume"] = [col for col in columns if col.lower() in VOLUME_COLUMNS]

    # Market indicators

    feature_groups["market"] = [
        col for col in columns if any(mi in col.lower() for mi in MARKET_INDICATORS)
    ]

    # Technical indicator groups

    # Helper function to check if a column belongs to a specific indicator group
    def belongs_to_group(col: str, indicators: List[str]) -> bool:
        col_lower = col.lower()
        for indicator in indicators:
            # Match indicator name followed by an underscore or as the entire string
            if f"{indicator}_" in col_lower or col_lower == indicator:
                return True
        return False

    # Assign columns to technical indicator groups
    for group_name, indicators in TECHNICAL_INDICATOR_GROUPS.items():
        feature_groups[group_name] = [
            col for col in columns if belongs_to_group(col, indicators)
        ]

    # Check for unassigned columns (except excluded ones)
    assigned_columns = set()
    for group_cols in feature_groups.values():
        assigned_columns.update(group_cols)

    unassigned = [
        col
        for col in columns
        if col not in assigned_columns and col not in EXCLUDE_COLUMNS
    ]

    if unassigned:
        feature_groups["other"] = unassigned

    # Remove empty groups
    feature_groups = {k: v for k, v in feature_groups.items() if v}

    return feature_groups


def print_feature_groups(feature_groups: Dict[str, List[str]]) -> None:
    """Print feature groups in a readable format.

    Args:
        feature_groups: Dictionary mapping feature categories to lists of column names
    """
    print("Feature Groups for Normalization:")
    print("=================================")

    for group_name, columns in feature_groups.items():
        print(f"\n{group_name.upper()} ({len(columns)} features)")
        print("-" * (len(group_name) + 2 + len(str(len(columns))) + 10))
        for col in sorted(columns):
            print(f"  - {col}")

    # Count total features
    total_features = sum(len(cols) for cols in feature_groups.values())
    print(f"\nTotal: {total_features} features in {len(feature_groups)} groups")

def log_feature_groups(feature_groups: Dict[str, List[str]]) -> None:
    """Log feature groups in a readable format.

    Args:
        feature_groups: Dictionary mapping feature categories to lists of column names
    """
    logger.info("Feature Groups for Normalization:")
    logger.info("=================================")

    for group_name, columns in feature_groups.items():
        logger.info(f"\n{group_name.upper()} ({len(columns)} features)")
        logger.info("-" * (len(group_name) + 2 + len(str(len(columns))) + 10))
        for col in sorted(columns):
            logger.info(f"  - {col}")

    # Count total features
    total_features = sum(len(cols) for cols in feature_groups.values())
    logger.info(f"\nTotal: {total_features} features in {len(feature_groups)} groups")
