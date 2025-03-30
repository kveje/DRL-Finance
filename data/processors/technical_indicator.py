"""Technical indicator processor for financial data."""

# Initialize the logger
from utils.logger import Logger
logger = Logger.get_logger()


from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np

class TechnicalIndicatorProcessor:
    """Processor for calculating technical indicators on financial data.

    This class provides methods to calculate various technical indicators
    commonly used in financial analysis and algorithmic trading.
    """

    def __init__(self):
        """Initialize the technical indicator processor."""
        # Define all available indicators
        self.available_indicators = {
            "sma": self.calc_sma,
            "ema": self.calc_ema,
            "rsi": self.calc_rsi,
            "macd": self.calc_macd,
            "bollinger": self.calc_bollinger_bands,
            "atr": self.calc_atr,
            "adx": self.calc_adx,
            "cci": self.calc_cci,
            "stoch": self.calc_stochastic,
            "obv": self.calc_obv,
            "mfi": self.calc_mfi,
            "roc": self.calc_roc,
            "williams_r": self.calc_williams_r,
            "vwap": self.calc_vwap,
            "ichimoku": self.calc_ichimoku,
            "keltner": self.calc_keltner_channels,
        }

    def process(
        self,
        data: pd.DataFrame,
        indicators: List[str] = None,
        ticker_column: str = "ticker",
        params: Dict[str, Any] = None,
    ) -> pd.DataFrame:
        """Process data to add technical indicators.

        Args:
            data: DataFrame with price data (must include date, OHLCV and ticker columns)
            indicators: List of indicator names to calculate.
                        If None, calculates all available indicators.
            ticker_column: Name of the column containing ticker symbols
            params: Dictionary of parameters for specific indicators
                    e.g. {'sma': {'windows': [10, 20, 50]}}

        Returns:
            DataFrame with added technical indicators

        Raises:
            ValueError: If data is empty or doesn't have required columns
        """
        if data.empty:
            raise ValueError("Empty DataFrame provided")

        # Check required columns
        required_cols = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            ticker_column,
        ]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Make a copy to avoid modifying the original
        df = data.copy()

        # Use parameters if provided, otherwise empty dict
        params = params or {}

        # Determine which indicators to calculate
        if indicators is None:
            # Calculate all available indicators
            indicators_to_calc = list(self.available_indicators.keys())
        else:
            # Check that all requested indicators are available
            unknown_indicators = [
                ind for ind in indicators if ind not in self.available_indicators
            ]
            if unknown_indicators:
                logger.warning(f"Unknown indicators requested: {unknown_indicators}")

            # Calculate only requested indicators that are available
            indicators_to_calc = [
                ind for ind in indicators if ind in self.available_indicators
            ]

        logger.info(f"Calculating indicators: {indicators_to_calc}")

        # Process each ticker separately
        unique_tickers = df[ticker_column].unique()
        result_dfs = []

        for ticker in unique_tickers:
            ticker_data = df[df[ticker_column] == ticker].copy().sort_values("date")

            # Apply each indicator calculation
            for indicator in indicators_to_calc:
                try:
                    # Get specific parameters for this indicator if available
                    indicator_params = params.get(indicator, {})

                    # Call the indicator calculation function with params
                    ticker_data = self.available_indicators[indicator](
                        ticker_data, **indicator_params
                    )
                except Exception as e:
                    logger.warning(f"Error calculating {indicator} for {ticker}: {e}")

            result_dfs.append(ticker_data)

        # Combine results
        if result_dfs:
            result = pd.concat(result_dfs, ignore_index=True)

            # Sort by date and ticker
            result.sort_values(["date", ticker_column], inplace=True)

            return result
        else:
            return df  # Return original data if processing failed

    # --------- Technical Indicator Calculation Methods ---------

    def calc_sma(
        self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50, 200]
    ) -> pd.DataFrame:
        """Calculate Simple Moving Average (SMA) for multiple periods.

        Args:
            df: DataFrame with price data for a single ticker
            windows: List of period lengths to calculate

        Returns:
            DataFrame with added SMA columns
        """
        for window in windows:
            df[f"sma_{window}"] = df["close"].rolling(window=window).mean()
        return df

    def calc_ema(
        self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50, 200]
    ) -> pd.DataFrame:
        """Calculate Exponential Moving Average (EMA) for multiple periods.

        Args:
            df: DataFrame with price data for a single ticker
            windows: List of period lengths to calculate

        Returns:
            DataFrame with added EMA columns
        """
        for window in windows:
            df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
        return df

    def calc_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index (RSI).

        Args:
            df: DataFrame with price data for a single ticker
            window: Period length to calculate

        Returns:
            DataFrame with added RSI column
        """
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        # Calculate RSI
        rs = avg_gain / avg_loss
        df[f"rsi_{window}"] = 100 - (100 / (1 + rs))

        return df

    def calc_macd(
        self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """Calculate Moving Average Convergence Divergence (MACD).

        Args:
            df: DataFrame with price data for a single ticker
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with added MACD columns
        """
        # Calculate MACD line
        fast_ema = df["close"].ewm(span=fast, adjust=False).mean()
        slow_ema = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd_line"] = fast_ema - slow_ema

        # Calculate signal line
        df["macd_signal"] = df["macd_line"].ewm(span=signal, adjust=False).mean()

        # Calculate histogram
        df["macd_hist"] = df["macd_line"] - df["macd_signal"]

        return df

    def calc_bollinger_bands(
        self, df: pd.DataFrame, window: int = 20, num_std: float = 2.0
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands.

        Args:
            df: DataFrame with price data for a single ticker
            window: Period length for moving average
            num_std: Number of standard deviations for bands

        Returns:
            DataFrame with added Bollinger Bands columns
        """
        # Calculate middle band (SMA)
        df[f"bb_middle_{window}"] = df["close"].rolling(window=window).mean()

        # Calculate standard deviation
        rolling_std = df["close"].rolling(window=window).std()

        # Calculate upper and lower bands
        df[f"bb_upper_{window}"] = df[f"bb_middle_{window}"] + (rolling_std * num_std)
        df[f"bb_lower_{window}"] = df[f"bb_middle_{window}"] - (rolling_std * num_std)

        # Calculate %B (position within bands)
        df[f"bb_pct_b_{window}"] = (df["close"] - df[f"bb_lower_{window}"]) / (
            df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"]
        )

        return df

    def calc_atr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Average True Range (ATR).

        Args:
            df: DataFrame with price data for a single ticker
            window: Period length for ATR calculation

        Returns:
            DataFrame with added ATR column
        """
        # Calculate the three components of True Range
        high_low = df["high"] - df["low"]
        high_close_prev = abs(df["high"] - df["close"].shift(1))
        low_close_prev = abs(df["low"] - df["close"].shift(1))

        # True Range is the maximum of these three
        df["tr"] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(
            axis=1
        )

        # ATR is the simple moving average of the True Range
        df[f"atr_{window}"] = df["tr"].rolling(window=window).mean()

        # Remove temporary column
        df.drop(columns=["tr"], inplace=True)

        return df

    def calc_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume (OBV).

        Args:
            df: DataFrame with price data for a single ticker

        Returns:
            DataFrame with added OBV column
        """
        # Calculate daily price change direction
        price_direction = np.sign(df["close"].diff())

        # Replace 0 with 1 (if price unchanged, count volume as positive)
        price_direction = price_direction.replace(0, 1)

        # Calculate daily OBV contribution
        daily_obv = df["volume"] * price_direction

        # Cumulative sum gives us OBV
        df["obv"] = daily_obv.cumsum()

        return df

    def calc_adx(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX).

        Args:
            df: DataFrame with price data for a single ticker
            window: Period length for calculation

        Returns:
            DataFrame with added ADX columns
        """
        # Calculate True Range
        high_low = df["high"] - df["low"]
        high_close_prev = abs(df["high"] - df["close"].shift(1))
        low_close_prev = abs(df["low"] - df["close"].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(
            axis=1
        )

        # Calculate Directional Movement
        pos_dm = df["high"].diff()
        neg_dm = df["low"].diff() * -1
        pos_dm = pos_dm.where((pos_dm > neg_dm) & (pos_dm > 0), 0)
        neg_dm = neg_dm.where((neg_dm > pos_dm) & (neg_dm > 0), 0)

        # Calculate smoothed values
        tr_smooth = true_range.rolling(window).mean()
        pos_dm_smooth = pos_dm.rolling(window).mean()
        neg_dm_smooth = neg_dm.rolling(window).mean()

        # Calculate Directional Indicators
        pdi = 100 * pos_dm_smooth / tr_smooth
        ndi = 100 * neg_dm_smooth / tr_smooth

        # Calculate Directional Index and ADX
        dx = 100 * abs(pdi - ndi) / (pdi + ndi)
        df[f"adx_{window}"] = dx.rolling(window).mean()
        df[f"pdi_{window}"] = pdi
        df[f"ndi_{window}"] = ndi

        return df

    def calc_cci(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate Commodity Channel Index (CCI).

        Args:
            df: DataFrame with price data for a single ticker
            window: Period length for calculation

        Returns:
            DataFrame with added CCI column
        """
        # Calculate typical price
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        # Calculate SMA of typical price
        tp_sma = typical_price.rolling(window).mean()

        # Calculate Mean Deviation
        mean_deviation = abs(typical_price - tp_sma).rolling(window).mean()

        # Calculate CCI
        df[f"cci_{window}"] = (typical_price - tp_sma) / (0.015 * mean_deviation)

        return df

    def calc_stochastic(
        self, df: pd.DataFrame, k_window: int = 14, d_window: int = 3
    ) -> pd.DataFrame:
        """Calculate Stochastic Oscillator.

        Args:
            df: DataFrame with price data for a single ticker
            k_window: K period
            d_window: D period (SMA of K)

        Returns:
            DataFrame with added Stochastic columns
        """
        # Calculate %K
        lowest_low = df["low"].rolling(k_window).min()
        highest_high = df["high"].rolling(k_window).max()
        df[f"stoch_k_{k_window}"] = (
            100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
        )

        # Calculate %D
        df[f"stoch_d_{k_window}_{d_window}"] = (
            df[f"stoch_k_{k_window}"].rolling(d_window).mean()
        )

        return df

    def calc_mfi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Money Flow Index (MFI).

        Args:
            df: DataFrame with price data for a single ticker
            window: Period length for calculation

        Returns:
            DataFrame with added MFI column
        """
        # Calculate typical price
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        # Calculate raw money flow
        raw_money_flow = typical_price * df["volume"]

        # Calculate positive and negative money flow
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

        # Calculate money flow ratio
        positive_sum = positive_flow.rolling(window).sum()
        negative_sum = negative_flow.rolling(window).sum()
        money_flow_ratio = positive_sum / negative_sum

        # Calculate MFI
        df[f"mfi_{window}"] = 100 - (100 / (1 + money_flow_ratio))

        return df

    def calc_roc(self, df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
        """Calculate Rate of Change (ROC).

        Args:
            df: DataFrame with price data for a single ticker
            window: Period length for calculation

        Returns:
            DataFrame with added ROC column
        """
        # Calculate ROC
        df[f"roc_{window}"] = ((df["close"] / df["close"].shift(window)) - 1) * 100

        return df

    def calc_williams_r(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Williams %R.

        Args:
            df: DataFrame with price data for a single ticker
            window: Period length for calculation

        Returns:
            DataFrame with added Williams %R column
        """
        # Calculate highest high and lowest low
        highest_high = df["high"].rolling(window).max()
        lowest_low = df["low"].rolling(window).min()

        # Calculate Williams %R
        df[f"williams_r_{window}"] = (
            -100 * (highest_high - df["close"]) / (highest_high - lowest_low)
        )

        return df

    def calc_vwap(self, df: pd.DataFrame, reset_period: str = "daily") -> pd.DataFrame:
        """Calculate Volume Weighted Average Price (VWAP).

        Args:
            df: DataFrame with price data for a single ticker
            reset_period: When to reset the VWAP calculation ('daily', 'weekly', 'monthly')

        Returns:
            DataFrame with added VWAP column
        """
        # Make a copy to avoid modifying input
        result = df.copy()

        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(result["date"]):
            result["date"] = pd.to_datetime(result["date"])

        # Create grouping based on reset_period
        if reset_period == "daily":
            result["vwap_group"] = result["date"].dt.date
        elif reset_period == "weekly":
            result["vwap_group"] = result["date"].dt.isocalendar().week
        elif reset_period == "monthly":
            result["vwap_group"] = result["date"].dt.month
        else:
            result["vwap_group"] = 1  # No reset (one group)

        # Calculate typical price
        result["typical_price"] = (result["high"] + result["low"] + result["close"]) / 3

        # Calculate VWAP components
        result["price_volume"] = result["typical_price"] * result["volume"]

        # Group by the reset period and calculate VWAP
        result["cumulative_price_volume"] = result.groupby("vwap_group")[
            "price_volume"
        ].cumsum()
        result["cumulative_volume"] = result.groupby("vwap_group")["volume"].cumsum()

        # VWAP calculation
        result["vwap"] = result["cumulative_price_volume"] / result["cumulative_volume"]

        # Clean up temporary columns
        result.drop(
            columns=[
                "vwap_group",
                "typical_price",
                "price_volume",
                "cumulative_price_volume",
                "cumulative_volume",
            ],
            inplace=True,
        )

        return result

    def calc_ichimoku(
        self,
        df: pd.DataFrame,
        tenkan_window: int = 9,
        kijun_window: int = 26,
        senkou_span_b_window: int = 52,
        displacement: int = 26,
    ) -> pd.DataFrame:
        """Calculate Ichimoku Cloud components.

        Args:
            df: DataFrame with price data for a single ticker
            tenkan_window: Tenkan-sen (Conversion Line) period
            kijun_window: Kijun-sen (Base Line) period
            senkou_span_b_window: Senkou Span B (Leading Span B) period
            displacement: Displacement period for cloud

        Returns:
            DataFrame with added Ichimoku Cloud columns
        """
        # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past tenkan_window periods
        df["tenkan_sen"] = (
            df["high"].rolling(window=tenkan_window).max()
            + df["low"].rolling(window=tenkan_window).min()
        ) / 2

        # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for the past kijun_window periods
        df["kijun_sen"] = (
            df["high"].rolling(window=kijun_window).max()
            + df["low"].rolling(window=kijun_window).min()
        ) / 2

        # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 displaced forward displacement periods
        df["senkou_span_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(
            displacement
        )

        # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for the past senkou_span_b_window periods, displaced forward displacement periods
        df["senkou_span_b"] = (
            (
                df["high"].rolling(window=senkou_span_b_window).max()
                + df["low"].rolling(window=senkou_span_b_window).min()
            )
            / 2
        ).shift(displacement)

        # Calculate Chikou Span (Lagging Span): Current closing price, displaced backwards displacement periods
        df["chikou_span"] = df["close"].shift(-displacement)

        return df

    def calc_keltner_channels(
        self,
        df: pd.DataFrame,
        ema_window: int = 20,
        atr_window: int = 14,
        multiplier: float = 2.0,
    ) -> pd.DataFrame:
        """Calculate Keltner Channels.

        Args:
            df: DataFrame with price data for a single ticker
            ema_window: Period for the EMA calculation
            atr_window: Period for the ATR calculation
            multiplier: Multiplier for the ATR

        Returns:
            DataFrame with added Keltner Channels columns
        """
        # Calculate the EMA of the typical price
        df["keltner_middle"] = df["close"].ewm(span=ema_window, adjust=False).mean()

        # Calculate the ATR
        high_low = df["high"] - df["low"]
        high_close_prev = abs(df["high"] - df["close"].shift(1))
        low_close_prev = abs(df["low"] - df["close"].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(
            axis=1
        )
        atr = true_range.ewm(span=atr_window, adjust=False).mean()

        # Calculate the upper and lower bands
        df["keltner_upper"] = df["keltner_middle"] + (multiplier * atr)
        df["keltner_lower"] = df["keltner_middle"] - (multiplier * atr)

        return df
