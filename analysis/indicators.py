import logging

import pandas as pd
import pandas_ta as ta

from utils.constants import (
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MA_PERIODS,
    RSI_PERIOD,
    VOLUME_MA_PERIOD,
)

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Computes all technical indicators needed for signal generation and charting.
    All methods are pure static functions - no side effects, no state.
    """

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Master method: compute all indicators and add columns to a df copy.

        Input: DataFrame with Open, High, Low, Close, Volume columns.
        Returns: New DataFrame with indicator columns added.
        """
        df = df.copy()
        df = TechnicalIndicators.compute_moving_averages(df)
        df = TechnicalIndicators.compute_rsi(df)
        df = TechnicalIndicators.compute_macd(df)
        df = TechnicalIndicators.compute_bollinger_bands(df)
        df = TechnicalIndicators.compute_atr(df)
        df = TechnicalIndicators.compute_volume_profile(df)
        return df

    @staticmethod
    def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Compute SMA for each period in MA_PERIODS (5, 10, 20, 60)."""
        for period in MA_PERIODS:
            result = ta.sma(df["Close"], length=period)
            if result is not None:
                df[f"MA{period}"] = result
            else:
                df[f"MA{period}"] = float("nan")
        return df

    @staticmethod
    def compute_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
        """Compute RSI using Wilder's smoothing."""
        result = ta.rsi(df["Close"], length=period)
        df["RSI"] = result if result is not None else float("nan")
        return df

    @staticmethod
    def compute_macd(
        df: pd.DataFrame,
        fast: int = MACD_FAST,
        slow: int = MACD_SLOW,
        signal: int = MACD_SIGNAL,
    ) -> pd.DataFrame:
        """Compute MACD line, signal line, and histogram."""
        result = ta.macd(df["Close"], fast=fast, slow=slow, signal=signal)
        if result is not None and not result.empty:
            # pandas-ta names: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            cols = result.columns.tolist()
            macd_col = next((c for c in cols if c.startswith("MACD_")), None)
            hist_col = next((c for c in cols if c.startswith("MACDh_")), None)
            sig_col = next((c for c in cols if c.startswith("MACDs_")), None)

            df["MACD"] = result[macd_col] if macd_col else float("nan")
            df["MACD_Hist"] = result[hist_col] if hist_col else float("nan")
            df["MACD_Signal"] = result[sig_col] if sig_col else float("nan")
        else:
            df["MACD"] = float("nan")
            df["MACD_Hist"] = float("nan")
            df["MACD_Signal"] = float("nan")
        return df

    @staticmethod
    def compute_bollinger_bands(
        df: pd.DataFrame,
        period: int = BB_PERIOD,
        std: float = BB_STD,
    ) -> pd.DataFrame:
        """Compute Bollinger Bands: upper, mid, lower, width, %B."""
        result = ta.bbands(df["Close"], length=period, std=std)
        if result is not None and not result.empty:
            cols = result.columns.tolist()
            lower_col = next((c for c in cols if c.startswith("BBL_")), None)
            mid_col = next((c for c in cols if c.startswith("BBM_")), None)
            upper_col = next((c for c in cols if c.startswith("BBU_")), None)
            bw_col = next((c for c in cols if c.startswith("BBB_")), None)
            bp_col = next((c for c in cols if c.startswith("BBP_")), None)

            df["BB_Lower"] = result[lower_col] if lower_col else float("nan")
            df["BB_Mid"] = result[mid_col] if mid_col else float("nan")
            df["BB_Upper"] = result[upper_col] if upper_col else float("nan")
            df["BB_Width"] = result[bw_col] if bw_col else float("nan")
            df["BB_Pct"] = result[bp_col] if bp_col else float("nan")
        else:
            for col in ["BB_Lower", "BB_Mid", "BB_Upper", "BB_Width", "BB_Pct"]:
                df[col] = float("nan")
        return df

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
        """Compute Average True Range for volatility context."""
        result = ta.atr(df["High"], df["Low"], df["Close"], length=period)
        df["ATR"] = result if result is not None else float("nan")
        return df

    @staticmethod
    def compute_volume_profile(df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume_MA20 and Volume_Ratio columns."""
        vol_ma = ta.sma(df["Volume"].astype(float), length=VOLUME_MA_PERIOD)
        df["Volume_MA20"] = vol_ma if vol_ma is not None else float("nan")
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"].replace(0, float("nan"))
        return df

    @staticmethod
    def get_latest_values(df: pd.DataFrame) -> dict:
        """Extract the most recent row's indicator values as a clean dict."""
        if df.empty:
            return {}
        row = df.iloc[-1]
        return row.to_dict()
