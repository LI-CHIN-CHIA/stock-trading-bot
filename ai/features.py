"""
Feature engineering for LightGBM trading model.
Transforms OHLCV + indicator DataFrame into ML-ready feature matrix.
"""

import logging

import numpy as np
import pandas as pd

from analysis.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


def build_feature_matrix(df: pd.DataFrame, code: str = "") -> pd.DataFrame:
    """
    Build feature matrix from a DataFrame that already has indicators added.
    Returns a new DataFrame with all feature columns. NaN rows are dropped.
    """
    df = df.copy()

    # Ensure indicators are computed
    if "RSI" not in df.columns:
        df = TechnicalIndicators.add_all(df)

    feats = pd.DataFrame(index=df.index)

    # ── Group 1: Raw indicator levels ─────────────────────────────────────────
    for col in ["RSI", "MACD", "MACD_Signal", "MACD_Hist",
                "BB_Upper", "BB_Lower", "BB_Mid", "BB_Pct", "BB_Width",
                "MA5", "MA10", "MA20", "MA60", "ATR"]:
        feats[col] = df.get(col, np.nan)

    # ── Group 2: Price-relative (scale-invariant) ─────────────────────────────
    close = df["Close"]
    feats["close_vs_ma5"]    = (close - df["MA5"])  / df["MA5"].replace(0, np.nan)
    feats["close_vs_ma10"]   = (close - df["MA10"]) / df["MA10"].replace(0, np.nan)
    feats["close_vs_ma20"]   = (close - df["MA20"]) / df["MA20"].replace(0, np.nan)
    feats["close_vs_ma60"]   = (close - df["MA60"]) / df["MA60"].replace(0, np.nan)
    feats["close_vs_bb_mid"] = (close - df["BB_Mid"]) / df["BB_Mid"].replace(0, np.nan)
    feats["atr_pct"]         = df["ATR"] / close.replace(0, np.nan)
    feats["high_low_pct"]    = (df["High"] - df["Low"]) / close.replace(0, np.nan)
    feats["body_pct"]        = abs(close - df["Open"]) / close.replace(0, np.nan)

    # ── Group 3: Momentum / rate-of-change ───────────────────────────────────
    feats["roc_1"]  = close.pct_change(1)
    feats["roc_3"]  = close.pct_change(3)
    feats["roc_5"]  = close.pct_change(5)
    feats["roc_10"] = close.pct_change(10)
    feats["macd_hist_change"] = df["MACD_Hist"].diff(1)
    feats["rsi_change"]       = df["RSI"].diff(1)

    # ── Group 4: MA alignment (continuous) ───────────────────────────────────
    feats["ma5_vs_ma10"]  = (df["MA5"]  - df["MA10"]) / df["MA10"].replace(0, np.nan)
    feats["ma10_vs_ma20"] = (df["MA10"] - df["MA20"]) / df["MA20"].replace(0, np.nan)
    feats["ma20_vs_ma60"] = (df["MA20"] - df["MA60"]) / df["MA60"].replace(0, np.nan)
    feats["ma_fan_score"] = (
        np.sign(df["MA5"]  - df["MA10"]) +
        np.sign(df["MA10"] - df["MA20"]) +
        np.sign(df["MA20"] - df["MA60"])
    )

    # ── Group 5: Volume features ──────────────────────────────────────────────
    vol = df["Volume"].astype(float)
    feats["volume_ratio"]    = df.get("Volume_Ratio", vol / vol.rolling(20).mean())
    feats["volume_change_1"] = vol.pct_change(1)
    feats["volume_ma5"]      = vol.rolling(5).mean()
    feats["vol_price_trend"] = np.sign(close - df["Open"]) * feats["volume_ratio"]

    # ── Group 6: Crossover event binary features ──────────────────────────────
    macd     = df["MACD"]
    macd_sig = df["MACD_Signal"]
    ma5      = df["MA5"]
    ma20     = df["MA20"]
    rsi      = df["RSI"]

    feats["macd_golden_cross"] = (
        (macd.shift(1) < macd_sig.shift(1)) & (macd >= macd_sig)
    ).astype(float)
    feats["macd_death_cross"] = (
        (macd.shift(1) > macd_sig.shift(1)) & (macd <= macd_sig)
    ).astype(float)
    feats["ma5_golden_cross"] = (
        (ma5.shift(1) < ma20.shift(1)) & (ma5 >= ma20)
    ).astype(float)
    feats["ma5_death_cross"] = (
        (ma5.shift(1) > ma20.shift(1)) & (ma5 <= ma20)
    ).astype(float)
    feats["rsi_above_50"]    = (rsi > 50).astype(float)
    feats["price_above_ma20"] = (close > ma20).astype(float)

    # ── Group 7: Lag features ─────────────────────────────────────────────────
    feats["RSI_lag1"]             = rsi.shift(1)
    feats["RSI_lag2"]             = rsi.shift(2)
    feats["MACD_Hist_lag1"]       = df["MACD_Hist"].shift(1)
    feats["MACD_Hist_lag2"]       = df["MACD_Hist"].shift(2)
    feats["roc_1_lag1"]           = feats["roc_1"].shift(1)
    feats["BB_Pct_lag1"]          = df["BB_Pct"].shift(1)
    feats["volume_ratio_lag1"]    = feats["volume_ratio"].shift(1)
    feats["volume_ratio_lag2"]    = feats["volume_ratio"].shift(2)
    feats["close_vs_ma20_lag1"]   = feats["close_vs_ma20"].shift(1)
    feats["macd_hist_change_lag1"] = feats["macd_hist_change"].shift(1)

    # ── Group 8: 三大法人（外資、投信、自營商）────────────────────────────────
    if code:
        try:
            from data.institutional import build_institutional_features
            inst = build_institutional_features(code, df)
            for col in ["foreign_net_ratio", "trust_net_ratio",
                        "institution_net_ratio", "foreign_buy_streak",
                        "institution_buy_streak"]:
                feats[col] = inst.get(col, 0.0)
        except Exception as e:
            logger.debug(f"三大法人特徵略過 {code}: {e}")
            for col in ["foreign_net_ratio", "trust_net_ratio",
                        "institution_net_ratio", "foreign_buy_streak",
                        "institution_buy_streak"]:
                feats[col] = 0.0
    else:
        for col in ["foreign_net_ratio", "trust_net_ratio",
                    "institution_net_ratio", "foreign_buy_streak",
                    "institution_buy_streak"]:
            feats[col] = 0.0

    # Drop warmup rows with NaN (need MA60 + lags + ROC10)
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.dropna()

    return feats


def build_target(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.03,
) -> pd.Series:
    """
    3-class target: 1=BUY (>+threshold), -1=SELL (<-threshold), 0=HOLD.
    Uses forward return over `horizon` trading days.
    Avoids look-ahead bias: target for day t uses close[t+horizon].
    """
    close = df["Close"]
    forward_return = close.shift(-horizon) / close - 1
    target = pd.Series(0, index=df.index, name="target")
    target[forward_return > threshold]  = 1
    target[forward_return < -threshold] = -1
    # Drop rows where forward data is unavailable
    target = target[forward_return.notna()]
    return target


def prepare_inference_row(df: pd.DataFrame, code: str = "") -> pd.DataFrame:
    """
    Build feature matrix and return only the last row for live inference.
    code: stock code (e.g. '2330') used to fetch institutional data.
    """
    feats = build_feature_matrix(df, code=code)
    if feats.empty:
        return feats
    return feats.iloc[[-1]]


def get_feature_names() -> list[str]:
    """Return canonical feature column names (for column-order validation)."""
    return [
        # Group 1: raw indicators
        "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Lower", "BB_Mid", "BB_Pct", "BB_Width",
        "MA5", "MA10", "MA20", "MA60", "ATR",
        # Group 2: price-relative
        "close_vs_ma5", "close_vs_ma10", "close_vs_ma20", "close_vs_ma60",
        "close_vs_bb_mid", "atr_pct", "high_low_pct", "body_pct",
        # Group 3: momentum
        "roc_1", "roc_3", "roc_5", "roc_10", "macd_hist_change", "rsi_change",
        # Group 4: MA alignment
        "ma5_vs_ma10", "ma10_vs_ma20", "ma20_vs_ma60", "ma_fan_score",
        # Group 5: volume
        "volume_ratio", "volume_change_1", "volume_ma5", "vol_price_trend",
        # Group 6: crossover events
        "macd_golden_cross", "macd_death_cross", "ma5_golden_cross", "ma5_death_cross",
        "rsi_above_50", "price_above_ma20",
        # Group 7: lags
        "RSI_lag1", "RSI_lag2", "MACD_Hist_lag1", "MACD_Hist_lag2",
        "roc_1_lag1", "BB_Pct_lag1", "volume_ratio_lag1", "volume_ratio_lag2",
        "close_vs_ma20_lag1", "macd_hist_change_lag1",
        # Group 8: 三大法人
        "foreign_net_ratio", "trust_net_ratio", "institution_net_ratio",
        "foreign_buy_streak", "institution_buy_streak",
    ]
