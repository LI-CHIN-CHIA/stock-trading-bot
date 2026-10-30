"""
Intraday feature engineering for day-trading model.
Input:  2-min OHLCV DataFrame (yfinance, current day)
        analysis: dict from day_trader._analyze_bars()
Output: dict of ~22 scalar tabular features
        np.ndarray (5, 20) for CNN input
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_CNN_WINDOW = 20  # bars for CNN input


# ── Internal helpers (mirror day_trader, no cross-import) ────────────────────

def _vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cumvol = df["Volume"].cumsum()
    return (typical * df["Volume"]).cumsum() / cumvol.replace(0, np.nan)


def _volume_delta(df: pd.DataFrame) -> pd.Series:
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    bull = df["Volume"] * (df["Close"] - df["Low"]) / hl
    bear = df["Volume"] * (df["High"] - df["Close"]) / hl
    return (bull - bear).fillna(0)


def _rsi(series: pd.Series, period: int = 5) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    val = rsi.iloc[-1]
    return float(val) if not np.isnan(val) else 50.0


# ── Public API ───────────────────────────────────────────────────────────────

def compute_tabular_features(df: pd.DataFrame, analysis: dict | None = None) -> dict:
    """
    Compute ~22 tabular features from a 2-min OHLCV DataFrame.
    `analysis` is optional (pre-computed dict from _analyze_bars); if None, we
    derive the needed values directly.
    """
    if len(df) < 5:
        return {}

    close = df["Close"]
    price = float(close.iloc[-1])
    cur   = df.iloc[-1]
    prv   = df.iloc[-2] if len(df) >= 2 else cur

    vwap_ser  = _vwap(df)
    delta_ser = _volume_delta(df)

    vwap_val  = float(vwap_ser.iloc[-1]) if not np.isnan(vwap_ser.iloc[-1]) else price
    avg_vol   = float(df["Volume"].iloc[-20:].mean()) if len(df) >= 20 else float(df["Volume"].mean())
    last_vol  = float(df["Volume"].iloc[-1])
    delta_sum = float(delta_ser.iloc[-10:].sum())

    # ── K-line shape of current bar ──────────────────────────────────────────
    hl = cur["High"] - cur["Low"]
    if hl == 0:
        hl = 1e-8
    body_high = max(cur["Close"], cur["Open"])
    body_low  = min(cur["Close"], cur["Open"])
    body      = body_high - body_low

    upper_shadow = (cur["High"] - body_high) / hl
    lower_shadow = (body_low - cur["Low"])   / hl
    body_ratio   = body / hl
    body_dir     = 1.0 if cur["Close"] >= cur["Open"] else -1.0

    prv_hl   = prv["High"] - prv["Low"]
    prv_body = abs(prv["Close"] - prv["Open"])
    prev_dir = 1.0 if prv["Close"] >= prv["Open"] else -1.0

    # ── Momentum ─────────────────────────────────────────────────────────────
    day_open      = float(df["Open"].iloc[0])
    intraday_roc  = (price - day_open) / (day_open + 1e-8)
    momentum_3bar = (float(close.iloc[-1]) - float(close.iloc[-4])) / (float(close.iloc[-4]) + 1e-8) if len(df) >= 4 else 0.0

    # ── Volume acceleration ───────────────────────────────────────────────────
    vol5  = float(df["Volume"].iloc[-5:].mean())  if len(df) >= 5  else avg_vol
    vol15 = float(df["Volume"].iloc[-20:-5].mean()) if len(df) >= 20 else avg_vol
    vol_accel = vol5 / (vol15 + 1e-8)

    # ── Price vs recent range ─────────────────────────────────────────────────
    w = min(10, len(df))
    recent_high = float(df["High"].iloc[-w:].max())
    recent_low  = float(df["Low"].iloc[-w:].min())
    price_vs_high = (price - recent_high) / (price + 1e-8)
    price_vs_low  = (price - recent_low)  / (price + 1e-8)

    # ── Time bucket (09:00=0 … 13:30=1) ──────────────────────────────────────
    last_ts = df.index[-1]
    if hasattr(last_ts, "hour"):
        t_min = last_ts.hour * 60 + last_ts.minute
    else:
        t_min = 0
    time_norm = max(0.0, min(1.0, (t_min - 9 * 60) / (4 * 60 + 30)))

    # ── From analysis dict (or recompute) ────────────────────────────────────
    if analysis:
        above_vwap   = float(analysis.get("above_vwap", price > vwap_val))
        vol_surge    = float(analysis.get("vol_surge", last_vol > 2.5 * avg_vol))
        breakout     = float(analysis.get("breakout", False))
        pat          = analysis.get("pattern", "none")
        rule_score   = float(analysis.get("score", 0))
    else:
        above_vwap   = float(price > vwap_val)
        vol_surge    = float(last_vol > 2.5 * avg_vol)
        breakout_hl  = float(df["High"].iloc[-w - 1:-1].max()) if len(df) > w else float(df["High"].max())
        breakout     = float(price > breakout_hl and last_vol > 2.5 * avg_vol)
        pat          = "none"
        rule_score   = 0.0

    return {
        # VWAP / volume
        "vwap_dev":         (price - vwap_val) / (vwap_val + 1e-8),
        "vol_delta_ratio":  delta_sum / (sum(abs(delta_ser.iloc[-10:])) + 1e-8),
        "vol_surge_ratio":  last_vol / (avg_vol + 1e-8),
        "vol_accel":        vol_accel,
        # Momentum
        "intraday_roc":     intraday_roc,
        "momentum_3bar":    momentum_3bar,
        "rsi_5":            _rsi(close) / 100.0,
        # Price range
        "price_vs_10h":     price_vs_high,
        "price_vs_10l":     price_vs_low,
        # K-line shape
        "body_ratio":       body_ratio,
        "upper_shadow":     upper_shadow,
        "lower_shadow":     lower_shadow,
        "body_direction":   body_dir,
        "prev_body_dir":    prev_dir,
        # Time
        "time_norm":        time_norm,
        # Rule signals (binary)
        "above_vwap":       above_vwap,
        "vol_surge":        vol_surge,
        "breakout":         breakout,
        "rule_score_norm":  rule_score / 4.0,
        # K-line patterns
        "pat_hammer":       float(pat == "hammer"),
        "pat_engulf_bull":  float(pat == "engulf_bull"),
        "pat_engulf_bear":  float(pat == "engulf_bear"),
    }


def compute_cnn_input(df: pd.DataFrame, window: int = _CNN_WINDOW) -> np.ndarray | None:
    """
    Build (5, window) array for CNN.
    Channels: open_ret, high_ret, low_ret, close_ret, vol_norm
    All OHLC normalized as % change vs ref_close (close of bar -window).
    Volume normalized by window mean.
    """
    if len(df) < window:
        return None

    w = df.iloc[-window:]
    ref_close = float(w["Close"].iloc[0]) or 1e-8
    mean_vol  = float(w["Volume"].mean()) or 1e-8

    o_ret = ((w["Open"]  - ref_close) / ref_close).values
    h_ret = ((w["High"]  - ref_close) / ref_close).values
    l_ret = ((w["Low"]   - ref_close) / ref_close).values
    c_ret = ((w["Close"] - ref_close) / ref_close).values
    v_nrm = (w["Volume"] / mean_vol).values

    arr = np.stack([o_ret, h_ret, l_ret, c_ret, v_nrm], axis=0).astype(np.float32)
    return arr  # (5, 20)
