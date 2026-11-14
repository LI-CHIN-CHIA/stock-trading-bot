"""
加密貨幣技術訊號產生器
======================
直接複用 analysis/indicators.py 的指標計算，
輸出 -1.0 ~ 1.0 的複合訊號強度。

  StableSignal  — 4H K棒，適合 BTC/ETH/SOL
  NewCoinSignal — 1m K棒，捕捉新幣初爆
"""

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)

MIN_BARS_STABLE  = 80   # 4H K 至少 80 根（約 14 天）
MIN_BARS_NEWCOIN = 30   # 1m K 至少 30 根


class BaseSignal(ABC):
    @abstractmethod
    def score(self, df: pd.DataFrame) -> float:
        """回傳 -1.0（強烈賣出）~ +1.0（強烈買入）。"""


class StableSignal(BaseSignal):
    """
    穩定幣種訊號（4H K棒）。
    使用 RSI、MACD、布林通道、ATR 動能加權投票。
    """

    def score(self, df: pd.DataFrame) -> float:
        if len(df) < MIN_BARS_STABLE:
            return 0.0
        try:
            from analysis.indicators import TechnicalIndicators
            df = TechnicalIndicators.add_all(df.copy())
        except Exception as e:
            logger.warning(f"指標計算失敗: {e}")
            return 0.0

        votes = []

        # ── RSI ───────────────────────────────────────────────────────────
        rsi = df["RSI"].iloc[-1] if "RSI" in df.columns else None
        if rsi is not None and not pd.isna(rsi):
            if rsi < 30:
                votes.append(1.0)
            elif rsi < 45:
                votes.append(0.5)
            elif rsi > 70:
                votes.append(-1.0)
            elif rsi > 60:
                votes.append(-0.5)
            else:
                votes.append(0.0)

        # ── MACD 柱狀圖方向 ───────────────────────────────────────────────
        if "MACD_Hist" in df.columns:
            hist_now  = df["MACD_Hist"].iloc[-1]
            hist_prev = df["MACD_Hist"].iloc[-2] if len(df) > 1 else 0
            if not pd.isna(hist_now) and not pd.isna(hist_prev):
                if hist_now > 0 and hist_now > hist_prev:
                    votes.append(0.8)
                elif hist_now > 0:
                    votes.append(0.3)
                elif hist_now < 0 and hist_now < hist_prev:
                    votes.append(-0.8)
                elif hist_now < 0:
                    votes.append(-0.3)
                else:
                    votes.append(0.0)

        # ── 布林通道位置 ──────────────────────────────────────────────────
        if all(c in df.columns for c in ["BB_Upper", "BB_Lower", "BB_Mid"]):
            close   = df["Close"].iloc[-1]
            bb_up   = df["BB_Upper"].iloc[-1]
            bb_low  = df["BB_Lower"].iloc[-1]
            bb_mid  = df["BB_Mid"].iloc[-1]
            if not any(pd.isna(x) for x in [bb_up, bb_low, bb_mid]):
                bb_range = bb_up - bb_low
                if bb_range > 0:
                    pos = (close - bb_low) / bb_range   # 0~1
                    if pos < 0.2:
                        votes.append(0.8)
                    elif pos < 0.4:
                        votes.append(0.3)
                    elif pos > 0.8:
                        votes.append(-0.8)
                    elif pos > 0.6:
                        votes.append(-0.3)
                    else:
                        votes.append(0.0)

        # ── MA 趨勢 ───────────────────────────────────────────────────────
        if "MA20" in df.columns and "MA60" in df.columns:
            ma20 = df["MA20"].iloc[-1]
            ma60 = df["MA60"].iloc[-1]
            close = df["Close"].iloc[-1]
            if not any(pd.isna(x) for x in [ma20, ma60]):
                if close > ma20 > ma60:
                    votes.append(0.6)
                elif close < ma20 < ma60:
                    votes.append(-0.6)
                else:
                    votes.append(0.0)

        if not votes:
            return 0.0
        return max(-1.0, min(1.0, sum(votes) / len(votes)))


class NewCoinSignal(BaseSignal):
    """
    新幣策略訊號（1m K棒）。
    主要看成交量爆量 + 價格突破 + RSI 未超買。
    """

    def score(self, df: pd.DataFrame) -> float:
        if len(df) < MIN_BARS_NEWCOIN:
            return 0.0
        try:
            from analysis.indicators import TechnicalIndicators
            df = TechnicalIndicators.add_all(df.copy())
        except Exception as e:
            logger.warning(f"指標計算失敗: {e}")
            return 0.0

        close   = df["Close"].iloc[-1]
        volume  = df["Volume"].iloc[-1]
        votes   = []

        # ── 成交量爆量（> 5x 均量）───────────────────────────────────────
        if "Volume_MA" in df.columns:
            vol_ma = df["Volume_MA"].iloc[-1]
            if vol_ma and vol_ma > 0 and not pd.isna(vol_ma):
                ratio = volume / vol_ma
                if ratio >= 10:
                    votes.append(1.0)
                elif ratio >= 5:
                    votes.append(0.7)
                elif ratio >= 2:
                    votes.append(0.3)
                else:
                    votes.append(-0.3)

        # ── 價格突破近期高點 ──────────────────────────────────────────────
        recent_high = df["High"].iloc[-30:-1].max() if len(df) >= 30 else df["High"].iloc[:-1].max()
        if not pd.isna(recent_high) and recent_high > 0:
            if close > recent_high * 1.03:
                votes.append(1.0)
            elif close > recent_high:
                votes.append(0.5)
            elif close < recent_high * 0.97:
                votes.append(-0.5)

        # ── RSI：尚未超買（< 75）─────────────────────────────────────────
        rsi = df["RSI"].iloc[-1] if "RSI" in df.columns else None
        if rsi is not None and not pd.isna(rsi):
            if rsi < 50:
                votes.append(0.5)
            elif rsi < 65:
                votes.append(0.8)   # 動能強但未過熱
            elif rsi < 75:
                votes.append(0.3)
            else:
                votes.append(-0.5)  # 過熱，不追高

        if not votes:
            return 0.0
        return max(-1.0, min(1.0, sum(votes) / len(votes)))


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, float("nan")))


def completed_daily(df: pd.DataFrame) -> pd.DataFrame:
    """去掉還沒收完的當日 K 棒（index 為該棒開盤時間，UTC）。"""
    if df is None or df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(days=1) <= now]


class DipSignal(BaseSignal):
    """
    多頭回檔買進（日線）：收盤在 200 日均線之上，且 RSI(2) < 門檻時買進；
    收盤站回 5 日均線就賣（出場判斷在 exit_signal）。

    回測（backtest/crypto_research.py，2018~2026、來回手續費 0.3%）：
      BTC 71 筆 勝率 69% 每筆 +0.93%；ETH 65 筆 勝率 66% 每筆 +0.85%；前後半段皆為正。
      門檻 5~20、趨勢線 50/100/200 日、出場 3/5/10 日線結果相近；加停損反而變差。
    """
    RSI_N, RSI_TH, TREND_N, EXIT_N = 2, 10.0, 200, 5

    def score(self, df: pd.DataFrame) -> float:
        d = completed_daily(df)
        if d is None or len(d) < self.TREND_N + 1:
            return 0.0
        c = d["Close"]
        r = _rsi(c, self.RSI_N).iloc[-1]
        if pd.isna(r):
            return 0.0
        return 1.0 if (r < self.RSI_TH and c.iloc[-1] > c.rolling(self.TREND_N).mean().iloc[-1]) else 0.0

    def exit_signal(self, df: pd.DataFrame, entry_time: "pd.Timestamp | None" = None) -> bool:
        """最近一根已收完的日線收盤 > 5 日均線（且該棒在進場之後才收完）。"""
        d = completed_daily(df)
        if d is None or len(d) < self.EXIT_N:
            return False
        if entry_time is not None and d.index[-1] + pd.Timedelta(days=1) <= entry_time:
            return False
        c = d["Close"]
        return bool(c.iloc[-1] > c.rolling(self.EXIT_N).mean().iloc[-1])


def get_signal_engine(mode: str, strategy: str = "") -> BaseSignal:
    if mode.lower() == "newcoin":
        return NewCoinSignal()
    return DipSignal() if strategy.lower() == "dip" else StableSignal()
