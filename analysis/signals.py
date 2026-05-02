import math
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from utils.constants import (
    HIGH_VOLUME_RATIO,
    MODERATE_BUY_THRESHOLD,
    MODERATE_SELL_THRESHOLD,
    RSI_NEAR_OVERBOUGHT,
    RSI_NEAR_OVERSOLD,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    STRONG_BUY_THRESHOLD,
    STRONG_SELL_THRESHOLD,
    WEAK_BUY_THRESHOLD,
    WEAK_SELL_THRESHOLD,
)

SignalDirection = Literal["買進 BUY", "賣出 SELL", "觀望 NEUTRAL"]
ConfidenceLevel = Literal["強力", "中度", "弱", "觀望"]


@dataclass
class Signal:
    """A single atomic trading signal from one indicator or rule."""
    source: str
    direction: SignalDirection
    score: int          # positive = bullish, negative = bearish
    reason: str         # Bilingual explanation
    value: float = 0.0  # Indicator value that triggered this


@dataclass
class AnalysisResult:
    """Aggregated analysis output for one stock at one point in time."""
    ticker: str
    date: str
    signals: list = field(default_factory=list)
    total_score: int = 0
    direction: SignalDirection = "觀望 NEUTRAL"
    confidence: ConfidenceLevel = "觀望"
    summary: str = ""
    price: float = 0.0
    indicators: dict = field(default_factory=dict)


def _safe(val) -> float:
    """Return float or NaN-safe 0.0 for calculations."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return float("nan")
    return float(val)


def _valid(*vals) -> bool:
    """Return True if all values are non-NaN floats."""
    return all(not math.isnan(_safe(v)) for v in vals)


class SignalEngine:
    """
    Rule-based engine that evaluates technical indicators and produces
    scored, human-readable buy/sell signals.

    Scoring:
      total_score >= +4  → 強力買進 STRONG BUY
      total_score +2,+3  → 中度買進 MODERATE BUY
      total_score +1     → 弱買進   WEAK BUY
      total_score 0      → 觀望     NEUTRAL
      total_score -1     → 弱賣出   WEAK SELL
      total_score -2,-3  → 中度賣出 MODERATE SELL
      total_score <= -4  → 強力賣出 STRONG SELL
    """

    def analyze(self, df: pd.DataFrame, ticker: str) -> AnalysisResult:
        """Run all signal rules against the latest two data rows."""
        if len(df) < 2:
            return AnalysisResult(ticker=ticker, date="N/A", summary="資料不足")

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        rules = [
            self._rule_rsi(latest, prev),
            self._rule_macd_crossover(latest, prev),
            self._rule_macd_zero_line(latest),
            self._rule_bollinger_bands(latest, prev),
            self._rule_ma_crossover(latest, prev),
            self._rule_ma_alignment(latest),
            self._rule_volume_confirmation(latest),
        ]

        # Flatten: each rule may return a list or a single Signal
        signals = []
        for r in rules:
            if isinstance(r, list):
                signals.extend(r)
            elif r is not None:
                signals.append(r)

        # Filter out neutral zero-score signals for cleaner display
        active_signals = [s for s in signals if s.score != 0]
        total_score = sum(s.score for s in active_signals)

        direction, confidence = self._compute_confidence(total_score)

        # Get latest indicator snapshot
        indicators = {
            "RSI": _safe(latest.get("RSI")),
            "MACD": _safe(latest.get("MACD")),
            "MACD_Signal": _safe(latest.get("MACD_Signal")),
            "MACD_Hist": _safe(latest.get("MACD_Hist")),
            "BB_Upper": _safe(latest.get("BB_Upper")),
            "BB_Mid": _safe(latest.get("BB_Mid")),
            "BB_Lower": _safe(latest.get("BB_Lower")),
            "BB_Pct": _safe(latest.get("BB_Pct")),
            "MA5": _safe(latest.get("MA5")),
            "MA10": _safe(latest.get("MA10")),
            "MA20": _safe(latest.get("MA20")),
            "MA60": _safe(latest.get("MA60")),
            "Volume_Ratio": _safe(latest.get("Volume_Ratio")),
        }

        result = AnalysisResult(
            ticker=ticker,
            date=str(latest.name)[:10] if hasattr(latest, "name") else "",
            signals=active_signals,
            total_score=total_score,
            direction=direction,
            confidence=confidence,
            price=_safe(latest.get("Close")),
            indicators=indicators,
        )
        result.summary = self._build_summary(result)
        return result

    # ─────────────────────────────── RSI Rules ────────────────────────────────

    def _rule_rsi(self, latest: pd.Series, prev: pd.Series) -> list:
        """RSI oversold/overbought and 50-line crossover rules."""
        signals = []
        rsi = _safe(latest.get("RSI"))
        prev_rsi = _safe(prev.get("RSI"))

        if not _valid(rsi):
            return signals

        # Oversold / Overbought
        if rsi < RSI_OVERSOLD:
            signals.append(Signal("RSI", "買進 BUY", +2,
                f"RSI超賣 (RSI Oversold): RSI={rsi:.1f}", rsi))
        elif rsi < RSI_NEAR_OVERSOLD:
            signals.append(Signal("RSI", "買進 BUY", +1,
                f"RSI接近超賣 (RSI Near Oversold): RSI={rsi:.1f}", rsi))
        elif rsi > RSI_OVERBOUGHT:
            signals.append(Signal("RSI", "賣出 SELL", -2,
                f"RSI超買 (RSI Overbought): RSI={rsi:.1f}", rsi))
        elif rsi > RSI_NEAR_OVERBOUGHT:
            signals.append(Signal("RSI", "賣出 SELL", -1,
                f"RSI接近超買 (RSI Near Overbought): RSI={rsi:.1f}", rsi))

        # 50-line crossover
        if _valid(prev_rsi):
            if prev_rsi < 50 and rsi >= 50:
                signals.append(Signal("RSI穿越", "買進 BUY", +1,
                    f"RSI突破50均線向上 (RSI crossed above 50): RSI={rsi:.1f}", rsi))
            elif prev_rsi > 50 and rsi <= 50:
                signals.append(Signal("RSI穿越", "賣出 SELL", -1,
                    f"RSI跌破50均線向下 (RSI crossed below 50): RSI={rsi:.1f}", rsi))

        return signals

    # ─────────────────────────────── MACD Rules ───────────────────────────────

    def _rule_macd_crossover(self, latest: pd.Series, prev: pd.Series) -> list:
        """MACD line/signal crossover and histogram direction change rules."""
        signals = []
        macd = _safe(latest.get("MACD"))
        sig = _safe(latest.get("MACD_Signal"))
        hist = _safe(latest.get("MACD_Hist"))
        prev_macd = _safe(prev.get("MACD"))
        prev_sig = _safe(prev.get("MACD_Signal"))
        prev_hist = _safe(prev.get("MACD_Hist"))

        if not _valid(macd, sig):
            return signals

        # Golden/Death cross
        if _valid(prev_macd, prev_sig):
            if prev_macd < prev_sig and macd >= sig:
                signals.append(Signal("MACD交叉", "買進 BUY", +2,
                    f"MACD黃金交叉 (MACD Golden Cross): MACD={macd:.4f}", macd))
            elif prev_macd > prev_sig and macd <= sig:
                signals.append(Signal("MACD交叉", "賣出 SELL", -2,
                    f"MACD死亡交叉 (MACD Death Cross): MACD={macd:.4f}", macd))

        # Histogram direction change
        if _valid(hist, prev_hist):
            if prev_hist < 0 and hist >= 0:
                signals.append(Signal("MACD動能", "買進 BUY", +1,
                    f"MACD動能轉強 (Momentum turning bullish): Hist={hist:.4f}", hist))
            elif prev_hist > 0 and hist <= 0:
                signals.append(Signal("MACD動能", "賣出 SELL", -1,
                    f"MACD動能轉弱 (Momentum turning bearish): Hist={hist:.4f}", hist))

        return signals

    def _rule_macd_zero_line(self, latest: pd.Series) -> Signal | None:
        """MACD above or below zero line."""
        macd = _safe(latest.get("MACD"))
        if not _valid(macd):
            return None
        if macd > 0:
            return Signal("MACD零軸", "買進 BUY", +1,
                f"MACD在零軸上方 (MACD above zero, bullish momentum): {macd:.4f}", macd)
        elif macd < 0:
            return Signal("MACD零軸", "賣出 SELL", -1,
                f"MACD在零軸下方 (MACD below zero, bearish momentum): {macd:.4f}", macd)
        return None

    # ────────────────────────── Bollinger Bands Rules ─────────────────────────

    def _rule_bollinger_bands(self, latest: pd.Series, prev: pd.Series) -> list:
        """Bollinger Band breakout and bounce rules."""
        signals = []
        close = _safe(latest.get("Close"))
        prev_close = _safe(prev.get("Close"))
        upper = _safe(latest.get("BB_Upper"))
        lower = _safe(latest.get("BB_Lower"))
        prev_upper = _safe(prev.get("BB_Upper"))
        prev_lower = _safe(prev.get("BB_Lower"))
        bb_width = _safe(latest.get("BB_Width"))

        if not _valid(close, upper, lower):
            return signals

        # Price vs bands
        if close < lower:
            signals.append(Signal("布林通道", "買進 BUY", +2,
                f"收盤價跌破布林下軌 (Price below BB Lower): Close={close:.2f}, Lower={lower:.2f}", close))
        elif _valid(prev_close, prev_lower) and prev_close < prev_lower and close >= lower:
            signals.append(Signal("布林通道", "買進 BUY", +1,
                f"價格從布林下軌反彈 (BB Lower bounce): Close={close:.2f}", close))

        if close > upper:
            signals.append(Signal("布林通道", "賣出 SELL", -2,
                f"收盤價突破布林上軌 (Price above BB Upper): Close={close:.2f}, Upper={upper:.2f}", close))
        elif _valid(prev_close, prev_upper) and prev_close > prev_upper and close <= upper:
            signals.append(Signal("布林通道", "賣出 SELL", -1,
                f"價格從布林上軌回落 (BB Upper retreat): Close={close:.2f}", close))

        # Bollinger squeeze (informational, score=0)
        if _valid(bb_width) and bb_width < 5.0:
            signals.append(Signal("布林收縮", "觀望 NEUTRAL", 0,
                f"布林帶收窄 (Bollinger Squeeze - breakout pending): Width={bb_width:.1f}%", bb_width))

        return signals

    # ──────────────────────── Moving Average Rules ────────────────────────────

    def _rule_ma_crossover(self, latest: pd.Series, prev: pd.Series) -> list:
        """MA crossover (golden/death cross) and price vs MA20 rules."""
        signals = []
        close = _safe(latest.get("Close"))
        ma5 = _safe(latest.get("MA5"))
        ma10 = _safe(latest.get("MA10"))
        ma20 = _safe(latest.get("MA20"))
        ma60 = _safe(latest.get("MA60"))
        prev_ma5 = _safe(prev.get("MA5"))
        prev_ma10 = _safe(prev.get("MA10"))
        prev_ma20 = _safe(prev.get("MA20"))
        prev_ma60 = _safe(prev.get("MA60"))

        # MA5 / MA20 crossover
        if _valid(ma5, ma20, prev_ma5, prev_ma20):
            if prev_ma5 < prev_ma20 and ma5 >= ma20:
                signals.append(Signal("均線交叉", "買進 BUY", +2,
                    f"MA5突破MA20黃金交叉 (MA5 Golden Cross above MA20)", ma5))
            elif prev_ma5 > prev_ma20 and ma5 <= ma20:
                signals.append(Signal("均線交叉", "賣出 SELL", -2,
                    f"MA5跌破MA20死亡交叉 (MA5 Death Cross below MA20)", ma5))

        # MA10 / MA60 crossover
        if _valid(ma10, ma60, prev_ma10, prev_ma60):
            if prev_ma10 < prev_ma60 and ma10 >= ma60:
                signals.append(Signal("均線交叉", "買進 BUY", +2,
                    f"MA10突破MA60黃金交叉 (MA10 Golden Cross above MA60)", ma10))
            elif prev_ma10 > prev_ma60 and ma10 <= ma60:
                signals.append(Signal("均線交叉", "賣出 SELL", -2,
                    f"MA10跌破MA60死亡交叉 (MA10 Death Cross below MA60)", ma10))

        # Price vs MA20
        if _valid(close, ma20):
            if close > ma20:
                signals.append(Signal("股價位置", "買進 BUY", +1,
                    f"股價在20日均線上方 (Price above MA20): Close={close:.2f}, MA20={ma20:.2f}", close))
            else:
                signals.append(Signal("股價位置", "賣出 SELL", -1,
                    f"股價在20日均線下方 (Price below MA20): Close={close:.2f}, MA20={ma20:.2f}", close))

        return signals

    def _rule_ma_alignment(self, latest: pd.Series) -> Signal | None:
        """MA alignment (trend direction): bullish or bearish fan pattern."""
        ma5 = _safe(latest.get("MA5"))
        ma10 = _safe(latest.get("MA10"))
        ma20 = _safe(latest.get("MA20"))
        ma60 = _safe(latest.get("MA60"))

        if not _valid(ma5, ma10, ma20, ma60):
            return None

        if ma5 > ma10 > ma20 > ma60:
            return Signal("均線排列", "買進 BUY", +1,
                f"均線多頭排列 (Bullish MA alignment): MA5>{ma5:.1f}>MA10>{ma10:.1f}>MA20>{ma20:.1f}>MA60>{ma60:.1f}", ma5)
        elif ma5 < ma10 < ma20 < ma60:
            return Signal("均線排列", "賣出 SELL", -1,
                f"均線空頭排列 (Bearish MA alignment): MA5<{ma5:.1f}<MA10<{ma10:.1f}<MA20<{ma20:.1f}<MA60<{ma60:.1f}", ma5)
        return None

    # ──────────────────────── Volume Confirmation ─────────────────────────────

    def _rule_volume_confirmation(self, latest: pd.Series) -> Signal | None:
        """Volume confirmation rule."""
        close = _safe(latest.get("Close"))
        open_ = _safe(latest.get("Open"))
        vol_ratio = _safe(latest.get("Volume_Ratio"))

        if not _valid(close, open_, vol_ratio):
            return None

        if vol_ratio > HIGH_VOLUME_RATIO:
            if close > open_:
                return Signal("成交量", "買進 BUY", +1,
                    f"放量上漲 (High volume rally): Vol ratio={vol_ratio:.1f}x", vol_ratio)
            else:
                return Signal("成交量", "賣出 SELL", -1,
                    f"放量下跌 (High volume decline): Vol ratio={vol_ratio:.1f}x", vol_ratio)
        return None

    # ─────────────────────── Confidence Mapping ───────────────────────────────

    def _compute_confidence(self, total_score: int) -> tuple[SignalDirection, ConfidenceLevel]:
        if total_score >= STRONG_BUY_THRESHOLD:
            return "買進 BUY", "強力"
        elif total_score >= MODERATE_BUY_THRESHOLD:
            return "買進 BUY", "中度"
        elif total_score >= WEAK_BUY_THRESHOLD:
            return "買進 BUY", "弱"
        elif total_score <= STRONG_SELL_THRESHOLD:
            return "賣出 SELL", "強力"
        elif total_score <= MODERATE_SELL_THRESHOLD:
            return "賣出 SELL", "中度"
        elif total_score <= WEAK_SELL_THRESHOLD:
            return "賣出 SELL", "弱"
        else:
            return "觀望 NEUTRAL", "觀望"

    def _build_summary(self, result: AnalysisResult) -> str:
        lines = [
            f"綜合分析 (Overall Analysis): {result.direction}",
            f"信心度: {result.confidence} | 信號強度: {result.total_score:+d}",
            "",
            "觸發信號 (Triggered Signals):",
        ]
        for s in sorted(result.signals, key=lambda x: abs(x.score), reverse=True):
            direction_tag = "BUY " if s.score > 0 else "SELL" if s.score < 0 else "INFO"
            lines.append(f"  [{direction_tag} {s.score:+d}] {s.reason}")

        if not result.signals:
            lines.append("  無明確訊號 (No active signals)")

        lines.append("")
        lines.append("⚠️ 本分析僅供參考，不構成投資建議 (For reference only, not investment advice)")
        return "\n".join(lines)

    # ───────────────────── Historical Signal Generation ───────────────────────

    def generate_historical_signals(
        self, df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """
        Generate a time-series of signals for chart marker overlays.

        Iterates over each row (starting from row 61 to ensure MA60 warmup)
        and records the total score and direction for that date.
        """
        records = []
        min_idx = 61  # Need MA60 + 1 prev row

        for i in range(min_idx, len(df)):
            window = df.iloc[: i + 1]
            result = self.analyze(window, ticker)
            records.append(
                {
                    "Date": df.index[i],
                    "Close": df["Close"].iloc[i],
                    "High": df["High"].iloc[i],
                    "Low": df["Low"].iloc[i],
                    "total_score": result.total_score,
                    "direction": result.direction,
                    "confidence": result.confidence,
                }
            )

        if not records:
            return pd.DataFrame(columns=["Date", "Close", "High", "Low", "total_score", "direction", "confidence"])

        signal_df = pd.DataFrame(records).set_index("Date")
        return signal_df
