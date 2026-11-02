"""
台指期夜盤方向門控
==================
每日 08:30 開盤前呼叫，取得夜盤結束後的台指期漲跌幅，
作為當日是否允許開倉的前置過濾條件。

資料來源:
  1. TAIFEX 即時行情 API（近月台指期 TX）
  2. 備援: iShares MSCI Taiwan ETF (EWT) 前日美股漲跌

Gate 邏輯:
  夜盤跌 > 1.5%  → 全日禁止買入
  夜盤跌 0.5~1.5% → 買入門檻額外 +0.03
  其他            → 正常放行
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)
HEADERS = {"User-Agent": "Mozilla/5.0"}

_night_cache: tuple | None = None  # (timestamp, change_pct_or_None)
_CACHE_TTL = 3600  # 1 hour


def _fetch_taifex() -> float | None:
    try:
        r = requests.get(
            "https://mis.taifex.com.tw/futures/api/getQuoteList",
            params={"MarketType": "0", "SymbolType": "F", "queryType": "1",
                    "commodity_id": "TX"},
            headers=HEADERS,
            timeout=8,
        )
        r.raise_for_status()
        items = r.json().get("RTList") or r.json().get("data", [])
        if not items:
            return None
        row = items[0]
        close = float(row.get("Close") or row.get("LS") or row.get("lastPrice") or 0)
        ref   = float(row.get("RefPrice") or row.get("RO") or row.get("refPrice") or 0)
        if close > 0 and ref > 0:
            return (close - ref) / ref
    except Exception as e:
        logger.debug(f"TAIFEX 夜盤 API 失敗: {e}")
    return None


def _fetch_ewt() -> float | None:
    try:
        import yfinance as yf
        df = yf.Ticker("EWT").history(period="3d")
        if len(df) >= 2:
            return float(
                (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]
            )
    except Exception as e:
        logger.debug(f"EWT 備援失敗: {e}")
    return None


def get_night_futures_change() -> float | None:
    """
    取得夜盤漲跌幅。主要源: TAIFEX；備援: EWT。
    快取 1 小時。回傳 None 代表無資料（預設放行）。
    """
    global _night_cache
    if _night_cache is not None:
        ts, chg = _night_cache
        if (time.time() - ts) < _CACHE_TTL:
            return chg

    chg = _fetch_taifex()
    if chg is None:
        chg = _fetch_ewt()

    _night_cache = (time.time(), chg)
    if chg is not None:
        arrow = "⬆️" if chg > 0 else "⬇️"
        logger.info(f"夜盤方向: {arrow} {chg:+.2%}")
    else:
        logger.info("夜盤方向: 無資料（放行）")
    return chg


def night_gate_ok() -> tuple[bool, float]:
    """
    開盤前門控。
    Returns:
      (gate_open: bool, extra_threshold: float)
      gate_open=False  → 跳過今日所有買入
      extra_threshold  → 買入機率門檻額外加碼
    """
    chg = get_night_futures_change()
    if chg is None:
        return True, 0.0

    if chg <= -0.015:
        logger.warning(f"🚫 夜盤大跌 {chg:+.2%}，跳過今日所有買入")
        return False, 0.0
    elif chg <= -0.005:
        extra = 0.03
        logger.info(f"⚠️  夜盤下跌 {chg:+.2%}，買入門檻提高 +{extra:.0%}")
        return True, extra
    else:
        return True, 0.0
