"""
企業重要事件標記
================
減資 (capital_reduction_flag) — 近 45 天內有減資 → 緊縮停損至 -15%、禁止新買
庫藏股 (buyback_active)       — 正在執行庫藏股買回 → 偏多訊號

資料來源: TWSE 公開觀測站
快取: 6 小時更新一次
"""

import logging
import time
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0"}

_cache: dict = {}
_cache_ts: float = 0.0
_CACHE_TTL = 6 * 3600


def _fetch_capital_reductions() -> set:
    codes: set = set()
    try:
        r = requests.get(
            "https://www.twse.com.tw/rwd/zh/reducedCapital/twtasuReducedCapital",
            params={"response": "json"},
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        rows = r.json().get("data", [])
        today = datetime.now()
        cutoff = today + timedelta(days=45)
        for row in rows:
            try:
                code = str(row[0]).strip().replace(" ", "")
                for cell in row[1:6]:
                    cell = str(cell).strip()
                    if len(cell) >= 7 and cell[:3].isdigit():
                        parts = cell.replace("/", "-").split("-")
                        ex_date = datetime(int(parts[0]) + 1911, int(parts[1]), int(parts[2]))
                        if today <= ex_date <= cutoff:
                            codes.add(code)
                        break
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"減資資料抓取失敗: {e}")
    return codes


def _fetch_buybacks() -> set:
    codes: set = set()
    try:
        r = requests.get(
            "https://www.twse.com.tw/rwd/zh/stockBuyBack/twtb4u",
            params={"response": "json"},
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        for row in r.json().get("data", []):
            try:
                code = str(row[0]).strip().replace(" ", "")
                if code:
                    codes.add(code)
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"庫藏股資料抓取失敗: {e}")
    return codes


def _refresh() -> None:
    global _cache, _cache_ts
    cap = _fetch_capital_reductions()
    time.sleep(0.3)
    buy = _fetch_buybacks()
    _cache = {"capital_reductions": cap, "buybacks": buy}
    _cache_ts = time.time()
    logger.info(f"企業事件更新: 減資 {len(cap)} 支, 庫藏股 {len(buy)} 支")


def get_event_flags(code: str) -> dict:
    """
    回傳 {"capital_reduction_flag": 0/1, "buyback_active": 0/1}
    capital_reduction_flag=1 → 禁止買入、使用 -15% 固定停損
    buyback_active=1         → 偏多加分訊號
    """
    global _cache_ts
    if not _cache or (time.time() - _cache_ts) > _CACHE_TTL:
        try:
            _refresh()
        except Exception as e:
            logger.warning(f"企業事件快取刷新失敗: {e}")
            if not _cache:
                return {"capital_reduction_flag": 0, "buyback_active": 0}
    return {
        "capital_reduction_flag": int(code in _cache.get("capital_reductions", set())),
        "buyback_active":         int(code in _cache.get("buybacks", set())),
    }
