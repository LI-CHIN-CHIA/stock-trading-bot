"""
三大法人買賣超快取與進場過濾
==============================
每日收盤後 TWSE 公布 T86 資料；本模組快取當日資料供 _scan_and_buy 使用。

公開 API:
  get_institutional_today() -> dict[code, {foreign_net, trust_net, dealer_net, total_net}]
  is_foreign_buying(code)   -> bool | None   (None = 無資料，放行)
"""

import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

_cache: dict[str, dict] = {}   # date_str -> {code: {...}}
_cache_lock = threading.Lock()


def get_institutional_today(date_str: str = "") -> dict[str, dict]:
    """
    取得當日三大法人買賣超資料（有快取，同一天只打一次 TWSE）。
    date_str: YYYYMMDD，空字串代表今天。
    """
    from utils.stock_discovery import _fetch_twse_institutional
    if not date_str:
        date_str = datetime.now().strftime("%Y%m%d")

    with _cache_lock:
        if date_str in _cache:
            return _cache[date_str]

    data = _fetch_twse_institutional(date_str)

    with _cache_lock:
        _cache[date_str] = data
        # 只保留最近 5 個交易日
        if len(_cache) > 5:
            oldest = sorted(_cache)[0]
            del _cache[oldest]

    return data


def is_foreign_buying(code: str) -> bool | None:
    """
    外資今日是否淨買超（foreign_net >= 0）。
    回傳 None 表示 TWSE 無此股資料（OTC / 盤前尚未公布），視為放行。
    """
    data = get_institutional_today()
    if not data:
        return None
    entry = data.get(code)
    if entry is None:
        return None
    return entry.get("foreign_net", 0) >= 0
