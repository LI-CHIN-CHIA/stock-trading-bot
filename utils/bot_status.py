"""Bot → DB：把伺服器端的交易狀態寫進 platform_bot_status，讓 App 看得到。

App 上的開關只是「使用者想要」；這裡回報的是伺服器「實際能不能」做真實交易，
例如 DAY_REAL_TRADING 有沒有開、富邦有沒有登入、即時報價通不通。
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime

import pytz

logger = logging.getLogger(__name__)
TZ = pytz.timezone("Asia/Taipei")
_last: dict = {"ts": 0.0, "payload": None}


def build_status(bot, real_day_bot) -> dict:
    from trader.day_trader import DAY_REAL_TRADING, DAY_REAL_MAX_DEVIATION
    logged_in = bool(getattr(bot, "sdk", None) is not None and getattr(bot, "account", None) is not None)
    quote_ok, sample = None, None
    if logged_in:
        try:
            from trader.day_trader import _quote_prices
            sample = _quote_prices(bot.live_quote("2330"))   # 用 bot 實際的解析方式驗證格式
            quote_ok = sample["last"] is not None
        except Exception:
            quote_ok = False
    connected = real_day_bot is not None and getattr(real_day_bot, "_place_order_fn", None) is not None
    return {
        "real_trading_enabled": DAY_REAL_TRADING,   # 伺服器是否允許真實當沖
        "fubon_logged_in":      logged_in,
        "live_quote_ok":        quote_ok,           # None = 未登入無法測
        "quote_sample_2330":    sample,
        "real_order_connected": connected,          # 真實帳戶已接上富邦下單
        "real_short_supported": False,              # 真實先賣後買尚未實作
        "paper_only_main_bot":  bool(getattr(bot, "paper_trading", True)),
        "max_signal_deviation": DAY_REAL_MAX_DEVIATION,
        "updated_at":           datetime.now(TZ).isoformat(timespec="seconds"),
    }


def report_status(bot, real_day_bot, min_interval: float = 0.0) -> None:
    """寫入（或更新）目前使用者的狀態列；失敗只記 log，不影響交易。"""
    uid = int(os.getenv("TRADING_USER_ID", "0") or 0)
    if not uid:
        return
    now = time.time()
    if now - _last["ts"] < min_interval:
        return
    _last["ts"] = now
    try:
        payload = build_status(bot, real_day_bot)
    except Exception as e:
        logger.debug(f"bot status 建立失敗: {e}")
        return
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("PLATFORM_DB_HOST", "postgres"),
            port=int(os.getenv("PLATFORM_DB_PORT", "5432")),
            dbname=os.getenv("PLATFORM_DB_NAME", "trading"),
            user=os.getenv("PLATFORM_DB_USER", "trading"),
            password=os.getenv("PLATFORM_DB_PASS", ""),
            connect_timeout=3,
        )
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS platform_bot_status ("
                    " user_id integer PRIMARY KEY, data text NOT NULL,"
                    " updated_at timestamptz NOT NULL DEFAULT now())")
                cur.execute(
                    "INSERT INTO platform_bot_status (user_id, data, updated_at) VALUES (%s, %s, now())"
                    " ON CONFLICT (user_id) DO UPDATE SET data = EXCLUDED.data, updated_at = now()",
                    (uid, json.dumps(payload, ensure_ascii=False)))
        conn.close()
        _last["payload"] = payload
    except Exception as e:
        logger.warning(f"bot status 寫入失敗: {e}")
