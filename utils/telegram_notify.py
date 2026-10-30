"""
Telegram 推播通知
================
環境變數:
  TELEGRAM_BOT_TOKEN  Bot token (由 @BotFather 取得)
  TELEGRAM_CHAT_ID    你的 chat_id（用 @userinfobot 查詢）
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")


def _name(code: str) -> str:
    """回傳 '代號 中文名'，查不到就只回傳代號。"""
    try:
        from utils.constants import STOCK_DB
        full = STOCK_DB.get(code, "")
        # full 格式: "台積電 TSMC" → 只取中文部分
        zh = full.split()[0] if full else ""
        return f"{code} {zh}" if zh else code
    except Exception:
        return code

# 去重：同一則訊息 key 在 cooldown 秒內只發一次
_sent: dict[str, datetime] = {}
_sent_lock = threading.Lock()
_DEFAULT_COOLDOWN = 300  # 5 分鐘


def send(msg: str, key: str | None = None, cooldown: int = _DEFAULT_COOLDOWN) -> bool:
    """發送 Telegram 訊息。key 相同的訊息在 cooldown 秒內只發一次。"""
    if not BOT_TOKEN or not CHAT_ID:
        return False

    if key:
        now = datetime.now()
        with _sent_lock:
            if key in _sent and now - _sent[key] < timedelta(seconds=cooldown):
                return False
            _sent[key] = now

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=8,
        )
        if not resp.ok:
            logger.warning(f"Telegram 發送失敗: {resp.status_code} {resp.text[:100]}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Telegram 連線失敗: {e}")
        return False


def notify_buy(source: str, code: str, shares: int, price: float, reason: str = "") -> None:
    name = _name(code)
    msg = (
        f"🟢 <b>{source} 買入</b>\n"
        f"股票: <b>{name}</b>\n"
        f"數量: {shares:,} 股 / {shares//1000}張{shares%1000}股\n"
        f"價格: {price:.2f}\n"
        f"金額: {shares * price:,.0f}"
        + (f"\n備註: {reason}" if reason else "")
    )
    send(msg, key=f"buy:{code}:{price:.0f}", cooldown=60)


def notify_sell(source: str, code: str, shares: int, price: float, pnl: float, reason: str = "") -> None:
    emoji = "🔴" if pnl < 0 else "💰"
    name = _name(code)
    msg = (
        f"{emoji} <b>{source} 賣出</b>\n"
        f"股票: <b>{name}</b>\n"
        f"數量: {shares:,} 股\n"
        f"價格: {price:.2f}\n"
        f"損益: {pnl:+,.0f} TWD\n"
        f"原因: {reason}"
    )
    send(msg, key=f"sell:{code}:{price:.0f}", cooldown=60)


def notify_volume_surge(code: str, price: float, vol_ratio: float, score: int, action: str = "觀察") -> None:
    name = _name(code)
    msg = (
        f"📊 <b>大量預警 {name}</b>\n"
        f"現價: {price:.2f}\n"
        f"量比: {vol_ratio:.1f}x 均量\n"
        f"技術分: {score}\n"
        f"動作: {action}"
    )
    send(msg, key=f"surge:{code}", cooldown=600)  # 同一股票 10 分鐘最多一次


def notify_ta_signal(code: str, signal: str, confidence: float, summary: str) -> None:
    emoji = {"BUY": "🤖✅", "SELL": "🤖❌", "HOLD": "🤖⏸"}.get(signal, "🤖")
    name = _name(code)
    msg = (
        f"{emoji} <b>TradingAgents {name}</b>\n"
        f"訊號: {signal}  信心: {confidence:.0%}\n"
        f"{summary[:200]}"
    )
    send(msg, key=f"ta:{code}:{signal}", cooldown=3600)
