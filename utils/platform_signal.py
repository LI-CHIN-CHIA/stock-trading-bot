"""
Broadcast trading signals and trade records from trading_bot → Platform API.
Only active when PLATFORM_API_URL and PLATFORM_SYSTEM_KEY env vars are set.
"""
import os
import logging
import threading

logger = logging.getLogger("platform_signal")

_URL = os.getenv("PLATFORM_API_URL", "").rstrip("/")
_KEY = os.getenv("PLATFORM_SYSTEM_KEY", "")


def broadcast(code: str, action: str, score: int, price: float, reason: str = "") -> None:
    """POST a signal to the platform; silently skips if not configured."""
    if not _URL or not _KEY:
        return
    try:
        import requests
        resp = requests.post(
            f"{_URL}/api/signals/broadcast",
            json={
                "code":   code,
                "action": action,
                "score":  score,
                "price":  round(float(price), 2),
                "reason": reason[:200],
            },
            headers={"X-System-Key": _KEY},
            timeout=2,
        )
        if resp.status_code == 201:
            logger.debug(f"[PlatformSignal] broadcast ok {action} {code}")
        else:
            logger.warning(f"[PlatformSignal] broadcast failed {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        logger.warning(f"[PlatformSignal] broadcast error: {e}")


def report_trade(
    user_id: int,
    code: str,
    action: str,
    shares: int,
    price: float,
    pnl: float = 0.0,
    reason: str = "",
    order_no: str = "",
    is_real: bool = False,
) -> None:
    """POST a trade record to the platform in a background thread. Silently skips if not configured."""
    if not _URL or not _KEY or not user_id:
        return

    def _post():
        try:
            import requests
            resp = requests.post(
                f"{_URL}/api/system/trades",
                json={
                    "user_id":  user_id,
                    "code":     code,
                    "action":   action,
                    "shares":   shares,
                    "price":    round(float(price), 2),
                    "pnl":      round(float(pnl), 0),
                    "reason":   reason[:200],
                    "order_no": order_no,
                    "is_real":  is_real,
                },
                headers={"X-System-Key": _KEY},
                timeout=3,
            )
            if resp.status_code == 201:
                logger.debug(f"[PlatformSignal] trade reported {action} {code} is_real={is_real}")
            else:
                logger.warning(f"[PlatformSignal] trade report failed {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            logger.warning(f"[PlatformSignal] trade report error: {e}")

    threading.Thread(target=_post, daemon=True).start()


def report_position(
    user_id: int,
    code: str,
    direction: str,       # "LONG" / "SHORT"
    shares: int,          # 0 = 平倉
    entry_price: float,
    is_real: bool = False,
) -> None:
    """POST 開倉/平倉到 platform。Background thread，失敗不影響主流程。"""
    if not _URL or not _KEY or not user_id:
        return

    def _post():
        try:
            import requests
            resp = requests.post(
                f"{_URL}/api/system/positions/upsert",
                json={
                    "user_id":     user_id,
                    "code":        code,
                    "direction":   direction,
                    "shares":      shares,
                    "entry_price": round(float(entry_price), 2),
                    "is_real":     is_real,
                },
                headers={"X-System-Key": _KEY},
                timeout=3,
            )
            action = "close" if shares == 0 else "open"
            if resp.status_code in (200, 201):
                logger.debug(f"[PlatformSignal] position {action} ok {direction} {code} shares={shares}")
            else:
                logger.warning(f"[PlatformSignal] position {action} failed {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            logger.warning(f"[PlatformSignal] position report error: {e}")

    threading.Thread(target=_post, daemon=True).start()
