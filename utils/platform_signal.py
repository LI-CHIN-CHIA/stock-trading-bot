"""
Broadcast trading signals from trading_bot → Platform API.
Only active when PLATFORM_API_URL and PLATFORM_SYSTEM_KEY env vars are set.
"""
import os
import logging

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
