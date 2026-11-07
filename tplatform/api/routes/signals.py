import os
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..database import get_db
from ..models import Signal, User
from ..auth import get_current_user_id  # used by list_signals
from ..schemas import SignalIn, SignalOut
from ..ws_hub import hub

router = APIRouter(prefix="/api", tags=["signals"])

_SYSTEM_KEY = os.getenv("PLATFORM_SYSTEM_KEY", "")
_TZ_TAIPEI = timezone(timedelta(hours=8))


def _today_taipei() -> str:
    return datetime.now(_TZ_TAIPEI).strftime("%Y-%m-%d")


@router.get("/signals", response_model=list[SignalOut])
async def list_signals(
    signal_date: str = "",
    _: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    d = signal_date or _today_taipei()
    rows = await db.scalars(
        select(Signal).where(Signal.signal_date == d).order_by(Signal.created_at.desc())
    )
    return rows.all()


@router.post("/signals/broadcast", response_model=SignalOut, status_code=201)
async def broadcast_signal(
    body: SignalIn,
    db: AsyncSession = Depends(get_db),
    x_system_key: str = Header(default=""),
    authorization: str = Header(default=""),
):
    """儲存信號並廣播給所有已連線的 Local Agent。需系統 key 或 admin JWT。"""
    # System key path: trading_bot server → no JWT required
    if x_system_key and _SYSTEM_KEY and x_system_key == _SYSTEM_KEY:
        pass
    else:
        # JWT path: must be admin
        from ..auth import decode_token as _decode
        token = authorization.removeprefix("Bearer ").strip() if authorization.startswith("Bearer ") else ""
        uid = _decode(token) if token else None
        if not uid:
            raise HTTPException(status_code=401, detail="Authentication required")
        user = await db.scalar(select(User).where(User.id == uid))
        if not user or not user.is_admin:
            raise HTTPException(status_code=403, detail="Admin or system key required")

    sig = Signal(
        code=body.code,
        action=body.action,
        score=body.score,
        price=body.price,
        reason=body.reason,
        signal_date=body.signal_date or _today_taipei(),
    )
    db.add(sig)
    await db.commit()
    await db.refresh(sig)

    await hub.broadcast({
        "type": "signal",
        "data": {
            "id":    sig.id,
            "code":  sig.code,
            "action":sig.action,
            "score": sig.score,
            "price": sig.price,
            "reason":sig.reason,
        },
    })

    return sig
