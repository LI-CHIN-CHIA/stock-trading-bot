from datetime import date
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..database import get_db
from ..models import Signal
from ..auth import get_current_user_id
from ..schemas import SignalIn, SignalOut
from ..ws_hub import hub

router = APIRouter(prefix="/api", tags=["signals"])


@router.get("/signals", response_model=list[SignalOut])
async def list_signals(
    signal_date: str = "",
    _: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    d = signal_date or str(date.today())
    rows = await db.scalars(
        select(Signal).where(Signal.signal_date == d).order_by(Signal.created_at.desc())
    )
    return rows.all()


@router.post("/signals/broadcast", response_model=SignalOut, status_code=201)
async def broadcast_signal(
    body: SignalIn,
    _: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """儲存信號並廣播給所有已連線的 Local Agent。"""
    sig = Signal(
        code=body.code,
        action=body.action,
        score=body.score,
        price=body.price,
        reason=body.reason,
        signal_date=body.signal_date or str(date.today()),
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
