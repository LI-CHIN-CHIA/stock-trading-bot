from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from ..database import get_db
from ..models import Position
from ..auth import get_current_user_id
from ..schemas import PositionOut, PositionUpsert
from ..ws_hub import hub

router = APIRouter(prefix="/api", tags=["positions"])


@router.get("/positions", response_model=list[PositionOut])
async def list_positions(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    rows = await db.scalars(
        select(Position)
        .where(Position.user_id == user_id)
        .order_by(Position.opened_at.desc())
    )
    return rows.all()


@router.post("/positions/upsert", response_model=PositionOut | None, status_code=200)
async def upsert_position(
    body: PositionUpsert,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Open (shares > 0) or close (shares == 0) a position."""
    existing = await db.scalar(
        select(Position).where(
            Position.user_id == user_id,
            Position.code == body.code,
            Position.direction == body.direction,
        )
    )

    if body.shares == 0:
        # Close: remove the position
        if existing:
            await db.execute(delete(Position).where(Position.id == existing.id))
            await db.commit()
        await hub.send(user_id, {"type": "position_update", "data": {
            "code": body.code, "direction": body.direction, "shares": 0,
        }})
        return None

    if existing:
        existing.shares = body.shares
        existing.entry_price = body.entry_price
        existing.is_real = body.is_real
        await db.commit()
        await db.refresh(existing)
        pos = existing
    else:
        pos = Position(
            user_id=user_id,
            code=body.code,
            direction=body.direction,
            shares=body.shares,
            entry_price=body.entry_price,
            is_real=body.is_real,
        )
        db.add(pos)
        await db.commit()
        await db.refresh(pos)

    await hub.send(user_id, {"type": "position_update", "data": {
        "id": pos.id, "code": pos.code, "direction": pos.direction,
        "shares": pos.shares, "entry_price": pos.entry_price,
        "is_real": pos.is_real,
    }})
    return pos
