import os
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from ..database import get_db
from ..models import Position, UserConfig
from ..auth import get_current_user_id
from ..schemas import PositionOut, PositionUpsert, SystemPositionUpsert
from ..ws_hub import hub
from .. import accounts as acct

_SYSTEM_KEY = os.getenv("PLATFORM_SYSTEM_KEY", "")

router = APIRouter(prefix="/api", tags=["positions"])


async def _cfg(db: AsyncSession, user_id: int) -> UserConfig | None:
    return await db.scalar(select(UserConfig).where(UserConfig.user_id == user_id))


@router.get("/positions", response_model=list[PositionOut])
async def list_positions(
    account: str = "",
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    only = acct.read_filter(account, await _cfg(db, user_id))
    conds = [Position.user_id == user_id]
    if only is not None:
        conds.append(Position.account.in_(only))
    rows = await db.scalars(select(Position).where(*conds).order_by(Position.opened_at.desc()))
    return rows.all()


async def _upsert(user_id: int, body: PositionUpsert | SystemPositionUpsert,
                  db: AsyncSession) -> Position | None:
    """開倉（shares > 0）或平倉（shares == 0）；同一檔股票在不同帳戶各自獨立。"""
    account = acct.resolve_write(body.account, body.is_real, await _cfg(db, user_id))
    existing = await db.scalar(
        select(Position).where(
            Position.user_id == user_id,
            Position.code == body.code,
            Position.direction == body.direction,
            Position.account == account,
        )
    )

    if body.shares == 0:
        if existing:
            await db.execute(delete(Position).where(Position.id == existing.id))
            await db.commit()
        await hub.send(user_id, {"type": "position_update", "data": {
            "code": body.code, "direction": body.direction, "shares": 0, "account": account,
        }})
        return None

    if existing:
        existing.shares = body.shares
        existing.entry_price = body.entry_price
        existing.is_real = body.is_real
        pos = existing
    else:
        pos = Position(
            user_id=user_id,
            code=body.code,
            direction=body.direction,
            shares=body.shares,
            entry_price=body.entry_price,
            is_real=body.is_real,
            account=account,
        )
        db.add(pos)
    await db.commit()
    await db.refresh(pos)

    await hub.send(user_id, {"type": "position_update", "data": {
        "id": pos.id, "code": pos.code, "direction": pos.direction,
        "shares": pos.shares, "entry_price": pos.entry_price,
        "is_real": pos.is_real, "account": pos.account,
    }})
    return pos


@router.post("/positions/upsert", response_model=PositionOut | None, status_code=200)
async def upsert_position(
    body: PositionUpsert,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    return await _upsert(user_id, body, db)


@router.post("/system/positions/upsert", status_code=200)
async def system_upsert_position(
    body: SystemPositionUpsert,
    db: AsyncSession = Depends(get_db),
    x_system_key: str = Header(default=""),
):
    """Bot 開倉/平倉回報（系統 key 認證）。"""
    if not _SYSTEM_KEY or x_system_key != _SYSTEM_KEY:
        raise HTTPException(status_code=401, detail="System key required")
    return await _upsert(body.user_id, body, db)
