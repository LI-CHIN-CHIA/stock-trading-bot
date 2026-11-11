import os
from datetime import datetime, timezone, timedelta, date as date_type
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from ..database import get_db
from ..models import Trade, UserConfig
from ..auth import get_current_user_id
from ..schemas import TradeIn, TradeOut, DailySummary, SystemTradeIn
from ..ws_hub import hub

_SYSTEM_KEY = os.getenv("PLATFORM_SYSTEM_KEY", "")

_TZ_TAIPEI = timezone(timedelta(hours=8))


def _today_taipei() -> str:
    return datetime.now(_TZ_TAIPEI).strftime("%Y-%m-%d")

router = APIRouter(prefix="/api", tags=["trades"])


@router.get("/trades", response_model=list[TradeOut])
async def list_trades(
    trade_date: str = "",
    days: int = 1,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    trade_date 指定時：只回傳該日。
    未指定時依 days 決定範圍（days=1 只回今天，days=7 回最近 7 天）。
    """
    if trade_date:
        rows = await db.scalars(
            select(Trade)
            .where(and_(Trade.trade_date == trade_date, Trade.user_id == user_id))
            .order_by(Trade.created_at.desc())
        )
    else:
        today = datetime.now(_TZ_TAIPEI).date()
        start = (today - timedelta(days=days - 1)).isoformat()
        rows = await db.scalars(
            select(Trade)
            .where(and_(Trade.trade_date >= start, Trade.user_id == user_id))
            .order_by(Trade.trade_date.desc(), Trade.created_at.desc())
        )
    return rows.all()


@router.post("/trades", response_model=TradeOut, status_code=201)
async def report_trade(
    body: TradeIn,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    trade = Trade(
        user_id=user_id,
        trade_date=body.trade_date or _today_taipei(),
        code=body.code,
        action=body.action,
        shares=body.shares,
        price=body.price,
        pnl=body.pnl,
        reason=body.reason,
        order_no=body.order_no,
        is_real=body.is_real,
    )
    db.add(trade)
    await db.commit()
    await db.refresh(trade)
    return trade


@router.get("/summary", response_model=DailySummary)
async def daily_summary(
    trade_date: str = "",
    days: int = 1,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    if trade_date:
        condition = and_(Trade.trade_date == trade_date, Trade.user_id == user_id)
        d = trade_date
    else:
        today = datetime.now(_TZ_TAIPEI).date()
        start = (today - timedelta(days=days - 1)).isoformat()
        condition = and_(Trade.trade_date >= start, Trade.user_id == user_id)
        d = _today_taipei()
    trades = (await db.scalars(select(Trade).where(condition))).all()

    cfg = await db.scalar(select(UserConfig).where(UserConfig.user_id == user_id))
    budget = cfg.day_budget if cfg else 50000.0

    total_pnl  = sum(t.pnl for t in trades if t.action == "SELL")
    sell_trades = [t for t in trades if t.action == "SELL"]
    win_count  = sum(1 for t in sell_trades if t.pnl > 0)
    target_pnl = budget * 0.02

    return DailySummary(
        date=d,
        total_pnl=round(total_pnl, 0),
        trade_count=len(sell_trades),
        win_count=win_count,
        win_rate=round(win_count / len(sell_trades), 4) if sell_trades else 0.0,
        target_pnl=round(target_pnl, 0),
        hit_target=total_pnl >= target_pnl,
    )


@router.post("/system/trades", response_model=TradeOut, status_code=201)
async def system_report_trade(
    body: SystemTradeIn,
    db: AsyncSession = Depends(get_db),
    x_system_key: str = Header(default=""),
):
    """Bot 回報交易紀錄（系統 key 認證，不需要 JWT）。"""
    if not _SYSTEM_KEY or x_system_key != _SYSTEM_KEY:
        raise HTTPException(status_code=401, detail="System key required")

    trade = Trade(
        user_id=body.user_id,
        trade_date=body.trade_date or _today_taipei(),
        code=body.code,
        action=body.action,
        shares=body.shares,
        price=body.price,
        pnl=body.pnl,
        reason=body.reason,
        order_no=body.order_no,
        is_real=body.is_real,
    )
    db.add(trade)
    await db.commit()
    await db.refresh(trade)

    await hub.send(body.user_id, {
        "type": "trade",
        "data": {
            "id":         trade.id,
            "trade_date": trade.trade_date,
            "code":       trade.code,
            "action":     trade.action,
            "shares":     trade.shares,
            "price":      float(trade.price),
            "pnl":        float(trade.pnl),
            "reason":     trade.reason,
            "is_real":    trade.is_real,
        },
    })
    return trade
