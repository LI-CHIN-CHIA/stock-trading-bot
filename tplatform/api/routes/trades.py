from datetime import date
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from ..database import get_db
from ..models import Trade, UserConfig
from ..auth import get_current_user_id
from ..schemas import TradeIn, TradeOut, DailySummary

router = APIRouter(prefix="/api", tags=["trades"])


@router.get("/trades", response_model=list[TradeOut])
async def list_trades(
    trade_date: str = "",
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    d = trade_date or str(date.today())
    rows = await db.scalars(
        select(Trade)
        .where(Trade.user_id == user_id, Trade.trade_date == d)
        .order_by(Trade.created_at.desc())
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
        trade_date=body.trade_date or str(date.today()),
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
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    d = trade_date or str(date.today())
    trades = (await db.scalars(
        select(Trade).where(Trade.user_id == user_id, Trade.trade_date == d)
    )).all()

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
