import os
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from ..database import get_db
from ..models import Trade, Position, UserConfig
from ..auth import get_current_user_id
from ..schemas import TradeIn, TradeOut, DailySummary, SystemTradeIn, AccountOut
from ..ws_hub import hub
from .. import accounts as acct

_SYSTEM_KEY = os.getenv("PLATFORM_SYSTEM_KEY", "")

_TZ_TAIPEI = timezone(timedelta(hours=8))
_TARGET_PCT = 0.01   # 每日目標：額度的 1%（使用者目標區間 1%~2% 的下緣）


def _today_taipei() -> str:
    return datetime.now(_TZ_TAIPEI).strftime("%Y-%m-%d")


def _date_conds(trade_date: str, days: int, start_date: str, end_date: str) -> list:
    """日期條件，優先順序：trade_date（單日）> start_date/end_date（區間，含頭尾）> 最近 days 天。"""
    for d in (trade_date, start_date, end_date):
        if d:
            try:
                datetime.strptime(d, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=422, detail=f"日期格式需為 YYYY-MM-DD: {d}")
    if trade_date:
        return [Trade.trade_date == trade_date]
    if start_date or end_date:
        if start_date and end_date and start_date > end_date:
            start_date, end_date = end_date, start_date
        conds = []
        if start_date:
            conds.append(Trade.trade_date >= start_date)
        if end_date:
            conds.append(Trade.trade_date <= end_date)
        return conds
    today = datetime.now(_TZ_TAIPEI).date()
    return [Trade.trade_date >= (today - timedelta(days=max(days, 1) - 1)).isoformat()]


async def _cfg(db: AsyncSession, user_id: int) -> UserConfig | None:
    return await db.scalar(select(UserConfig).where(UserConfig.user_id == user_id))


def _stats(trades: list[Trade]) -> tuple[float, int, int]:
    closes = [t for t in trades if t.action in ("SELL", "COVER")]
    return sum(t.pnl for t in closes), len(closes), sum(1 for t in closes if t.pnl > 0)


router = APIRouter(prefix="/api", tags=["trades"])


@router.get("/trades", response_model=list[TradeOut])
async def list_trades(
    trade_date: str = "",
    days: int = 1,
    start_date: str = "",
    end_date: str = "",
    account: str = "",
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    trade_date 指定時只回傳該日；start_date/end_date 指定區間；否則回傳最近 days 天。
    account: "real" / "paper-<額度>" / "all"；未指定時回傳真實＋主要模擬帳戶（見 accounts.read_filter）。
    """
    only = acct.read_filter(account, await _cfg(db, user_id))
    conds = [Trade.user_id == user_id, *_date_conds(trade_date, days, start_date, end_date)]
    if only is not None:
        conds.append(Trade.account.in_(only))
    rows = await db.scalars(
        select(Trade).where(and_(*conds)).order_by(Trade.trade_date.desc(), Trade.created_at.desc())
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
        account=acct.resolve_write(body.account, body.is_real, await _cfg(db, user_id)),
    )
    db.add(trade)
    await db.commit()
    await db.refresh(trade)
    return trade


@router.get("/summary", response_model=DailySummary)
async def daily_summary(
    trade_date: str = "",
    days: int = 1,
    start_date: str = "",
    end_date: str = "",
    account: str = "",
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    cfg = await _cfg(db, user_id)
    only = acct.read_filter(account, cfg)
    conds = [Trade.user_id == user_id, *_date_conds(trade_date, days, start_date, end_date)]
    if only is not None:
        conds.append(Trade.account.in_(only))
    trades = (await db.scalars(select(Trade).where(and_(*conds)))).all()

    budget = acct.budget_of(account, cfg)
    total_pnl, n, wins = _stats(trades)
    target_pnl = budget * _TARGET_PCT

    return DailySummary(
        account=account,
        budget=budget,
        date=trade_date or (f"{start_date}~{end_date}" if (start_date or end_date) else _today_taipei()),
        total_pnl=round(total_pnl, 0),
        trade_count=n,
        win_count=wins,
        win_rate=round(wins / n, 4) if n else 0.0,
        target_pct=_TARGET_PCT,
        target_pnl=round(target_pnl, 0),
        hit_target=total_pnl >= target_pnl,
    )


@router.get("/accounts", response_model=list[AccountOut])
async def list_accounts(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """真實帳戶 + 每個模擬額度各一個帳戶，附今日損益與持倉佔用。"""
    cfg = await _cfg(db, user_id)
    today = _today_taipei()
    trades = (await db.scalars(
        select(Trade).where(Trade.user_id == user_id, Trade.trade_date == today))).all()
    positions = (await db.scalars(select(Position).where(Position.user_id == user_id))).all()

    ids = [acct.REAL] + [acct.paper_id(b) for b in acct.paper_budgets(cfg)]
    # 近 30 日有成交或目前有持倉的其他帳戶（策略帳戶如 fade-*、已從設定移除的模擬帳戶）也列出
    since = (datetime.now(_TZ_TAIPEI).date() - timedelta(days=30)).isoformat()
    recent = (await db.scalars(
        select(Trade.account).where(Trade.user_id == user_id, Trade.trade_date >= since).distinct())).all()
    for a in sorted(set(recent) | {p.account for p in positions}):
        if a and a not in ids:
            ids.append(a)

    out = []
    for a in ids:
        budget = acct.budget_of(a, cfg)
        pnl, n, wins = _stats([t for t in trades if t.account == a])
        used = sum(p.shares * p.entry_price for p in positions if p.account == a)
        traded = sum(t.shares * t.price for t in trades
                     if t.account == a and t.action in ("BUY", "SHORT"))
        target = budget * _TARGET_PCT
        out.append(AccountOut(
            id=a, kind="real" if a == acct.REAL else "paper", label=acct.label_of(a),
            budget=budget, used=round(used, 0), traded_today=round(traded, 0),
            today_pnl=round(pnl, 0),
            pnl_pct=round(pnl / budget, 5) if budget else 0.0,
            trade_count=n, win_count=wins, win_rate=round(wins / n, 4) if n else 0.0,
            target_pnl=round(target, 0), hit_target=pnl >= target,
        ))
    return out


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
        account=acct.resolve_write(body.account, body.is_real, await _cfg(db, body.user_id)),
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
            "account":    trade.account,
        },
    })
    return trade
