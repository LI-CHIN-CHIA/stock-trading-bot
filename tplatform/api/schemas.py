from datetime import datetime
from pydantic import BaseModel


# ── Auth ─────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str = ""

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    display_name: str


# ── Config ───────────────────────────────────────────────────────────────────

class ConfigResponse(BaseModel):
    enable_day: bool
    enable_short: bool
    enable_overnight: bool
    auto_trade: bool
    day_budget: float
    max_positions: int
    stop_loss_pct: float
    take_profit_pct: float
    updated_at: datetime

    class Config:
        from_attributes = True

class ConfigUpdate(BaseModel):
    enable_day: bool | None = None
    enable_short: bool | None = None
    enable_overnight: bool | None = None
    auto_trade: bool | None = None
    day_budget: float | None = None
    max_positions: int | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None


# ── Trades ───────────────────────────────────────────────────────────────────

class TradeIn(BaseModel):
    code: str
    action: str           # BUY / SELL
    shares: int
    price: float
    pnl: float = 0.0
    reason: str = ""
    order_no: str = ""
    is_real: bool = True
    trade_date: str = ""  # YYYY-MM-DD; empty → today

class TradeOut(BaseModel):
    id: int
    trade_date: str
    code: str
    action: str
    shares: int
    price: float
    pnl: float
    reason: str
    order_no: str
    is_real: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ── Signals ──────────────────────────────────────────────────────────────────

class SignalIn(BaseModel):
    code: str
    action: str        # BUY / SELL / SHORT / COVER
    score: int = 0
    price: float
    reason: str = ""
    signal_date: str = ""

class SignalOut(BaseModel):
    id: int
    code: str
    action: str
    score: int
    price: float
    reason: str
    signal_date: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Positions ─────────────────────────────────────────────────────────────────

class PositionOut(BaseModel):
    id: int
    code: str
    direction: str
    shares: int
    entry_price: float
    is_real: bool
    opened_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class PositionUpsert(BaseModel):
    """Sent by Local Agent to open or close a position."""
    code: str
    direction: str = "LONG"   # LONG / SHORT
    shares: int               # 0 = close the position
    entry_price: float
    is_real: bool = False


# ── System (bot → platform) ───────────────────────────────────────────────────

class SystemPositionUpsert(BaseModel):
    user_id: int
    code: str
    direction: str = "LONG"   # LONG / SHORT
    shares: int               # 0 = close the position
    entry_price: float
    is_real: bool = False


class SystemTradeIn(BaseModel):
    user_id: int
    code: str
    action: str           # BUY / SELL / SHORT / COVER
    shares: int
    price: float
    pnl: float = 0.0
    reason: str = ""
    order_no: str = ""
    is_real: bool = False
    trade_date: str = ""  # YYYY-MM-DD; empty → today


# ── Summary ──────────────────────────────────────────────────────────────────

class DailySummary(BaseModel):
    date: str
    total_pnl: float
    trade_count: int
    win_count: int
    win_rate: float
    target_pct: float = 0.02
    target_pnl: float
    hit_target: bool
