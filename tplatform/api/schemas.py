from datetime import datetime
from pydantic import BaseModel, EmailStr


# ── Auth ─────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: str = ""

class LoginRequest(BaseModel):
    email: EmailStr
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
