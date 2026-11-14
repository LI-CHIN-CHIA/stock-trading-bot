from datetime import datetime
from pydantic import BaseModel, field_validator


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
    paper_budgets: list[int] = [500000]
    updated_at: datetime

    @field_validator("paper_budgets", mode="before")
    @classmethod
    def _split_budgets(cls, v):
        from .accounts import parse_budgets
        return parse_budgets(v)

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
    paper_budgets: list[int] | None = None


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
    account: str = ""     # "real" / "paper-<額度>"；空字串由伺服器依 is_real 歸戶
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
    account: str = ""     # "real" / "paper-<額度>"；空字串由伺服器依 is_real 歸戶
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
    account: str = ""     # "real" / "paper-<額度>"；空字串由伺服器依 is_real 歸戶
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
    account: str = ""     # "real" / "paper-<額度>"；空字串由伺服器依 is_real 歸戶


# ── System (bot → platform) ───────────────────────────────────────────────────

class SystemPositionUpsert(BaseModel):
    user_id: int
    code: str
    direction: str = "LONG"   # LONG / SHORT
    shares: int               # 0 = close the position
    entry_price: float
    is_real: bool = False
    account: str = ""     # "real" / "paper-<額度>"；空字串由伺服器依 is_real 歸戶


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
    account: str = ""     # "real" / "paper-<額度>"；空字串由伺服器依 is_real 歸戶
    trade_date: str = ""  # YYYY-MM-DD; empty → today


# ── Summary ──────────────────────────────────────────────────────────────────

class AccountOut(BaseModel):
    id: str               # "real" / "paper-500000"
    kind: str             # "real" / "paper"
    label: str
    budget: float         # 真實=單筆額度；模擬=帳戶總額度
    used: float           # 目前持倉佔用金額（收盤平倉後為 0）
    traded_today: float = 0.0   # 今日進場成交金額合計（BUY + SHORT）
    today_pnl: float
    pnl_pct: float        # today_pnl / budget
    trade_count: int
    win_count: int
    win_rate: float
    target_pnl: float
    hit_target: bool


class DayUsage(BaseModel):
    date: str
    traded: float = 0.0          # 當日進場成交金額合計
    peak_used: float = 0.0       # 當日最高同時佔用
    peak_used_pct: float = 0.0   # 佔額度比例
    pnl: float = 0.0
    trade_count: int = 0


class DailySummary(BaseModel):
    traded: float = 0.0          # 區間進場成交金額合計
    avg_traded: float = 0.0      # 每個交易日平均
    peak_used: float = 0.0       # 區間內單日最高同時佔用
    peak_used_pct: float = 0.0
    trading_days: int = 0
    days: list[DayUsage] = []
    account: str = ""
    budget: float = 0.0
    date: str
    total_pnl: float
    trade_count: int
    win_count: int
    win_rate: float
    target_pct: float = 0.01
    target_pnl: float
    hit_target: bool
