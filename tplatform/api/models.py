from datetime import datetime, timezone
from sqlalchemy import String, Float, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "platform_users"

    id:           Mapped[int]      = mapped_column(Integer, primary_key=True)
    email:        Mapped[str]      = mapped_column(String(255), unique=True, nullable=False)
    password_hash:Mapped[str]      = mapped_column(String(255), nullable=False)
    display_name: Mapped[str]      = mapped_column(String(100), default="")
    is_active:    Mapped[bool]     = mapped_column(Boolean, default=True)
    is_admin:     Mapped[bool]     = mapped_column(Boolean, default=False)
    created_at:   Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    config:    Mapped["UserConfig"]     = relationship("UserConfig",  back_populates="user", uselist=False, cascade="all, delete-orphan")
    trades:    Mapped[list["Trade"]]    = relationship("Trade",     back_populates="user", cascade="all, delete-orphan")
    positions: Mapped[list["Position"]] = relationship("Position",  back_populates="user", cascade="all, delete-orphan")


class UserConfig(Base):
    __tablename__ = "platform_configs"

    id:           Mapped[int]   = mapped_column(Integer, primary_key=True)
    user_id:      Mapped[int]   = mapped_column(Integer, ForeignKey("platform_users.id"), unique=True)
    enable_day:      Mapped[bool]  = mapped_column(Boolean, default=True)
    enable_short:    Mapped[bool]  = mapped_column(Boolean, default=True)
    enable_overnight:Mapped[bool]  = mapped_column(Boolean, default=False)
    auto_trade:      Mapped[bool]  = mapped_column(Boolean, default=False)
    day_budget:      Mapped[float] = mapped_column(Float, default=50000.0)
    max_positions:   Mapped[int]   = mapped_column(Integer, default=3)
    stop_loss_pct:   Mapped[float] = mapped_column(Float, default=-0.015)
    take_profit_pct: Mapped[float] = mapped_column(Float, default=0.025)
    paper_budgets:   Mapped[str]   = mapped_column(String(200), default="500000", server_default="500000")
    updated_at: Mapped[datetime]   = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    user: Mapped["User"] = relationship("User", back_populates="config")


class Signal(Base):
    __tablename__ = "platform_signals"

    id:         Mapped[int]      = mapped_column(Integer, primary_key=True)
    code:       Mapped[str]      = mapped_column(String(10))
    action:     Mapped[str]      = mapped_column(String(10))
    score:      Mapped[int]      = mapped_column(Integer, default=0)
    price:      Mapped[float]    = mapped_column(Float)
    reason:     Mapped[str]      = mapped_column(String(200), default="")
    signal_date:Mapped[str]      = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Trade(Base):
    __tablename__ = "platform_trades"

    id:         Mapped[int]   = mapped_column(Integer, primary_key=True)
    user_id:    Mapped[int]   = mapped_column(Integer, ForeignKey("platform_users.id"))
    trade_date: Mapped[str]   = mapped_column(String(10))
    code:       Mapped[str]   = mapped_column(String(10))
    action:     Mapped[str]   = mapped_column(String(10))
    shares:     Mapped[int]   = mapped_column(Integer)
    price:      Mapped[float] = mapped_column(Float)
    pnl:        Mapped[float] = mapped_column(Float, default=0.0)
    reason:     Mapped[str]   = mapped_column(String(100), default="")
    order_no:   Mapped[str]   = mapped_column(String(50), default="")
    is_real:    Mapped[bool]  = mapped_column(Boolean, default=True)
    account:    Mapped[str]   = mapped_column(String(24), default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped["User"] = relationship("User", back_populates="trades")


class Position(Base):
    """Live positions reported by the Local Agent."""
    __tablename__ = "platform_positions"

    id:          Mapped[int]      = mapped_column(Integer, primary_key=True)
    user_id:     Mapped[int]      = mapped_column(Integer, ForeignKey("platform_users.id"))
    code:        Mapped[str]      = mapped_column(String(10))
    direction:   Mapped[str]      = mapped_column(String(5), default="LONG")  # LONG / SHORT
    shares:      Mapped[int]      = mapped_column(Integer)
    entry_price: Mapped[float]    = mapped_column(Float)
    is_real:     Mapped[bool]     = mapped_column(Boolean, default=False)
    account:     Mapped[str]      = mapped_column(String(24), default="", server_default="")
    opened_at:   Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    user: Mapped["User"] = relationship("User", back_populates="positions")
