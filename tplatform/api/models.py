from datetime import datetime
from sqlalchemy import String, Float, Integer, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


class User(Base):
    __tablename__ = "platform_users"

    id:           Mapped[int]      = mapped_column(Integer, primary_key=True)
    email:        Mapped[str]      = mapped_column(String(255), unique=True, nullable=False)
    password_hash:Mapped[str]      = mapped_column(String(255), nullable=False)
    display_name: Mapped[str]      = mapped_column(String(100), default="")
    is_active:    Mapped[bool]     = mapped_column(Boolean, default=True)
    created_at:   Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    config:  Mapped["UserConfig"]  = relationship("UserConfig",  back_populates="user", uselist=False, cascade="all, delete-orphan")
    trades:  Mapped[list["Trade"]] = relationship("Trade",  back_populates="user", cascade="all, delete-orphan")


class UserConfig(Base):
    __tablename__ = "platform_configs"

    id:           Mapped[int]   = mapped_column(Integer, primary_key=True)
    user_id:      Mapped[int]   = mapped_column(Integer, ForeignKey("platform_users.id"), unique=True)
    # 交易模式
    enable_day:      Mapped[bool]  = mapped_column(Boolean, default=True)   # 當沖
    enable_short:    Mapped[bool]  = mapped_column(Boolean, default=True)   # 反沖
    enable_overnight:Mapped[bool]  = mapped_column(Boolean, default=False)  # 隔日沖
    auto_trade:      Mapped[bool]  = mapped_column(Boolean, default=False)  # 自動下單
    # 資金設定
    day_budget:      Mapped[float] = mapped_column(Float, default=50000.0)  # 單筆額度
    max_positions:   Mapped[int]   = mapped_column(Integer, default=3)
    stop_loss_pct:   Mapped[float] = mapped_column(Float, default=-0.015)
    take_profit_pct: Mapped[float] = mapped_column(Float, default=0.025)
    updated_at: Mapped[datetime]   = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user: Mapped["User"] = relationship("User", back_populates="config")


class Signal(Base):
    __tablename__ = "platform_signals"

    id:         Mapped[int]      = mapped_column(Integer, primary_key=True)
    code:       Mapped[str]      = mapped_column(String(10))
    action:     Mapped[str]      = mapped_column(String(10))   # BUY / SELL / SHORT / COVER
    score:      Mapped[int]      = mapped_column(Integer, default=0)
    price:      Mapped[float]    = mapped_column(Float)
    reason:     Mapped[str]      = mapped_column(String(200), default="")
    signal_date:Mapped[str]      = mapped_column(String(10))   # YYYY-MM-DD
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Trade(Base):
    __tablename__ = "platform_trades"

    id:         Mapped[int]   = mapped_column(Integer, primary_key=True)
    user_id:    Mapped[int]   = mapped_column(Integer, ForeignKey("platform_users.id"))
    trade_date: Mapped[str]   = mapped_column(String(10))   # YYYY-MM-DD
    code:       Mapped[str]   = mapped_column(String(10))
    action:     Mapped[str]   = mapped_column(String(10))   # BUY / SELL
    shares:     Mapped[int]   = mapped_column(Integer)
    price:      Mapped[float] = mapped_column(Float)
    pnl:        Mapped[float] = mapped_column(Float, default=0.0)
    reason:     Mapped[str]   = mapped_column(String(100), default="")
    order_no:   Mapped[str]   = mapped_column(String(50), default="")
    is_real:    Mapped[bool]  = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship("User", back_populates="trades")
