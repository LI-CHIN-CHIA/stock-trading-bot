"""
加密貨幣風險管理模組
====================
定義兩種策略的風險設定檔：
  STABLE  — 穩定幣種（BTC/ETH/SOL），持倉較長
  NEWCOIN — 新幣策略，捕捉初爆後快速出場

與 trader/risk.py 完全獨立，不共用任何類別。
"""

from dataclasses import dataclass

TAKER_FEE = 0.0015   # MAX taker 0.15%
MAKER_FEE = 0.0005   # MAX maker 0.05%


@dataclass(frozen=True)
class CryptoRiskProfile:
    stop_loss_pct: float        # 固定停損（ATR 停損為主，此為備用）
    take_profit_pct: float      # 停利門檻
    atr_multiplier: float       # ATR 追蹤停損倍率
    max_hold_hours: int         # 最大持有小時數（超過強制出場）
    max_per_coin_pct: float     # 單幣最大佔資金比例
    min_signal_strength: float  # 最低訊號強度門檻（-1.0 ~ 1.0）
    candle_interval: str        # K棒時間粒度（"4h" / "1m"）
    scan_interval_sec: int      # 主循環掃描間隔（秒）
    label: str


# 穩定幣種（BTC / ETH / SOL 等主流幣）
STABLE = CryptoRiskProfile(
    stop_loss_pct      = 0.15,    # -15%
    take_profit_pct    = 0.30,    # +30%
    atr_multiplier     = 3.0,
    max_hold_hours     = 336,     # 14 天
    max_per_coin_pct   = 0.33,    # 最多佔 33% 資金
    min_signal_strength= 0.3,
    candle_interval    = "4h",
    scan_interval_sec  = 900,     # 每 15 分鐘掃描
    label              = "穩定",
)

# 新幣策略（捕捉初爆）
NEWCOIN = CryptoRiskProfile(
    stop_loss_pct      = 0.15,    # -15%
    take_profit_pct    = 0.50,    # +50%
    atr_multiplier     = 3.0,
    max_hold_hours     = 24,      # 最多持有 1 天
    max_per_coin_pct   = 0.25,    # 最多佔 25% 資金
    min_signal_strength= 0.5,     # 門檻更高（新幣風險大）
    candle_interval    = "1m",
    scan_interval_sec  = 60,      # 每分鐘掃描
    label              = "新幣",
)


# 多頭回檔買進（BTC / ETH，日線）；出場靠訊號（站回 5 日線）與最長持有 10 天，不設停損停利
DIP = CryptoRiskProfile(
    stop_loss_pct      = 9.99,    # 不啟用（回測中加停損會變差）
    take_profit_pct    = 9.99,    # 不啟用
    atr_multiplier     = 0.0,     # 不啟用（持倉 atr 記為 0）
    max_hold_hours     = 240,     # 10 天
    max_per_coin_pct   = 0.5,
    min_signal_strength= 0.5,
    candle_interval    = "1d",
    scan_interval_sec  = 900,
    label              = "回檔買進",
)


def get_profile(mode: str, strategy: str = "") -> CryptoRiskProfile:
    """根據 CRYPTO_MODE / CRYPTO_STRATEGY 回傳對應 profile。"""
    if mode.lower() == "newcoin":
        return NEWCOIN
    return DIP if strategy.lower() == "dip" else STABLE


# ── 交易成本計算 ──────────────────────────────────────────────────────────────

def buy_cost_usdt(amount_usdt: float) -> float:
    """買入實際扣款（含 taker 手續費）。"""
    return amount_usdt * (1 + TAKER_FEE)


def sell_proceeds_usdt(amount_coin: float, price: float) -> float:
    """賣出實際入帳（含 taker 手續費）。"""
    return amount_coin * price * (1 - TAKER_FEE)


def position_size_usdt(
    available_usdt: float,
    total_capital: float,
    profile: CryptoRiskProfile,
    n_slots: int = 1,
) -> float:
    """計算單筆買入金額（USDT）。"""
    cap_base = total_capital if total_capital > 0 else available_usdt
    budget = min(
        available_usdt / max(n_slots, 1),
        cap_base * profile.max_per_coin_pct,
    )
    return max(budget, 0.0)


def should_stop_loss(
    entry_price: float,
    current_price: float,
    profile: CryptoRiskProfile,
) -> bool:
    if entry_price <= 0:
        return False
    return (current_price - entry_price) / entry_price <= -profile.stop_loss_pct


def should_atr_stop(
    peak_price: float,
    current_price: float,
    atr: float,
    profile: CryptoRiskProfile,
) -> bool:
    if peak_price <= 0 or atr <= 0:
        return False
    stop_level = peak_price - profile.atr_multiplier * atr
    return current_price <= stop_level


def should_take_profit(
    entry_price: float,
    current_price: float,
    profile: CryptoRiskProfile,
) -> bool:
    if entry_price <= 0:
        return False
    return (current_price - entry_price) / entry_price >= profile.take_profit_pct


def should_time_exit(entry_time_iso: str, profile: CryptoRiskProfile) -> bool:
    """超過 max_hold_hours 強制出場。"""
    from datetime import datetime, timezone
    try:
        entry = datetime.fromisoformat(entry_time_iso)
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=timezone.utc)
        elapsed_hours = (datetime.now(timezone.utc) - entry).total_seconds() / 3600
        return elapsed_hours >= profile.max_hold_hours
    except Exception:
        return False


def pnl(entry_price: float, current_price: float, amount_coin: float) -> tuple[float, float]:
    """回傳 (絕對損益 USDT, 損益百分比)。"""
    cost     = buy_cost_usdt(amount_coin * entry_price)
    proceeds = sell_proceeds_usdt(amount_coin, current_price)
    abs_pnl  = proceeds - cost
    pct_pnl  = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
    return round(abs_pnl, 4), round(pct_pnl, 4)
