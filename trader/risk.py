"""
Risk management module.
Enforces position sizing, stop-loss, take-profit, and capital limits.

支援兩種風險設定檔:
  RiskProfile.NORMAL   — 一般上市/上櫃股票
  RiskProfile.EMERGING — 興櫃高風險股票（更嚴格的保護機制）
"""

from dataclasses import dataclass


# ── 一般股票參數 ──────────────────────────────────────────────────────────────
TOTAL_CAPITAL = 20_000      # TWD
MAX_POSITIONS = 5           # Max concurrent stock holdings
STOP_LOSS_PCT = 0.07        # Fallback fixed stop-loss: exit if loss >= 7%
TAKE_PROFIT_PCT = 0.10      # Exit if gain >= 10% (Strategy B: AI confirms first)
MIN_BUY_PROBA = 0.45        # Minimum AI buy probability
MAX_HOLD_DAYS = 10          # Force exit after 10 trading days (~2 weeks)
MAX_PER_STOCK_PCT = 0.40    # Max 40% of capital in one stock
TRANSACTION_FEE = 0.001425  # 0.1425%
SELL_TAX = 0.003            # 0.3%
ATR_MULTIPLIER = 2.0        # ATR trailing stop: stop = peak_price - 2 × ATR


# ── 風險設定檔 ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskProfile:
    """Per-stock risk parameters. Use get_risk_profile(code) to obtain."""
    stop_loss_pct: float       # 固定停損（ATR 不可用時備用）
    take_profit_pct: float     # 停利門檻
    atr_multiplier: float      # ATR 追蹤停損倍率
    max_hold_days: int         # 最大持有天數
    max_per_stock_pct: float   # 單支最大佔資金比例
    min_buy_proba: float       # 最低 AI 買入信心
    require_ta: bool           # 是否強制要求 TradingAgents 確認才能買
    ta_ttl_min: int            # 持倉時 TradingAgents 快取時間（分鐘）
    is_emerging: bool          # 是否為興櫃
    label: str                 # 顯示標籤


# 一般上市/上櫃
NORMAL = RiskProfile(
    stop_loss_pct    = 0.07,
    take_profit_pct  = 0.10,
    atr_multiplier   = 2.0,
    max_hold_days    = 10,
    max_per_stock_pct= 0.40,
    min_buy_proba    = 0.45,
    require_ta       = False,
    ta_ttl_min       = 90,
    is_emerging      = False,
    label            = "一般",
)

# 興櫃高風險
EMERGING = RiskProfile(
    stop_loss_pct    = 0.05,   # 更緊停損 -5%
    take_profit_pct  = 0.08,   # 停利 +8%（不等 AI 確認，直接出場）
    atr_multiplier   = 1.5,    # ATR 追蹤更緊
    max_hold_days    = 5,      # 最多持有 5 個交易日
    max_per_stock_pct= 0.20,   # 最多佔總資金 20%
    min_buy_proba    = 0.60,   # 買入門檻更高
    require_ta       = True,   # 必須有 TradingAgents 確認才能買
    ta_ttl_min       = 30,     # 每 30 分鐘重新詢問 TA
    is_emerging      = True,
    label            = "興櫃",
)


def get_risk_profile(code: str) -> RiskProfile:
    """Return the appropriate RiskProfile for a given stock code."""
    from utils.constants import EMERGING_SYMBOLS
    return EMERGING if code in EMERGING_SYMBOLS else NORMAL


# ── 共用計算函式（接受 RiskProfile 參數）────────────────────────────────────

def position_size(
    available_cash: float,
    price: float,
    n_slots: int = 1,
    total_capital: float = 0.0,
    profile: RiskProfile = NORMAL,
) -> int:
    """
    Calculate number of shares to buy (zero-lot, minimum 1 share).
    Budget = available_cash / n_slots, capped at profile.max_per_stock_pct of total_capital.
    Returns integer number of shares (can be 0 if not affordable).
    """
    cap_base = total_capital if total_capital > 0 else available_cash
    budget = min(
        available_cash / max(n_slots, 1),
        cap_base * profile.max_per_stock_pct,
    )
    cost_per_share = price * (1 + TRANSACTION_FEE)
    if cost_per_share <= 0:
        return 0
    shares = int(budget / cost_per_share)
    return max(shares, 0)


def should_stop_loss(
    entry_price: float,
    current_price: float,
    profile: RiskProfile = NORMAL,
) -> bool:
    """Return True if position has hit fixed stop-loss threshold."""
    if entry_price <= 0:
        return False
    return (current_price - entry_price) / entry_price <= -profile.stop_loss_pct


def should_atr_stop(
    peak_price: float,
    current_price: float,
    atr: float,
    profile: RiskProfile = NORMAL,
) -> bool:
    """
    ATR trailing stop: triggers when price falls profile.atr_multiplier × ATR below the peak.
    Falls back gracefully if ATR is unavailable (atr <= 0).
    """
    if peak_price <= 0 or atr <= 0:
        return False
    stop_level = peak_price - profile.atr_multiplier * atr
    return current_price <= stop_level


def should_take_profit(
    entry_price: float,
    current_price: float,
    profile: RiskProfile = NORMAL,
) -> bool:
    """Return True if position has hit take-profit threshold."""
    if entry_price <= 0:
        return False
    return (current_price - entry_price) / entry_price >= profile.take_profit_pct


def buy_cost(shares: int, price: float) -> float:
    """Total cost to buy including commission."""
    return shares * price * (1 + TRANSACTION_FEE)


def sell_proceeds(shares: int, price: float) -> float:
    """Net cash received after selling (commission + tax)."""
    return shares * price * (1 - TRANSACTION_FEE - SELL_TAX)


def pnl(entry_price: float, current_price: float, shares: int) -> tuple[float, float]:
    """Return (absolute PnL in TWD, PnL as percentage)."""
    cost = buy_cost(shares, entry_price)
    proceeds = sell_proceeds(shares, current_price)
    abs_pnl = proceeds - cost
    pct_pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
    return round(abs_pnl, 2), round(pct_pnl, 4)


def can_open_position(
    holdings: dict,
    available_cash: float,
    price: float,
    profile: RiskProfile = NORMAL,
) -> bool:
    """Return True if we are allowed to open a new position."""
    if len(holdings) >= MAX_POSITIONS:
        return False
    if available_cash < price:
        return False
    # 興櫃：全部持倉中最多 1 支
    if profile.is_emerging:
        from utils.constants import EMERGING_SYMBOLS
        current_emerging = sum(1 for c in holdings if c in EMERGING_SYMBOLS)
        if current_emerging >= 1:
            return False
    return True
