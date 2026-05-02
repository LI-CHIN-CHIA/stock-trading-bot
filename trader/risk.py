"""
Risk management module.
Enforces position sizing, stop-loss, take-profit, and capital limits.
"""

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


def position_size(available_cash: float, price: float, n_slots: int = 1, total_capital: float = 0.0) -> int:
    """
    Calculate number of shares to buy (zero-lot, minimum 1 share).
    Budget = available_cash / n_slots, capped at MAX_PER_STOCK_PCT of total_capital.
    If total_capital is not provided, uses available_cash as the reference.
    Returns integer number of shares (can be 0 if not affordable).
    """
    cap_base = total_capital if total_capital > 0 else available_cash
    budget = min(
        available_cash / max(n_slots, 1),
        cap_base * MAX_PER_STOCK_PCT,
    )
    cost_per_share = price * (1 + TRANSACTION_FEE)
    if cost_per_share <= 0:
        return 0
    shares = int(budget / cost_per_share)
    return max(shares, 0)


def should_stop_loss(entry_price: float, current_price: float) -> bool:
    """Return True if position has hit fixed stop-loss threshold (fallback when ATR unavailable)."""
    if entry_price <= 0:
        return False
    return (current_price - entry_price) / entry_price <= -STOP_LOSS_PCT


def should_atr_stop(peak_price: float, current_price: float, atr: float) -> bool:
    """
    ATR trailing stop: triggers when price falls ATR_MULTIPLIER × ATR below the peak.
    The stop level rises as the stock goes up (trailing), but never moves down.
    Falls back to fixed stop-loss check if ATR is unavailable (atr <= 0).
    """
    if peak_price <= 0 or atr <= 0:
        return False
    stop_level = peak_price - ATR_MULTIPLIER * atr
    return current_price <= stop_level


def should_take_profit(entry_price: float, current_price: float) -> bool:
    """Return True if position has hit take-profit threshold."""
    if entry_price <= 0:
        return False
    return (current_price - entry_price) / entry_price >= TAKE_PROFIT_PCT


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
) -> bool:
    """Return True if we are allowed to open a new position."""
    if len(holdings) >= MAX_POSITIONS:
        return False
    if available_cash < price:
        return False
    return True
