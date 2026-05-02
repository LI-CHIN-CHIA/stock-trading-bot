"""
Risk management: position sizing, stop-loss, take-profit, budget tracking.
"""

import logging
import math

from data.state import StateStore

logger = logging.getLogger(__name__)

TOTAL_BUDGET = 20_000.0
MAX_POSITIONS = 5
STOP_LOSS_PCT = 0.10    # -10% from entry
TAKE_PROFIT_PCT = 0.15  # +15% from entry
MAX_HOLD_DAYS = 15      # Force sell after N trading days


class RiskManager:
    def __init__(
        self,
        total_budget: float = TOTAL_BUDGET,
        max_positions: int = MAX_POSITIONS,
        stop_loss_pct: float = STOP_LOSS_PCT,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        max_hold_days: int = MAX_HOLD_DAYS,
    ):
        self.total_budget = total_budget
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_days = max_hold_days

    # ── Position Sizing ───────────────────────────────────────────────────────

    def per_position_budget(self) -> float:
        return self.total_budget / self.max_positions  # 4,000 TWD

    def get_available_budget(self, state: StateStore) -> float:
        used = state.get_total_cost()
        return max(0.0, self.total_budget - used)

    def calculate_position_size(self, price: float, available_budget: float) -> int:
        """
        Return number of shares to buy (odd-lot: 1 to 999).
        Targets per-position budget but never exceeds available cash.
        Returns 0 if price > available_budget.
        """
        if price <= 0 or available_budget <= 0:
            return 0

        target = min(self.per_position_budget(), available_budget)
        shares = math.floor(target / price)

        # Ensure we never exceed available budget
        while shares > 0 and shares * price > available_budget:
            shares -= 1

        # Fugle odd-lot: max 999 shares per order
        shares = min(shares, 999)
        return max(shares, 0)

    # ── Entry Gate ────────────────────────────────────────────────────────────

    def can_open_position(self, ticker: str, price: float, state: StateStore) -> tuple[bool, str]:
        positions = state.get_positions()

        if len(positions) >= self.max_positions:
            return False, f"已達最大持倉數 ({self.max_positions})"

        if ticker in positions:
            return False, "已持有此股票"

        available = self.get_available_budget(state)
        if price > available:
            return False, f"資金不足 (需 {price:.0f}, 可用 {available:.0f})"

        shares = self.calculate_position_size(price, available)
        if shares == 0:
            return False, "計算持股數為 0"

        return True, ""

    # ── Exit Conditions ───────────────────────────────────────────────────────

    def calculate_stop_loss(self, entry_price: float) -> float:
        return round(entry_price * (1 - self.stop_loss_pct), 2)

    def calculate_take_profit(self, entry_price: float) -> float:
        return round(entry_price * (1 + self.take_profit_pct), 2)

    def check_exit_conditions(
        self, ticker: str, current_price: float, state: StateStore
    ) -> tuple[str, str]:
        """
        Returns ("STOP_LOSS"|"TAKE_PROFIT"|"TIMEOUT"|"HOLD", reason_str).
        """
        from datetime import datetime, timezone

        positions = state.get_positions()
        if ticker not in positions:
            return "HOLD", ""

        pos = positions[ticker]
        stop = pos.get("stop_loss", 0)
        target = pos.get("take_profit", 999999)

        if current_price <= stop:
            pnl = (current_price - pos["entry_price"]) * pos["shares"]
            return "STOP_LOSS", f"觸發停損 {current_price:.2f} ≤ {stop:.2f} | 損益 {pnl:+.0f} TWD"

        if current_price >= target:
            pnl = (current_price - pos["entry_price"]) * pos["shares"]
            return "TAKE_PROFIT", f"觸發停利 {current_price:.2f} ≥ {target:.2f} | 損益 {pnl:+.0f} TWD"

        # Timeout check
        entry_date_str = pos.get("entry_date", "")
        if entry_date_str:
            try:
                entry_dt = datetime.fromisoformat(entry_date_str)
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                now = datetime.now(tz=timezone.utc)
                hold_days = (now - entry_dt).days
                if hold_days >= self.max_hold_days:
                    return "TIMEOUT", f"持倉超過 {self.max_hold_days} 天，強制平倉"
            except Exception:
                pass

        return "HOLD", ""

    # ── P&L ───────────────────────────────────────────────────────────────────

    def calculate_pnl(
        self, ticker: str, current_price: float, state: StateStore
    ) -> dict:
        positions = state.get_positions()
        if ticker not in positions:
            return {}

        pos = positions[ticker]
        entry = pos["entry_price"]
        shares = pos["shares"]
        cost = pos["cost_basis"]
        current_value = current_price * shares
        unrealized_pnl = current_value - cost
        unrealized_pct = (unrealized_pnl / cost * 100) if cost > 0 else 0.0

        return {
            "ticker": ticker,
            "shares": shares,
            "entry_price": entry,
            "current_price": current_price,
            "cost_basis": cost,
            "current_value": round(current_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "unrealized_pct": round(unrealized_pct, 2),
        }
