"""
Unit tests for trader/risk.py
Tests RiskProfile, position sizing, stop conditions, emerging constraints.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from trader.risk import (
    NORMAL, EMERGING, get_risk_profile,
    position_size, should_stop_loss, should_atr_stop,
    should_take_profit, can_open_position, buy_cost, sell_proceeds, pnl,
)


# ── RiskProfile 基本屬性 ──────────────────────────────────────────────────────

class TestRiskProfiles:
    def test_normal_params(self):
        assert NORMAL.stop_loss_pct == 0.07
        assert NORMAL.take_profit_pct == 0.10
        assert NORMAL.atr_multiplier == 2.0
        assert NORMAL.max_hold_days == 10
        assert NORMAL.max_per_stock_pct == 0.40
        assert NORMAL.min_buy_proba == 0.45
        assert not NORMAL.require_ta
        assert not NORMAL.is_emerging

    def test_emerging_params(self):
        assert EMERGING.stop_loss_pct == 0.05
        assert EMERGING.take_profit_pct == 0.15
        assert EMERGING.atr_multiplier == 1.5
        assert EMERGING.max_hold_days == 5
        assert EMERGING.max_per_stock_pct == 0.20
        assert EMERGING.min_buy_proba == 0.60
        assert EMERGING.require_ta
        assert EMERGING.is_emerging

    def test_emerging_risk_parameters(self):
        # 停損更緊（更快截損）
        assert EMERGING.stop_loss_pct < NORMAL.stop_loss_pct
        # 停利更高（給題材行情更多空間，R/R = 15/5 = 3x）
        assert EMERGING.take_profit_pct > NORMAL.take_profit_pct
        # ATR 追蹤更緊
        assert EMERGING.atr_multiplier < NORMAL.atr_multiplier
        # 持有天數更短
        assert EMERGING.max_hold_days < NORMAL.max_hold_days
        # 單支上限更低
        assert EMERGING.max_per_stock_pct < NORMAL.max_per_stock_pct
        # 買入門檻更高
        assert EMERGING.min_buy_proba > NORMAL.min_buy_proba

    def test_get_risk_profile_normal(self):
        p = get_risk_profile("2330")  # TSMC not in EMERGING_SYMBOLS
        assert not p.is_emerging
        assert p.stop_loss_pct == NORMAL.stop_loss_pct

    def test_get_risk_profile_emerging(self, monkeypatch):
        """模擬 EMERGING_SYMBOLS 包含 9999"""
        import utils.constants as c
        monkeypatch.setitem(c.EMERGING_SYMBOLS, "9999", "測試興櫃")
        c.EMERGING_TRADEABLE.add("9999")
        p = get_risk_profile("9999")
        assert p.is_emerging
        c.EMERGING_TRADEABLE.discard("9999")
        del c.EMERGING_SYMBOLS["9999"]


# ── 停損條件 ──────────────────────────────────────────────────────────────────

class TestStopLoss:
    def test_normal_no_stop(self):
        assert not should_stop_loss(100, 94, NORMAL)   # -6% < 7% 不觸發

    def test_normal_stop_triggered(self):
        assert should_stop_loss(100, 92, NORMAL)        # -8% >= 7% 觸發

    def test_normal_at_threshold(self):
        assert should_stop_loss(100, 93.0, NORMAL)      # 精確 -7% 觸發

    def test_emerging_stop_less_drop(self):
        # 興櫃 -5% 就停損
        assert should_stop_loss(100, 94.9, EMERGING)   # -5.1% 觸發
        assert not should_stop_loss(100, 95.5, EMERGING)  # -4.5% 不觸發

    def test_entry_price_zero(self):
        assert not should_stop_loss(0, 50, NORMAL)

    def test_price_higher_than_entry(self):
        assert not should_stop_loss(100, 120, NORMAL)


# ── ATR 追蹤停損 ──────────────────────────────────────────────────────────────

class TestATRStop:
    def test_no_atr(self):
        assert not should_atr_stop(100, 90, 0, NORMAL)

    def test_normal_not_triggered(self):
        # peak=110, atr=5, stop_level=110-2×5=100; price=101 → 不觸發
        assert not should_atr_stop(110, 101, 5.0, NORMAL)

    def test_normal_triggered(self):
        # peak=110, atr=5, stop_level=100; price=99.9 → 觸發
        assert should_atr_stop(110, 99.9, 5.0, NORMAL)

    def test_emerging_tighter(self):
        # peak=110, atr=5, EMERGING stop=110-1.5×5=102.5; price=103 → 不觸發
        assert not should_atr_stop(110, 103, 5.0, EMERGING)
        # price=102 → 觸發
        assert should_atr_stop(110, 102, 5.0, EMERGING)

    def test_same_atr_emerging_triggers_earlier(self):
        """相同 ATR，興櫃比一般更早停損"""
        peak, atr = 100.0, 5.0
        for price in [92.0, 93.5, 94.0]:
            normal_stop   = should_atr_stop(peak, price, atr, NORMAL)    # stop=90
            emerging_stop = should_atr_stop(peak, price, atr, EMERGING)  # stop=92.5
            if price <= 90:
                assert normal_stop and emerging_stop
            elif price <= 92.5:
                assert not normal_stop and emerging_stop
            else:
                assert not normal_stop and not emerging_stop


# ── 停利條件 ──────────────────────────────────────────────────────────────────

class TestTakeProfit:
    def test_normal_not_triggered(self):
        assert not should_take_profit(100, 109, NORMAL)  # +9% < 10%

    def test_normal_triggered(self):
        assert should_take_profit(100, 111, NORMAL)      # +11%

    def test_emerging_threshold(self):
        assert should_take_profit(100, 116, EMERGING)     # +16% >= 15% 觸發
        assert not should_take_profit(100, 114, EMERGING) # +14% 不觸發

    def test_at_threshold(self):
        assert should_take_profit(100, 115.0, EMERGING)  # 精確 +15%


# ── 倉位大小 ──────────────────────────────────────────────────────────────────

class TestPositionSize:
    def test_normal_uses_40pct(self):
        cash = 20000
        price = 50
        shares = position_size(cash, price, 1, cash, NORMAL)
        cost = shares * price
        # 不超過 40% of 20000 = 8000
        assert cost <= 8000 * 1.01  # 含手續費容差

    def test_emerging_uses_20pct(self):
        cash = 20000
        price = 50
        shares = position_size(cash, price, 1, cash, EMERGING)
        cost = shares * price
        # 不超過 20% of 20000 = 4000
        assert cost <= 4000 * 1.01

    def test_emerging_half_of_normal(self):
        cash = 20000
        price = 100
        n = position_size(cash, price, 1, cash, NORMAL)
        e = position_size(cash, price, 1, cash, EMERGING)
        assert abs(e - n / 2) <= 1  # 大約一半

    def test_zero_shares_if_too_expensive(self):
        assert position_size(100, 200, 1, 100, NORMAL) == 0


# ── 開倉條件 ─────────────────────────────────────────────────────────────────

class TestCanOpenPosition:
    def test_max_positions_reached(self):
        holdings = {str(i): {} for i in range(5)}  # 5 持倉
        assert not can_open_position(holdings, 10000, 50, NORMAL)

    def test_insufficient_cash(self):
        assert not can_open_position({}, 10, 100, NORMAL)

    def test_normal_ok(self):
        assert can_open_position({}, 10000, 50, NORMAL)

    def test_emerging_max_1(self, monkeypatch):
        import utils.constants as c
        monkeypatch.setitem(c.EMERGING_SYMBOLS, "7777", "測試")
        c.EMERGING_TRADEABLE.add("7777")
        # 已有 1 支興櫃 → 不能再買
        holdings = {"7777": {}}
        assert not can_open_position(holdings, 10000, 50, EMERGING)
        # 無興櫃持倉 → 可以買
        assert can_open_position({}, 10000, 50, EMERGING)
        c.EMERGING_TRADEABLE.discard("7777")
        del c.EMERGING_SYMBOLS["7777"]


# ── 成本/收益計算 ─────────────────────────────────────────────────────────────

class TestCostProceeds:
    def test_buy_cost(self):
        cost = buy_cost(100, 50.0)
        assert abs(cost - 100 * 50.0 * 1.001425) < 0.01

    def test_sell_proceeds(self):
        proceeds = sell_proceeds(100, 50.0)
        assert abs(proceeds - 100 * 50.0 * (1 - 0.001425 - 0.003)) < 0.01

    def test_pnl_positive(self):
        abs_pnl, pct_pnl = pnl(50.0, 60.0, 100)
        assert abs_pnl > 0
        assert pct_pnl == pytest.approx(0.20, abs=0.001)

    def test_pnl_negative(self):
        abs_pnl, pct_pnl = pnl(50.0, 45.0, 100)
        assert abs_pnl < 0
        assert pct_pnl == pytest.approx(-0.10, abs=0.001)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
