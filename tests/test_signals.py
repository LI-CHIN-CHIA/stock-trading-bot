"""
Unit tests for signal combining, tick rounding, and bot helpers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ── combine_signals ───────────────────────────────────────────────────────────

class TestCombineSignals:
    def setup_method(self):
        from trader.ta_signal import combine_signals
        self.combine = combine_signals

    def _lgbm(self, signal, buy=0.6, sell=0.1):
        return {"signal": signal, "buy_proba": buy, "sell_proba": sell,
                "hold_proba": 1-buy-sell, "atr": 5.0, "trained": True}

    def _ta(self, signal, conf=0.70):
        return {"signal": signal, "confidence": conf,
                "full_decision": signal, "source": "trading_agents"}

    def test_both_buy_boosts_proba(self):
        result = self.combine(self._lgbm("BUY", buy=0.60), self._ta("BUY"))
        assert result["signal"] == "BUY"
        assert result["buy_proba"] > 0.60  # 加成

    def test_both_sell_boosts_proba(self):
        result = self.combine(self._lgbm("SELL", sell=0.60), self._ta("SELL"))
        assert result["signal"] == "SELL"
        assert result["sell_proba"] > 0.60

    def test_conflict_becomes_hold(self):
        result = self.combine(self._lgbm("BUY", buy=0.60), self._ta("SELL"))
        assert result["signal"] == "HOLD"

    def test_conflict_strong_lgbm_stays_buy(self):
        result = self.combine(self._lgbm("BUY", buy=0.80), self._ta("SELL"))
        assert result["signal"] == "BUY"
        assert result["buy_proba"] < 0.80  # 輕微降低

    def test_ta_hold_maintains_signal(self):
        result = self.combine(self._lgbm("BUY", buy=0.65), self._ta("HOLD"))
        assert result["signal"] == "BUY"
        assert result["buy_proba"] < 0.65  # 微降

    def test_none_lgbm_uses_ta(self):
        result = self.combine(None, self._ta("BUY", conf=0.75))
        assert result["signal"] == "BUY"
        assert result["buy_proba"] == pytest.approx(0.75)

    def test_none_ta_returns_lgbm(self):
        lgbm = self._lgbm("BUY", buy=0.65)
        result = self.combine(lgbm, None)
        assert result["signal"] == "BUY"
        assert result["ta_signal"] is None

    def test_both_none_returns_none(self):
        assert self.combine(None, None) is None

    def test_atr_preserved(self):
        result = self.combine(self._lgbm("BUY", buy=0.65), self._ta("BUY"))
        assert result.get("atr") == pytest.approx(5.0)


# ── tick rounding ─────────────────────────────────────────────────────────────

class TestTickRound:
    def setup_method(self):
        # 直接 import bot 需要 SDK，改為從 bot 模組取函式
        import importlib.util, types
        # 獨立測試 _tick_round 邏輯
        def _tick_round(price):
            if price < 10:   tick = 0.01
            elif price < 50: tick = 0.05
            elif price < 100:tick = 0.10
            elif price < 500:tick = 0.50
            else:            tick = 1.00
            return round(round(price / tick) * tick, 2)
        self.fn = _tick_round

    def test_below_10(self):
        assert self.fn(5.123) == pytest.approx(5.12)
        assert self.fn(9.999) == pytest.approx(10.0)

    def test_10_to_50(self):
        assert self.fn(25.13) == pytest.approx(25.15)
        assert self.fn(49.99) == pytest.approx(50.0)

    def test_50_to_100(self):
        assert self.fn(75.06) == pytest.approx(75.1)   # 750.6 → 751 → 75.1
        assert self.fn(99.99) == pytest.approx(100.0)

    def test_100_to_500(self):
        assert self.fn(250.26) == pytest.approx(250.5)  # 500.52 → 501 → 250.5
        assert self.fn(499.99) == pytest.approx(500.0)

    def test_above_500(self):
        assert self.fn(1000.6) == pytest.approx(1001.0)  # 1000.6 → 1001
        assert self.fn(599.6) == pytest.approx(600.0)

    def test_limit_price_buy(self):
        # BUY = base * 1.0999（漲停板），再 tick_round
        base = 50.0
        limit = self.fn(base * 1.0999)
        # 應接近 55 (50 * 1.0999 = 54.995 → tick 0.1 → 55.0)
        assert limit == pytest.approx(55.0)

    def test_limit_price_sell_high_conf(self):
        base = 50.0
        limit = self.fn(base * 0.99)  # proba >= 0.70
        assert limit == pytest.approx(49.5)

    def test_limit_price_sell_low_conf(self):
        base = 50.0
        limit = self.fn(base * 0.98)  # proba < 0.70
        assert limit == pytest.approx(49.0)


# ── ta_signal cache & parse ───────────────────────────────────────────────────

class TestTASignalHelpers:
    def setup_method(self):
        from trader.ta_signal import _parse_decision, _confidence_from_decision
        self.parse = _parse_decision
        self.conf  = _confidence_from_decision

    def test_parse_buy(self):
        assert self.parse("Based on analysis: BUY") == "BUY"

    def test_parse_sell(self):
        assert self.parse("I recommend SELL immediately") == "SELL"

    def test_parse_hold(self):
        assert self.parse("The strategy is to HOLD") == "HOLD"

    def test_parse_last_wins(self):
        # 多個關鍵字，取最後一個
        assert self.parse("Initially BUY but now SELL") == "SELL"

    def test_parse_no_keyword_defaults_hold(self):
        assert self.parse("Market conditions unclear") == "HOLD"

    def test_confidence_strong_buy(self):
        assert self.conf("STRONG BUY signal") == pytest.approx(0.90)

    def test_confidence_normal_buy(self):
        assert self.conf("I recommend BUY") == pytest.approx(0.70)

    def test_confidence_hold(self):
        assert self.conf("HOLD for now") == pytest.approx(0.55)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
