"""真實帳戶與模擬帳戶分離（trader/day_trader.py）。

不連富邦、不連 DB：下單 / 報價 / 設定全部用假的。
"""
import os
import tempfile
import threading

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
os.environ["TRADING_USER_ID"] = "0"   # 不查 DB，直接用 _user_config_cache

import pytest

import trader.day_trader as d


@pytest.fixture(autouse=True)
def cfg(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(d, "DAY_REAL_TRADING", True)
    c = d._user_config_cache
    saved = dict(c)
    c.update(auto_trade=True, real_enable_day=True, real_enable_short=True,
             day_budget=100_000.0, max_positions=3, enable_day=True, enable_short=True,
             stop_loss_pct=-0.015, take_profit_pct=0.025, paper_budgets=[500_000])
    yield c
    c.clear(); c.update(saved)


def _bare(real: bool) -> d.DayTrader:
    t = d.DayTrader.__new__(d.DayTrader)
    t.real = real
    t._quote_fn = None
    t._lock = threading.RLock()
    t._place_order_fn = t._check_filled_fn = t._cancel_order_fn = None
    t._inventories_fn = t._unrealized_fn = t._daytrade_info_fn = None
    t._daytrade_cache = {}
    t._pending_buy_orders, t._pending_sell_orders = {}, {}
    t._state = t._new_state()
    t._log_trade = lambda *a, **k: None
    t._start_fast_monitor = lambda: None
    t._report_shadow = lambda *a, **k: None
    return t


def _real(quote=None, orders=None):
    t = _bare(True)
    orders = [] if orders is None else orders

    def place(code, side, shares, price):
        orders.append((code, side, shares, price))
        return f"NO{len(orders)}"
    t._place_order_fn = place
    t._quote_fn = lambda code: quote
    return t, orders


Q = {"lastPrice": 50.1, "bids": [{"price": 50.0}], "asks": [{"price": 50.1}]}


def test_paper_account_cannot_get_real_orders():
    t = _bare(False)
    with pytest.raises(RuntimeError):
        t.set_real_trading(place_order_fn=lambda *a: "X", check_filled_fn=lambda *a: (0, 0))


def test_paper_buy_never_sends_orders():
    t = _bare(False)
    assert t._paper_buy("2337", 50.0, {}) is True
    assert t._state["holdings"]["2337"]["is_real"] is False
    assert t._pending_buy_orders == {}


def test_real_buy_uses_live_ask_and_app_budget():
    t, orders = _real(Q)
    assert t._paper_buy("2337", 50.0, {}) is True
    code, side, shares, price = orders[0]
    assert (code, side, price) == ("2337", "BUY", 50.1)          # 富邦即時賣一價
    assert shares == 1000                                          # 10 萬 / 50.1 → 1 張
    assert t._state["holdings"] == {}                              # 成交前不建倉
    assert len(t._pending_buy_orders) == 1


@pytest.mark.parametrize("key", ["auto_trade", "real_enable_day"])
def test_real_buy_blocked_by_app_switch(cfg, key):
    cfg[key] = False
    t, orders = _real(Q)
    assert t._paper_buy("2337", 50.0, {}) is False
    assert orders == [] and t._state["holdings"] == {}             # 也不轉紙上


def test_real_buy_blocked_when_server_switch_off(monkeypatch):
    monkeypatch.setattr(d, "DAY_REAL_TRADING", False)
    t, orders = _real(Q)
    assert t._paper_buy("2337", 50.0, {}) is False and orders == []


def test_real_buy_skips_without_live_quote():
    t, orders = _real(None)
    assert t._paper_buy("2337", 50.0, {}) is False and orders == []


def test_real_buy_skips_stale_signal():
    t, orders = _real({"lastPrice": 52.0, "asks": [{"price": 52.0}]})   # +4% vs 訊號價
    assert t._paper_buy("2337", 50.0, {}) is False and orders == []


def test_real_daily_total_is_budget_times_positions(cfg):
    cfg["day_budget"], cfg["max_positions"] = 60_000.0, 2
    t, orders = _real(Q)
    assert t._state["initial"] == 120_000
    for i in range(5):
        t._paper_buy(f"C{i}", 50.0, {})
        t._pending_buy_orders.clear()      # 模擬已成交並平倉，額度不回補
    assert len(orders) == 2                # 第 3 筆起當日額度用完


def test_real_short_is_not_sent_and_not_papered():
    t, orders = _real(Q)
    assert t._paper_short("2337", 50.0, {}) is False
    assert orders == [] and t._state["holdings"] == {}


def test_real_sell_uses_live_bid():
    t, orders = _real(Q)
    t._state["holdings"]["2337"] = {"shares": 1000, "entry_price": 49.0, "cost": 49_070,
                                    "is_real": True, "stop_loss": 48.0, "take_profit": 51.0}
    t._paper_sell("2337", 51.0, "停利")
    assert orders == [("2337", "SELL", 1000, 50.0)]


def test_quote_parsing():
    assert d._quote_prices(Q) == {"last": 50.1, "bid": 50.0, "ask": 50.1}
    assert d._quote_prices({"closePrice": 30}) == {"last": 30.0, "bid": None, "ask": None}
    assert d._quote_prices(None)["last"] is None


def test_accounts_and_state_files_are_separate(tmp_path):
    assert d._state_path(True) != d._state_path(False)
    assert _bare(True)._account() == "real"
    assert _bare(False)._account().startswith("paper-")
    assert _bare(True)._state["shadows"] == {}
