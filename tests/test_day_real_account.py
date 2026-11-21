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
    monkeypatch.setattr(d, "_prev_day", lambda code: (49.0, 0.0))   # 前日平盤：不受限空令影響
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
    t._pending_short_orders, t._pending_cover_orders = {}, {}
    t._candles_fn = None
    t._state = t._new_state()
    t._log_trade = lambda *a, **k: None
    t._start_fast_monitor = lambda: None
    t._report_shadow = lambda *a, **k: None
    return t


def _real(quote=None, orders=None):
    t = _bare(True)
    orders = [] if orders is None else orders

    def place(code, side, shares, price, order_type="Stock"):
        orders.append((code, side, shares, price) if order_type == "Stock"
                      else (code, side, shares, price, order_type))
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


def test_real_short_sends_daytrade_sell_at_live_bid():
    t, orders = _real(Q)
    assert t._paper_short("2337", 50.0, {}) is True
    assert orders == [("2337", "SELL", 2000, 50.0, "DayTrade")]       # 現股當沖先賣、富邦買一價
    assert t._state["holdings"] == {}                                   # 成交前不建倉
    assert len(t._pending_short_orders) == 1 and t._state["traded"] == 100_000


def test_real_short_blocked_by_app_switch(cfg):
    cfg["real_enable_short"] = False
    t, orders = _real(Q)
    assert t._paper_short("2337", 50.0, {}) is False
    assert orders == [] and t._state["holdings"] == {}


def _filled_short(fills):
    t, orders = _real(Q)
    t._check_filled_fn = lambda code, no: fills.get(no, (0, 0.0))
    t._paper_short("2337", 50.0, {})
    return t, orders


def test_real_short_fill_then_cover_pnl():
    fills = {"NO1": (2000, 50.0)}
    t, orders = _filled_short(fills)
    t._confirm_pending_orders()
    h = t._state["holdings"]["SHORT_2337"]
    assert h["is_real"] and h["shares"] == 2000 and h["entry_price"] == 50.0
    cash_open = t._state["cash"]
    t._paper_cover("2337", 49.0, "停利")
    assert orders[-1] == ("2337", "BUY", 2000, 50.1)                    # 富邦賣一價買回
    assert "SHORT_2337" not in t._state["holdings"]                     # 等成交
    fills["NO2"] = (2000, 49.0)
    t._confirm_pending_orders()
    assert t._pending_cover_orders == {}
    pnl = t._state["daily_pnl"]
    assert 1500 < pnl < 2000                                              # 1000 元價差扣手續費/證交稅
    assert abs(t._state["cash"] - (cash_open + 100_000 + pnl)) < 1e-6


def test_real_short_reject_refunds():
    t, orders = _filled_short({"NO1": (-1, 0.0)})
    t._confirm_pending_orders()
    assert t._pending_short_orders == {} and t._state["holdings"] == {}
    assert t._state["traded"] == 0 and t._state["cash"] == t._state["initial"]


def test_real_cover_timeout_retries_until_bought_back(monkeypatch):
    fills = {"NO1": (2000, 50.0)}
    t, orders = _filled_short(fills)
    t._confirm_pending_orders()
    t._paper_cover("2337", 49.0, "停損")
    no = next(iter(t._pending_cover_orders))
    t._pending_cover_orders[no]["placed_at"] -= 999
    t._confirm_pending_orders()              # 第一輪：送取消
    t._confirm_pending_orders()              # 第二輪：確認沒成交 → 放回持倉並標記重下
    h = t._state["holdings"]["SHORT_2337"]
    assert h["force_exit"] == "停損" and h["cover_retries"] == 1
    t._check_holdings = d.DayTrader._check_holdings.__get__(t)
    t._bars = lambda code: None
    t._px = lambda code: 49.5
    t._check_holdings()
    assert orders[-1][1] == "BUY" and orders[-1][3] > 50.1                 # 重試加價買回


def test_inventory_sync_does_not_drop_real_shorts():
    t, _ = _real(Q)
    t._state["holdings"]["SHORT_2337"] = {"code": "2337", "direction": "short", "shares": 1000,
                                          "entry_price": 50.0, "is_real": True, "entry_ts": 0}
    class R:  # 富邦現股庫存沒有先賣部位
        is_success, data = True, []
    t._inventories_fn = lambda: R()
    for _ in range(3):
        t._sync_day_holdings_from_fubon()
    assert "SHORT_2337" in t._state["holdings"]


def test_candles_parsing():
    resp = {"symbol": "2330", "data": [
        {"date": "2026-11-20T09:00:00.000+08:00", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 0},
        {"date": "2026-11-20T09:01:00.000+08:00", "open": 10.5, "high": 11, "low": 10, "close": 10.8, "volume": 120},
    ]}
    df = d._candles_df(resp)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"] and len(df) == 1
    assert d._candles_df(None) is None and d._candles_df({"data": []}) is None


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


def test_cancel_failure_blocks_resell_and_refund():
    """取消失敗 → 不標記已取消、不重下賣單（避免重複賣出）。"""
    t, orders = _real(Q)
    t._check_filled_fn = lambda code, no: (0, 0.0)
    t._cancel_order_fn = lambda code, no: False
    t._pending_sell_orders["S1"] = {"side": "SELL", "code": "2337", "shares": 1000, "price": 50.0,
                                    "cost": 49_070, "reason": "停損", "placed_at": 0, "retries": 0,
                                    "entry_price": 49.0}
    for _ in range(3):
        t._confirm_pending_orders()
    assert orders == [] and "S1" in t._pending_sell_orders
    assert not t._pending_sell_orders["S1"].get("cancel_sent")


def test_query_failure_does_not_refund_buy():
    t, orders = _real(Q)
    t._paper_buy("2337", 50.0, {})
    no = next(iter(t._pending_buy_orders))
    t._pending_buy_orders[no]["placed_at"] = 0
    t._pending_buy_orders[no]["cancel_sent"] = True
    t._check_filled_fn = lambda code, n: (-2, 0.0)          # 富邦查詢失敗
    cash = t._state["cash"]
    t._confirm_pending_orders()
    assert no in t._pending_buy_orders and t._state["cash"] == cash


def test_no_long_and_short_on_same_code():
    t, orders = _real(Q)
    assert t._paper_short("2337", 50.0, {}) is True
    assert t._paper_buy("2337", 50.0, {}) is False
    assert len(orders) == 1


def test_real_short_respects_fsc_price_limit(monkeypatch):
    monkeypatch.setattr(d, "_prev_day", lambda code: (51.0, -0.05))   # 前日跌 5%，先賣價 50 < 前收 51
    t, orders = _real(Q)
    assert t._paper_short("2337", 50.0, {}) is False and orders == []
    monkeypatch.setattr(d, "_prev_day", lambda code: (49.0, -0.05))   # 先賣價高於前收 → 允許
    t, orders = _real(Q)
    assert t._paper_short("2337", 50.0, {}) is True
