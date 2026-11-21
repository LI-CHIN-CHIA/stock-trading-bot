"""順勢當沖（紙上）：選股、成交價、損益、無即時報價不交易。"""
import os, tempfile
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
os.environ["TRADING_USER_ID"] = "0"
from datetime import datetime
import pytest
import trader.trend_trader as tt


def Q(o, last, bid=None, ask=None, value=1e9):
    return {"openPrice": o, "lastPrice": last, "bids": [{"price": bid or last}], "asks": [{"price": ask or last}],
            "total": {"tradeValue": value}}


def test_pick_needs_market_move():
    qs = {c: tt.parse_quote(Q(100, 100.5)) for c in "ABC"}
    assert tt.pick(qs)[1] == 0                                   # +0.5% 未達 1%


def test_pick_down_market_shorts_same_direction_by_value():
    qs = {"A": tt.parse_quote(Q(100, 97, value=5e8)), "B": tt.parse_quote(Q(100, 98, value=9e8)),
          "C": tt.parse_quote(Q(100, 101, value=1e10))}
    mkt, d, chosen = tt.pick(qs, n=5)
    assert d == -1 and [c for c, _ in chosen] == ["B", "A"]       # C 逆向不選；成交值大的優先


@pytest.fixture
def bot(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(tt, "WATCH_STOCKS", {"A", "B"})
    t = tt.TrendTrader(); t._state = t._new_state()
    return t


def _at(monkeypatch, h, m):
    class D(datetime):
        @classmethod
        def now(cls, tz=None): return datetime(2026, 11, 23, h, m, tzinfo=tz)
    monkeypatch.setattr(tt, "datetime", D)


def test_no_quote_fn_no_trade(bot, monkeypatch):
    _at(monkeypatch, 9, 30); bot._state["date"] = "2026-11-23"
    bot.run_cycle()
    assert bot._state["holdings"] == {} and bot._state["scanned"]


def test_short_round_trip_uses_bid_ask_and_fees(bot, monkeypatch):
    book = {"A": Q(100, 97, bid=96.9, ask=97.0), "B": Q(50, 49, bid=48.95, ask=49.0)}
    bot.set_quote_fn(lambda c: book[c])
    _at(monkeypatch, 9, 30); bot._state["date"] = "2026-11-23"
    bot.run_cycle()
    h = bot._state["holdings"]
    assert h["A"]["dir"] == -1 and h["A"]["entry_price"] == 96.9          # 放空用買一價
    assert h["A"]["shares"] == 2000                                         # 20 萬 / 96.9
    book["A"] = Q(100, 95, bid=94.9, ask=95.0); book["B"] = Q(50, 49, bid=48.95, ask=49.0)
    _at(monkeypatch, 13, 20)
    bot.run_cycle()
    assert bot._state["holdings"] == {}
    a = [x for x in bot._state["trade_log"] if x["code"] == "A" and x["side"] == "COVER"][0]
    assert a["price"] == 95.0                                               # 回補用賣一價
    gross = 2000 * (96.9 - 95.0)
    assert 0 < a["pnl"] < gross
