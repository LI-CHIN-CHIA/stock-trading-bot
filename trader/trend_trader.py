"""順勢當沖（紙上）策略 — 與當沖/反沖分開記帳，只用來前瞻驗證。

假設（使用者 2026-11-21）：方向明確的日子當沖好賺。
規則：09:30 用富邦即時報價算股票池（WATCH_STOCKS）的平均「開盤→現價」漲跌，
  |平均| ≥ TREND_MKT_MIN（預設 1%）就順著大盤方向，挑同方向、成交值最大的 TREND_MAX_POS 檔，
  大盤漲 → 做多、大盤跌 → 放空；13:20 平倉。不設停損停利（與回測相同）。
回測（23 天 1 分 K、樣本很少）：09:30 進場 +87bp/筆，但 10:00、10:30 進場為負 → 可能只是雜訊，
  所以只做紙上、用即時報價成交，累積一兩個月再評估。

成交價：做多用賣一價、放空用買一價進場；平倉反之（比 yfinance 延遲報價貼近真實）。
拿不到富邦報價（未登入）就不交易，絕不用延遲資料成交。
成交以帳戶 "trend-<額度>" 回報到 platform，App 上獨立顯示。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path

import pytz

from trader.day_trader import (TRADING_USER_ID, WATCH_STOCKS, _COMMISSION_RATE, _MIN_FEE, _STT_RATE)

logger = logging.getLogger(__name__)
TZ = pytz.timezone("Asia/Taipei")

ENABLE_TREND    = os.getenv("ENABLE_TREND_TRADING", "true").lower() == "true"
TREND_CAPITAL   = float(os.getenv("TREND_CAPITAL", "1000000"))
TREND_MAX_POS   = int(os.getenv("TREND_MAX_POSITIONS", "5"))
TREND_MKT_MIN   = float(os.getenv("TREND_MKT_MIN", "0.01"))
_ENTRY_AT, _ENTRY_UNTIL, _CLOSE_AT = (9, 30), (9, 36), (13, 20)

ACCOUNT = f"trend-{int(TREND_CAPITAL)}"


def _state_path() -> Path:
    return Path(os.getenv("DATA_DIR", Path(__file__).parent.parent)) / "trend_trader_state.json"


def parse_quote(q) -> dict | None:
    """富邦/Fugle intraday quote → {open, last, bid, ask, value}；沒有開盤價或現價回傳 None。"""
    if q is None:
        return None
    g = (lambda o, k: o.get(k) if isinstance(o, dict) else getattr(o, k, None))
    f = (lambda v: float(v) if v not in (None, "", 0) else None)
    bids, asks = g(q, "bids") or [], g(q, "asks") or []
    total = g(q, "total") or {}
    out = {"open": f(g(q, "openPrice")), "last": f(g(q, "lastPrice") or g(q, "closePrice")),
           "bid": f(g(bids[0], "price")) if bids else None,
           "ask": f(g(asks[0], "price")) if asks else None,
           "value": f(g(total, "tradeValue")) or 0.0}
    return out if out["open"] and out["last"] else None


def pick(quotes: dict[str, dict], mkt_min: float = TREND_MKT_MIN, n: int = TREND_MAX_POS):
    """回傳 (大盤平均漲跌, 方向 +1/-1/0, 選出的 [(code, quote)])。純函式，方便測試。"""
    moves = {c: q["last"] / q["open"] - 1 for c, q in quotes.items()}
    if not moves:
        return 0.0, 0, []
    mkt = sum(moves.values()) / len(moves)
    if abs(mkt) < mkt_min:
        return mkt, 0, []
    d = 1 if mkt > 0 else -1
    same = [(c, quotes[c]) for c, m in moves.items() if m * d > 0]
    same.sort(key=lambda x: x[1]["value"], reverse=True)
    return mkt, d, same[:n]


class TrendTrader:
    def __init__(self):
        self._lock = threading.RLock()
        self._quote_fn = None            # 富邦即時報價（bot.live_quote）；None = 不交易
        self._state = self._load()

    def set_quote_fn(self, fn):
        self._quote_fn = fn

    # ── state ────────────────────────────────────────────────────────────────
    def _new_state(self) -> dict:
        return {"date": str(datetime.now(TZ).date()), "holdings": {}, "daily_pnl": 0.0,
                "scanned": False, "market_move": None, "trade_log": []}

    def _load(self) -> dict:
        try:
            return json.loads(_state_path().read_text(encoding="utf-8"))
        except Exception:
            return self._new_state()

    def _save(self):
        try:
            _state_path().write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[Trend] state 儲存失敗: {e}")

    def _report(self, code, side, shares, price, reason, pnl=0.0, direction="LONG", open_shares=0):
        self._state["trade_log"].append({
            "time": datetime.now(TZ).strftime("%H:%M"), "code": code, "side": side,
            "shares": shares, "price": price, "reason": reason, "pnl": round(pnl, 0)})
        try:
            from utils.platform_signal import report_position, report_trade
            report_trade(user_id=TRADING_USER_ID, code=code, action=side, shares=shares,
                         price=price, pnl=pnl, reason=reason, is_real=False, account=ACCOUNT)
            report_position(TRADING_USER_ID, code, direction, open_shares, price,
                            is_real=False, account=ACCOUNT)
        except Exception:
            pass

    def _quotes(self, codes) -> dict[str, dict]:
        out = {}
        for c in codes:
            try:
                q = parse_quote(self._quote_fn(c))
            except Exception:
                q = None
            if q:
                out[c] = q
        return out

    # ── 進出場 ───────────────────────────────────────────────────────────────
    def _enter(self, code: str, q: dict, d: int, mkt: float) -> bool:
        h = self._state["holdings"]
        if code in h or len(h) >= TREND_MAX_POS:
            return False
        price = (q["ask"] if d > 0 else q["bid"]) or q["last"]
        shares = int((TREND_CAPITAL / TREND_MAX_POS) // (price * 1000)) * 1000
        if shares <= 0:
            return False
        value = shares * price
        fee = max(_MIN_FEE, round(value * _COMMISSION_RATE))
        h[code] = {"dir": d, "shares": shares, "entry_price": price, "fee": fee,
                   "entry_time": datetime.now(TZ).strftime("%H:%M")}
        early = q["last"] / q["open"] - 1
        reason = f"順勢{'做多' if d > 0 else '放空'} 大盤{mkt*100:+.1f}% 個股{early*100:+.1f}%"
        self._report(code, "BUY" if d > 0 else "SHORT", shares, price, reason,
                     direction="LONG" if d > 0 else "SHORT", open_shares=shares)
        logger.info(f"[Trend] {'🟢' if d > 0 else '🔴'} {code} {shares}股 @{price:.2f}  {reason}")
        try:
            from utils.telegram_notify import notify_buy
            notify_buy("順勢當沖(紙)", code, shares, price, reason)
        except Exception:
            pass
        return True

    def _exit(self, code: str, q: dict | None, reason: str):
        h = self._state["holdings"].get(code)
        if h is None:
            return
        d = h["dir"]
        price = None
        if q:
            price = (q["bid"] if d > 0 else q["ask"]) or q["last"]
        if price is None:
            return                                  # 沒報價就下一輪再試
        locked = q is not None and (q["bid"] is None if d > 0 else q["ask"] is None)
        self._state["holdings"].pop(code)
        value = h["shares"] * price
        exit_fee = max(_MIN_FEE, round(value * _COMMISSION_RATE))
        entry_value = h["shares"] * h["entry_price"]
        stt = round((value if d > 0 else entry_value) * _STT_RATE)   # 證交稅在賣出那一邊
        pnl = d * (value - entry_value) - h["fee"] - exit_fee - stt
        self._state["daily_pnl"] += pnl
        why = reason + ("【漲跌停鎖住，實盤可能無法平倉】" if locked else "")
        self._report(code, "SELL" if d > 0 else "COVER", h["shares"], price, why, pnl,
                     direction="LONG" if d > 0 else "SHORT", open_shares=0)
        logger.info(f"[Trend] {'🟩' if pnl >= 0 else '🟥'} 平倉 {code} @{price:.2f} 損益={pnl:+,.0f} {why}")
        try:
            from utils.telegram_notify import notify_sell
            notify_sell("順勢當沖(紙)", code, h["shares"], price, pnl, why)
        except Exception:
            pass

    def run_cycle(self):
        if not ENABLE_TREND:
            return
        now = datetime.now(TZ)
        hm = (now.hour, now.minute)
        with self._lock:
            if self._state.get("date") != str(now.date()):
                self._state = self._new_state()
        if self._quote_fn is None:
            if _ENTRY_AT <= hm < _ENTRY_UNTIL and not self._state["scanned"]:
                logger.info("[Trend] 富邦即時報價未接上，今日不交易（不使用延遲報價）")
                self._state["scanned"] = True
            return

        if not self._state["scanned"] and _ENTRY_AT <= hm < _ENTRY_UNTIL:
            quotes = self._quotes(sorted(WATCH_STOCKS))
            mkt, d, chosen = pick(quotes)
            with self._lock:
                self._state["market_move"] = round(mkt, 5)
                if d == 0:
                    logger.info(f"[Trend] 09:30 股票池平均 {mkt*100:+.2f}%（{len(quotes)} 檔），"
                                f"未達 ±{TREND_MKT_MIN:.0%}，今日不進場")
                for code, q in chosen:
                    self._enter(code, q, d, mkt)
                self._state["scanned"] = True
                self._save()

        if self._state["holdings"] and hm >= _CLOSE_AT:
            quotes = self._quotes(list(self._state["holdings"]))
            with self._lock:
                for code in list(self._state["holdings"]):
                    self._exit(code, quotes.get(code), "收盤前平倉 13:20")
                self._save()

    def get_summary(self) -> dict:
        s = self._state
        return {"account": ACCOUNT, "date": s.get("date"), "daily_pnl": s.get("daily_pnl", 0.0),
                "positions": len(s.get("holdings", {})), "market_move": s.get("market_move")}
