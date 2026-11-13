"""過熱股放空（紙上）策略 — 與當沖/反沖分開記帳。

規則（backtest/daily_edge.py，6 年日線驗證）：
  近 5 日漲幅 ≥ FADE_RET5_MIN 且今日開盤跳空介於 FADE_GAP_MIN~FADE_GAP_MAX 的股票，
  開盤後放空，13:20 回補。回測中加停損會讓結果變差，故預設不設停損。

只做紙上交易；成交以帳戶 "fade-<額度>" 回報到 platform，App 上獨立顯示。
已知風險：約 16% 的訊號當日收在漲停附近，真實空單可能無法回補。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import date, datetime
from pathlib import Path

import pytz
import yfinance as yf

from trader.day_trader import (
    TRADING_USER_ID, WATCH_STOCKS, _COMMISSION_RATE, _MIN_FEE, _STT_RATE,
    _fetch_intraday, _fetch_user_config, _symbol,
)

logger = logging.getLogger(__name__)
TZ = pytz.timezone("Asia/Taipei")

ENABLE_FADE     = os.getenv("ENABLE_FADE_TRADING", "true").lower() == "true"
FADE_CAPITAL    = float(os.getenv("FADE_CAPITAL", "1000000"))
FADE_PER_TRADE  = float(os.getenv("FADE_PER_TRADE_PCT", "0.33"))
FADE_MAX_POS    = int(os.getenv("FADE_MAX_POSITIONS", "3"))
FADE_RET5_MIN   = float(os.getenv("FADE_RET5_MIN", "0.10"))
FADE_GAP_MIN    = float(os.getenv("FADE_GAP_MIN", "0.015"))
FADE_GAP_MAX    = float(os.getenv("FADE_GAP_MAX", "0.09"))    # 開盤即漲停的不做（回測 74% 鎖漲停，空單回補不了）
FADE_STOP_PCT   = float(os.getenv("FADE_STOP_PCT", "0"))      # 0 = 不設停損
_SCAN_FROM, _SCAN_UNTIL, _CLOSE_AT = (9, 5), (9, 30), (13, 20)

ACCOUNT = f"fade-{int(FADE_CAPITAL)}"


def _state_path() -> Path:
    return Path(os.getenv("DATA_DIR", Path(__file__).parent.parent)) / "fade_trader_state.json"


def find_candidates(codes=None) -> list[dict]:
    """回傳今日符合條件的股票，依跳空幅度由大到小。"""
    today = datetime.now(TZ).date()
    out = []
    for code in sorted(codes or WATCH_STOCKS):
        try:
            d = yf.Ticker(_symbol(code)).history(period="15d", interval="1d")
            d = d[[x.date() < today for x in d.index]]["Close"].dropna()
            if len(d) < 6:
                continue
            prev_close = float(d.iloc[-1])
            ret5 = prev_close / float(d.iloc[-6]) - 1
            if ret5 < FADE_RET5_MIN:
                continue
            bars = _fetch_intraday(code)
            if bars is None or bars.empty:
                continue
            bars = bars[[x.date() == today for x in bars.index.tz_convert(TZ)]]
            if bars.empty:
                continue
            day_open = float(bars["Open"].iloc[0])
            gap = day_open / prev_close - 1
            if not (FADE_GAP_MIN <= gap < FADE_GAP_MAX):
                continue
            out.append({"code": code, "price": float(bars["Close"].iloc[-1]),
                        "ret5": ret5, "gap": gap})
        except Exception as e:
            logger.debug(f"[Fade] {code} 資料失敗: {e}")
        time.sleep(0.05)
    return sorted(out, key=lambda x: -x["gap"])


class FadeTrader:
    def __init__(self):
        self._lock = threading.RLock()
        self._state = self._load()

    def _new_state(self) -> dict:
        return {"date": str(date.today()), "cash": FADE_CAPITAL, "holdings": {},
                "daily_pnl": 0.0, "scanned": False, "trade_log": []}

    def _load(self) -> dict:
        try:
            return json.loads(_state_path().read_text(encoding="utf-8"))
        except Exception:
            return self._new_state()

    def _save(self):
        try:
            _state_path().write_text(json.dumps(self._state, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        except Exception as e:
            logger.warning(f"[Fade] state 儲存失敗: {e}")

    def _report(self, code, side, shares, price, reason, pnl=0.0):
        self._state["trade_log"].append({
            "time": datetime.now(TZ).strftime("%H:%M"), "code": code, "side": side,
            "shares": shares, "price": price, "reason": reason, "pnl": round(pnl, 0)})
        try:
            from utils.platform_signal import report_position, report_trade
            report_trade(user_id=TRADING_USER_ID, code=code, action=side, shares=shares,
                         price=price, pnl=pnl, reason=reason, is_real=False, account=ACCOUNT)
            report_position(TRADING_USER_ID, code, "SHORT", shares if side == "SHORT" else 0,
                            price, is_real=False, account=ACCOUNT)
        except Exception:
            pass

    def _short(self, c: dict) -> bool:
        code, price = c["code"], c["price"]
        h = self._state["holdings"]
        if code in h or len(h) >= FADE_MAX_POS:
            return False
        shares = int((FADE_CAPITAL * FADE_PER_TRADE) // (price * 1000)) * 1000
        if shares <= 0:
            logger.info(f"[Fade] {code} @{price:.2f} 超過單筆預算，略過")
            return False
        value = shares * price
        proceeds = value - max(_MIN_FEE, round(value * _COMMISSION_RATE)) - round(value * _STT_RATE)
        h[code] = {"shares": shares, "entry_price": price, "proceeds": proceeds,
                   "entry_time": datetime.now(TZ).strftime("%H:%M"),
                   "stop": round(price * (1 + FADE_STOP_PCT), 2) if FADE_STOP_PCT > 0 else None}
        reason = f"過熱放空 5日{c['ret5']*100:+.1f}% 跳空{c['gap']*100:+.1f}%"
        self._report(code, "SHORT", shares, price, reason)
        logger.info(f"[Fade] 🔴 放空 {code} {shares}股 @{price:.2f}  {reason}")
        try:
            from utils.telegram_notify import notify_buy
            notify_buy("過熱放空(紙)", code, shares, price, reason)
        except Exception:
            pass
        return True

    def _cover(self, code: str, price: float, reason: str):
        h = self._state["holdings"].pop(code, None)
        if h is None:
            return
        cost = h["shares"] * price
        pnl = h["proceeds"] - cost - max(_MIN_FEE, round(cost * _COMMISSION_RATE))
        self._state["daily_pnl"] += pnl
        self._report(code, "COVER", h["shares"], price, reason, pnl)
        logger.info(f"[Fade] {'🟩' if pnl >= 0 else '🟥'} 回補 {code} @{price:.2f}"
                    f"  損益={pnl:+,.0f}  原因={reason}")
        try:
            from utils.telegram_notify import notify_sell
            notify_sell("過熱放空(紙)", code, h["shares"], price, pnl, reason)
        except Exception:
            pass

    def run_cycle(self):
        if not ENABLE_FADE:
            return
        now = datetime.now(TZ)
        hm = (now.hour, now.minute)
        with self._lock:
            if self._state.get("date") != str(date.today()):
                self._state = self._new_state()

        # 進場：09:05~09:30 掃描一次（資料沒抓到會在視窗內重試）
        if (not self._state["scanned"] and _SCAN_FROM <= hm < _SCAN_UNTIL
                and _fetch_user_config().get("enable_short", True)):
            cands = find_candidates()
            with self._lock:
                for c in cands:
                    self._short(c)
                self._state["scanned"] = True
                logger.info(f"[Fade] 今日候選 {len(cands)} 檔: "
                            + ", ".join(f"{c['code']}(gap{c['gap']*100:+.1f}%)" for c in cands))
                self._save()

        # 出場
        for code in list(self._state["holdings"]):
            df = _fetch_intraday(code)
            if df is None or df.empty:
                continue
            price = float(df["Close"].iloc[-1])
            h = self._state["holdings"][code]
            with self._lock:
                if hm >= _CLOSE_AT:
                    self._cover(code, price, "收盤前回補 13:20")
                elif h.get("stop") and price >= h["stop"]:
                    self._cover(code, price, f"停損 {h['stop']:.2f}")
                self._save()

    def get_summary(self) -> dict:
        s = self._state
        return {"account": ACCOUNT, "date": s.get("date"), "daily_pnl": s.get("daily_pnl", 0.0),
                "positions": len(s.get("holdings", {})), "trades_today": len(s.get("trade_log", []))}
