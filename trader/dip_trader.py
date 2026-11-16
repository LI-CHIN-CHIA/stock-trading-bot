"""超跌股開盤做多（紙上）策略 — 與當沖/反沖分開記帳。

規則（backtest/oversold_research.py、winrate_research.py；400 檔、6 年日線）：
  近 5 日跌幅 ≥ DIP_RET5_MIN 且今日開盤向下跳空介於 DIP_GAP_MIN~DIP_GAP_MAX、股價 ≤ DIP_PRICE_MAX，
  開盤後買進；漲到進場價 +DIP_TP_PCT 停利，否則 13:20 賣出。不設停損。
  回測：停利 3% 時勝率 77%、每筆約 +0.9%（已扣成本），但獲利集中在全市場大跌日，
  且樣本有存活者偏誤 → 只做紙上交易，累積實盤樣本後再評估。

成交以帳戶 "dip-<額度>" 回報到 platform，App 上獨立顯示。
已知風險：約 17% 的訊號當日收在跌停，真實多單可能賣不掉（紙上會標記）。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path

import pytz
import yfinance as yf

from trader.day_trader import TRADING_USER_ID, _COMMISSION_RATE, _MIN_FEE, _STT_RATE

logger = logging.getLogger(__name__)
TZ = pytz.timezone("Asia/Taipei")

ENABLE_DIP     = os.getenv("ENABLE_DIP_TRADING", "true").lower() == "true"
DIP_CAPITAL    = float(os.getenv("DIP_CAPITAL", "1000000"))
DIP_PER_TRADE  = float(os.getenv("DIP_PER_TRADE_PCT", "0.33"))
DIP_MAX_POS    = int(os.getenv("DIP_MAX_POSITIONS", "3"))
DIP_RET5_MIN   = float(os.getenv("DIP_RET5_MIN", "0.08"))     # 5 日跌幅門檻（正數）
DIP_GAP_MIN    = float(os.getenv("DIP_GAP_MIN", "0.03"))      # 向下跳空下限（正數）
DIP_GAP_MAX    = float(os.getenv("DIP_GAP_MAX", "0.09"))      # 開盤即跌停的不做
DIP_PRICE_MAX  = float(os.getenv("DIP_PRICE_MAX", "330"))
DIP_TP_PCT     = float(os.getenv("DIP_TP_PCT", "0.03"))
_SCAN_FROM, _SCAN_UNTIL, _CLOSE_AT = (9, 5), (9, 30), (13, 20)
_SCAN_GIVE_UP = (9, 28)   # 到這時間報價還沒到齊，就用已取得的決定

ACCOUNT = f"dip-{int(DIP_CAPITAL)}"


def _universe() -> dict[str, str]:
    """{代號: 'TW'|'TWO'}，成交值前 400 大（2026-11-14 產生）。"""
    try:
        return json.loads((Path(__file__).parent / "dip_universe.json").read_text())
    except Exception as e:
        logger.warning(f"[Dip] 股票池讀取失敗: {e}")
        return {}


def _state_path() -> Path:
    return Path(os.getenv("DATA_DIR", Path(__file__).parent.parent)) / "dip_trader_state.json"


def _last_price(sym: str) -> tuple[float, float] | None:
    """回傳 (今日開盤價, 最新價)；今日尚無成交則 None。"""
    today = datetime.now(TZ).date()
    df = yf.Ticker(sym).history(period="1d", interval="1m")
    if df is None or df.empty:
        return None
    df = df[df["Volume"] > 0].dropna(subset=["Close"])
    df = df[[x.date() == today for x in df.index.tz_convert(TZ)]]
    if df.empty:
        return None
    return float(df["Open"].iloc[0]), float(df["Close"].iloc[-1])


def select(rows: list[dict]) -> list[dict]:
    """套用進場條件並依 5 日跌幅由深到淺排序（純函式，方便測試）。"""
    out = [r for r in rows
           if r["ret5"] <= -DIP_RET5_MIN and -DIP_GAP_MAX <= r["gap"] <= -DIP_GAP_MIN
           and r["price"] <= DIP_PRICE_MAX]
    return sorted(out, key=lambda r: r["ret5"])


def prefilter() -> list[tuple[str, str, float, float]]:
    """用日線篩出 5 日跌幅達標的股票：[(代號, symbol, 昨收, 5日報酬)]。"""
    today = datetime.now(TZ).date()
    syms = {f"{c}.{s}": c for c, s in _universe().items()}
    pre = []
    names = list(syms)
    for i in range(0, len(names), 100):
        chunk = names[i:i + 100]
        try:
            d = yf.download(chunk, period="15d", interval="1d", group_by="ticker",
                            auto_adjust=False, progress=False, threads=True)
        except Exception as e:
            logger.warning(f"[Dip] 日線下載失敗: {e}")
            continue
        for sym in chunk:
            try:
                c = d[sym]["Close"].dropna()
                c = c[[x.date() < today for x in c.index]]
                if len(c) < 6:
                    continue
                pc = float(c.iloc[-1])
                ret5 = pc / float(c.iloc[-6]) - 1
                if ret5 <= -DIP_RET5_MIN:
                    pre.append((syms[sym], sym, pc, ret5))
            except Exception:
                continue
    return pre


def quote(pre: list[tuple[str, str, float, float]]) -> list[dict]:
    """抓今日分 K；今日還沒有報價的股票不會出現在結果裡（呼叫端稍後重試）。"""
    rows = []
    for code, sym, pc, ret5 in pre:
        try:
            q = _last_price(sym)
            if q is None:
                continue
            day_open, price = q
            rows.append({"code": code, "symbol": sym, "price": price, "ret5": ret5,
                         "gap": day_open / pc - 1, "prev_close": pc})
        except Exception as e:
            logger.debug(f"[Dip] {sym} 資料失敗: {e}")
    return rows


def find_candidates() -> list[dict]:
    pre = prefilter()
    rows = quote(pre)
    logger.info(f"[Dip] 5日跌≥{DIP_RET5_MIN:.0%} 共 {len(pre)} 檔，取得報價 {len(rows)} 檔")
    return select(rows)


class DipTrader:
    def __init__(self):
        self._lock = threading.RLock()
        self._state = self._load()
        self._pre, self._pre_date = None, None   # 當日日線初篩結果（只算一次）

    def _new_state(self) -> dict:
        return {"date": str(datetime.now(TZ).date()), "holdings": {}, "daily_pnl": 0.0,
                "scanned": False, "trade_log": []}

    def _load(self) -> dict:
        try:
            return json.loads(_state_path().read_text(encoding="utf-8"))
        except Exception:
            return self._new_state()

    def _save(self):
        try:
            _state_path().write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[Dip] state 儲存失敗: {e}")

    def _report(self, code, side, shares, price, reason, pnl=0.0):
        self._state["trade_log"].append({
            "time": datetime.now(TZ).strftime("%H:%M"), "code": code, "side": side,
            "shares": shares, "price": price, "reason": reason, "pnl": round(pnl, 0)})
        try:
            from utils.platform_signal import report_position, report_trade
            report_trade(user_id=TRADING_USER_ID, code=code, action=side, shares=shares,
                         price=price, pnl=pnl, reason=reason, is_real=False, account=ACCOUNT)
            report_position(TRADING_USER_ID, code, "LONG", shares if side == "BUY" else 0,
                            price, is_real=False, account=ACCOUNT)
        except Exception:
            pass

    def _buy(self, c: dict) -> bool:
        code, price = c["code"], c["price"]
        h = self._state["holdings"]
        if code in h or len(h) >= DIP_MAX_POS:
            return False
        shares = int((DIP_CAPITAL * DIP_PER_TRADE) // (price * 1000)) * 1000
        if shares <= 0:
            return False
        value = shares * price
        cost = value + max(_MIN_FEE, round(value * _COMMISSION_RATE))
        h[code] = {"symbol": c["symbol"], "shares": shares, "entry_price": price, "cost": cost,
                   "prev_close": c["prev_close"], "tp": round(price * (1 + DIP_TP_PCT), 2),
                   "entry_time": datetime.now(TZ).strftime("%H:%M")}
        reason = f"超跌做多 5日{c['ret5']*100:+.1f}% 跳空{c['gap']*100:+.1f}%"
        self._report(code, "BUY", shares, price, reason)
        logger.info(f"[Dip] 🟢 買進 {code} {shares}股 @{price:.2f}  {reason}")
        try:
            from utils.telegram_notify import notify_buy
            notify_buy("超跌做多(紙)", code, shares, price, reason)
        except Exception:
            pass
        return True

    def _sell(self, code: str, price: float, reason: str):
        h = self._state["holdings"].pop(code, None)
        if h is None:
            return
        value = h["shares"] * price
        pnl = value - max(_MIN_FEE, round(value * _COMMISSION_RATE)) - round(value * _STT_RATE) - h["cost"]
        self._state["daily_pnl"] += pnl
        self._report(code, "SELL", h["shares"], price, reason, pnl)
        logger.info(f"[Dip] {'🟩' if pnl >= 0 else '🟥'} 賣出 {code} @{price:.2f} 損益={pnl:+,.0f} 原因={reason}")
        try:
            from utils.telegram_notify import notify_sell
            notify_sell("超跌做多(紙)", code, h["shares"], price, pnl, reason)
        except Exception:
            pass

    def run_cycle(self):
        if not ENABLE_DIP:
            return
        now = datetime.now(TZ)
        hm = (now.hour, now.minute)
        with self._lock:
            if self._state.get("date") != str(now.date()):
                self._state = self._new_state()

        # 進場掃描：yfinance 分 K 會延遲（2026-11-16 實測 09:05 時 5 檔候選一檔報價都沒有），
        # 所以報價沒到齊就每個週期重試，到齊或到 _SCAN_GIVE_UP 才決定，以保留「跌最深優先」的排序。
        if not self._state["scanned"] and _SCAN_FROM <= hm < _SCAN_UNTIL:
            if self._pre is None or self._pre_date != str(now.date()):
                self._pre, self._pre_date = prefilter(), str(now.date())
                logger.info(f"[Dip] 5日跌≥{DIP_RET5_MIN:.0%} 共 {len(self._pre)} 檔")
            rows = quote(self._pre)
            if len(rows) >= len(self._pre) or hm >= _SCAN_GIVE_UP:
                cands = select(rows)
                with self._lock:
                    for c in cands:
                        self._buy(c)
                    self._state["scanned"] = True
                    logger.info(f"[Dip] 取得報價 {len(rows)}/{len(self._pre)} 檔，今日符合 {len(cands)} 檔: "
                                + ", ".join(f"{c['code']}(5日{c['ret5']*100:+.0f}% gap{c['gap']*100:+.1f}%)" for c in cands[:10]))
                    self._save()
            else:
                logger.info(f"[Dip] 報價尚未到齊（{len(rows)}/{len(self._pre)}），下個週期重試")

        for code in list(self._state["holdings"]):
            h = self._state["holdings"][code]
            try:
                q = _last_price(h["symbol"])
            except Exception:
                q = None
            if q is None:
                continue
            price = q[1]
            with self._lock:
                if price >= h["tp"]:
                    self._sell(code, price, f"停利 +{DIP_TP_PCT*100:.0f}%")
                elif hm >= _CLOSE_AT:
                    locked = price <= h["prev_close"] * 0.905
                    self._sell(code, price, "收盤前賣出 13:20"
                               + ("【跌停鎖死，實盤可能賣不掉】" if locked else ""))
                self._save()

    def get_summary(self) -> dict:
        s = self._state
        return {"account": ACCOUNT, "date": s.get("date"), "daily_pnl": s.get("daily_pnl", 0.0),
                "positions": len(s.get("holdings", {})), "trades_today": len(s.get("trade_log", []))}
