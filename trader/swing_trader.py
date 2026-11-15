"""個股深度回檔小波段（紙上）策略 — 抱 1~10 個交易日，與當沖/超跌做多分開記帳。

規則（backtest/swing_research.py；400 檔、6 年日線、收盤前進場）：
  進場（13:15~13:25 掃描一次，以當時價格近似收盤價）：
    個股收盤 > 200 日均線、RSI(2) < SWING_RSI_MAX、近 3 日跌幅 ≥ SWING_RET3_MIN、
    股價介於 SWING_PRICE_MIN~SWING_PRICE_MAX，且「大盤多頭」（股票池等權指數 > 其 200 日均線）。
    同日多檔取 RSI(2) 最低者，最多同時 SWING_MAX_POSITIONS 檔。
  出場（09:05~09:30，以前一日收盤判斷）：前一日收盤 > 5 日均線，或進場後已過 SWING_MAX_DAYS−1 根日K。
  不設停損（回測加停損變差）。
  回測（手續費 6 折、來回 47.1bp）：2,141 筆、勝率 61%、每筆 +105bp、6 年僅 2022 為負；
  100 萬/3 檔模擬約 +9.7%/年、最大回撤 22 萬。股票池以今日成交值選出，有存活者偏誤 → 只做紙上。

成交以帳戶 "swing-<額度>" 回報到 platform。部位跨日保存於 swing_trader_state.json。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from trader.day_trader import TRADING_USER_ID, _COMMISSION_RATE, _MIN_FEE

logger = logging.getLogger(__name__)
TZ = pytz.timezone("Asia/Taipei")

ENABLE_SWING    = os.getenv("ENABLE_SWING_TRADING", "true").lower() == "true"
SWING_CAPITAL   = float(os.getenv("SWING_CAPITAL", "1000000"))
SWING_MAX_POS   = int(os.getenv("SWING_MAX_POSITIONS", "3"))
SWING_RSI_MAX   = float(os.getenv("SWING_RSI_MAX", "5"))
SWING_RET3_MIN  = float(os.getenv("SWING_RET3_MIN", "0.06"))    # 近 3 日跌幅門檻（正數）
SWING_PRICE_MAX = float(os.getenv("SWING_PRICE_MAX", "330"))
SWING_PRICE_MIN = float(os.getenv("SWING_PRICE_MIN", "10"))
SWING_MAX_DAYS  = int(os.getenv("SWING_MAX_DAYS", "10"))
_TAX = 0.003                                                    # 非當沖證交稅
_EXIT_FROM, _EXIT_UNTIL = (9, 5), (9, 30)
_SCAN_FROM, _SCAN_UNTIL = (13, 15), (13, 25)

ACCOUNT = f"swing-{int(SWING_CAPITAL)}"


def _universe() -> dict[str, str]:
    try:
        return json.loads((Path(__file__).parent / "dip_universe.json").read_text())
    except Exception as e:
        logger.warning(f"[Swing] 股票池讀取失敗: {e}")
        return {}


def _state_path() -> Path:
    return Path(os.getenv("DATA_DIR", Path(__file__).parent.parent)) / "swing_trader_state.json"


def rsi(c: pd.Series, n: int = 2) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def market_up(closes: pd.DataFrame) -> bool:
    """股票池等權指數是否在其 200 日均線之上（closes: 日期 × 代號，最後一列為今日）。"""
    eq = closes.pct_change(fill_method=None).mean(axis=1).add(1).cumprod()
    ma = eq.rolling(200).mean()
    return bool(len(eq) >= 200 and eq.iloc[-1] > ma.iloc[-1])


def entry_signals(closes: pd.DataFrame) -> list[dict]:
    """套用進場條件（純函式）。closes 最後一列視為今日收盤；回傳依 RSI(2) 由低到高。"""
    if not market_up(closes):
        return []
    out = []
    for code in closes.columns:
        c = closes[code].dropna()
        if len(c) < 201 or c.index[-1] != closes.index[-1]:
            continue
        px = float(c.iloc[-1])
        r2 = float(rsi(c).iloc[-1])
        ret3 = px / float(c.iloc[-4]) - 1
        if (SWING_PRICE_MIN <= px <= SWING_PRICE_MAX and px > float(c.iloc[-200:].mean())
                and r2 < SWING_RSI_MAX and ret3 <= -SWING_RET3_MIN):
            out.append({"code": code, "price": px, "rsi2": r2, "ret3": ret3})
    return sorted(out, key=lambda r: r["rsi2"])


def exit_reason(c: pd.Series, entry_date: str) -> str | None:
    """c: 已收完的日收盤（不含今日）。回傳出場原因或 None。"""
    c = c.dropna()
    if len(c) < 5:
        return None
    after = int((c.index > pd.Timestamp(entry_date)).sum())
    if after >= 1 and float(c.iloc[-1]) > float(c.iloc[-5:].mean()):
        return "站回5日線"
    if after >= SWING_MAX_DAYS - 1:
        return f"持有滿{SWING_MAX_DAYS}日"
    return None


def download_closes(symbols: list[str], period: str = "2y") -> pd.DataFrame:
    """還原權息日收盤（與回測一致），欄位為 yfinance 代號，索引為無時區日期。"""
    frames = []
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        try:
            d = yf.download(chunk, period=period, interval="1d", group_by="ticker",
                            auto_adjust=True, progress=False, threads=True)
        except Exception as e:
            logger.warning(f"[Swing] 日線下載失敗: {e}")
            continue
        for s in chunk:
            try:
                frames.append(d[s]["Close"].rename(s))
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    return out.sort_index()


def _last_price(sym: str) -> float | None:
    today = datetime.now(TZ).date()
    df = yf.Ticker(sym).history(period="1d", interval="1m")
    if df is None or df.empty:
        return None
    df = df[df["Volume"] > 0].dropna(subset=["Close"])
    df = df[[x.date() == today for x in df.index.tz_convert(TZ)]]
    return float(df["Close"].iloc[-1]) if not df.empty else None


class SwingTrader:
    def __init__(self):
        self._lock = threading.RLock()
        self._state = self._load()

    def _load(self) -> dict:
        try:
            return json.loads(_state_path().read_text(encoding="utf-8"))
        except Exception:
            return {"holdings": {}, "scan_date": "", "exit_date": "", "total_pnl": 0.0, "trade_log": []}

    def _save(self):
        try:
            self._state["trade_log"] = self._state["trade_log"][-200:]
            _state_path().write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[Swing] state 儲存失敗: {e}")

    def _report(self, code, side, shares, price, reason, pnl=0.0):
        self._state["trade_log"].append({
            "time": datetime.now(TZ).strftime("%Y-%m-%d %H:%M"), "code": code, "side": side,
            "shares": shares, "price": price, "reason": reason, "pnl": round(pnl, 0)})
        try:
            from utils.platform_signal import report_position, report_trade
            report_trade(user_id=TRADING_USER_ID, code=code, action=side, shares=shares,
                         price=price, pnl=pnl, reason=reason, is_real=False, account=ACCOUNT)
            report_position(TRADING_USER_ID, code, "LONG", shares if side == "BUY" else 0,
                            price, is_real=False, account=ACCOUNT)
        except Exception:
            pass

    def _buy(self, c: dict, symbol: str) -> bool:
        code, price = c["code"], c["price"]
        h = self._state["holdings"]
        if code in h or len(h) >= SWING_MAX_POS:
            return False
        shares = int((SWING_CAPITAL / SWING_MAX_POS) // (price * 1000)) * 1000
        if shares <= 0:
            return False
        value = shares * price
        h[code] = {"symbol": symbol, "shares": shares, "entry_price": price,
                   "cost": value + max(_MIN_FEE, round(value * _COMMISSION_RATE)),
                   "entry_date": str(datetime.now(TZ).date())}
        reason = f"回檔波段 RSI2={c['rsi2']:.1f} 3日{c['ret3']*100:+.1f}%"
        self._report(code, "BUY", shares, price, reason)
        logger.info(f"[Swing] 🟢 買進 {code} {shares}股 @{price:.2f}  {reason}")
        try:
            from utils.telegram_notify import notify_buy
            notify_buy("回檔波段(紙)", code, shares, price, reason)
        except Exception:
            pass
        return True

    def _sell(self, code: str, price: float, reason: str):
        h = self._state["holdings"].pop(code, None)
        if h is None:
            return
        value = h["shares"] * price
        pnl = value - max(_MIN_FEE, round(value * _COMMISSION_RATE)) - round(value * _TAX) - h["cost"]
        self._state["total_pnl"] = self._state.get("total_pnl", 0.0) + pnl
        self._report(code, "SELL", h["shares"], price, reason, pnl)
        logger.info(f"[Swing] {'🟩' if pnl >= 0 else '🟥'} 賣出 {code} @{price:.2f} 損益={pnl:+,.0f} 原因={reason}")
        try:
            from utils.telegram_notify import notify_sell
            notify_sell("回檔波段(紙)", code, h["shares"], price, pnl, reason)
        except Exception:
            pass

    def _check_exits(self, today):
        hold = self._state["holdings"]
        if not hold:
            return True
        closes = download_closes([h["symbol"] for h in hold.values()], period="3mo")
        if closes.empty:
            return False
        closes = closes[closes.index < pd.Timestamp(today)]
        for code in list(hold):
            h = hold[code]
            if h["entry_date"] >= str(today) or h["symbol"] not in closes:
                continue
            why = exit_reason(closes[h["symbol"]], h["entry_date"])
            if not why:
                continue
            try:
                price = _last_price(h["symbol"])
            except Exception:
                price = None
            if price is None:
                return False                      # 還沒有今日成交，稍後重試
            self._sell(code, price, why)
        return True

    def _scan(self, today) -> bool:
        if len(self._state["holdings"]) >= SWING_MAX_POS:
            return True
        uni = _universe()
        syms = {f"{c}.{s}": c for c, s in uni.items()}
        closes = download_closes(list(syms))
        if closes.empty or closes.index[-1].date() != today:
            logger.info("[Swing] 日線尚無今日資料，稍後重試")
            return False
        cands = entry_signals(closes.rename(columns=syms))
        back = {v: k for k, v in syms.items()}
        logger.info(f"[Swing] 大盤多頭={market_up(closes)}  符合 {len(cands)} 檔: "
                    + ", ".join(f"{c['code']}(RSI2 {c['rsi2']:.0f}, 3日{c['ret3']*100:+.0f}%)" for c in cands[:10]))
        for c in cands:
            if len(self._state["holdings"]) >= SWING_MAX_POS:
                break
            try:
                px = _last_price(back[c["code"]])       # 以 1 分K 最新價成交（日線當日價可能延遲）
            except Exception:
                px = None
            if px:
                self._buy({**c, "price": px}, back[c["code"]])
        return True

    def run_cycle(self):
        if not ENABLE_SWING:
            return
        now = datetime.now(TZ)
        hm, today = (now.hour, now.minute), now.date()
        with self._lock:
            if self._state.get("exit_date") != str(today) and _EXIT_FROM <= hm < _EXIT_UNTIL:
                if self._check_exits(today):
                    self._state["exit_date"] = str(today)
                self._save()
            if self._state.get("scan_date") != str(today) and _SCAN_FROM <= hm < _SCAN_UNTIL:
                if self._scan(today):
                    self._state["scan_date"] = str(today)
                self._save()

    def get_summary(self) -> dict:
        s = self._state
        return {"account": ACCOUNT, "total_pnl": s.get("total_pnl", 0.0),
                "positions": len(s.get("holdings", {}))}
