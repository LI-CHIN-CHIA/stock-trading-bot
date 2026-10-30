"""
隔日沖模組
==========
尾盤（14:00~14:25）選股佈局，隔日早盤（09:05~11:00）出場。

特性：
- 獨立 state file：overnight_trader_state.json（不影響其他帳戶）
- 初始資金 OVERNIGHT_INITIAL_CAPITAL，每日不重置（持倉可跨日）
- 選股：LightGBM buy_proba > 0.55 + 量能/趨勢過濾
- 停損：-2.0%（略寬，因為跨夜波動較大）
- 停利：+3.0%
- 11:00 強制平倉（確保當天出場，不變成多日持倉）
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from utils.constants import OTC_SYMBOLS, TRADEABLE

logger = logging.getLogger(__name__)

TZ = pytz.timezone("Asia/Taipei")

# ── 參數 ─────────────────────────────────────────────────────────────────────
OVERNIGHT_INITIAL_CAPITAL = float(os.getenv("OVERNIGHT_INITIAL_CAPITAL", "300000"))
OVERNIGHT_PER_TRADE_PCT   = float(os.getenv("OVERNIGHT_PER_TRADE_PCT",   "0.25"))
OVERNIGHT_MAX_POSITIONS   = int(os.getenv("OVERNIGHT_MAX_POSITIONS",     "3"))
OVERNIGHT_STOP_LOSS_PCT   = float(os.getenv("OVERNIGHT_STOP_LOSS_PCT",   "-0.02"))
OVERNIGHT_TAKE_PROFIT_PCT = float(os.getenv("OVERNIGHT_TAKE_PROFIT_PCT", "0.03"))
OVERNIGHT_BUY_PROBA_MIN   = float(os.getenv("OVERNIGHT_BUY_PROBA_MIN",   "0.55"))
ENABLE_OVERNIGHT_TRADING  = os.getenv("ENABLE_OVERNIGHT_TRADING", "false").lower() == "true"

# 量能門檻：日成交量至少 3,000,000 股（3000張），且今日量 >= 昨日量（真正放量）
OVERNIGHT_MIN_DAILY_VOL   = int(os.getenv("OVERNIGHT_MIN_DAILY_VOL", "3000000"))
OVERNIGHT_VOL_RATIO_MIN   = float(os.getenv("OVERNIGHT_VOL_RATIO_MIN", "1.0"))

_ENTRY_START  = (14, 0)
_ENTRY_END    = (14, 25)
_EXIT_FORCE_H = 11
_EXIT_FORCE_M = 0
_GAP_UP_INSTANT_SELL = 0.04   # 跳空 > +4% 立即賣

# 手續費 & 證交稅（隔日沖）
_COMMISSION_RATE = 0.001425   # 券商手續費 0.1425%
_STT_RATE        = 0.003      # 證交稅 0.3%（非當沖，全額課徵）
_MIN_FEE         = 20         # 最低手續費 20 元


def _symbol(code: str) -> str:
    return f"{code}.TWO" if code in OTC_SYMBOLS else f"{code}.TW"


def _state_path() -> Path:
    data_dir = Path(os.getenv("DATA_DIR", Path(__file__).parent.parent))
    return data_dir / "overnight_trader_state.json"


# ── 資料 ─────────────────────────────────────────────────────────────────────

def _fetch_daily(code: str, period: str = "6mo") -> pd.DataFrame | None:
    """取得日線資料（用於 LightGBM inference）。"""
    try:
        df = yf.Ticker(_symbol(code)).history(period=period)
        if len(df) < 60:
            return None
        return df.dropna(subset=["Close"])
    except Exception as e:
        logger.debug(f"[Overnight] {code} 日線資料擷取失敗: {e}")
        return None


def _fetch_intraday(code: str) -> pd.DataFrame | None:
    """取得當日 2 分鐘 K 線（用於盯盤出場）。"""
    try:
        df = yf.Ticker(_symbol(code)).history(period="1d", interval="2m")
        if df.empty:
            return None
        return df.dropna(subset=["Close"])
    except Exception as e:
        logger.debug(f"[Overnight] {code} 分鐘資料擷取失敗: {e}")
        return None


def _current_price(code: str) -> float | None:
    """快速取得最新成交價（先試分鐘 K，再試日線）。"""
    df = _fetch_intraday(code)
    if df is not None and not df.empty:
        return float(df["Close"].iloc[-1])
    df_d = _fetch_daily(code, period="5d")
    if df_d is not None and not df_d.empty:
        return float(df_d["Close"].iloc[-1])
    return None


# ── 選股訊號 ──────────────────────────────────────────────────────────────────

def _lgbm_buy_signal(code: str, model) -> tuple[bool, float]:
    """
    使用現有日線 LightGBM model 判斷是否買進。
    回傳 (is_buy, buy_proba)
    """
    if model is None:
        return True, 1.0   # 無 model 時退化為純規則

    try:
        from ai.features import prepare_inference_row
        df = _fetch_daily(code)
        if df is None:
            return False, 0.0
        row = prepare_inference_row(df, code=code)
        sig = model.predict(row)
        buy_proba = sig.get("buy_proba", 0.0)
        return buy_proba >= OVERNIGHT_BUY_PROBA_MIN, buy_proba
    except Exception as e:
        logger.debug(f"[Overnight] {code} model inference 失敗: {e}")
        return False, 0.0


def _tail_session_check(code: str) -> dict | None:
    """
    尾盤技術面過濾（14:00 附近的日線快照）：
    - 今日量 > 昨日量（放量）
    - 今日收盤 > 5MA（短線趨勢向上）
    - 今日漲幅 > -1%（非重跌股）
    """
    df = _fetch_daily(code, period="3mo")
    if df is None or len(df) < 10:
        return None

    close = df["Close"]
    vol   = df["Volume"]
    ma5   = close.rolling(5).mean()

    today_close  = float(close.iloc[-1])
    today_vol    = float(vol.iloc[-1])
    prev_vol     = float(vol.iloc[-2]) if len(vol) >= 2 else today_vol
    today_ret    = float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) >= 2 else 0.0
    ma5_val      = float(ma5.iloc[-1]) if not np.isnan(ma5.iloc[-1]) else today_close

    above_ma5    = today_close > ma5_val
    vol_up       = today_vol >= prev_vol * OVERNIGHT_VOL_RATIO_MIN  # 真正放量（≥昨日）
    not_crashing = today_ret > -0.01
    vol_ok       = today_vol >= OVERNIGHT_MIN_DAILY_VOL              # 最低日成交量 3000張

    if not (above_ma5 and vol_up and not_crashing and vol_ok):
        return None

    return {
        "price":      today_close,
        "today_ret":  today_ret,
        "above_ma5":  above_ma5,
        "vol_ratio":  today_vol / (prev_vol + 1e-8),
    }


# ── 主類別 ────────────────────────────────────────────────────────────────────

class OvernightTrader:
    """隔日沖紙上交易，每 2 分鐘 run_cycle() 一次。"""

    def __init__(self, model=None):
        self._lock  = threading.Lock()
        self._model = model
        self._state = self._load_state()

    def set_model(self, model) -> None:
        self._model = model

    # ── 狀態 I/O ─────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        path = _state_path()
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    s = json.load(f)
                # 確保 cash 欄位存在（首次 migration）
                if "cash" not in s:
                    s["cash"] = OVERNIGHT_INITIAL_CAPITAL
                return s
            except Exception:
                pass
        return self._new_state()

    def _new_state(self) -> dict:
        return {
            "cash":        OVERNIGHT_INITIAL_CAPITAL,
            "holdings":    {},
            "trade_log":   [],
            "total_pnl":   0.0,
        }

    def _save_state(self):
        try:
            with open(_state_path(), "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[Overnight] state 儲存失敗: {e}")

    # ── 時間判斷 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _now_tw() -> datetime:
        return datetime.now(TZ)

    def _is_trading_day(self) -> bool:
        now = self._now_tw()
        return now.weekday() < 5

    def _is_entry_window(self) -> bool:
        now = self._now_tw()
        return _ENTRY_START <= (now.hour, now.minute) <= _ENTRY_END

    def _is_exit_window(self) -> bool:
        """09:05~11:00：監控停利停損"""
        now = self._now_tw()
        return (9, 5) <= (now.hour, now.minute) < (_EXIT_FORCE_H, _EXIT_FORCE_M)

    def _is_force_close_time(self) -> bool:
        now = self._now_tw()
        return (now.hour, now.minute) >= (_EXIT_FORCE_H, _EXIT_FORCE_M)

    # ── 買賣執行（紙上）──────────────────────────────────────────────────────

    def _paper_buy(self, code: str, price: float, buy_proba: float, info: dict) -> bool:
        holdings = self._state["holdings"]
        if code in holdings:
            return False
        if len(holdings) >= OVERNIGHT_MAX_POSITIONS:
            return False

        budget = self._state["cash"] * OVERNIGHT_PER_TRADE_PCT
        shares = int(budget // (price * 1000)) * 1000
        if shares <= 0:
            shares = int(budget // price)   # 零股
        if shares <= 0:
            return False
        cost     = shares * price
        buy_fee  = max(_MIN_FEE, round(cost * _COMMISSION_RATE))
        total_cost = cost + buy_fee
        if total_cost > self._state["cash"]:
            return False

        self._state["cash"] -= total_cost
        try:
            from utils.telegram_notify import notify_buy
            notify_buy("隔日沖", code, shares, price, f"buy_proba={buy_proba:.3f} ret={info.get('today_ret',0):.2%}")
        except Exception:
            pass
        holdings[code] = {
            "shares":      shares,
            "entry_price": price,
            "entry_date":  str(date.today()),
            "entry_time":  self._now_tw().strftime("%H:%M"),
            "cost":        total_cost,   # 含買進手續費
            "stop_loss":   round(price * (1 + OVERNIGHT_STOP_LOSS_PCT), 2),
            "take_profit": round(price * (1 + OVERNIGHT_TAKE_PROFIT_PCT), 2),
            "buy_proba":   round(buy_proba, 3),
        }
        self._log_trade(code, "BUY", shares, price, "隔日沖買入", buy_proba=buy_proba)
        logger.info(
            f"[Overnight] 🌙 買入 {code}  {shares}股 @{price:.2f}"
            f"  buy_proba={buy_proba:.3f}  ret={info.get('today_ret', 0):.2%}"
        )
        return True

    def _paper_sell(self, code: str, price: float, reason: str) -> float:
        holdings = self._state["holdings"]
        if code not in holdings:
            return 0.0
        h        = holdings.pop(code)
        proceeds = h["shares"] * price
        sell_fee = max(_MIN_FEE, round(proceeds * _COMMISSION_RATE))
        stt      = round(proceeds * _STT_RATE)
        net_proceeds = proceeds - sell_fee - stt
        pnl      = net_proceeds - h["cost"]
        self._state["cash"]       += net_proceeds
        self._state["total_pnl"]  += pnl
        self._log_trade(code, "SELL", h["shares"], price, reason)
        emoji = "🌅" if pnl >= 0 else "🌧"
        logger.info(
            f"[Overnight] {emoji} 賣出 {code}  {h['shares']}股 @{price:.2f}"
            f"  手續費={sell_fee+stt:,.0f}  損益={pnl:+,.0f} TWD  原因={reason}"
        )
        try:
            from utils.telegram_notify import notify_sell
            notify_sell("隔日沖", code, h["shares"], price, pnl, reason)
        except Exception:
            pass
        return pnl

    def _log_trade(self, code: str, side: str, shares: int, price: float,
                   reason: str, buy_proba: float | None = None):
        self._state["trade_log"].append({
            "time":      self._now_tw().strftime("%Y-%m-%d %H:%M"),
            "code":      code,
            "side":      side,
            "shares":    shares,
            "price":     price,
            "reason":    reason,
            "buy_proba": buy_proba,
        })

    # ── 出場監控 ─────────────────────────────────────────────────────────────

    def _check_exit(self):
        """檢查現有持倉的停利/停損/強制平倉。"""
        if not self._state["holdings"]:
            return
        force = self._is_force_close_time()
        for code in list(self._state["holdings"]):
            price = _current_price(code)
            if price is None:
                continue
            h = self._state["holdings"][code]

            # 跳空大幅高開 → 立即賣
            gap = (price - h["entry_price"]) / h["entry_price"]
            if gap >= _GAP_UP_INSTANT_SELL:
                self._paper_sell(code, price, f"跳空高開 +{gap:.1%}")
                continue

            if force:
                self._paper_sell(code, price, "強制平倉 11:00")
            elif price <= h["stop_loss"]:
                self._paper_sell(code, price, f"停損 {h['stop_loss']:.2f}")
            elif price >= h["take_profit"]:
                self._paper_sell(code, price, f"停利 {h['take_profit']:.2f}")

    # ── 尾盤選股 ─────────────────────────────────────────────────────────────

    def _scan_entries(self):
        """14:00~14:25 掃描尾盤佈局機會。"""
        if not self._is_entry_window():
            return
        if len(self._state["holdings"]) >= OVERNIGHT_MAX_POSITIONS:
            return

        already_held = set(self._state["holdings"].keys())
        candidates = []

        for code in TRADEABLE:
            if code in already_held:
                continue
            # 基本技術面過濾
            info = _tail_session_check(code)
            if info is None:
                continue
            # LightGBM buy signal
            is_buy, buy_proba = _lgbm_buy_signal(code, self._model)
            if not is_buy:
                continue
            candidates.append((code, buy_proba, info))

        # 依 buy_proba 排序，取最高的
        candidates.sort(key=lambda x: x[1], reverse=True)
        for code, buy_proba, info in candidates:
            if len(self._state["holdings"]) >= OVERNIGHT_MAX_POSITIONS:
                break
            price = info["price"]
            logger.info(
                f"[Overnight] 📋 {code} buy_proba={buy_proba:.3f}"
                f"  ret={info.get('today_ret', 0):.2%}"
                f"  vol_ratio={info.get('vol_ratio', 1):.2f}"
                f"  @{price:.2f}"
            )
            self._paper_buy(code, price, buy_proba, info)

    # ── 主迴圈 ───────────────────────────────────────────────────────────────

    def run_cycle(self):
        with self._lock:
            if not self._is_trading_day():
                return

            now = self._now_tw()
            hour, minute = now.hour, now.minute

            # 早盤：出場監控
            if self._is_exit_window() or self._is_force_close_time():
                self._check_exit()

            # 尾盤：選股佈局
            if ENABLE_OVERNIGHT_TRADING:
                self._scan_entries()

            # 持倉狀態 log
            if (9, 0) <= (hour, minute) <= (14, 30) and self._state["holdings"]:
                holdings_str = ", ".join(
                    f"{c}@{h['entry_price']:.2f}(SL:{h['stop_loss']:.2f})"
                    for c, h in self._state["holdings"].items()
                )
                logger.info(f"[Overnight] 持倉: {holdings_str}  現金={self._state['cash']:,.0f}")

            self._save_state()

    # ── 外部查詢 ─────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        with self._lock:
            s = self._state
            return {
                "cash":           s.get("cash", 0.0),
                "total_pnl":      s.get("total_pnl", 0.0),
                "positions":      len(s.get("holdings", {})),
                "holdings":       dict(s.get("holdings", {})),
                "trades_total":   len(s.get("trade_log", [])),
                "mode":           "paper_overnight" if ENABLE_OVERNIGHT_TRADING else "watch_only",
            }
