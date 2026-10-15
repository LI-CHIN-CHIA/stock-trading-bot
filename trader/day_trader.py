"""
當沖紙上交易模組
================
每 2 分鐘掃描 TRADEABLE 候選股票，依量能 + K 線形態決策。
即使不交易也會持續記錄盤況（持續盯盤模式）。

特性：
- 獨立 state file：day_trader_state.json（不影響原有帳戶）
- 當日初始資金 DAY_INITIAL_CAPITAL，每日重置
- 13:20 強制平倉（確保當沖 T+0）
- 停損 -1.5%，停利 +2.5%

訊號邏輯：
- VWAP：價格在 VWAP 上方為偏多
- 量能方向：上漲 K 棒買方量估計 vs 下跌 K 棒賣方量
- 突破：最新收盤突破近 10 根 K 棒高點（含量確認）
- K 線形態：錘子線 / 吞噬形態
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
DAY_INITIAL_CAPITAL = float(os.getenv("DAY_INITIAL_CAPITAL", "300000"))
DAY_PER_TRADE_PCT   = float(os.getenv("DAY_PER_TRADE_PCT",   "0.20"))   # 每筆最多 20%
DAY_MAX_POSITIONS   = int(os.getenv("DAY_MAX_POSITIONS",     "3"))
DAY_STOP_LOSS_PCT   = float(os.getenv("DAY_STOP_LOSS_PCT",   "-0.015")) # -1.5%
DAY_TAKE_PROFIT_PCT = float(os.getenv("DAY_TAKE_PROFIT_PCT", "0.025"))  # +2.5%
DAY_FORCE_CLOSE_H   = 13
DAY_FORCE_CLOSE_M   = 20
ENABLE_DAY_TRADING  = os.getenv("ENABLE_DAY_TRADING", "false").lower() == "true"

_INTERVAL = "2m"   # yfinance 2-minute bars (stable for intraday)
_PERIOD   = "1d"

WATCH_STOCKS: set[str] = TRADEABLE   # 用現有候選清單


def _symbol(code: str) -> str:
    return f"{code}.TWO" if code in OTC_SYMBOLS else f"{code}.TW"


def _state_path() -> Path:
    data_dir = Path(os.getenv("DATA_DIR", Path(__file__).parent.parent))
    return data_dir / "day_trader_state.json"


# ── 技術計算 ──────────────────────────────────────────────────────────────────

def _vwap(df: pd.DataFrame) -> pd.Series:
    """累計 VWAP（從當日開盤算起）。"""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    cum_tp_vol = (typical * df["Volume"]).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def _volume_delta(df: pd.DataFrame) -> pd.Series:
    """
    估算每根 K 棒的買賣方量差值（Volume Delta）。
    Bull candle: buy_vol ≈ volume * (Close-Low)/(High-Low)
    Bear candle: sell_vol ≈ volume * (High-Close)/(High-Low)
    delta = buy_vol - sell_vol
    """
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    bull = df["Volume"] * (df["Close"] - df["Low"]) / hl
    bear = df["Volume"] * (df["High"] - df["Close"]) / hl
    return (bull - bear).fillna(0)


def _candle_pattern(df: pd.DataFrame) -> str:
    """
    根據最近兩根 K 棒判斷形態，回傳 'hammer' | 'engulf_bull' | 'engulf_bear' | 'none'。
    只看最新兩根。
    """
    if len(df) < 2:
        return "none"
    cur = df.iloc[-1]
    prv = df.iloc[-2]
    body_cur = abs(cur["Close"] - cur["Open"])
    body_prv = abs(prv["Close"] - prv["Open"])
    hl = cur["High"] - cur["Low"]
    if hl == 0:
        return "none"

    # 錘子線（看多反轉）：下影線 >= 2x 實體，上影線 <= 10% hl，小實體
    lower_shadow = min(cur["Open"], cur["Close"]) - cur["Low"]
    upper_shadow = cur["High"] - max(cur["Open"], cur["Close"])
    if (lower_shadow >= 2 * body_cur and
            upper_shadow <= 0.1 * hl and
            body_cur <= 0.4 * hl):
        return "hammer"

    # 看漲吞噬：前一根收黑，當根收紅且實體完全包住前一根實體
    if (prv["Close"] < prv["Open"] and
            cur["Close"] > cur["Open"] and
            cur["Close"] > prv["Open"] and
            cur["Open"] < prv["Close"]):
        return "engulf_bull"

    # 看跌吞噬：前一根收紅，當根收黑且實體完全包住前一根實體
    if (prv["Close"] > prv["Open"] and
            cur["Close"] < cur["Open"] and
            cur["Open"] > prv["Close"] and
            cur["Close"] < prv["Open"]):
        return "engulf_bear"

    return "none"


def _analyze_bars(df: pd.DataFrame) -> dict:
    """
    給定當日分鐘 K，回傳盤況分析結果：
    {
      price, vwap, above_vwap,
      vol_delta_sum,    # 近10根累積量差
      vol_delta_signal, # 'buy' | 'sell' | 'neutral'
      avg_volume,       # 近20根平均量
      last_volume,      # 最新一根量
      vol_surge,        # 最新量 > 2.5x 均量
      breakout,         # 突破近10根高點
      pattern,          # candle pattern
      score,            # 綜合分數 -4~+4，正數偏多
    }
    """
    if len(df) < 5:
        return {}

    vwap_ser   = _vwap(df)
    delta_ser  = _volume_delta(df)
    price      = float(df["Close"].iloc[-1])
    vwap_val   = float(vwap_ser.iloc[-1]) if not np.isnan(vwap_ser.iloc[-1]) else price
    above_vwap = price > vwap_val

    window = min(10, len(df))
    delta_sum  = float(delta_ser.iloc[-window:].sum())
    avg_vol    = float(df["Volume"].iloc[-20:].mean()) if len(df) >= 20 else float(df["Volume"].mean())
    last_vol   = float(df["Volume"].iloc[-1])
    vol_surge  = last_vol > 2.5 * avg_vol

    recent_high = float(df["High"].iloc[-window-1:-1].max()) if len(df) > window else float(df["High"].max())
    breakout    = price > recent_high and vol_surge

    pattern = _candle_pattern(df)

    # ── 評分（-4 ~ +4）─────────────────────────────────────────────────────
    score = 0
    if above_vwap:
        score += 1
    else:
        score -= 1

    if delta_sum > 0:
        score += 1
        vol_delta_signal = "buy"
    elif delta_sum < 0:
        score -= 1
        vol_delta_signal = "sell"
    else:
        vol_delta_signal = "neutral"

    if breakout:
        score += 1
    if vol_surge:
        score += 1  # 大量一律加分（方向已由 delta_sum 決定）
    if pattern == "hammer":
        score += 1
    elif pattern == "engulf_bull":
        score += 1
    elif pattern == "engulf_bear":
        score -= 1

    return {
        "price":            price,
        "vwap":             round(vwap_val, 2),
        "above_vwap":       above_vwap,
        "vol_delta_sum":    round(delta_sum, 0),
        "vol_delta_signal": vol_delta_signal,
        "avg_volume":       round(avg_vol, 0),
        "last_volume":      round(last_vol, 0),
        "vol_surge":        vol_surge,
        "breakout":         breakout,
        "pattern":          pattern,
        "score":            score,
    }


def _fetch_intraday(code: str) -> pd.DataFrame | None:
    """取得當日 2 分鐘 K 線資料。"""
    try:
        symbol = _symbol(code)
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=_PERIOD, interval=_INTERVAL)
        if df.empty:
            return None
        df = df.dropna(subset=["Close"])
        df = df[df["Volume"] > 0]
        return df
    except Exception as e:
        logger.debug(f"[DayTrader] {code} 資料擷取失敗: {e}")
        return None


# ── 主類別 ────────────────────────────────────────────────────────────────────

class DayTrader:
    """當沖紙上交易，每 2 分鐘 run_cycle() 一次。"""

    def __init__(self):
        self._lock  = threading.Lock()
        self._state = self._load_state()

    # ── 狀態 I/O ─────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        path = _state_path()
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return self._new_state()

    def _new_state(self) -> dict:
        return {
            "date":         str(date.today()),
            "cash":         DAY_INITIAL_CAPITAL,
            "holdings":     {},
            "trade_log":    [],
            "daily_pnl":    0.0,
            "watch_log":    {},  # code -> last analysis snapshot
        }

    def _save_state(self):
        try:
            with open(_state_path(), "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[DayTrader] state 儲存失敗: {e}")

    def _reset_if_new_day(self):
        today = str(date.today())
        if self._state.get("date") != today:
            logger.info(f"[DayTrader] 新的一天（{today}），重置帳戶")
            closed_pnl = self._state.get("daily_pnl", 0.0)
            self._state = self._new_state()
            self._state["date"] = today
            logger.info(f"[DayTrader] 昨日損益 {closed_pnl:+,.0f} TWD")

    # ── 時間判斷 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _now_tw() -> datetime:
        return datetime.now(TZ)

    def _is_market_open(self) -> bool:
        now = self._now_tw()
        if now.weekday() >= 5:
            return False
        return (9, 0) <= (now.hour, now.minute) <= (13, 30)

    def _is_buy_window(self) -> bool:
        """只在 09:05 ~ 11:00 買進，避免尾盤追高。"""
        now = self._now_tw()
        return (9, 5) <= (now.hour, now.minute) < (11, 0)

    def _is_force_close_time(self) -> bool:
        now = self._now_tw()
        return (now.hour, now.minute) >= (DAY_FORCE_CLOSE_H, DAY_FORCE_CLOSE_M)

    # ── 買賣執行（紙上）──────────────────────────────────────────────────────

    def _paper_buy(self, code: str, price: float, analysis: dict) -> bool:
        holdings = self._state["holdings"]
        if code in holdings:
            return False
        if len(holdings) >= DAY_MAX_POSITIONS:
            return False

        budget  = self._state["cash"] * DAY_PER_TRADE_PCT
        shares  = int(budget // (price * 1000)) * 1000  # 整張
        if shares <= 0:
            return False
        cost    = shares * price
        if cost > self._state["cash"]:
            return False

        self._state["cash"] -= cost
        holdings[code] = {
            "shares":      shares,
            "entry_price": price,
            "entry_time":  self._now_tw().strftime("%H:%M"),
            "cost":        cost,
            "stop_loss":   round(price * (1 + DAY_STOP_LOSS_PCT), 2),
            "take_profit": round(price * (1 + DAY_TAKE_PROFIT_PCT), 2),
        }
        self._log_trade(code, "BUY", shares, price, "當沖買入", analysis)
        logger.info(
            f"[DayTrader] 🟢 買入 {code}  {shares}股 @{price:.2f}"
            f"  score={analysis.get('score')}  delta={analysis.get('vol_delta_signal')}"
        )
        return True

    def _paper_sell(self, code: str, price: float, reason: str) -> float:
        holdings = self._state["holdings"]
        if code not in holdings:
            return 0.0
        h       = holdings.pop(code)
        shares  = h["shares"]
        cost    = h["cost"]
        proceeds= shares * price
        pnl     = proceeds - cost
        self._state["cash"]      += proceeds
        self._state["daily_pnl"] += pnl
        self._log_trade(code, "SELL", shares, price, reason, {})
        emoji = "🟩" if pnl >= 0 else "🟥"
        logger.info(
            f"[DayTrader] {emoji} 賣出 {code}  {shares}股 @{price:.2f}"
            f"  損益={pnl:+,.0f} TWD  原因={reason}"
        )
        return pnl

    def _log_trade(self, code: str, side: str, shares: int,
                   price: float, reason: str, analysis: dict):
        self._state["trade_log"].append({
            "time":     self._now_tw().strftime("%H:%M"),
            "code":     code,
            "side":     side,
            "shares":   shares,
            "price":    price,
            "reason":   reason,
            "score":    analysis.get("score"),
            "delta":    analysis.get("vol_delta_signal"),
            "pattern":  analysis.get("pattern"),
        })

    # ── 持倉檢查 ─────────────────────────────────────────────────────────────

    def _check_holdings(self):
        if not self._state["holdings"]:
            return
        force_close = self._is_force_close_time()
        for code in list(self._state["holdings"]):
            df = _fetch_intraday(code)
            price = float(df["Close"].iloc[-1]) if df is not None and not df.empty else None
            if price is None:
                continue
            h = self._state["holdings"][code]
            if force_close:
                self._paper_sell(code, price, "強制平倉 13:20")
                continue
            if price <= h["stop_loss"]:
                self._paper_sell(code, price, f"停損 {h['stop_loss']:.2f}")
            elif price >= h["take_profit"]:
                self._paper_sell(code, price, f"停利 {h['take_profit']:.2f}")

    # ── 盤況記錄（持續盯盤）──────────────────────────────────────────────────

    def _update_watch_log(self, code: str, analysis: dict):
        self._state["watch_log"][code] = {
            "time":  self._now_tw().strftime("%H:%M"),
            **{k: analysis[k] for k in
               ("price","vwap","above_vwap","vol_delta_signal",
                "last_volume","vol_surge","breakout","pattern","score")
               if k in analysis},
        }

    # ── 掃描新倉位 ────────────────────────────────────────────────────────────

    def _scan_entries(self):
        if not self._is_buy_window():
            return
        if len(self._state["holdings"]) >= DAY_MAX_POSITIONS:
            return

        candidates = []
        for code in WATCH_STOCKS:
            if code in self._state["holdings"]:
                continue
            df = _fetch_intraday(code)
            if df is None or len(df) < 5:
                continue
            analysis = _analyze_bars(df)
            if not analysis:
                continue
            self._update_watch_log(code, analysis)
            if analysis["score"] >= 3:
                candidates.append((code, analysis["score"], analysis))

        # 取得分最高的進場
        candidates.sort(key=lambda x: x[1], reverse=True)
        for code, score, analysis in candidates:
            if len(self._state["holdings"]) >= DAY_MAX_POSITIONS:
                break
            price = analysis["price"]
            logger.info(
                f"[DayTrader] 📊 {code} score={score}"
                f"  vwap={'↑' if analysis['above_vwap'] else '↓'}"
                f"  delta={analysis['vol_delta_signal']}"
                f"  surge={'✓' if analysis['vol_surge'] else '✗'}"
                f"  breakout={'✓' if analysis['breakout'] else '✗'}"
                f"  pattern={analysis['pattern']}"
                f"  @{price:.2f}"
            )
            self._paper_buy(code, price, analysis)

    # ── 只盯盤不交易（ENABLE_DAY_TRADING=false 時）────────────────────────────

    def _watch_only(self):
        """盤中持續記錄所有候選股票的量能狀態，不下單。"""
        updated = 0
        for code in WATCH_STOCKS:
            df = _fetch_intraday(code)
            if df is None or len(df) < 5:
                continue
            analysis = _analyze_bars(df)
            if not analysis:
                continue
            self._update_watch_log(code, analysis)
            updated += 1

        # 每輪印出高分候選
        tops = sorted(
            [(c, w) for c, w in self._state["watch_log"].items() if w.get("score", 0) >= 2],
            key=lambda x: x[1].get("score", 0),
            reverse=True
        )[:5]
        if tops:
            logger.info(
                "[DayTrader] 盯盤 Top5: " +
                "  ".join(
                    f"{c}({w.get('score')}) @{w.get('price'):.2f}"
                    f" delta={w.get('vol_delta_signal')}"
                    for c, w in tops
                )
            )

    # ── 主迴圈 ───────────────────────────────────────────────────────────────

    def run_cycle(self):
        with self._lock:
            self._reset_if_new_day()
            if not self._is_market_open():
                return

            if ENABLE_DAY_TRADING:
                self._check_holdings()
                self._scan_entries()
                self._log_portfolio_status()
            else:
                # 盯盤模式：不下單，但持續分析並記錄
                self._watch_only()

            self._save_state()

    def _log_portfolio_status(self):
        holdings = self._state["holdings"]
        if holdings:
            lines = []
            for code, h in holdings.items():
                df = _fetch_intraday(code)
                cur = float(df["Close"].iloc[-1]) if df is not None else h["entry_price"]
                pnl = (cur - h["entry_price"]) * h["shares"]
                lines.append(f"{code} {cur:.2f}({pnl:+,.0f})")
            logger.info(f"[DayTrader] 持倉: {', '.join(lines)}  現金={self._state['cash']:,.0f}")
        else:
            logger.info(f"[DayTrader] 無持倉  現金={self._state['cash']:,.0f}  今日損益={self._state['daily_pnl']:+,.0f}")

    # ── 外部查詢 ─────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        with self._lock:
            s = self._state
            return {
                "date":       s.get("date"),
                "cash":       s.get("cash", 0.0),
                "daily_pnl":  s.get("daily_pnl", 0.0),
                "positions":  len(s.get("holdings", {})),
                "holdings":   dict(s.get("holdings", {})),
                "trades_today": len([t for t in s.get("trade_log", [])
                                     if True]),  # same-day already
                "watch_count":  len(s.get("watch_log", {})),
                "mode":        "paper_day_trade" if ENABLE_DAY_TRADING else "watch_only",
            }
