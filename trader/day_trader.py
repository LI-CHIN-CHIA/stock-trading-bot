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
import time
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from utils.constants import OTC_SYMBOLS, TRADEABLE

logger = logging.getLogger(__name__)

# Lazy-load DayModel to avoid import errors if ai package isn't ready
try:
    from ai.day_model import DayModel as _DayModel
    _DAY_MODEL_AVAILABLE = True
except ImportError:
    _DAY_MODEL_AVAILABLE = False

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

# 手續費 & 證交稅（當沖）
_COMMISSION_RATE = 0.001425   # 券商手續費 0.1425%（可折扣）
_STT_RATE        = 0.0015     # 證交稅 0.15%（當沖減半）
_MIN_FEE         = 20         # 最低手續費 20 元

# 量能門檻：近20根1min K 均量至少 15,000 股（≈日均3000張@1min），進場那根至少均量 80%
DAY_MIN_AVG_VOLUME    = int(os.getenv("DAY_MIN_AVG_VOLUME",    "15000"))
DAY_MIN_ENTRY_VOL_PCT = float(os.getenv("DAY_MIN_ENTRY_VOL_PCT", "0.8"))

_INTERVAL = "1m"   # 1-minute bars for tighter monitoring
_PERIOD   = "1d"

WATCH_STOCKS: set[str] = TRADEABLE   # 用現有候選清單

# 大量告警門檻：當前K棒量 >= 均量 × N 倍觸發
VOL_SURGE_ALERT_X = float(os.getenv("VOL_SURGE_ALERT_X", "3.0"))


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

_FAST_MONITOR_INTERVAL = 10   # 持倉時每 10 秒查一次即時價


def _fast_price(code: str) -> float | None:
    """用 yfinance fast_info 快速取得最新成交價（不下載完整 K 棒）。"""
    try:
        info = yf.Ticker(_symbol(code)).fast_info
        p = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
        return float(p) if p else None
    except Exception:
        return None


class DayTrader:
    """當沖紙上交易，每 1 分鐘 run_cycle() 一次；持倉時背景執行緒每 10 秒盯盤。"""

    def __init__(self):
        self._lock         = threading.RLock()  # RLock 允許同執行緒重入（run_cycle → _paper_sell）
        self._state        = self._load_state()
        self._model        = self._load_model()
        self._monitor_stop = threading.Event()
        self._monitor_thr: threading.Thread | None = None
        # 若啟動時已有持倉（跨週期 reload），立即啟動背景監控
        if self._state.get("holdings"):
            self._start_fast_monitor()

    # ── 10 秒背景盯盤 ─────────────────────────────────────────────────────────

    def _start_fast_monitor(self):
        if self._monitor_thr and self._monitor_thr.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thr = threading.Thread(
            target=self._fast_monitor_loop, daemon=True, name="DayTrader-FastMonitor"
        )
        self._monitor_thr.start()
        logger.info("[DayTrader] ⏱ 10秒盯盤執行緒已啟動")

    def _stop_fast_monitor(self):
        self._monitor_stop.set()

    def _fast_monitor_loop(self):
        """背景執行緒：有持倉時每 10 秒檢查停損/停利。
        設計原則：網路 I/O (_fast_price) 在鎖外執行，避免與 run_cycle 互相卡死。
        """
        while not self._monitor_stop.is_set():
            time.sleep(_FAST_MONITOR_INTERVAL)
            # 快速快照（短暫持鎖）
            with self._lock:
                holdings_snap = dict(self._state.get("holdings", {}))
            if not holdings_snap:
                self._monitor_stop.set()
                logger.info("[DayTrader] ⏱ 無持倉，10秒盯盤執行緒停止")
                break
            now = self._now_tw()
            if not ((9, 0) <= (now.hour, now.minute) <= (13, 30)):
                continue
            # 網路取價在鎖外（避免佔鎖 0.5~2 秒）
            for code in list(holdings_snap):
                try:
                    price = _fast_price(code)
                    if price is None:
                        continue
                    h = holdings_snap[code]
                    entry = h["entry_price"]
                    pct   = (price - entry) / entry
                    sl    = h["stop_loss"]
                    tp    = h["take_profit"]

                    # 推播持倉狀態（每分鐘第一次 tick，避免過多訊息）
                    if now.second < 15:
                        try:
                            from utils.telegram_notify import send, _name
                            send(
                                f"📍 <b>釘盤 {_name(code)}</b>  {price:.2f}"
                                f"  ({pct:+.2%})\n"
                                f"SL {sl:.2f} / TP {tp:.2f}",
                                key=f"watch:{code}:{now.minute}",
                                cooldown=50,
                            )
                        except Exception:
                            pass

                    # 觸發停損/停利（_paper_sell 內部持鎖）
                    if price <= sl:
                        self._paper_sell(code, price, f"停損 {sl:.2f}")
                    elif price >= tp:
                        self._paper_sell(code, price, f"停利 {tp:.2f}")
                    # 即將觸碰停損 (距 SL < 0.3%) → 緊急預警
                    elif (price - sl) / entry < 0.003:
                        try:
                            from utils.telegram_notify import send, _name
                            send(f"⚠️ <b>接近停損 {_name(code)}</b>  現價 {price:.2f}  SL {sl:.2f}",
                                 key=f"nearsl:{code}", cooldown=120)
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(f"[DayTrader] fast_monitor {code} 異常: {e}")
            with self._lock:
                self._save_state()

    def _load_model(self):
        if not _DAY_MODEL_AVAILABLE:
            return None
        try:
            m = _DayModel.load_latest()
            if m:
                logger.info(f"[DayTrader] 當沖 model 已載入，trained_at={m.trained_at}")
            else:
                logger.info("[DayTrader] 無當沖 model，使用純規則選股")
            return m
        except Exception as e:
            logger.warning(f"[DayTrader] model 載入失敗: {e}")
            return None

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
            # 若有未平倉部位，先告警再清除（#4：避免靜默吞掉昨日倉位）
            leftover = self._state.get("holdings", {})
            if leftover:
                logger.warning(f"[DayTrader] ⚠️ 昨日有 {len(leftover)} 個未平倉部位被清除: {list(leftover)}")
                try:
                    from utils.telegram_notify import send
                    send(f"⚠️ <b>當沖未平倉警告</b>\n昨日收盤後仍有 {len(leftover)} 筆持倉被清除:\n"
                         + "\n".join(f"  {c}: {h['shares']}股 @{h['entry_price']:.2f}"
                                     for c, h in leftover.items()),
                         key=f"leftover:{today}", cooldown=0)
                except Exception:
                    pass
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
        """09:05 ~ 12:30 均可買進（大量突破盤中隨時出現）。"""
        now = self._now_tw()
        return (9, 5) <= (now.hour, now.minute) < (12, 30)

    def _is_force_close_time(self) -> bool:
        now = self._now_tw()
        return (now.hour, now.minute) >= (DAY_FORCE_CLOSE_H, DAY_FORCE_CLOSE_M)

    # ── 買賣執行（紙上）──────────────────────────────────────────────────────

    def _paper_buy(self, code: str, price: float, analysis: dict,
                   predicted_gain: float | None = None) -> bool:
        with self._lock:
            return self.__paper_buy(code, price, analysis, predicted_gain)

    def __paper_buy(self, code: str, price: float, analysis: dict,
                    predicted_gain: float | None = None) -> bool:
        holdings = self._state["holdings"]
        if code in holdings:
            return False
        if len(holdings) >= DAY_MAX_POSITIONS:
            return False

        budget  = self._state["cash"] * DAY_PER_TRADE_PCT
        # 先嘗試整張（1000股倍數），不足1張才買零股
        shares = int(budget // (price * 1000)) * 1000
        if shares <= 0:
            shares = int(budget // price)   # 零股：至少 1 股
        if shares <= 0:
            return False
        cost     = shares * price
        buy_fee  = max(_MIN_FEE, round(cost * _COMMISSION_RATE))
        total_cost = cost + buy_fee
        if total_cost > self._state["cash"]:
            return False

        self._state["cash"] -= total_cost
        holdings[code] = {
            "shares":         shares,
            "entry_price":    price,
            "entry_time":     self._now_tw().strftime("%H:%M"),
            "cost":           total_cost,   # 含買進手續費
            "stop_loss":      round(price * (1 + DAY_STOP_LOSS_PCT), 2),
            "take_profit":    round(price * (1 + DAY_TAKE_PROFIT_PCT), 2),
            "predicted_gain": round(predicted_gain * 100, 3) if predicted_gain is not None else None,
        }
        self._log_trade(code, "BUY", shares, price, "當沖買入", analysis, predicted_gain)
        gain_str = f"  model={predicted_gain*100:.2f}%" if predicted_gain is not None else ""
        logger.info(
            f"[DayTrader] 🟢 買入 {code}  {shares}股 @{price:.2f}"
            f"  score={analysis.get('score')}  delta={analysis.get('vol_delta_signal')}"
            f"{gain_str}"
        )
        try:
            from utils.telegram_notify import notify_buy
            notify_buy("當沖", code, shares, price,
                       f"score={analysis.get('score')} model={predicted_gain*100:.2f}%" if predicted_gain else f"score={analysis.get('score')}")
        except Exception:
            pass
        # 有持倉 → 啟動 10 秒盯盤執行緒
        self._start_fast_monitor()
        return True

    def _paper_sell(self, code: str, price: float, reason: str) -> float:
        with self._lock:
            return self.__paper_sell(code, price, reason)

    def __paper_sell(self, code: str, price: float, reason: str) -> float:
        holdings = self._state["holdings"]
        if code not in holdings:
            return 0.0
        h       = holdings.pop(code)
        shares  = h["shares"]
        cost    = h["cost"]
        proceeds = shares * price
        sell_fee = max(_MIN_FEE, round(proceeds * _COMMISSION_RATE))
        stt      = round(proceeds * _STT_RATE)
        net_proceeds = proceeds - sell_fee - stt
        pnl     = net_proceeds - cost
        self._state["cash"]      += net_proceeds
        self._state["daily_pnl"] += pnl
        self._log_trade(code, "SELL", shares, price, reason, {})
        emoji = "🟩" if pnl >= 0 else "🟥"
        logger.info(
            f"[DayTrader] {emoji} 賣出 {code}  {shares}股 @{price:.2f}"
            f"  手續費={sell_fee+stt:,.0f}  損益={pnl:+,.0f} TWD  原因={reason}"
        )
        try:
            from utils.telegram_notify import notify_sell
            notify_sell("當沖", code, shares, price, pnl, reason)
        except Exception:
            pass
        return pnl

    def _log_trade(self, code: str, side: str, shares: int,
                   price: float, reason: str, analysis: dict,
                   predicted_gain: float | None = None):
        entry = {
            "time":            self._now_tw().strftime("%H:%M"),
            "code":            code,
            "side":            side,
            "shares":          shares,
            "price":           price,
            "reason":          reason,
            "score":           analysis.get("score"),
            "delta":           analysis.get("vol_delta_signal"),
            "pattern":         analysis.get("pattern"),
            "predicted_gain":  round(predicted_gain * 100, 3) if predicted_gain is not None else None,
        }
        self._state["trade_log"].append(entry)

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
            avg_vol  = analysis.get("avg_volume", 0)
            last_vol = analysis.get("last_volume", 0)
            vol_ratio = (last_vol / avg_vol) if avg_vol > 0 else 0
            # 大量爆發（≥3x 均量）降低 score 門檻至 2；一般需 score≥3
            score_threshold = 2 if vol_ratio >= VOL_SURGE_ALERT_X else 3
            if analysis["score"] >= score_threshold:
                # 量能硬性門檻：流動性不足直接跳過
                if avg_vol < DAY_MIN_AVG_VOLUME:
                    logger.debug(f"[DayTrader] {code} 均量不足 ({avg_vol:.0f} < {DAY_MIN_AVG_VOLUME})，跳過")
                    continue
                if last_vol < avg_vol * DAY_MIN_ENTRY_VOL_PCT:
                    logger.debug(f"[DayTrader] {code} 進場K棒量太小 ({last_vol:.0f} < avg×{DAY_MIN_ENTRY_VOL_PCT})，跳過")
                    continue
                candidates.append((code, analysis["score"], analysis))

        # 取得分最高的進場
        candidates.sort(key=lambda x: x[1], reverse=True)
        for code, score, analysis in candidates:
            if len(self._state["holdings"]) >= DAY_MAX_POSITIONS:
                break
            price = analysis["price"]

            # ── Model confirmation ────────────────────────────────────────────
            predicted_gain: float | None = None
            if self._model is not None:
                df_for_model = _fetch_intraday(code)
                if df_for_model is not None:
                    go, predicted_gain = self._model.is_buy_signal(df_for_model, analysis)
                    if not go:
                        logger.info(
                            f"[DayTrader] ⛔ {code} score={score} "
                            f"model_pred={predicted_gain*100:.2f}% < threshold, skip"
                        )
                        continue

            logger.info(
                f"[DayTrader] 📊 {code} score={score}"
                f"  vwap={'↑' if analysis['above_vwap'] else '↓'}"
                f"  delta={analysis['vol_delta_signal']}"
                f"  surge={'✓' if analysis['vol_surge'] else '✗'}"
                f"  breakout={'✓' if analysis['breakout'] else '✗'}"
                f"  pattern={analysis['pattern']}"
                + (f"  model={predicted_gain*100:.2f}%" if predicted_gain is not None else "")
                + f"  @{price:.2f}"
            )
            self._paper_buy(code, price, analysis, predicted_gain)

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

    def _alert_volume_surges(self):
        """掃描大量股票並推播 Telegram（不受 ENABLE_DAY_TRADING 限制，純告警）。"""
        try:
            from utils.telegram_notify import notify_volume_surge
        except ImportError:
            return
        for code in WATCH_STOCKS:
            try:
                df = _fetch_intraday(code)
                if df is None or len(df) < 10:
                    continue
                vol = df["Volume"]
                avg_vol = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float(vol.mean())
                last_vol = float(vol.iloc[-1])
                if avg_vol <= 0:
                    continue
                ratio = last_vol / avg_vol
                if ratio >= VOL_SURGE_ALERT_X:
                    price = float(df["Close"].iloc[-1])
                    analysis = _analyze_bars(df) or {}
                    score = analysis.get("score", 0)
                    action = "⚡進場候選" if score >= 3 else "觀察"
                    notify_volume_surge(code, price, ratio, score, action)
                    logger.info(f"[DayTrader] 📢 大量 {code} vol_ratio={ratio:.1f}x price={price:.2f} score={score}")
            except Exception:
                pass

    def run_cycle(self):
        # 只在 reset/is_open 短暫持鎖，避免長時間佔鎖卡住 10 秒盯盤執行緒
        with self._lock:
            self._reset_if_new_day()
            if not self._is_market_open():
                return

        # 網路 I/O（下載 K 棒）在鎖外執行
        self._alert_volume_surges()

        if ENABLE_DAY_TRADING:
            self._check_holdings()
            self._scan_entries()
            self._log_portfolio_status()
        else:
            self._watch_only()

        with self._lock:
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
