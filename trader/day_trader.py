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
import psycopg2
import pytz
import yfinance as yf

from utils.constants import OTC_SYMBOLS, TRADEABLE

logger = logging.getLogger(__name__)

# yfinance 開盤前抓不到資料時會發 ERROR-level "possibly delisted" 訊息，對我們是雜訊
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

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
ENABLE_DAY_TRADING   = os.getenv("ENABLE_DAY_TRADING",   "false").lower() == "true"
ENABLE_SHORT_TRADING = os.getenv("ENABLE_SHORT_TRADING", "false").lower() == "true"

# 手續費 & 證交稅（當沖）
# 手續費折數：0.28 = 2.8 折；1.0 = 無折扣。依券商實際折數設定，回測與紙上損益才會貼近真實
DAY_FEE_DISCOUNT = float(os.getenv("DAY_FEE_DISCOUNT", "1.0"))
_COMMISSION_RATE = 0.001425 * DAY_FEE_DISCOUNT   # 券商手續費 0.1425% × 折數
_STT_RATE        = 0.0015     # 證交稅 0.15%（當沖減半）
_MIN_FEE         = 20         # 最低手續費 20 元

# 真實下單旗標
DAY_REAL_TRADING  = os.getenv("DAY_REAL_TRADING", "false").lower() == "true"
TRADING_USER_ID   = int(os.getenv("TRADING_USER_ID", "0"))   # 0 = 停用 DB 設定查詢
DAY_MAX_BUDGET    = float(os.getenv("DAY_MAX_BUDGET",    "50000"))   # 單筆最高金額 NTD（真實）
DAY_SHADOW_BUDGET = float(os.getenv("DAY_SHADOW_BUDGET", "160000"))  # 影子帳戶單筆預算（已棄用，保留相容）
_SHADOW_CAPS_STR  = os.getenv("DAY_SHADOW_CAPITALS", "500000,1000000,1500000,2000000,2500000,3000000,3500000,4000000,4500000,5000000")
SHADOW_CAPITALS: list[int] = [int(x.strip()) for x in _SHADOW_CAPS_STR.split(",") if x.strip().isdigit()]

# 移動停利：達 +1.5% 後啟動，trailing distance 1%
DAY_TRAIL_ACTIVATE_PCT = float(os.getenv("DAY_TRAIL_ACTIVATE_PCT", "0.015"))
DAY_TRAIL_DISTANCE_PCT = float(os.getenv("DAY_TRAIL_DISTANCE_PCT", "0.010"))
# 時間衰退：持倉 N 分鐘仍在 ±band 橫盤 → 主動出場
DAY_TIME_DECAY_MIN  = int(os.getenv("DAY_TIME_DECAY_MIN",  "30"))
DAY_TIME_DECAY_BAND = float(os.getenv("DAY_TIME_DECAY_BAND", "0.005"))
# 重播回測（backtest/replay_live.py，30 日）：VWAP 反向出場勝率 0%、空單時間衰退勝率 14%，
# 關閉後多空期望值皆改善，故預設關閉；保留開關以便對照。
DAY_VWAP_EXIT        = os.getenv("DAY_VWAP_EXIT",        "false").lower() == "true"
DAY_SHORT_TIME_DECAY = os.getenv("DAY_SHORT_TIME_DECAY", "false").lower() == "true"

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
    # 指數加權 Volume Delta：近期 K 棒權重較大（λ=1 → 最後一根權重最大）
    raw_delta = delta_ser.iloc[-window:].values
    weights   = np.exp(np.linspace(0, 1, len(raw_delta)))
    delta_sum = float((raw_delta * weights).sum())

    avg_vol    = float(df["Volume"].iloc[-20:].mean()) if len(df) >= 20 else float(df["Volume"].mean())
    last_vol   = float(df["Volume"].iloc[-1])
    vol_surge  = last_vol > 2.5 * avg_vol

    recent_high = float(df["High"].iloc[-window-1:-1].max()) if len(df) > window else float(df["High"].max())
    breakout    = price > recent_high and vol_surge

    pattern = _candle_pattern(df)

    # ── 評分（base -5 ~ +6，不含大盤加成）──────────────────────────────────
    # 注意：大盤加成 (±1/±2) 在 _scan_entries 加入，使評分與市場方向解耦
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
        score += 2   # 突破 = 價格 + 大量雙確認，可信度最高給 +2
    if vol_surge:
        score += 1   # 大量本身也加分（方向已由 delta_sum 決定）
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


_user_config_cache: dict = {
    "ts":             0.0,
    "auto_trade":     True,
    "enable_day":     True,    # App「當沖（做多）」開關
    "enable_short":   True,    # App「反沖（做空）」開關
    "stop_loss_pct":  DAY_STOP_LOSS_PCT,
    "take_profit_pct":DAY_TAKE_PROFIT_PCT,
    "day_budget":     DAY_INITIAL_CAPITAL * DAY_PER_TRADE_PCT,
    "max_positions":  DAY_MAX_POSITIONS,
    # 模擬帳戶額度清單：第一個是主要模擬帳戶，其餘為影子帳戶（App 設定優先）
    "paper_budgets":  [int(DAY_INITIAL_CAPITAL)] + (SHADOW_CAPITALS if DAY_REAL_TRADING else []),
}


def _fetch_user_config() -> dict:
    """查詢 platform DB 中該 user 的所有交易設定。快取 60 秒。
    若 TRADING_USER_ID=0 或 DB 不可達，回傳模組預設值（不中斷交易）。
    """
    if not TRADING_USER_ID:
        return _user_config_cache
    now = time.time()
    if now - _user_config_cache["ts"] < 60:
        return _user_config_cache
    try:
        conn = psycopg2.connect(
            host=os.getenv("PLATFORM_DB_HOST", "postgres"),
            port=int(os.getenv("PLATFORM_DB_PORT", "5432")),
            dbname=os.getenv("PLATFORM_DB_NAME", "trading"),
            user=os.getenv("PLATFORM_DB_USER", "trading"),
            password=os.getenv("PLATFORM_DB_PASS", ""),
            connect_timeout=3,
        )
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT auto_trade, stop_loss_pct, take_profit_pct, day_budget, max_positions,"
                    " paper_budgets, enable_day, enable_short"
                    " FROM platform_configs WHERE user_id = %s LIMIT 1",
                    (TRADING_USER_ID,),
                )
                row = cur.fetchone()
        conn.close()
        if row:
            prev_auto = _user_config_cache["auto_trade"]
            _user_config_cache.update({
                "ts":             now,
                "auto_trade":     bool(row[0]),
                "stop_loss_pct":  float(row[1]),
                "take_profit_pct":float(row[2]),
                "day_budget":     float(row[3]),
                "max_positions":  int(row[4]),
                "enable_day":     bool(row[6]),
                "enable_short":   bool(row[7]),
            })
            budgets = [int(float(x)) for x in str(row[5] or "").split(",") if x.strip()]
            if budgets:
                _user_config_cache["paper_budgets"] = budgets
            if not row[0] and prev_auto:
                logger.info(f"[DayTrader] 自動下單已由 user_id={TRADING_USER_ID} 在 App 關閉")
        else:
            _user_config_cache["ts"] = now
    except Exception as e:
        logger.debug(f"[DayTrader] DB 查詢 user_config 失敗，沿用上次設定: {e}")
    return _user_config_cache


def _fetch_user_auto_trade() -> bool:
    return _fetch_user_config()["auto_trade"]


_mkt_dir_cache: dict = {"ts": 0.0, "dir": "FLAT", "gap": 0.0}
_us_overnight_cache: dict = {"date": "", "dir": "NEUTRAL", "spx": 0.0, "ndx": 0.0}


def _fetch_mkt_direction() -> tuple[str, float, str]:
    """Return ('UP'|'DOWN'|'FLAT', gap_pct, intraday_signal). Cached 5 min.

    gap_pct       = (cur - prev_close) / prev_close
    intraday_signal:
      'FADE' = 盤中從高點回落 >1.5%（開高回落）
      'PUSH' = 盤中持續推升
      'NONE' = 無明顯盤中走勢
    """
    now = time.time()
    if now - _mkt_dir_cache["ts"] < 300:
        return _mkt_dir_cache["dir"], _mkt_dir_cache["gap"], _mkt_dir_cache.get("intraday", "NONE")
    try:
        df = yf.Ticker("^TWII").history(period="5d", interval="1d")
        if len(df) < 2:
            return "FLAT", 0.0, "NONE"
        prev_close = float(df["Close"].iloc[-2])
        cur = getattr(yf.Ticker("^TWII").fast_info, "last_price", None)
        if not cur:
            return "FLAT", 0.0, "NONE"
        cur = float(cur)
        gap = (cur - prev_close) / prev_close
        d = "UP" if gap >= 0.003 else ("DOWN" if gap <= -0.003 else "FLAT")

        # 盤中走勢：取今日 5m K 棒，判斷是否從高點回落
        intraday_signal = "NONE"
        try:
            idf = yf.Ticker("^TWII").history(period="1d", interval="5m")
            if not idf.empty:
                today_high = float(idf["High"].max())
                pullback = (cur - today_high) / today_high
                if pullback <= -0.015:          # 從高點回落 ≥1.5% → FADE
                    intraday_signal = "FADE"
                elif cur > float(idf["Open"].iloc[0]) * 1.005:
                    intraday_signal = "PUSH"    # 盤中持續推升
        except Exception:
            pass

        _mkt_dir_cache.update({"ts": now, "dir": d, "gap": gap, "intraday": intraday_signal})
        return d, gap, intraday_signal
    except Exception:
        return "FLAT", 0.0, "NONE"


def _fetch_us_overnight() -> tuple[str, float, float]:
    """Return ('BULL'|'BEAR'|'NEUTRAL', spx_pct, ndx_pct). Cached per calendar day.

    台灣交易時段（9:00-13:30 UTC+8）美股已收盤，取前一個美股交易日表現
    作為當日台股開盤方向的結構性參考。
    """
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    if _us_overnight_cache["date"] == today:
        return _us_overnight_cache["dir"], _us_overnight_cache["spx"], _us_overnight_cache["ndx"]
    try:
        spx_df = yf.Ticker("^GSPC").history(period="5d", interval="1d")
        ndx_df = yf.Ticker("^IXIC").history(period="5d", interval="1d")
        if len(spx_df) < 2 or len(ndx_df) < 2:
            return "NEUTRAL", 0.0, 0.0
        spx_pct = float((spx_df["Close"].iloc[-1] - spx_df["Close"].iloc[-2]) / spx_df["Close"].iloc[-2])
        ndx_pct = float((ndx_df["Close"].iloc[-1] - ndx_df["Close"].iloc[-2]) / ndx_df["Close"].iloc[-2])
        avg = (spx_pct + ndx_pct) / 2
        d = "BULL" if avg >= 0.003 else ("BEAR" if avg <= -0.003 else "NEUTRAL")
        _us_overnight_cache.update({"date": today, "dir": d, "spx": spx_pct, "ndx": ndx_pct})
        return d, spx_pct, ndx_pct
    except Exception:
        return "NEUTRAL", 0.0, 0.0


def _mkt_score(gap: float, us_dir: str = "NEUTRAL", intraday: str = "NONE") -> tuple[int, int]:
    """Return (long_bonus, short_bonus).

    語意：
      long_bonus  > 0 → 鼓勵做多（eff_long 更高，更容易過 >= 門檻）
      short_bonus > 0 → 抑制做空（eff_short 更高，更難過 <= 負門檻）
      short_bonus < 0 → 鼓勵做空（eff_short 更負，更容易過 <= 負門檻）

    大盤 UP → short_bonus 為正（抑制空單）；大盤 DOWN → short_bonus 為負（鼓勵空單）。
    注意：此符號邏輯與 long_bonus 相同，均為「順市場方向為正」。
    """
    if gap >= 0.010:
        base_long, base_short = +2, +2    # 大盤強彈：助多抑空
    elif gap >= 0.003:
        base_long, base_short = +1, +1
    elif gap <= -0.010:
        base_long, base_short = -2, -2   # 大盤強跌：抑多助空
    elif gap <= -0.003:
        base_long, base_short = -1, -1
    else:
        base_long, base_short = 0, 0

    # 盤中高點回落修正：開高後盤中明顯回落 → 空方機會增加，降低 1 點抑制
    if intraday == "FADE" and gap > 0.003:
        base_short -= 1

    # 美股方向衝突修正
    if gap >= 0.003 and us_dir == "BEAR":
        base_long = max(0, base_long - 1)        # 開高但美股熊：降低多方信心
    elif gap <= -0.003 and us_dir == "BULL":
        base_short = min(0, base_short + 1)      # 開低但美股牛：降低空方鼓勵力道

    return base_long, base_short


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


def _paper_budgets() -> list[int]:
    out: list[int] = []
    for b in _fetch_user_config()["paper_budgets"]:
        if b > 0 and b not in out:
            out.append(int(b))
    return out or [int(DAY_INITIAL_CAPITAL)]


def _new_shadow(cap: int) -> dict:
    return {"initial": int(cap), "cash": float(cap), "pnl": 0.0, "holdings": {}}


def _new_shadows() -> dict:
    """額外的模擬帳戶（設定清單中第一個以外的額度）。"""
    return {str(cap): _new_shadow(cap) for cap in _paper_budgets()[1:]}


def _paper_account(cap: float) -> str:
    return f"paper-{int(cap)}"


class DayTrader:
    """當沖紙上交易，每 1 分鐘 run_cycle() 一次；持倉時背景執行緒每 10 秒盯盤。"""

    def __init__(self):
        self._lock         = threading.RLock()
        self._state        = self._load_state()
        self._model        = self._load_model()
        self._monitor_stop = threading.Event()
        self._monitor_thr: threading.Thread | None = None
        # 真實下單 SDK 注入（預設為 None，即純紙上模式）
        self._place_order_fn:     object = None  # callable(code, side, shares, price) -> order_no|None
        self._check_filled_fn:    object = None  # callable(code, order_no) -> (qty, fill_price)
        self._cancel_order_fn:    object = None  # callable(code, order_no)
        self._inventories_fn:     object = None  # callable() -> SDK result (inventories)
        self._unrealized_fn:      object = None  # callable() -> SDK result (unrealized_gains)
        self._daytrade_info_fn:   object = None  # callable(code) -> SDK result (daytrade_and_stock_info)
        self._daytrade_cache:     dict   = {}    # code -> {date, status, quota, eligible}
        self._pending_buy_orders:  dict = {}   # order_no -> {code, shares, price, reserved_cash, analysis, predicted_gain, placed_at}
        self._pending_sell_orders: dict = {}   # order_no -> {code, shares, price, reason, placed_at}
        # 從 state 還原（啟動後恢復未成交委託，避免重啟後遺漏）
        self._pending_buy_orders  = self._state.pop("pending_buy",  {})
        self._pending_sell_orders = self._state.pop("pending_sell", {})
        self._last_monitoring_tg_ts: float = 0.0  # 上次盯盤 Telegram 發送時間
        # 若啟動時已有持倉（跨週期 reload），立即啟動背景監控
        if self._state.get("holdings"):
            self._start_fast_monitor()

    # ── SDK 注入（真實下單）─────────────────────────────────────────────────────

    def set_real_trading(self, place_order_fn, check_filled_fn, cancel_order_fn=None,
                         inventories_fn=None, unrealized_fn=None, daytrade_info_fn=None):
        """
        注入 TradingBot 的下單 / 查詢成交 / 取消委託 / 庫存查詢 / 當沖資格查詢函式。
        呼叫後若 DAY_REAL_TRADING=true，買賣將透過 Fubon SDK 執行。
        """
        self._place_order_fn    = place_order_fn
        self._check_filled_fn   = check_filled_fn
        self._cancel_order_fn   = cancel_order_fn
        self._inventories_fn    = inventories_fn
        self._unrealized_fn     = unrealized_fn
        self._daytrade_info_fn  = daytrade_info_fn
        mode = "真實下單" if DAY_REAL_TRADING else "紙上（DAY_REAL_TRADING=false）"
        logger.info(f"[DayTrader] set_real_trading 完成，模式={mode}  單筆上限={DAY_MAX_BUDGET:,.0f}")

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
            for key in list(holdings_snap):
                h = holdings_snap[key]
                is_short = h.get("direction") == "short"
                code = h.get("code", key) if is_short else key
                try:
                    price = _fast_price(code)
                    if price is None:
                        continue

                    if is_short:
                        sl = h["stop_loss"]
                        tp = h["take_profit"]
                        pct = (h["entry_price"] - price) / h["entry_price"]
                        if now.second < 15:
                            try:
                                from utils.telegram_notify import send, _name
                                send(
                                    f"📍 <b>反沖盯盤 {_name(code)}</b>  {price:.2f}"
                                    f"  ({pct:+.2%})\n"
                                    f"SL {sl:.2f} / TP {tp:.2f}",
                                    key=f"watch_s:{code}:{now.minute}",
                                    cooldown=50,
                                )
                            except Exception:
                                pass
                        if price >= sl:
                            self._paper_cover(code, price, f"停損 {sl:.2f}")
                        elif price <= tp:
                            self._paper_cover(code, price, f"停利 {tp:.2f}")
                        continue

                    # 多方邏輯
                    # 移動停利更新（在鎖內；holdings_snap[key] 與 _state["holdings"][key] 共享引用）
                    with self._lock:
                        self._update_trailing_stop(code, price)

                    h = holdings_snap[key]
                    entry = h["entry_price"]
                    pct   = (price - entry) / entry
                    sl    = h["stop_loss"]   # 可能已被 _update_trailing_stop 上移
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
                    state = json.load(f)
                # 遷移：舊單一影子帳戶欄位 → 新多帳戶 shadows dict
                if "shadow_cash" in state or "shadow_holdings" in state:
                    state.pop("shadow_cash",     None)
                    state.pop("shadow_pnl",      None)
                    state.pop("shadow_holdings", None)
                state.setdefault("shadows", _new_shadows())
                return state
            except Exception:
                pass
        return self._new_state()

    def _new_state(self) -> dict:
        return {
            "date":           str(datetime.now(TZ).date()),
            "initial":        float(_paper_budgets()[0]),
            "cash":           float(_paper_budgets()[0]),
            "holdings":       {},
            "trade_log":      [],
            "daily_pnl":      0.0,
            "watch_log":      {},
            "shadows":        _new_shadows(),
            "short_cooldown": {},   # {code: last_cover_ts}
        }

    def _account(self, is_real: bool = False) -> str:
        if is_real:
            return "real"
        return _paper_account(self._state.get("initial", DAY_INITIAL_CAPITAL))

    def _trade_budget(self, for_real: bool = False) -> float:
        """單筆預算：模擬 = 帳戶額度 × 比例；真實下單另受 DAY_MAX_BUDGET 限制。"""
        initial = self._state.get("initial", DAY_INITIAL_CAPITAL)
        budget = min(initial * DAY_PER_TRADE_PCT, self._state["cash"])
        return min(budget, DAY_MAX_BUDGET) if for_real else budget

    def _sync_shadows(self):
        """依 App 設定增減影子帳戶（需在 _lock 內呼叫）。有持倉的帳戶等平倉後才移除。"""
        shadows = self._state.setdefault("shadows", {})
        main_cap = int(self._state.get("initial", DAY_INITIAL_CAPITAL))
        want = {str(c) for c in _paper_budgets()[1:] if c != main_cap}
        for k in want - set(shadows):
            shadows[k] = _new_shadow(int(k))
            logger.info(f"[DayTrader] 新增模擬帳戶 {int(k)//10000}萬")
        for k in [k for k, acc in shadows.items() if k not in want and not acc["holdings"]]:
            del shadows[k]

    def _report_shadow(self, acc: dict, code: str, side: str, shares: int, price: float,
                       reason: str, pnl: float = 0.0):
        """影子帳戶成交與持倉回報到 platform（App 依帳戶區分顯示）。"""
        account = _paper_account(acc["initial"])
        try:
            from utils.platform_signal import report_trade, report_position
            report_trade(user_id=TRADING_USER_ID, code=code, action=side, shares=shares,
                         price=price, pnl=pnl, reason=reason, is_real=False, account=account)
            direction = "SHORT" if side in ("SHORT", "COVER") else "LONG"
            report_position(TRADING_USER_ID, code, direction,
                            shares if side in ("BUY", "SHORT") else 0, price,
                            is_real=False, account=account)
        except Exception:
            pass

    def _save_state(self):
        try:
            # 寫入前把 pending orders 一併存入 state（啟動後可還原）
            self._state["pending_buy"]  = dict(self._pending_buy_orders)
            self._state["pending_sell"] = dict(self._pending_sell_orders)
            with open(_state_path(), "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[DayTrader] state 儲存失敗: {e}")

    def _reset_if_new_day(self):
        today = str(datetime.now(TZ).date())
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
            self._pending_buy_orders  = {}
            self._pending_sell_orders = {}
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
        # 也排除正在等待成交的買單（避免重複下單）
        if any(o["code"] == code and o["side"] == "BUY" for o in self._pending_buy_orders.values()):
            return False
        if len(holdings) >= DAY_MAX_POSITIONS:
            return False

        # 單筆預算：資金比例 與 DAY_MAX_BUDGET 取較小值
        real_path = (DAY_REAL_TRADING and self._place_order_fn is not None
                     and _fetch_user_auto_trade())
        budget  = self._trade_budget(for_real=real_path)
        # 當沖只能整張（零股不支援當沖），不足1張直接跳過
        shares = int(budget // (price * 1000)) * 1000
        if shares <= 0:
            logger.debug(f"[DayTrader] ⛔ {code} 預算不足一張  需={price*1000:,.0f}  可用={budget:,.0f}，跳過")
            return False
        cost     = shares * price
        buy_fee  = max(_MIN_FEE, round(cost * _COMMISSION_RATE))
        total_cost = cost + buy_fee
        if total_cost > self._state["cash"]:
            return False

        # 真實下單路徑（非阻塞，等待下次 _confirm_pending_orders 確認成交）
        if DAY_REAL_TRADING and self._place_order_fn is not None:
            if not real_path:
                # App 已關閉自動下單 → fall-through 到紙上交易
                logger.info(f"[DayTrader] ⏸ {code} 自動下單已關閉，改為紙上交易")
            else:
                order_no = self._place_order_fn(code, "BUY", shares, price)
                if not order_no:
                    logger.warning(f"[DayTrader] ❌ {code} 真實買單送出失敗")
                    return False
                # 預扣資金，成交後才建倉；若失敗/超時則退款
                self._state["cash"] -= total_cost
                self._pending_buy_orders[order_no] = {
                    "side": "BUY", "code": code, "shares": shares, "price": price,
                    "reserved_cash": total_cost, "analysis": analysis,
                    "predicted_gain": predicted_gain, "placed_at": time.time(),
                }
                logger.info(
                    f"[DayTrader] ⏳ 真實買單送出 {code} {shares}股 @{price:.2f}"
                    f"  上限={DAY_MAX_BUDGET:,.0f}  單號={order_no}"
                )
                return True

        _cfg = _fetch_user_config()
        _sl  = _cfg["stop_loss_pct"]
        _tp  = _cfg["take_profit_pct"]
        self._state["cash"] -= total_cost
        holdings[code] = {
            "shares":         shares,
            "entry_price":    price,
            "entry_time":     self._now_tw().strftime("%H:%M"),
            "entry_ts":       time.time(),
            "is_real":        False,  # 紙上買入，賣出不走真實委託
            "cost":           total_cost,   # 含買進手續費
            "stop_loss":      round(price * (1 + _sl), 2),
            "take_profit":    round(price * (1 + _tp), 2),
            "predicted_gain": round(predicted_gain * 100, 3) if predicted_gain is not None else None,
        }
        self._log_trade(code, "BUY", shares, price, "當沖買入", analysis, predicted_gain, is_real=False)
        try:
            from utils.platform_signal import report_position as _rpos
            _rpos(TRADING_USER_ID, code, "LONG", shares, price, is_real=False, account=self._account())
        except Exception:
            pass
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

    # ── 反沖（空方）紙上交易 ──────────────────────────────────────────────────

    def _paper_short(self, code: str, price: float, analysis: dict) -> bool:
        """空方進場：模擬先賣後買（反沖），只走紙上，不送真實委託。"""
        with self._lock:
            holdings = self._state["holdings"]
            short_key = f"SHORT_{code}"
            if short_key in holdings or code in holdings:
                return False
            if len(holdings) >= DAY_MAX_POSITIONS:
                return False

            budget = self._trade_budget()
            shares = int(budget // (price * 1000)) * 1000
            if shares <= 0:
                logger.debug(f"[DayTrader] {code} 單價過高/預算不足，略過空單（需 {price*1000:,.0f}，上限 {budget:,.0f}）")
                return False

            sell_value = shares * price
            sell_fee   = max(_MIN_FEE, round(sell_value * _COMMISSION_RATE))
            stt        = round(sell_value * _STT_RATE)

            holdings[short_key] = {
                "code":         code,
                "direction":    "short",
                "shares":       shares,
                "entry_price":  price,
                "entry_time":   self._now_tw().strftime("%H:%M"),
                "entry_ts":     time.time(),
                "is_real":      False,
                "proceeds":     sell_value - sell_fee - stt,   # 賣出淨收
                "stop_loss":    round(price * (1 + abs(_fetch_user_config()["stop_loss_pct"])), 2),
                "take_profit":  round(price * (1 - abs(_fetch_user_config()["take_profit_pct"])), 2),
            }
            self._log_trade(code, "SHORT", shares, price, "反沖賣出", analysis, is_real=False)
            try:
                from utils.platform_signal import report_position as _rpos
                _rpos(TRADING_USER_ID, code, "SHORT", shares, price, is_real=False, account=self._account())
            except Exception:
                pass
            logger.info(
                f"[DayTrader] 🔴 反沖空頭 {code}  {shares}股 @{price:.2f}"
                f"  score={analysis.get('score')}  delta={analysis.get('vol_delta_signal')}"
            )
            try:
                from utils.telegram_notify import notify_buy
                notify_buy("反沖(空)", code, shares, price,
                           f"score={analysis.get('score')} 反沖空頭進場")
            except Exception:
                pass
            self._start_fast_monitor()
            return True

    def _paper_cover(self, code: str, price: float, reason: str) -> float:
        """空方平倉：買回，計算損益。"""
        with self._lock:
            holdings = self._state["holdings"]
            short_key = f"SHORT_{code}"
            if short_key not in holdings:
                return 0.0
            h = holdings.pop(short_key)
            shares = h["shares"]
            buy_back = shares * price
            buy_fee  = max(_MIN_FEE, round(buy_back * _COMMISSION_RATE))
            pnl = h["proceeds"] - (buy_back + buy_fee)
            self._state["daily_pnl"] += pnl
            # 記錄冷卻計時，避免同日反覆進場同一空標
            self._state.setdefault("short_cooldown", {})[code] = time.time()
            self._log_trade(code, "COVER", shares, price, reason, {}, is_real=False, pnl=pnl)
            try:
                from utils.platform_signal import report_position as _rpos
                _rpos(TRADING_USER_ID, code, "SHORT", 0, price, is_real=False, account=self._account())
            except Exception:
                pass
            emoji = "🟩" if pnl >= 0 else "🟥"
            logger.info(
                f"[DayTrader] {emoji} 反沖回補 {code}  {shares}股 @{price:.2f}"
                f"  損益={pnl:+,.0f} TWD  原因={reason}"
            )
            try:
                from utils.telegram_notify import notify_sell
                notify_sell("反沖(空)", code, shares, price, pnl, reason)
            except Exception:
                pass
            return pnl

    # ── 影子帳戶（多資本層模擬） ──────────────────────────────────────────────

    def __create_shadow_long(self, code: str, price: float, analysis: dict):
        """對每個影子帳戶嘗試多方進場（需在 _lock 內呼叫）。"""
        for cap_key, acc in self._state.get("shadows", {}).items():
            sh = acc["holdings"]
            if code in sh or len(sh) >= DAY_MAX_POSITIONS:
                continue
            budget = min(acc["initial"] * DAY_PER_TRADE_PCT, acc["cash"])
            shares = int(budget // (price * 1000)) * 1000
            if shares <= 0:
                continue
            cost  = shares * price
            fee   = max(_MIN_FEE, round(cost * _COMMISSION_RATE))
            total = cost + fee
            if total > acc["cash"]:
                continue
            acc["cash"] -= total
            sh[code] = {
                "shares":      shares,
                "entry_price": price,
                "entry_time":  self._now_tw().strftime("%H:%M"),
                "entry_ts":    time.time(),
                "cost":        total,
                "stop_loss":   round(price * (1 + _fetch_user_config()["stop_loss_pct"]), 2),
                "take_profit": round(price * (1 + _fetch_user_config()["take_profit_pct"]), 2),
            }
            cap = acc["initial"]
            logger.info(
                f"[Shadow-{cap//10000}萬] 🔵 買入 {code}  {shares}股 @{price:.2f}"
                f"  現金={acc['cash']:,.0f}"
            )
            self._report_shadow(acc, code, "BUY", shares, price, "當沖買入")

    def __create_shadow_short(self, code: str, price: float, analysis: dict):
        """對每個影子帳戶嘗試空方進場（需在 _lock 內呼叫）。"""
        for cap_key, acc in self._state.get("shadows", {}).items():
            sh = acc["holdings"]
            short_key = f"SHORT_{code}"
            if short_key in sh or len(sh) >= DAY_MAX_POSITIONS:
                continue
            budget   = min(acc["initial"] * DAY_PER_TRADE_PCT, acc["cash"])
            shares   = int(budget // (price * 1000)) * 1000
            if shares <= 0:
                continue
            sell_val = shares * price
            sell_fee = max(_MIN_FEE, round(sell_val * _COMMISSION_RATE))
            stt      = round(sell_val * _STT_RATE)
            sh[short_key] = {
                "code":        code,
                "direction":   "short",
                "shares":      shares,
                "entry_price": price,
                "entry_time":  self._now_tw().strftime("%H:%M"),
                "entry_ts":    time.time(),
                "proceeds":    sell_val - sell_fee - stt,
                "stop_loss":   round(price * (1 + abs(_fetch_user_config()["stop_loss_pct"])), 2),
                "take_profit": round(price * (1 - abs(_fetch_user_config()["take_profit_pct"])), 2),
            }
            cap = acc["initial"]
            logger.info(
                f"[Shadow-{cap//10000}萬] 🔴 反沖空頭 {code}  {shares}股 @{price:.2f}"
                f"  現金={acc['cash']:,.0f}"
            )
            self._report_shadow(acc, code, "SHORT", shares, price, "反沖賣出")

    def _shadow_sell_acc(self, acc: dict, code: str, price: float, reason: str) -> float:
        """單一影子帳戶多方出場（需在 _lock 內呼叫）。"""
        sh = acc["holdings"]
        if code not in sh:
            return 0.0
        h        = sh.pop(code)
        shares   = h["shares"]
        proceeds = shares * price
        sell_fee = max(_MIN_FEE, round(proceeds * _COMMISSION_RATE))
        stt      = round(proceeds * _STT_RATE)
        net      = proceeds - sell_fee - stt
        pnl      = net - h["cost"]
        acc["cash"] += net
        acc["pnl"]  += pnl
        cap = acc["initial"]
        logger.info(
            f"[Shadow-{cap//10000}萬] {'🟩' if pnl>=0 else '🟥'} 賣出 {code}"
            f"  {shares}股 @{price:.2f}  損益={pnl:+,.0f}  原因={reason}"
        )
        self._report_shadow(acc, code, "SELL", shares, price, reason, pnl)
        return pnl

    def _shadow_cover_acc(self, acc: dict, code: str, price: float, reason: str) -> float:
        """單一影子帳戶空方平倉（需在 _lock 內呼叫）。"""
        sh = acc["holdings"]
        short_key = f"SHORT_{code}"
        if short_key not in sh:
            return 0.0
        h       = sh.pop(short_key)
        shares  = h["shares"]
        buy_bk  = shares * price
        buy_fee = max(_MIN_FEE, round(buy_bk * _COMMISSION_RATE))
        pnl     = h["proceeds"] - (buy_bk + buy_fee)
        acc["pnl"] += pnl
        cap = acc["initial"]
        logger.info(
            f"[Shadow-{cap//10000}萬] {'🟩' if pnl>=0 else '🟥'} 反沖回補 {code}"
            f"  {shares}股 @{price:.2f}  損益={pnl:+,.0f}  原因={reason}"
        )
        self._report_shadow(acc, code, "COVER", shares, price, reason, pnl)
        return pnl

    def _shadow_sell(self, code: str, price: float, reason: str):
        with self._lock:
            for acc in self._state.get("shadows", {}).values():
                self._shadow_sell_acc(acc, code, price, reason)

    def _shadow_cover(self, code: str, price: float, reason: str):
        with self._lock:
            for acc in self._state.get("shadows", {}).values():
                self._shadow_cover_acc(acc, code, price, reason)

    def _check_shadow_holdings(self):
        """影子帳戶持倉監控（同等 SL/TP/時間衰退邏輯）。每個代碼只抓一次行情。"""
        shadows = self._state.get("shadows", {})
        if not shadows:
            return

        force_close = self._is_force_close_time()

        # 收集所有影子帳戶持有的唯一代碼
        long_codes:  set[str] = set()
        short_codes: set[str] = set()
        for acc in shadows.values():
            for key in acc["holdings"]:
                if key.startswith("SHORT_"):
                    short_codes.add(key[6:])
                else:
                    long_codes.add(key)

        if not long_codes and not short_codes:
            return

        # 每個代碼只下載一次
        price_map: dict[str, float | None] = {}
        df_map:    dict[str, object]        = {}
        for code in long_codes | short_codes:
            df = _fetch_intraday(code)
            df_map[code]    = df
            price_map[code] = float(df["Close"].iloc[-1]) if df is not None and not df.empty else None

        # 多方持倉
        for code in long_codes:
            price = price_map.get(code)
            if price is None:
                continue
            # 移動停利（每帳戶各自更新 SL）
            with self._lock:
                for acc in shadows.values():
                    _h = acc["holdings"].get(code)
                    if _h:
                        pct2 = (price - _h["entry_price"]) / _h["entry_price"]
                        if pct2 >= DAY_TRAIL_ACTIVATE_PCT:
                            new_sl = round(price * (1 - DAY_TRAIL_DISTANCE_PCT), 2)
                            if new_sl > _h["stop_loss"]:
                                _h["stop_loss"] = new_sl
            # 各帳戶獨立判斷出場
            for cap_key, acc in shadows.items():
                h = acc["holdings"].get(code)
                if h is None:
                    continue
                entry = h["entry_price"]
                pct   = (price - entry) / entry
                with self._lock:
                    if force_close:
                        self._shadow_sell_acc(acc, code, price, "強制平倉 13:20")
                    elif price <= h["stop_loss"]:
                        self._shadow_sell_acc(acc, code, price, f"停損 {h['stop_loss']:.2f}")
                    elif price >= h["take_profit"]:
                        self._shadow_sell_acc(acc, code, price, f"停利 {h['take_profit']:.2f}")
                    elif h.get("entry_ts") and (time.time() - h["entry_ts"]) >= DAY_TIME_DECAY_MIN * 60:
                        if abs(pct) < DAY_TIME_DECAY_BAND:
                            self._shadow_sell_acc(acc, code, price, f"時間衰退 {DAY_TIME_DECAY_MIN}分鐘")

        # 空方持倉
        for code in short_codes:
            price = price_map.get(code)
            if price is None:
                continue
            for cap_key, acc in shadows.items():
                short_key = f"SHORT_{code}"
                h = acc["holdings"].get(short_key)
                if h is None:
                    continue
                entry_ts = h.get("entry_ts")
                pct = (h["entry_price"] - price) / h["entry_price"]
                with self._lock:
                    if force_close:
                        self._shadow_cover_acc(acc, code, price, "強制平倉 13:20")
                    elif price >= h["stop_loss"]:
                        self._shadow_cover_acc(acc, code, price, f"停損 {h['stop_loss']:.2f}")
                    elif price <= h["take_profit"]:
                        self._shadow_cover_acc(acc, code, price, f"停利 {h['take_profit']:.2f}")
                    elif DAY_SHORT_TIME_DECAY and entry_ts and (time.time() - entry_ts) >= DAY_TIME_DECAY_MIN * 60:
                        if abs(pct) < DAY_TIME_DECAY_BAND:
                            self._shadow_cover_acc(acc, code, price, f"時間衰退 {DAY_TIME_DECAY_MIN}分鐘")

    def _paper_sell(self, code: str, price: float, reason: str) -> float:
        with self._lock:
            return self.__paper_sell(code, price, reason)

    def __paper_sell(self, code: str, price: float, reason: str) -> float:
        holdings = self._state["holdings"]
        if code not in holdings:
            return 0.0
        # 已有進行中的賣單（避免重複下單）
        if any(o["code"] == code and o["side"] == "SELL" for o in self._pending_sell_orders.values()):
            return 0.0

        h = holdings[code]

        # 真實下單路徑：只有真實買入的持倉（is_real=True）才走真實賣出，
        # 避免對紙上持倉下真實委託而出現「集保庫存不足」錯誤。
        if DAY_REAL_TRADING and self._place_order_fn is not None and h.get("is_real"):
            shares = h["shares"]
            order_no = self._place_order_fn(code, "SELL", shares, price)
            if not order_no:
                logger.warning(f"[DayTrader] ❌ {code} 真實賣單送出失敗，保留持倉等下次重試")
                try:
                    from utils.telegram_notify import send
                    send(f"❌ <b>真實賣單失敗</b> {code}\n原因={reason}\n持倉已保留，下次觸發再試",
                         key=f"sell_fail:{code}", cooldown=60)
                except Exception:
                    pass
                return 0.0  # 保留持倉，不做紙上結算
            else:
                # 從 holdings 移除（防止 fast monitor 觸發重複賣單），記錄 pending
                holdings.pop(code)
                self._pending_sell_orders[order_no] = {
                    "side": "SELL", "code": code, "shares": shares, "price": price,
                    "cost": h["cost"], "reason": reason, "placed_at": time.time(),
                    "retries": 0,
                    "entry_price": h["entry_price"],
                    "entry_ts":    h.get("entry_ts"),
                }
                logger.info(
                    f"[DayTrader] ⏳ 真實賣單送出 {code} {shares}股 @{price:.2f}"
                    f"  原因={reason}  單號={order_no}"
                )
                return 0.0   # 損益等成交後計算

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
        self._log_trade(code, "SELL", shares, price, reason, {}, is_real=False, pnl=pnl)
        try:
            from utils.platform_signal import report_position as _rpos
            _rpos(TRADING_USER_ID, code, "LONG", 0, price, is_real=False, account=self._account())
        except Exception:
            pass
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
                   predicted_gain: float | None = None,
                   is_real: bool = False, pnl: float = 0.0,
                   order_no: str = ""):
        entry = {
            "time":            self._now_tw().strftime("%H:%M"),
            "code":            code,
            "side":            side,
            "shares":          shares,
            "price":           price,
            "reason":          reason,
            "is_real":         is_real,
            "pnl":             round(pnl, 0),
            "score":           analysis.get("score"),
            "delta":           analysis.get("vol_delta_signal"),
            "pattern":         analysis.get("pattern"),
            "predicted_gain":  round(predicted_gain * 100, 3) if predicted_gain is not None else None,
        }
        self._state["trade_log"].append(entry)
        try:
            from utils.platform_signal import report_trade as _report
            _report(
                user_id=TRADING_USER_ID,
                code=code,
                action=side,
                shares=shares,
                price=price,
                pnl=pnl,
                reason=reason,
                order_no=order_no,
                is_real=is_real,
                account=self._account(is_real),
            )
        except Exception:
            pass

    # ── 真實委託確認 ──────────────────────────────────────────────────────────

    @staticmethod
    def _alert(msg: str, key: str, cooldown: int = 300):
        """真實下單異常一律 Telegram 告警（失敗不影響主流程）。"""
        try:
            from utils.telegram_notify import send
            send(msg, key=key, cooldown=cooldown)
        except Exception:
            pass

    def _requeue_real_holding(self, o: dict, shares: int | None = None, cost: float | None = None):
        """賣單沒賣掉：把真實持倉放回 holdings 並標記 force_exit，下一輪立即再賣（需在 _lock 內呼叫）。"""
        if o["code"] in self._state["holdings"]:
            return
        ep = o.get("entry_price", o["price"])
        cfg = _fetch_user_config()
        self._state["holdings"][o["code"]] = {
            "shares": shares if shares is not None else o["shares"],
            "entry_price": ep,
            "entry_time": "??:??",
            "entry_ts":   o.get("entry_ts", time.time()),
            "is_real": True,
            "cost": cost if cost is not None else o["cost"],
            "stop_loss":   round(ep * (1 + cfg["stop_loss_pct"]), 2),
            "take_profit": round(ep * (1 + cfg["take_profit_pct"]), 2),
            "force_exit": o.get("reason", "重試賣出"),
        }

    def _confirm_pending_orders(self):
        """
        查詢 pending 買/賣單是否成交（每 run_cycle 呼叫一次，在鎖外執行 API）。
        成交 → 更新 holdings/cash/pnl；超時 60s → 取消重試（賣單最多 1 次）；
        超時 120s 仍未成交 → 賣單紙上強制結算（確保 T+0）。
        """
        if not self._check_filled_fn:
            return

        # ── 買單確認 ──────────────────────────────────────────────────────────
        buy_done = []
        for order_no, o in list(self._pending_buy_orders.items()):
            filled, fill_price = self._check_filled_fn(o["code"], order_no)
            if filled > 0:
                # 以實際成交價建倉；fill_price=0 代表 SDK 未回報，退守委託價
                price = fill_price if fill_price > 0 else o["price"]
                if fill_price > 0 and abs(fill_price - o["price"]) > 0.01:
                    logger.info(
                        f"[DayTrader] 📋 買入滑價 {o['code']}  委託={o['price']:.2f} → 成交={fill_price:.2f}"
                        f"  差={fill_price - o['price']:+.2f}"
                    )
                cost  = filled * price
                fee   = max(_MIN_FEE, round(cost * _COMMISSION_RATE))
                total = cost + fee
                # 若實際金額 < 預扣，退還差額
                refund = o["reserved_cash"] - total
                with self._lock:
                    if refund > 0:
                        self._state["cash"] += refund
                    self._state["holdings"][o["code"]] = {
                        "shares":         filled,
                        "entry_price":    price,
                        "entry_time":     self._now_tw().strftime("%H:%M"),
                        "entry_ts":       time.time(),
                        "is_real":        True,
                        "cost":           total,
                        "stop_loss":      round(price * (1 + _fetch_user_config()["stop_loss_pct"]), 2),
                        "take_profit":    round(price * (1 + _fetch_user_config()["take_profit_pct"]), 2),
                        "predicted_gain": round(o["predicted_gain"] * 100, 3) if o.get("predicted_gain") else None,
                    }
                    self._log_trade(o["code"], "BUY", filled, price, "當沖買入(真實)", o.get("analysis") or {}, o.get("predicted_gain"), is_real=True, order_no=order_no)
                try:
                    from utils.platform_signal import report_position as _rpos
                    _rpos(TRADING_USER_ID, o["code"], "LONG", filled, price, is_real=True, account="real")
                except Exception:
                    pass
                logger.info(f"[DayTrader] ✅ 買入確認 {o['code']} {filled}股 @{price:.2f}")
                try:
                    from utils.telegram_notify import notify_buy
                    gain_str = f"model={o['predicted_gain']*100:.2f}%" if o.get("predicted_gain") else ""
                    score_str = o.get("analysis", {}).get("score", "")
                    notify_buy("當沖(真實)", o["code"], filled, price,
                               f"score={score_str} {gain_str}".strip())
                except Exception:
                    pass
                if filled < o["shares"]:
                    # 部分成交：取消剩餘委託，避免之後成交卻沒被追蹤
                    if self._cancel_order_fn and not o.get("cancel_sent"):
                        self._cancel_order_fn(o["code"], order_no)
                    logger.warning(f"[DayTrader] ⚠️ 買單部分成交 {o['code']} {filled}/{o['shares']}股，已取消剩餘委託")
                    self._alert(f"⚠️ <b>當沖買單部分成交</b> {o['code']}  {filled}/{o['shares']}股\n"
                                f"剩餘委託已送取消，請確認券商庫存", key=f"buy_partial:{order_no}")
                self._start_fast_monitor()
                buy_done.append(order_no)
            elif filled == -1:
                # 被拒：退還預扣資金，Telegram 告警
                with self._lock:
                    self._state["cash"] += o["reserved_cash"]
                logger.warning(f"[DayTrader] ❌ 買單被拒 {o['code']} {order_no}，已退款")
                try:
                    from utils.telegram_notify import send
                    send(f"❌ <b>當沖買入失敗</b> {o['code']}  委託價={o['price']:.2f}  單號={order_no}", key=f"buy_reject:{order_no}")
                except Exception:
                    pass
                buy_done.append(order_no)
            elif time.time() - o["placed_at"] > 120:
                # 2 分鐘無成交：先送取消，下一輪確認「取消前沒有成交」才退款
                # （取消與成交可能同時發生；直接退款會留下沒人盯的真實持倉）
                if not o.get("cancel_sent"):
                    if self._cancel_order_fn:
                        self._cancel_order_fn(o["code"], order_no)
                    o["cancel_sent"] = True
                    logger.warning(f"[DayTrader] ⏰ 買單超時，已送取消 {o['code']} {order_no}，下一輪確認")
                    continue
                with self._lock:
                    self._state["cash"] += o["reserved_cash"]
                logger.warning(f"[DayTrader] ⏰ 買單超時取消完成 {o['code']} {order_no}，已退款")
                buy_done.append(order_no)
        for no in buy_done:
            del self._pending_buy_orders[no]

        # ── 賣單確認 ──────────────────────────────────────────────────────────
        sell_done = []
        for order_no, o in list(self._pending_sell_orders.items()):
            filled, fill_price = self._check_filled_fn(o["code"], order_no)
            if filled > 0:
                price = fill_price if fill_price > 0 else o["price"]
                if fill_price > 0 and abs(fill_price - o["price"]) > 0.01:
                    logger.info(
                        f"[DayTrader] 📋 賣出滑價 {o['code']}  委託={o['price']:.2f} → 成交={fill_price:.2f}"
                        f"  差={fill_price - o['price']:+.2f}"
                    )
                proceeds = filled * price
                fee  = max(_MIN_FEE, round(proceeds * _COMMISSION_RATE))
                stt  = round(proceeds * _STT_RATE)
                net  = proceeds - fee - stt
                remaining = max(o["shares"] - filled, 0)
                # 部分成交：成本按比例分攤，剩餘股數放回持倉繼續賣
                cost_part = o["cost"] * filled / o["shares"] if remaining else o["cost"]
                pnl  = net - cost_part
                with self._lock:
                    self._state["cash"]      += net
                    self._state["daily_pnl"] += pnl
                    self._log_trade(o["code"], "SELL", filled, price, o["reason"], {}, is_real=True, pnl=pnl, order_no=order_no)
                    if remaining:
                        self._requeue_real_holding(o, shares=remaining, cost=o["cost"] - cost_part)
                if remaining:
                    if self._cancel_order_fn and not o.get("cancel_sent"):
                        self._cancel_order_fn(o["code"], order_no)
                    logger.warning(f"[DayTrader] ⚠️ 賣單部分成交 {o['code']} {filled}/{o['shares']}股，剩餘 {remaining} 股續賣")
                    self._alert(f"⚠️ <b>當沖賣單部分成交</b> {o['code']}  {filled}/{o['shares']}股\n"
                                f"剩餘 {remaining} 股已重新排入賣出", key=f"sell_partial:{order_no}")
                try:
                    from utils.platform_signal import report_position as _rpos
                    _rpos(TRADING_USER_ID, o["code"], "LONG", remaining, o.get("entry_price", price),
                          is_real=True, account="real")
                except Exception:
                    pass
                emoji = "🟩" if pnl >= 0 else "🟥"
                slip_str = (f"  滑價={fill_price - o['price']:+.2f}" if fill_price > 0 and abs(fill_price - o["price"]) > 0.01 else "")
                logger.info(
                    f"[DayTrader] {emoji} 賣出確認 {o['code']} {filled}股 @{price:.2f}{slip_str}"
                    f"  手續費={fee+stt:,.0f}  損益={pnl:+,.0f} TWD"
                )
                try:
                    from utils.telegram_notify import notify_sell
                    notify_sell("當沖(真實)", o["code"], filled, price, pnl,
                                f"{o['reason']}{slip_str}")
                except Exception:
                    pass
                sell_done.append(order_no)
            elif filled == -1:
                logger.warning(f"[DayTrader] ❌ 賣單被拒 {o['code']} {order_no}，重試")
                try:
                    from utils.telegram_notify import send
                    send(f"❌ <b>當沖賣出失敗</b> {o['code']}  委託價={o['price']:.2f}  重試中", key=f"sell_reject:{order_no}")
                except Exception:
                    pass
                sell_done.append(order_no)
                # 放回 holdings 並標記 force_exit，下一輪立即再賣
                with self._lock:
                    self._requeue_real_holding(o)
            elif time.time() - o["placed_at"] > 60:
                # 60 秒未成交：先送取消，下一輪確認「取消前沒有成交」才重下
                # （若原單其實已成交還重下，會變成多賣一次＝真實放空）
                if not o.get("cancel_sent"):
                    if self._cancel_order_fn:
                        self._cancel_order_fn(o["code"], order_no)
                    o["cancel_sent"] = True
                    logger.warning(f"[DayTrader] ⏰ 賣單超時，已送取消 {o['code']} {order_no}，下一輪確認")
                    continue
                # 動作前再查一次成交，降低查詢失敗被當成「未成交」的機率
                again, _ = self._check_filled_fn(o["code"], order_no)
                if again != 0:
                    continue   # 已成交或被拒 → 交給下一輪的正常路徑處理
                new_no = None
                if o.get("retries", 0) == 0 and self._place_order_fn:
                    df = _fetch_intraday(o["code"])
                    new_price = float(df["Close"].iloc[-1]) if df is not None and not df.empty else o["price"]
                    new_no = self._place_order_fn(o["code"], "SELL", o["shares"], new_price)
                    if new_no:
                        self._pending_sell_orders[new_no] = dict(
                            o, price=new_price, placed_at=time.time(), retries=1, cancel_sent=False)
                        logger.warning(f"[DayTrader] ⚠️ 賣單超時重試 {o['code']} @{new_price:.2f} 新單號={new_no}")
                if not new_no:
                    # 真實股票還在庫存：不做紙上結算（會讓帳面與券商不一致），放回持倉續賣並告警
                    with self._lock:
                        self._requeue_real_holding(o)
                    logger.error(f"[DayTrader] ❌ {o['code']} 賣單未成交（已重試 {o.get('retries', 0)} 次），持倉保留續賣")
                    self._alert(f"🚨 <b>當沖賣單未成交</b> {o['code']}  {o['shares']}股\n"
                                f"原因={o['reason']}  已重試 {o.get('retries', 0)} 次\n"
                                f"持倉已保留並持續嘗試賣出，請確認券商庫存，必要時手動處理",
                                key=f"sell_stuck:{o['code']}", cooldown=120)
                sell_done.append(order_no)
        for no in sell_done:
            del self._pending_sell_orders[no]

    # ── 當沖資格查詢 ─────────────────────────────────────────────────────────

    def _check_daytrade_eligible(self, code: str) -> bool:
        """
        查詢富邦 daytrade_and_stock_info，確認該股今日是否可現股當沖。
        結果每日快取，API 失敗時 fallback=True（不擋單，避免誤殺）。
        第一版先記錄完整 status / quota，讓 log 幫我們確認值域。
        """
        if not self._daytrade_info_fn:
            return True  # 未注入時不擋

        today = self._now_tw().strftime("%Y-%m-%d")
        cached = self._daytrade_cache.get(code)
        if cached and cached.get("date") == today:
            return cached["eligible"]

        try:
            result = self._daytrade_info_fn(code)
            if not result.is_success or not result.data:
                logger.warning(f"[DayTrader] 當沖資格查詢失敗 {code}，預設允許")
                return True

            d = result.data
            status  = getattr(d, "status",                  None)
            orig_q  = getattr(d, "daytrade_orig_quota",     None)
            trade_q = getattr(d, "daytrade_tradable_quota", None)
            pre_s   = getattr(d, "precollect_single",       None)
            pre_a   = getattr(d, "precollect_accumulate",   None)

            # 記錄完整回傳，供第一次觀察值域用
            logger.info(
                f"[DayTrader] 當沖資格 {code}"
                f"  status={status}  orig_quota={orig_q}  tradable_quota={trade_q}"
                f"  precollect_single={pre_s}  precollect_accumulate={pre_a}"
            )

            # status=0 通常代表「不可當沖」；其他值或 None 先放行
            # 待觀察真實值後再設精確門檻
            eligible = (status != 0)

            self._daytrade_cache[code] = {"date": today, "eligible": eligible,
                                          "status": status, "quota": trade_q}
            if not eligible:
                logger.info(f"[DayTrader] ⛔ {code} status={status} 不可現股當沖，跳過")
            return eligible

        except Exception as e:
            logger.warning(f"[DayTrader] 當沖資格查詢例外 {code}: {e}，預設允許")
            return True

    # ── 富邦庫存同步 ──────────────────────────────────────────────────────────

    def _sync_day_holdings_from_fubon(self):
        """
        每個 run_cycle 呼叫：用富邦 inventories + unrealized_gains 校正
        DayTrader 的 is_real 持倉數量與成本價，確保 Telegram 數字跟 App 一致。
        只同步 is_real=True 的持倉，不碰 paper 持倉，也不新增隔日沖部位。
        """
        if not self._inventories_fn:
            return
        try:
            inv = self._inventories_fn()
            if not inv.is_success or inv.data is None:
                return

            # 富邦庫存：code -> qty
            fubon_qty: dict[str, int] = {}
            for item in inv.data:
                code = str(getattr(item, "stock_no", None) or "").strip()
                odd  = getattr(item, "odd", None)
                qty  = int((getattr(odd, "today_qty", 0) if odd else 0)
                           or getattr(item, "today_qty", 0) or getattr(item, "tradable_qty", 0) or 0)
                if code and qty > 0:
                    fubon_qty[code] = qty

            # 富邦成本價：code -> cost_price
            cost_map: dict[str, float] = {}
            if self._unrealized_fn:
                try:
                    unreal = self._unrealized_fn()
                    if unreal.is_success and unreal.data:
                        for u in unreal.data:
                            c  = str(getattr(u, "stock_no", "") or "").strip()
                            cp = float(getattr(u, "cost_price", 0) or 0)
                            if c and cp > 0:
                                cost_map[c] = cp
                except Exception:
                    pass

            with self._lock:
                holdings = self._state["holdings"]
                for code in list(holdings):
                    h = holdings[code]
                    if not h.get("is_real"):
                        continue

                    # 富邦已無庫存 → 移除假持倉（可能已被其他路徑成交）
                    if code not in fubon_qty:
                        # 成交後庫存可能延遲出現：建倉 3 分鐘內不刪，之後連續 2 次查不到才刪
                        if time.time() - h.get("entry_ts", 0) < 180:
                            continue
                        h["inv_miss"] = h.get("inv_miss", 0) + 1
                        if h["inv_miss"] < 2:
                            continue
                        logger.warning(f"[DayTrader] ⚠️ 移除假持倉 {code}（富邦庫存連續 2 次為 0）")
                        self._alert(f"⚠️ <b>當沖持倉已移除</b> {code}\n富邦庫存連續 2 次查無此股，"
                                    f"本地持倉已清除（停損停利不再監控）", key=f"inv_gone:{code}")
                        del holdings[code]
                        continue
                    h.pop("inv_miss", None)

                    # 數量不符 → 以富邦為準
                    if h["shares"] != fubon_qty[code]:
                        logger.info(
                            f"[DayTrader] 📋 數量修正 {code}"
                            f"  本地={h['shares']} → 富邦={fubon_qty[code]}"
                        )
                        h["shares"] = fubon_qty[code]

                    # 成本價不符 → 以富邦為準，同步更新 SL/TP
                    if code in cost_map and abs(cost_map[code] - h["entry_price"]) > 0.05:
                        old_ep = h["entry_price"]
                        h["entry_price"] = cost_map[code]
                        h["stop_loss"]   = round(cost_map[code] * (1 + _fetch_user_config()["stop_loss_pct"]), 2)
                        h["take_profit"] = round(cost_map[code] * (1 + _fetch_user_config()["take_profit_pct"]), 2)
                        logger.info(
                            f"[DayTrader] 📋 成本修正 {code}"
                            f"  {old_ep:.2f} → {cost_map[code]:.2f}"
                            f"  SL={h['stop_loss']:.2f}  TP={h['take_profit']:.2f}"
                        )
        except Exception as e:
            logger.warning(f"[DayTrader] 富邦庫存同步失敗: {e}")

    # ── 持倉檢查 ─────────────────────────────────────────────────────────────

    def _update_trailing_stop(self, code: str, price: float) -> bool:
        """移動停利（必須在 _lock 內呼叫）。達 +1.5% 後以 1% trailing 上移 SL，SL 只升不降。"""
        h = self._state["holdings"].get(code)
        if h is None:
            return False
        entry = h["entry_price"]
        pct = (price - entry) / entry
        if pct < DAY_TRAIL_ACTIVATE_PCT:
            return False
        new_sl = round(price * (1 - DAY_TRAIL_DISTANCE_PCT), 2)
        if new_sl > h["stop_loss"]:
            old_sl = h["stop_loss"]
            h["stop_loss"] = new_sl
            logger.info(
                f"[DayTrader] 📈 移動停利 {code}  SL {old_sl:.2f} → {new_sl:.2f}"
                f"  現價 {price:.2f} ({pct:+.2%})"
            )
            return True
        return False

    def _check_holdings(self):
        if not self._state["holdings"]:
            return
        force_close = self._is_force_close_time()
        for key in list(self._state["holdings"]):
            h = self._state["holdings"].get(key)
            if h is None:
                continue

            # 空方持倉（key = "SHORT_{code}"）
            if h.get("direction") == "short":
                code = h["code"]
                df = _fetch_intraday(code)
                price = float(df["Close"].iloc[-1]) if df is not None and not df.empty else None
                if price is None:
                    continue
                entry_ts = h.get("entry_ts")
                pct = (h["entry_price"] - price) / h["entry_price"]  # 空方：跌=正收益

                if force_close:
                    self._paper_cover(code, price, "強制平倉 13:20")
                    continue
                if price >= h["stop_loss"]:
                    self._paper_cover(code, price, f"停損 {h['stop_loss']:.2f}")
                    continue
                if price <= h["take_profit"]:
                    self._paper_cover(code, price, f"停利 {h['take_profit']:.2f}")
                    continue
                if DAY_SHORT_TIME_DECAY and entry_ts and (time.time() - entry_ts) >= DAY_TIME_DECAY_MIN * 60:
                    if abs(pct) < DAY_TIME_DECAY_BAND:
                        self._paper_cover(code, price, f"時間衰退 {DAY_TIME_DECAY_MIN}分鐘橫盤")
                        continue
                # 理由消失：進場條件是 VWAP 下方；突破 VWAP 且已持倉 > 5 分鐘 → 出場
                if (DAY_VWAP_EXIT and df is not None and not df.empty
                        and entry_ts and (time.time() - entry_ts) > 300):
                    vwap_ser = _vwap(df)
                    vwap_val = float(vwap_ser.iloc[-1]) if not np.isnan(vwap_ser.iloc[-1]) else None
                    if vwap_val and price > vwap_val and pct > _fetch_user_config()["stop_loss_pct"]:
                        self._paper_cover(code, price, f"突破VWAP {vwap_val:.2f}")
                continue

            # 多方持倉（key = code）
            code = key
            df = _fetch_intraday(code)
            price = float(df["Close"].iloc[-1]) if df is not None and not df.empty else None
            if price is None:
                continue

            # 移動停利（每分鐘更新一次 SL）
            with self._lock:
                self._update_trailing_stop(code, price)

            h = self._state["holdings"].get(code)
            if h is None:
                continue
            entry = h["entry_price"]
            pct   = (price - entry) / entry

            if force_close:
                self._paper_sell(code, price, "強制平倉 13:20")
                continue
            if h.get("force_exit"):
                self._paper_sell(code, price, str(h["force_exit"]))
                continue
            if price <= h["stop_loss"]:
                self._paper_sell(code, price, f"停損 {h['stop_loss']:.2f}")
                continue
            elif price >= h["take_profit"]:
                self._paper_sell(code, price, f"停利 {h['take_profit']:.2f}")
                continue

            # 時間衰退：持倉超過 N 分鐘仍橫盤 → 換股
            entry_ts = h.get("entry_ts")
            if entry_ts and (time.time() - entry_ts) >= DAY_TIME_DECAY_MIN * 60:
                if abs(pct) < DAY_TIME_DECAY_BAND:
                    self._paper_sell(code, price, f"時間衰退 {DAY_TIME_DECAY_MIN}分鐘橫盤")
                    continue

            # 理由消失：進場條件是 VWAP 上方；跌破 VWAP 且已持倉 > 5 分鐘 → 出場
            if (DAY_VWAP_EXIT and df is not None and not df.empty
                    and entry_ts and (time.time() - entry_ts) > 300):
                vwap_ser = _vwap(df)
                vwap_val = float(vwap_ser.iloc[-1]) if not np.isnan(vwap_ser.iloc[-1]) else None
                if vwap_val and price < vwap_val and pct > _fetch_user_config()["stop_loss_pct"]:
                    self._paper_sell(code, price, f"跌破VWAP {vwap_val:.2f}")

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

        real_full = len(self._state["holdings"]) >= DAY_MAX_POSITIONS
        shadows   = self._state.get("shadows", {})
        all_shadow_full = (
            bool(shadows)
            and all(len(acc["holdings"]) >= DAY_MAX_POSITIONS for acc in shadows.values())
        )

        if real_full and (not shadows or all_shadow_full):
            return

        # 大盤方向加成：依 TWII 漲跌幅 + 盤中走勢 + 美股隔夜方向
        mkt_dir, mkt_gap, intraday_sig = _fetch_mkt_direction()
        us_dir, us_spx, us_ndx = _fetch_us_overnight()
        mkt_long_bonus, mkt_short_bonus = _mkt_score(mkt_gap, us_dir, intraday_sig)

        # 市場制動器：大盤強彈且尚未明顯回落 → 暫停新開空單
        # 條件：gap > +1.5% 且非 FADE（即 intraday_sig in PUSH/NONE）
        short_blocked_by_market = (mkt_gap > 0.015 and intraday_sig != "FADE")

        if mkt_dir != "FLAT" or us_dir != "NEUTRAL":
            logger.info(
                f"[DayTrader] 大盤 {mkt_dir}({intraday_sig})  gap={mkt_gap*100:+.2f}%"
                f"  美股 {us_dir}(SPX={us_spx*100:+.2f}% NDX={us_ndx*100:+.2f}%)"
                f"  多方加成={mkt_long_bonus:+d}  空方加成={mkt_short_bonus:+d}"
                + ("  🛑空單暫停" if short_blocked_by_market else "")
            )

        _ucfg = _fetch_user_config()
        allow_long  = _ucfg["enable_day"]
        allow_short = ENABLE_SHORT_TRADING and _ucfg["enable_short"]

        long_real    = []
        short_real   = []
        long_shadow  = []
        short_shadow = []

        for code in WATCH_STOCKS:
            holdings = self._state["holdings"]
            real_has = code in holdings or f"SHORT_{code}" in holdings
            need_real = not real_full and not real_has

            # 只要有任一影子帳戶還可入場就需要掃描
            need_shadow = any(
                code not in acc["holdings"]
                and f"SHORT_{code}" not in acc["holdings"]
                and len(acc["holdings"]) < DAY_MAX_POSITIONS
                for acc in shadows.values()
            )

            if not need_real and not need_shadow:
                continue

            df = _fetch_intraday(code)
            if df is None or len(df) < 5:
                continue
            analysis = _analyze_bars(df)
            if not analysis:
                continue
            self._update_watch_log(code, analysis)
            avg_vol   = analysis.get("avg_volume", 0)
            last_vol  = analysis.get("last_volume", 0)
            vol_ratio = (last_vol / avg_vol) if avg_vol > 0 else 0

            if avg_vol < DAY_MIN_AVG_VOLUME:
                continue
            if last_vol < avg_vol * DAY_MIN_ENTRY_VOL_PCT:
                continue

            # 加入大盤加成後的有效分數（用於進場判斷與排序）
            eff_long  = analysis["score"] + mkt_long_bonus
            eff_short = analysis["score"] + mkt_short_bonus

            # 門檻：爆量時降 1 分（大量訊號本身已納入評分）
            long_thr  = 2 if vol_ratio >= VOL_SURGE_ALERT_X else 3
            short_thr = -(2 if vol_ratio >= VOL_SURGE_ALERT_X else 3)

            # 空單冷卻：同支股票 COVER 後 60 分鐘內不再進場
            _SHORT_COOLDOWN_SECS = 3600
            cooldown_ts = self._state.get("short_cooldown", {}).get(code, 0)
            short_in_cooldown = (time.time() - cooldown_ts) < _SHORT_COOLDOWN_SECS

            if need_real:
                if allow_long and eff_long >= long_thr:
                    if self._check_daytrade_eligible(code):
                        long_real.append((code, eff_long, analysis))
                if allow_short and not short_blocked_by_market:
                    if short_in_cooldown:
                        logger.debug(f"[DayTrader] {code} 空單冷卻中（{int((cooldown_ts+_SHORT_COOLDOWN_SECS-time.time())/60)}分鐘後恢復），略過")
                    elif (eff_short <= short_thr
                            and not analysis["above_vwap"]
                            and analysis.get("vol_delta_signal") == "sell"):
                        if self._check_daytrade_eligible(code):
                            short_real.append((code, eff_short, analysis))

            if need_shadow:
                if allow_long and eff_long >= long_thr:
                    long_shadow.append((code, eff_long, analysis))
                if allow_short and not short_blocked_by_market:
                    if not short_in_cooldown:
                        if (eff_short <= short_thr
                                and not analysis["above_vwap"]
                                and analysis.get("vol_delta_signal") == "sell"):
                            short_shadow.append((code, eff_short, analysis))

        # ── 真實多方進場 ──────────────────────────────────────────────────────
        long_real.sort(key=lambda x: x[1], reverse=True)
        for code, score, analysis in long_real:
            if len(self._state["holdings"]) >= DAY_MAX_POSITIONS:
                break
            price = analysis["price"]
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
            try:
                from utils.platform_signal import broadcast as _bcast
                reason = (f"score={score} delta={analysis.get('vol_delta_signal')}"
                          + (f" model={predicted_gain*100:.2f}%" if predicted_gain else ""))
                _bcast(code, "BUY", score, price, reason)
            except Exception:
                pass

        # ── 真實空方進場 ──────────────────────────────────────────────────────
        if ENABLE_SHORT_TRADING:
            short_real.sort(key=lambda x: x[1])
            for code, score, analysis in short_real:
                if len(self._state["holdings"]) >= DAY_MAX_POSITIONS:
                    break
                price = analysis["price"]
                logger.info(
                    f"[DayTrader] 📊反沖 {code} score={score}"
                    f"  vwap={'↑' if analysis['above_vwap'] else '↓'}"
                    f"  delta={analysis['vol_delta_signal']}"
                    f"  surge={'✓' if analysis['vol_surge'] else '✗'}"
                    f"  @{price:.2f}"
                )
                self._paper_short(code, price, analysis)
                try:
                    from utils.platform_signal import broadcast as _bcast
                    reason = f"score={score} delta={analysis.get('vol_delta_signal')} 反沖空頭"
                    _bcast(code, "SHORT", score, price, reason)
                except Exception:
                    pass

        # ── 影子帳戶獨立進場（不依賴真實帳戶，所有資本層同時嘗試）──────────
        if shadows:
            long_shadow.sort(key=lambda x: x[1], reverse=True)
            # 與主帳戶相同的模型過濾（網路 I/O 在鎖外）
            passed = []
            for code, score, analysis in long_shadow:
                if self._model is not None:
                    df_m = _fetch_intraday(code)
                    if df_m is None or not self._model.is_buy_signal(df_m, analysis)[0]:
                        continue
                passed.append((code, analysis))
            with self._lock:
                for code, analysis in passed:
                    self.__create_shadow_long(code, analysis["price"], analysis)

            if ENABLE_SHORT_TRADING:
                short_shadow.sort(key=lambda x: x[1])
                with self._lock:
                    for code, score, analysis in short_shadow:
                        self.__create_shadow_short(code, analysis["price"], analysis)

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
            self._confirm_pending_orders()        # 確認上一輪委託是否成交
            self._sync_day_holdings_from_fubon()  # 以富邦庫存校正數量與成本價
            self._check_holdings()
            with self._lock:
                self._sync_shadows()
            self._check_shadow_holdings()         # 影子帳戶獨立監控
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
            for key, h in holdings.items():
                is_short  = h.get("direction") == "short"
                real_code = h.get("code", key) if is_short else key
                df  = _fetch_intraday(real_code)
                cur = float(df["Close"].iloc[-1]) if df is not None else h["entry_price"]
                pnl = ((h["entry_price"] - cur) if is_short else (cur - h["entry_price"])) * h["shares"]
                label    = f"S:{real_code}" if is_short else real_code
                is_real  = h.get("is_real", False)
                lines.append(f"{'[真]' if is_real else '[紙]'}{label} {cur:.2f}({pnl:+,.0f})")
            realized = self._state.get("daily_pnl", 0.0)
            logger.info(f"[DayTrader] 持倉: {', '.join(lines)}  已實現={realized:+,.0f}  現金={self._state['cash']:,.0f}")
        else:
            logger.info(
                f"[DayTrader] 無持倉  現金={self._state['cash']:,.0f}"
                f"  今日損益={self._state['daily_pnl']:+,.0f}"
            )

        if self._state.get("shadows"):
            shadows = self._state.get("shadows", {})
            for cap_key, acc in shadows.items():
                cap = acc["initial"]
                sh  = acc["holdings"]
                sh_lines = []
                for key, h in sh.items():
                    is_short  = h.get("direction") == "short"
                    real_code = h.get("code", key) if is_short else key
                    df  = _fetch_intraday(real_code)
                    cur = float(df["Close"].iloc[-1]) if df is not None else h["entry_price"]
                    pnl = ((h["entry_price"] - cur) if is_short else (cur - h["entry_price"])) * h["shares"]
                    label = f"S:{real_code}" if is_short else real_code
                    sh_lines.append(f"{label} {cur:.2f}({pnl:+,.0f})")
                if sh_lines:
                    logger.info(
                        f"[Shadow-{cap//10000}萬] 持倉: {', '.join(sh_lines)}"
                        f"  現金={acc['cash']:,.0f}  損益={acc['pnl']:+,.0f}"
                    )
                else:
                    logger.debug(
                        f"[Shadow-{cap//10000}萬] 無持倉"
                        f"  現金={acc['cash']:,.0f}  損益={acc['pnl']:+,.0f}"
                    )

        self._send_monitoring_telegram()

    def _send_monitoring_telegram(self):
        """每次盯盤週期發送一則包含所有帳戶的 Telegram 匯總（15 分鐘冷卻）。"""
        now = time.time()
        if now - self._last_monitoring_tg_ts < 15 * 60:
            return
        try:
            from utils.telegram_notify import send as tg_send
            tw = self._now_tw()
            lines = [f"📊 當沖盤況 {tw.strftime('%H:%M')}"]

            # 真實帳戶
            real_pnl  = self._state.get("daily_pnl", 0.0)
            real_cash = self._state.get("cash", 0.0)
            real_pos  = len(self._state.get("holdings", {}))
            lines.append(
                f"[{'真' if DAY_REAL_TRADING else '紙'} {int(self._state.get('initial', DAY_INITIAL_CAPITAL))//10000}萬]"
                f" 損益={real_pnl:+,.0f}"
                f" 現金={real_cash:,.0f} | {real_pos}倉"
            )

            # 各影子帳戶
            if self._state.get("shadows"):
                for cap_key, acc in self._state.get("shadows", {}).items():
                    cap = acc["initial"]
                    lines.append(
                        f"[紙{cap//10000:3d}萬]"
                        f" 損益={acc['pnl']:+,.0f}"
                        f" 現金={acc['cash']:,.0f}"
                        f" | {len(acc['holdings'])}倉"
                    )

            msg = "\n".join(lines)
            tg_send(msg, key="day_monitoring", cooldown=1)
            self._last_monitoring_tg_ts = now
        except Exception as e:
            logger.debug(f"[DayTrader] 盯盤 Telegram 發送失敗: {e}")

    # ── 外部查詢 ─────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        with self._lock:
            s = self._state
            shadow_summary = {
                cap_key: {
                    "initial":   acc["initial"],
                    "cash":      acc["cash"],
                    "pnl":       acc["pnl"],
                    "positions": len(acc["holdings"]),
                }
                for cap_key, acc in s.get("shadows", {}).items()
            }
            return {
                "date":          s.get("date"),
                "cash":          s.get("cash", 0.0),
                "daily_pnl":     s.get("daily_pnl", 0.0),
                "positions":     len(s.get("holdings", {})),
                "holdings":      dict(s.get("holdings", {})),
                "trades_today":  len(s.get("trade_log", [])),
                "watch_count":   len(s.get("watch_log", {})),
                "mode":          "paper_day_trade" if ENABLE_DAY_TRADING else "watch_only",
                "shadows":       shadow_summary,
            }
