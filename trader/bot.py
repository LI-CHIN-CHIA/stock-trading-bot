"""
Automated trading bot.
Runs daily scan, AI prediction, and executes orders via Fubon SDK.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, date
from pathlib import Path

import pytz
import yfinance as yf
from dotenv import load_dotenv

from ai.features import build_feature_matrix, prepare_inference_row
from ai.model import TradingModel
from trader.risk import (
    MAX_POSITIONS,
    MIN_BUY_PROBA,
    buy_cost,
    can_open_position,
    get_risk_profile,
    pnl,
    position_size,
    sell_proceeds,
    should_atr_stop,
    should_stop_loss,
    should_take_profit,
)
from utils.constants import EMERGING_SYMBOLS, EMERGING_TRADEABLE, OTC_SYMBOLS, STOCK_DB, TRADEABLE
from utils.stock_discovery import get_effective_tradeable
from utils.institutional_flow import is_foreign_buying
from utils.db import get_conn

load_dotenv()
logger = logging.getLogger(__name__)

_TZ = pytz.timezone("Asia/Taipei")

# DATA_DIR 可由環境變數指定，預設為專案根目錄
# Docker 部署時設為 /data，讓每個用戶掛載自己的 volume
DATA_DIR              = Path(os.getenv("DATA_DIR", Path(__file__).parent.parent))
RETRAIN_LOG           = DATA_DIR / "retrain_history.json"
MIN_CASH_RESERVE_PCT  = float(os.getenv("MIN_CASH_RESERVE_PCT",  "0.20"))
DRAWDOWN_BREAKER_PCT  = float(os.getenv("DRAWDOWN_BREAKER_PCT",  "0.15"))
ENABLE_TA_NORMAL      = os.getenv("ENABLE_TA_FOR_NORMAL", "false").lower() == "true"
SECTOR_MAX_POSITIONS  = int(os.getenv("SECTOR_MAX_POSITIONS", "2"))
MAX_DAILY_BUYS        = int(os.getenv("MAX_DAILY_BUYS", "3"))
MAX_DAILY_SPEND       = float(os.getenv("MAX_DAILY_SPEND", "50000"))

def _state_file(paper: bool) -> Path:
    """紙上交易與實盤使用不同 state 檔，避免互相污染。"""
    name = "paper_trader_state.json" if paper else "trader_state.json"
    return DATA_DIR / name


def get_symbol(code: str) -> str:
    return f"{code}.TWO" if code in OTC_SYMBOLS else f"{code}.TW"


class TradingBot:
    """
    Main trading bot. Call run_cycle() once per trading day.
    Uses fubon_neo SDK for real order execution.
    """

    def __init__(self):
        self.model = TradingModel()
        self.sdk = None
        self.account = None
        self.holdings: dict = {}   # code -> {shares, entry_price, entry_date, cost}
        self.cash = float(os.getenv("INITIAL_CAPITAL", "20000"))
        self.trade_log: list = []
        # 掛單追蹤: order_no -> {side, code, shares, price, placed_at, retries, reason}
        self.pending_orders: dict = {}
        # 保護 cash/holdings/trade_log/pending_orders 的可重入鎖（主執行緒 + SDK callback 共用）
        self._lock = threading.RLock()
        # 紙上交易模式：不送真實委託，模擬成交；使用獨立 state 檔
        self.paper_trading: bool = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self._state_file = _state_file(self.paper_trading)
        # 實盤資金不足時自動切換紙上交易（僅影響買入，賣出仍走實盤）
        self._auto_paper = False
        self._auto_paper_min_cash = float(os.getenv("AUTO_PAPER_MIN_CASH", "3000"))
        # 當日買入計數與花費金額（跨週期持久化，隔日自動重置）
        self._daily_buy_date: str = ""
        self._daily_buy_count: int = 0
        self._daily_spend: float = 0.0
        # 市場環境過濾快取（30 分鐘 TTL）
        self._market_uptrend_cache: bool | None = None
        self._market_cache_ts: float = 0.0
        # Portfolio 回撤熔斷：追蹤總資產峰值
        self._peak_capital: float = float(os.getenv("INITIAL_CAPITAL", "20000"))
        self._load_state()

    # ── Helpers (小工具) ──────────────────────────────────────────────────────

    @staticmethod
    def _days_held(h: dict) -> int:
        """計算持倉天數，從 entry_date 到今日（含容錯）。"""
        try:
            return (date.today() - date.fromisoformat(h["entry_date"])).days
        except Exception:
            return h.get("hold_days", 0)

    def _reset_daily_counters_if_needed(self) -> None:
        """若日期已跨日，重置當日買入計數與花費。"""
        today_str = datetime.now(_TZ).strftime("%Y-%m-%d")
        if self._daily_buy_date != today_str:
            self._daily_buy_date = today_str
            self._daily_buy_count = 0
            self._daily_spend = 0.0

    def _get_technical_snapshot(self, code: str) -> dict:
        """
        取得最新技術指標快照：RSI、close_vs_ma20、volume_ratio。
        優先從 DB market_cache 讀取（特徵已由 market-data service 預算），
        fallback 為 yfinance 短線抓取。
        回傳 {"rsi": float, "close_vs_ma20": float, "volume_ratio": float}
        """
        try:
            conn = get_conn()
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT features FROM market_cache
                    WHERE code = %s
                      AND fetched_at > NOW() - INTERVAL '60 minutes'
                """, (code,))
                row = cur.fetchone()
            finally:
                conn.close()
            if row:
                import json as _json
                feats = _json.loads(row[0])
                return {
                    "rsi":           float(feats.get("RSI", 50)),
                    "close_vs_ma20": float(feats.get("close_vs_ma20", 0)),
                    "volume_ratio":  float(feats.get("Volume_Ratio", 1.0)),
                }
        except Exception:
            pass

        # Fallback: 直接抓 yfinance 30 天資料計算
        try:
            sym = get_symbol(code)
            df = yf.Ticker(sym).history(period="40d")
            if len(df) < 20:
                return {}
            from analysis.indicators import TechnicalIndicators
            df = TechnicalIndicators.add_all(df)
            last = df.iloc[-1]
            ma20 = last.get("MA20", 0)
            vol5 = df["Volume"].iloc[-5:].mean()
            return {
                "rsi":           float(last.get("RSI", 50)),
                "close_vs_ma20": float((last["Close"] - ma20) / ma20) if ma20 > 0 else 0,
                "volume_ratio":  float(last["Volume"] / vol5) if vol5 > 0 else 1.0,
            }
        except Exception:
            return {}

    def _entry_quality_ok(self, code: str, snap: dict) -> bool:
        """
        進場品質三重確認：
          1. RSI < 65（非超買）
          2. 收盤 > MA20（多頭趨勢）
          3. 成交量 ≥ 5 日均量 × 0.6（非死水）
          4. 外資今日不賣超（或無三大法人資料 → 放行）
        任一不符合即回傳 False。
        """
        if not snap:
            return True  # 無資料時不阻擋

        rsi          = snap.get("rsi", 50)
        close_vs_ma20= snap.get("close_vs_ma20", 0)
        volume_ratio = snap.get("volume_ratio", 1.0)

        if rsi >= 65:
            logger.info(f"⚠️  {code} 進場品質: RSI={rsi:.1f} 超買，跳過")
            return False
        if close_vs_ma20 < -0.02:   # 收盤低於 MA20 超過 2%
            logger.info(f"⚠️  {code} 進場品質: 價格低於MA20 {close_vs_ma20:.1%}，跳過")
            return False
        if volume_ratio < 0.6:
            logger.info(f"⚠️  {code} 進場品質: 成交量萎縮 {volume_ratio:.1f}x，跳過")
            return False

        foreign_ok = is_foreign_buying(code)
        if foreign_ok is False:
            logger.info(f"⚠️  {code} 進場品質: 外資今日淨賣超，跳過")
            return False

        return True

    # ── SDK Login ─────────────────────────────────────────────────────────────

    def login(self) -> bool:
        """Login to Fubon SDK and register real-time fill callbacks."""
        if self.paper_trading:
            logger.info("📄 紙上交易模式：跳過 SDK 登入，使用模擬帳戶")
            return True
        try:
            from fubon_neo.sdk import FubonSDK
            self.sdk = FubonSDK()
            result = self.sdk.login(
                os.getenv("FUBON_ID"),
                os.getenv("FUBON_PASSWORD"),
                os.getenv("FUBON_CERT_PATH"),
                os.getenv("FUBON_CERT_PASSWORD", ""),
            )
            if result.is_success:
                self.account = result.data[0]
                logger.info(f"✅ 富邦登入成功: {self.account}")
                # 註冊即時成交回調，成交瞬間更新持倉
                self.sdk.set_on_filled(self._on_filled)
                self.sdk.set_on_order(self._on_order)
                logger.info("✅ 即時成交回調已註冊")
                return True
            else:
                logger.error(f"❌ 富邦登入失敗: {result.message}")
                return False
        except Exception as e:
            logger.error(f"❌ SDK 錯誤: {e}")
            return False

    def _on_filled(self, data) -> None:
        """
        即時成交回調 — 成交瞬間由 SDK 觸發（SDK callback thread）。
        用 _lock 確保與主執行緒的狀態更新互斥。
        先移除 pending_orders 再更新狀態，防止 process_pending_orders 重複結算。
        """
        try:
            seq_no = getattr(data, "seq_no", None) or getattr(data, "order_no", None)
            code   = str(getattr(data, "stock_no", "") or "").strip()
            side   = getattr(data, "buy_sell", None)
            filled = int(getattr(data, "filled_qty", 0) or getattr(data, "quantity", 0) or 0)
            price  = float(getattr(data, "filled_price", 0) or getattr(data, "price", 0) or 0)

            if not code or filled <= 0:
                return

            is_buy = str(side).upper() in ("BUY", "BSACTION.BUY")

            with self._lock:
                # 先從 pending_orders 移除，防止 process_pending_orders 重複結算
                removed = False
                if seq_no and seq_no in self.pending_orders:
                    self.pending_orders.pop(seq_no)
                    removed = True
                else:
                    # seq_no 為 None 或不在 dict 時，按 code+side 找
                    side_str = "BUY" if is_buy else "SELL"
                    stale = [k for k, v in self.pending_orders.items()
                             if v["code"] == code and v["side"] == side_str]
                    for k in stale:
                        self.pending_orders.pop(k)
                    removed = bool(stale)

                if is_buy:
                    actual_cost = buy_cost(filled, price)
                    self.cash -= actual_cost
                    if code in self.holdings:
                        h = self.holdings[code]
                        total = h["shares"] + filled
                        avg   = (h["entry_price"] * h["shares"] + price * filled) / total
                        h["shares"]      = total
                        h["entry_price"] = avg
                        h["cost"]        = h["cost"] + actual_cost
                    else:
                        self.holdings[code] = {
                            "shares": filled,
                            "entry_price": price,
                            "entry_date": datetime.now(_TZ).strftime("%Y-%m-%d"),
                            "cost": actual_cost,
                            "buy_proba": 0,
                            "peak_price": price,
                        }
                    self.trade_log.append({
                        "date":   datetime.now(_TZ).strftime("%Y-%m-%d"),
                        "action": "BUY",
                        "ticker": code,
                        "shares": filled,
                        "price":  price,
                    })
                    logger.info(f"🔔 即時成交(BUY): {code} {filled}股 @ {price:.2f} 成本={actual_cost:.0f}")
                else:
                    proceeds = sell_proceeds(filled, price)
                    self.cash += proceeds
                    if code in self.holdings:
                        h = self.holdings[code]
                        remain = h["shares"] - filled
                        if remain <= 0:
                            abs_pnl, pct_pnl = pnl(h["entry_price"], price, filled)
                            self.trade_log.append({
                                "date":    datetime.now(_TZ).strftime("%Y-%m-%d"),
                                "action":  "SELL",
                                "ticker":  code,
                                "shares":  filled,
                                "price":   price,
                                "pnl":     abs_pnl,
                                "pnl_pct": pct_pnl,
                            })
                            del self.holdings[code]
                        else:
                            h["shares"] = remain
                    logger.info(f"🔔 即時成交(SELL): {code} {filled}股 @ {price:.2f} 收入={proceeds:.0f}")

                self._save_state()

        except Exception as e:
            logger.error(f"即時成交回調錯誤: {e}", exc_info=True)

    def _on_order(self, data) -> None:
        """
        委託狀態變更回調 — 委託被拒或取消時清除 pending。
        """
        try:
            seq_no  = getattr(data, "seq_no", None)
            status  = getattr(data, "status", None)
            err_msg = getattr(data, "error_message", None) or ""
            code    = str(getattr(data, "stock_no", "") or "").strip()

            if status == 90 or err_msg.strip():
                if seq_no and seq_no in self.pending_orders:
                    info = self.pending_orders.pop(seq_no)
                    logger.warning(f"🔔 委託被拒/取消 {code} 序號={seq_no}: {err_msg.strip()}")
                    self._save_state()
        except Exception as e:
            logger.error(f"委託狀態回調錯誤: {e}", exc_info=True)

    # ── Main Cycle ────────────────────────────────────────────────────────────

    def run_cycle(self):
        """
        Main daily trading cycle:
        1. Load/retrain model
        2. Sync real account balance from Fubon
        3. Check holdings (stop-loss/take-profit/AI sell)
        4. Scan for new buy opportunities
        5. Execute orders
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        logger.info(f"=== 交易循環開始: {now} ===")

        # Load or train model
        if not self.model.load():
            logger.info("無模型，先進行訓練...")
            self._train_model()

        # Sync real balance and reconcile holdings with Fubon
        self._sync_account_balance()
        self._sync_holdings_from_fubon()

        # Check existing holdings first
        self._check_holdings()

        # Scan for new opportunities
        self._scan_and_buy()

        self._save_state()
        logger.info("=== 交易循環結束 ===")

    def _unsettled_buy_cost(self) -> float:
        """
        計算尚未交割的買入成本（T+2 交割制度）。
        台股買股後第 2 個交易日才正式扣款，因此富邦 available_balance
        在交割前仍會顯示較高的數字。
        取最近 2 個自然日內的 BUY 紀錄成本加總作為估算值。
        """
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        total = 0.0
        for t in self.trade_log:
            if t.get("action") == "BUY" and t.get("date", "") >= cutoff:
                shares = t.get("shares", 0)
                price  = t.get("price", 0.0)
                from trader.risk import TRANSACTION_FEE
                total += shares * price * (1 + TRANSACTION_FEE)
        return total

    def _sync_account_balance(self):
        """
        Query Fubon API for real cash balance and sync with local state.
        台股 T+2 交割：富邦 available_balance 在交割前不會反映近期買入扣款，
        因此用「富邦餘額 - 未交割買入成本」作為有效可用現金。
        Falls back gracefully if SDK is not connected.
        """
        if self.paper_trading or self.sdk is None or self.account is None:
            return  # paper mode 直接使用本地現金記錄
        try:
            result = self.sdk.accounting.bank_remain(self.account)
            if result.is_success and result.data:
                fubon_balance = float(result.data.available_balance)
                unsettled = self._unsettled_buy_cost()
                # 有效可用現金 = 富邦餘額 - 尚未交割的買入成本
                effective_cash = max(fubon_balance - unsettled, 0.0)
                diff = abs(effective_cash - self.cash)
                logger.info(
                    f"💰 富邦餘額={fubon_balance:,.0f} 未交割={unsettled:,.0f} "
                    f"有效可用={effective_cash:,.0f} 本地={self.cash:,.0f}"
                )
                if diff > 500:
                    logger.warning(
                        f"⚠️  餘額仍有差異 {diff:.0f} 元（扣除未交割後），使用本地記錄"
                    )
                    self.cash = min(effective_cash, self.cash)
                else:
                    self.cash = effective_cash

                # 資金不足時自動切換紙上交易
                if self.cash < self._auto_paper_min_cash:
                    if not self._auto_paper:
                        logger.warning(
                            f"💸 實盤資金不足 {self.cash:,.0f} 元（門檻 {self._auto_paper_min_cash:,.0f}），"
                            f"自動切換紙上交易模式"
                        )
                    self._auto_paper = True
                else:
                    if self._auto_paper:
                        logger.info(
                            f"💰 實盤資金恢復 {self.cash:,.0f} 元，回到實盤交易模式"
                        )
                    self._auto_paper = False
        except Exception as e:
            logger.warning(f"餘額同步失敗（使用本地記錄）: {e}")

    def retrain(self, reason: str = "排程重訓"):
        """Retrain model with latest 2 years of data. Records metrics to retrain_history.json."""
        started_at = datetime.now()
        logger.info(f"{'='*50}")
        logger.info(f"🔄 開始重新訓練  原因: {reason}")
        logger.info(f"   時間: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            from ai.backtest import run_backtest
            result = run_backtest(self.model, period="2y", verbose=False)
            self.model.save()
            self._last_retrain = datetime.now()
            duration = (datetime.now() - started_at).seconds

            # 記錄訓練結果
            record = {
                "timestamp": started_at.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": reason,
                "duration_sec": duration,
                "model_version": self.model.version,
                "backtest": {
                    "total_return_pct": result.total_return_pct,
                    "annualised_return_pct": result.annualised_return_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "win_rate_pct": result.win_rate_pct,
                    "total_trades": result.total_trades,
                },
            }
            self._append_retrain_log(record)

            logger.info(f"✅ 重訓完成  耗時: {duration}s")
            logger.info(f"   模型版本:   {self.model.version}")
            logger.info(f"   回測報酬:   {result.total_return_pct:+.2f}%")
            logger.info(f"   夏普比率:   {result.sharpe_ratio:.2f}")
            logger.info(f"   最大回撤:   {result.max_drawdown_pct:.2f}%")
            logger.info(f"   勝率:       {result.win_rate_pct:.1f}%")
            logger.info(f"{'='*50}")

        except Exception as e:
            logger.error(f"❌ 重訓失敗: {e}", exc_info=True)
        finally:
            self._save_state()

    def _append_retrain_log(self, record: dict):
        """Append a retrain record to retrain_history.json."""
        history = []
        if RETRAIN_LOG.exists():
            try:
                with open(RETRAIN_LOG) as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(record)
        with open(RETRAIN_LOG, "w") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def should_retrain(self) -> tuple[bool, str]:
        """
        Check whether any condition warrants an emergency retrain.
        Returns (should_retrain: bool, reason: str).

        Conditions:
          1. 模型過期     — 上次訓練超過 7 天
          2. 績效惡化     — 近 5 筆勝率 < 40%
          3. 連續虧損     — 最近 3 筆都虧損
          4. 市場劇變     — 大盤 (0050) 單日漲跌 > 3%
        """
        # ── 1. 模型過期 ───────────────────────────────────────────────────────
        last = getattr(self, "_last_retrain", None)
        if last is None:
            # 從模型版本字串推算（格式 YYYYMMDD_HHMM）
            ver = self.model.version
            if ver and len(ver) >= 8:
                try:
                    last = datetime.strptime(ver[:13], "%Y%m%d_%H%M")
                except ValueError:
                    last = None
        if last is None or (datetime.now() - last).days >= 7:
            return True, "模型已超過 7 天未更新"

        # ── 2. 績效惡化：近 5 筆勝率 < 40% ──────────────────────────────────
        recent = [t for t in self.trade_log if t.get("action") == "SELL"][-5:]
        if len(recent) >= 5:
            wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
            if wins / 5 < 0.40:
                return True, f"近5筆勝率僅 {wins}/5 ({wins/5:.0%}) 低於 40%"

        # ── 3. 連續虧損：最近 3 筆都虧 ───────────────────────────────────────
        sell_log = [t for t in self.trade_log if t.get("action") == "SELL"][-3:]
        if len(sell_log) >= 3 and all(t.get("pnl", 0) <= 0 for t in sell_log):
            return True, "連續 3 筆交易虧損"

        # ── 4. 市場劇變：0050 單日漲跌 > 3%（每日最多觸發一次）────────────
        last_emergency = getattr(self, "_last_emergency_retrain", None)
        today_str = datetime.now().strftime("%Y-%m-%d")
        if last_emergency == today_str:
            return False, ""  # 今天已緊急重訓過，不再重複
        try:
            df = yf.Ticker("0050.TW").history(period="3d")
            if len(df) >= 2:
                chg = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]
                if abs(chg) > 0.03:
                    return True, f"大盤單日波動 {chg:+.1%}，市場劇變"
        except Exception:
            pass

        return False, ""

    # ── Holdings Management ───────────────────────────────────────────────────

    def _check_holdings(self):
        # 先處理所有未成交掛單
        self.process_pending_orders()

        for code in list(self.holdings.keys()):
            with self._lock:
                h = self.holdings.get(code)
            if h is None:
                continue  # 已被 _on_filled 移除
            # 跳過已有掛賣單的持倉，避免重複下單
            with self._lock:
                has_pending_sell = any(
                    o.get("side") == "SELL" and o.get("code") == code
                    for o in self.pending_orders.values()
                )
            if has_pending_sell:
                logger.debug(f"{code} 已有掛賣單，跳過本次監控")
                continue

            price = self._get_latest_price(code)
            if price <= 0:
                continue

            profile = get_risk_profile(code)

            # Update trailing peak price (never moves down)
            h["peak_price"] = max(h.get("peak_price", h["entry_price"]), price)

            abs_pnl, pct_pnl = pnl(h["entry_price"], price, h["shares"])
            hold_days = self._days_held(h)
            h["hold_days"] = hold_days
            reason = None
            sell_proba = 0.0

            # Get AI signal once — reuse for both ATR stop and strategy B
            signal = self._get_ai_signal(code)
            atr = signal.get("atr", 0.0) if signal else h.get("atr", 0.0)

            # ── 1. ATR trailing stop (primary stop mechanism) ─────────────────
            if should_atr_stop(h["peak_price"], price, atr, profile):
                reason = (f"ATR追蹤停損[{profile.label}] {pct_pnl:.1%} "
                          f"(peak={h['peak_price']:.2f}, atr={atr:.2f}, x{profile.atr_multiplier})")
                sell_proba = 0.95
            # ── 2. Fixed stop-loss fallback (when ATR unavailable) ────────────
            elif atr <= 0 and should_stop_loss(h["entry_price"], price, profile):
                reason = f"停損[{profile.label}] {pct_pnl:.1%} (門檻{profile.stop_loss_pct:.0%})"
                sell_proba = 0.95
            # ── 3. Take-profit（興櫃：直接出場；一般：Strategy B AI 確認）────
            elif should_take_profit(h["entry_price"], price, profile):
                if not profile.is_emerging:
                    buy_proba = signal.get("buy_proba", 0) if signal else 0
                    if signal and signal.get("signal") == "BUY" and buy_proba >= 0.70:
                        logger.info(
                            f"📈 {code} 達停利 {pct_pnl:.1%} 但 AI 仍看多 ({buy_proba:.0%})，繼續持有"
                        )
                    else:
                        reason = f"停利 {pct_pnl:.1%}"
                        sell_proba = 0.80
                else:
                    # 興櫃：到達停利直接出場，不等 AI 確認
                    reason = f"停利[興櫃] {pct_pnl:.1%} (門檻{profile.take_profit_pct:.0%})"
                    sell_proba = 0.90
            # ── 4. 最大持有天數 ───────────────────────────────────────────────
            elif hold_days >= profile.max_hold_days:
                reason = f"超過持有期[{profile.label}] {hold_days}天 ({pct_pnl:.1%})"
                sell_proba = 0.80
            # ── 5. LightGBM sell signal ───────────────────────────────────────
            elif signal and signal.get("signal") == "SELL" and signal.get("sell_proba", 0) >= 0.55:
                sell_proba = signal["sell_proba"]
                reason = f"AI賣出 {sell_proba:.0%}"
            # ── 6. TradingAgents 主動詢問（每次監控都問，不受門檻限制）───────
            else:
                ta_opinion = self._get_ta_holding_opinion(code, h, price, pct_pnl, profile)
                if ta_opinion:
                    reason, sell_proba = ta_opinion

            if reason:
                proceeds = self._execute_sell(code, h["shares"], price, reason, proba=sell_proba)
                if self.paper_trading:
                    # 紙上交易：立即結算
                    if proceeds > 0:
                        with self._lock:
                            self.cash += proceeds
                            self.holdings.pop(code, None)
                            self.trade_log.append({
                                "date":    datetime.now(_TZ).strftime("%Y-%m-%d"),
                                "action":  "SELL",
                                "ticker":  code,
                                "shares":  h["shares"],
                                "price":   price,
                                "pnl":     abs_pnl,
                                "pnl_pct": pct_pnl,
                                "reason":  reason,
                            })
                        logger.info(f"賣出 {code}: {reason}, 損益 {abs_pnl:+.0f} 元")
                    else:
                        logger.warning(f"⚠️  {code} 賣出未成交，繼續持有")
                else:
                    # 實盤：委託已送出（proceeds=0），_on_filled 會處理 cash/holdings
                    if proceeds == 0.0 and code not in [o.get("code") for o in self.pending_orders.values()]:
                        logger.warning(f"⚠️  {code} 賣出委託失敗，繼續持有")
                    else:
                        logger.info(f"賣出委託送出 {code}: {reason}, 等待成交")

    def _get_ta_holding_opinion(
        self, code: str, h: dict, price: float, pct_pnl: float, profile=None
    ) -> tuple[str, float] | None:
        """
        每次監控都詢問 TradingAgents 對持倉股票的看法。
        不受停損/停利門檻限制，只要 TA 明確建議賣就執行。
        興櫃股票使用更低的信心門檻（0.50）和更短的 TTL（30分鐘）。

        回傳 (reason, sell_proba) 或 None（不賣）。
        """
        from trader.ta_signal import ENABLE_TA, get_ta_signal
        if not ENABLE_TA:
            return None
        if profile is None:
            profile = get_risk_profile(code)

        days_held = self._days_held(h)

        # 持有 1 天以上且損益未達 ±3%：讓部位有時間發展（當天不受此限）
        if days_held >= 1 and days_held < 5 and abs(pct_pnl) < 0.03:
            logger.info(
                f"🤖 TA {code}: 持有 {days_held} 天，損益 {pct_pnl:+.1%} 未達介入門檻（±3%），繼續持有"
            )
            return None

        # 興櫃信心門檻 0.65；一般股票提高至 0.75（避免輕微賣出訊號過度交易）
        sell_confidence_threshold = 0.65 if profile.is_emerging else 0.75
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            ta = get_ta_signal(code, today, mode="holding",
                               ttl_override=profile.ta_ttl_min)
            if ta is None:
                return None

            ta_sig  = ta.get("signal", "HOLD")
            ta_conf = ta.get("confidence", 0.0)

            if ta_sig == "SELL" and ta_conf >= sell_confidence_threshold:
                label = f"[{profile.label}]" if profile.is_emerging else ""
                reason = f"TradingAgents建議賣出{label} (信心{ta_conf:.0%}, 損益{pct_pnl:+.1%})"
                logger.info(f"🤖 TA {code}: SELL 信心={ta_conf:.0%} → 賣出")
                return reason, ta_conf

            logger.info(
                f"🤖 TA {code}[{profile.label}]: {ta_sig} 信心={ta_conf:.0%} | "
                f"持有{days_held}天 損益{pct_pnl:+.1%} → 繼續持有"
            )
            return None
        except Exception as e:
            logger.debug(f"_get_ta_holding_opinion {code} 失敗: {e}")
            return None

    def _is_market_uptrend(self) -> bool:
        """
        台股加權指數 (^TWII) 收盤 > 20 日均線 → 多頭，允許開倉。
        結果快取 30 分鐘；無法取得資料時預設允許開倉（保守 fallback）。
        """
        CACHE_TTL = 1800
        if (self._market_uptrend_cache is not None
                and time.time() - self._market_cache_ts < CACHE_TTL):
            return self._market_uptrend_cache
        try:
            df = yf.Ticker("^TWII").history(period="60d")
            if len(df) < 21:
                return True
            ma20  = df["Close"].rolling(20).mean().iloc[-1]
            price = df["Close"].iloc[-1]
            uptrend = bool(price > ma20)
            self._market_uptrend_cache = uptrend
            self._market_cache_ts = time.time()
            tag = "多頭" if uptrend else "空頭"
            logger.info(f"📊 市場環境: ^TWII {price:,.0f} vs MA20 {ma20:,.0f} → {tag}")
            return uptrend
        except Exception as e:
            logger.warning(f"市場環境判斷失敗（預設允許開倉）: {e}")
            return True

    def _sector_for(self, code: str) -> str:
        from utils.constants import SECTOR_MAP
        for sector, codes in SECTOR_MAP.items():
            if code in codes:
                return sector
        return "其他"

    def _sector_count_in_holdings(self, sector: str) -> int:
        return sum(1 for c in self.holdings if self._sector_for(c) == sector)

    def _scan_and_buy(self):
        if not can_open_position(self.holdings, self.cash, 1):
            return

        # ── 當日買入上限 ──────────────────────────────────────────────────────────
        self._reset_daily_counters_if_needed()
        if self._daily_buy_count >= MAX_DAILY_BUYS:
            logger.info(
                f"📅 當日買入筆數已達上限 {MAX_DAILY_BUYS} 筆（今日已買 {self._daily_buy_count} 筆），停止掃股"
            )
            return
        if self._daily_spend >= MAX_DAILY_SPEND:
            logger.info(
                f"📅 當日買入金額已達上限 {MAX_DAILY_SPEND:,.0f} 元"
                f"（今日已花 {self._daily_spend:,.0f} 元），停止掃股"
            )
            return

        # ── 市場環境過濾 ──────────────────────────────────────────────────────────
        if not self._is_market_uptrend():
            logger.info("📊 大盤處於空頭（^TWII < MA20），暫停所有新開倉")
            return

        # 總資產（用於計算每筆上限）
        # 若 _get_latest_price 回傳 0（盤前/盤後/資料不可用），以 entry_price 作為保守估算
        holding_value = 0.0
        prices_available = True
        for c, h in self.holdings.items():
            price = self._get_latest_price(c)
            if price <= 0:
                price = h.get("entry_price", 0.0)
                prices_available = False
            holding_value += h["shares"] * price
        total_capital = self.cash + holding_value

        # ── 最低現金保留 ───────────────────────────────────────────────────────────
        min_reserve = total_capital * MIN_CASH_RESERVE_PCT
        if self.cash <= min_reserve:
            logger.info(
                f"💰 現金 {self.cash:,.0f} ≤ 保留門檻 {min_reserve:,.0f} "
                f"({total_capital:,.0f} × {MIN_CASH_RESERVE_PCT:.0%})，停止買入"
            )
            return

        # ── Portfolio 回撤熔斷 ────────────────────────────────────────────────────
        # 只在價格資料完整時才更新峰值並檢查回撤，避免盤前/盤後價格缺失造成誤觸發
        if prices_available or not self.holdings:
            self._peak_capital = max(self._peak_capital, total_capital)
        if self._peak_capital > 0 and (prices_available or not self.holdings):
            drawdown = (self._peak_capital - total_capital) / self._peak_capital
            if drawdown >= DRAWDOWN_BREAKER_PCT:
                logger.warning(
                    f"🔴 Portfolio 回撤 {drawdown:.1%} ≥ 熔斷門檻 {DRAWDOWN_BREAKER_PCT:.0%}"
                    f"（峰值 {self._peak_capital:,.0f} → 現值 {total_capital:,.0f}），暫停新開倉"
                )
                return

        # 興櫃：13:00 後不新開倉（流動性差，來不及成交）
        now_tw = datetime.now(_TZ)
        allow_emerging = now_tw.hour < 13

        # 當日已賣出的股票（避免同日買回，Fubon 零股不允許）
        today = now_tw.strftime("%Y-%m-%d")
        sold_today = {t["ticker"] for t in self.trade_log if t.get("action") == "SELL" and t.get("date") == today}

        # ── 一般股票候選 ───────────────────────────────────────────────────────
        tradeable = self._get_tradeable_from_db() or get_effective_tradeable()
        candidates = []
        for code in tradeable:
            if code in self.holdings or code in EMERGING_TRADEABLE or code in sold_today:
                continue
            profile = get_risk_profile(code)
            signal = self._get_ai_signal(code)
            if signal and signal.get("signal") == "BUY" and signal.get("buy_proba", 0) >= profile.min_buy_proba:
                price = self._get_latest_price(code)
                if price > 0:
                    snap = self._get_technical_snapshot(code)
                    if not self._entry_quality_ok(code, snap):
                        continue
                    candidates.append((code, signal["buy_proba"], price, signal.get("atr", 0.0), profile))

        # ── 興櫃候選（需 TA 確認 + 時間限制 + 現有興櫃持倉 < 1）────────────
        if allow_emerging and EMERGING_TRADEABLE:
            current_emerging = sum(1 for c in self.holdings if c in EMERGING_TRADEABLE)
            if current_emerging < 1:
                for code in EMERGING_TRADEABLE:
                    if code in self.holdings or code in sold_today:
                        continue
                    profile = get_risk_profile(code)  # always EMERGING
                    signal = self._get_ai_signal(code)
                    if not (signal and signal.get("signal") == "BUY"
                            and signal.get("buy_proba", 0) >= profile.min_buy_proba):
                        continue
                    # 興櫃：額外要求 TradingAgents 確認
                    ta_ok = self._ta_confirm_buy(code)
                    if not ta_ok:
                        logger.info(f"⚠️  興櫃 {code} LightGBM BUY 但 TradingAgents 未確認，跳過")
                        continue
                    price = self._get_latest_price(code)
                    if price > 0:
                        candidates.append((code, signal["buy_proba"], price, signal.get("atr", 0.0), profile))

        candidates.sort(key=lambda x: x[1], reverse=True)
        slots = MAX_POSITIONS - len(self.holdings)

        # ── 一般股票 TA 二次確認（ENABLE_TA_FOR_NORMAL=true 時啟用）────────────
        if ENABLE_TA_NORMAL:
            ta_passed: list = []
            ta_skipped: list[str] = []
            for item in candidates:
                if item[4].is_emerging:
                    ta_passed.append(item)  # 興櫃已在上方做過 TA 確認
                elif len(ta_passed) < slots and self._ta_confirm_buy(item[0]):
                    ta_passed.append(item)
                else:
                    ta_skipped.append(item[0])
            if ta_skipped:
                logger.info(f"🤖 TA 一般股過濾跳過: {', '.join(ta_skipped)}")
            candidates = ta_passed

        for code, buy_proba, price, atr, profile in candidates[:slots]:
            if not can_open_position(self.holdings, self.cash, price, profile):
                continue
            # ── 類股分散限制 ──────────────────────────────────────────────────────
            sector = self._sector_for(code)
            if self._sector_count_in_holdings(sector) >= SECTOR_MAX_POSITIONS:
                logger.info(f"⚖️  {code}[{sector}] 類股已達上限 {SECTOR_MAX_POSITIONS} 支，跳過")
                continue
            shares = position_size(self.cash, price, slots, total_capital, profile)
            if shares < 1:
                continue
            cost = buy_cost(shares, price)
            available_cash = self.cash - min_reserve
            if cost > available_cash:
                continue
            # ── 當日花費上限：這筆買入後是否超標 ────────────────────────────────
            if self._daily_spend + cost > MAX_DAILY_SPEND:
                logger.info(
                    f"📅 {code} 成本 {cost:,.0f} 元會超出當日上限 {MAX_DAILY_SPEND:,.0f} 元"
                    f"（今日已花 {self._daily_spend:,.0f}），跳過"
                )
                continue

            filled_shares = self._execute_buy(code, shares, price, proba=buy_proba)
            label = "[興櫃]" if profile.is_emerging else ""
            if self.paper_trading or self._auto_paper:
                # 紙上交易：立即入帳
                if filled_shares > 0:
                    actual_cost = buy_cost(filled_shares, price)
                    with self._lock:
                        self.cash -= actual_cost
                        self.holdings[code] = {
                            "shares": filled_shares,
                            "entry_price": price,
                            "entry_date": datetime.now(_TZ).strftime("%Y-%m-%d"),
                            "cost": actual_cost,
                            "buy_proba": buy_proba,
                            "atr": atr,
                            "peak_price": price,
                            "hold_days": 0,
                        }
                        self.trade_log.append({
                            "date":    datetime.now(_TZ).strftime("%Y-%m-%d"),
                            "action":  "BUY",
                            "ticker":  code,
                            "shares":  filled_shares,
                            "price":   price,
                            "buy_proba": buy_proba,
                            "market":  "emerging" if profile.is_emerging else "normal",
                        })
                    self._daily_buy_count += 1
                    self._daily_spend += actual_cost
                    logger.info(
                        f"買入{label} {code}: {filled_shares}股 @ {price:.2f}, AI信心 {buy_proba:.0%} "
                        f"（今日第 {self._daily_buy_count}/{MAX_DAILY_BUYS} 筆，"
                        f"今日已花 {self._daily_spend:,.0f}/{MAX_DAILY_SPEND:,.0f} 元）"
                    )
            else:
                # 實盤：委託已送出（filled_shares=0），_on_filled 負責入帳
                # 仍計入當日限制（已送出即視為用掉額度）
                if code in [o.get("code") for o in self.pending_orders.values()]:
                    self._daily_buy_count += 1
                    self._daily_spend += cost
                    logger.info(
                        f"買入委託{label} {code}: {shares}股 @ {price:.2f}, AI信心 {buy_proba:.0%} "
                        f"（今日第 {self._daily_buy_count}/{MAX_DAILY_BUYS} 筆，等待成交）"
                    )

    def _ta_confirm_buy(self, code: str) -> bool:
        """
        向 TradingAgents 確認是否買入。
        用於興櫃股票的額外把關（require_ta=True）。
        若 ENABLE_TA=false 或 TA 無回應，預設拒絕（保守策略）。
        """
        from trader.ta_signal import ENABLE_TA, get_ta_signal
        if not ENABLE_TA:
            logger.info(f"興櫃 {code}: ENABLE_TRADING_AGENTS=false，跳過（不買）")
            return False
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            ta = get_ta_signal(code, today, mode="scan")
            if ta is None:
                return False
            sig  = ta.get("signal", "HOLD")
            conf = ta.get("confidence", 0.0)
            approved = sig == "BUY" and conf >= 0.60
            logger.info(f"🤖 TA 興櫃確認 {code}: {sig} 信心={conf:.0%} → {'✅核准' if approved else '❌拒絕'}")
            return approved
        except Exception as e:
            logger.warning(f"TA 興櫃確認 {code} 失敗: {e}")
            return False

    # ── Order Execution ───────────────────────────────────────────────────────

    def _odd_market_type(self):
        """盤中(09:00~13:30)用 IntradayOdd，盤後(13:40~14:30)用 Odd。"""
        from fubon_neo.constant import MarketType
        from datetime import time
        now = datetime.now(_TZ).time()
        if time(9, 0) <= now <= time(13, 30):
            return MarketType.IntradayOdd
        return MarketType.Odd

    @staticmethod
    def _tick_round(price: float, floor: bool = False) -> float:
        """
        Round price to Taiwan stock exchange tick size:
          < 10:    0.01
          10–50:   0.05
          50–100:  0.10
          100–500: 0.50
          ≥ 500:   1.00
        If floor=True, always round down (used for BUY limit to avoid exceeding exchange limit-up).
        """
        import math
        if price < 10:
            tick = 0.01
        elif price < 50:
            tick = 0.05
        elif price < 100:
            tick = 0.10
        elif price < 500:
            tick = 0.50
        else:
            tick = 1.00
        if floor:
            return round(math.floor(price / tick) * tick, 2)
        return round(round(price / tick) * tick, 2)

    def _ai_limit_price(self, code: str, side: str, base_price: float, proba: float = 0.0) -> float:
        """
        BUY: 使用接近漲停板價（+9.8%，向下取整），確保盤中零股撮合成交。
             零股是逐筆撮合，出高限價代表「願意接受任何低於此價的成交」，
             實際成交價是當下市場賣價，不會真的用限價買進。
             向下取整避免因當下市價≠昨收而超過交易所漲停板上限。
        SELL: 依 AI 信心決定折讓幅度。
        所有價格均對齊台股最小跳動單位，避免「單價輸入錯誤」。
        """
        if side == "BUY":
            limit_price = self._tick_round(base_price * 1.098, floor=True)
            logger.info(f"AI限價: {code} BUY base={base_price} proba={proba:.0%} → {limit_price} (接近漲停，確保成交)")
        else:  # SELL
            offset = 0.99 if proba >= 0.70 else 0.98
            limit_price = self._tick_round(base_price * offset)
            logger.info(f"AI限價: {code} SELL base={base_price} proba={proba:.0%} → {limit_price} (offset={offset})")
        return limit_price

    def _is_market_open(self) -> bool:
        """盤中零股 09:00~13:30，盤後零股 13:40~14:30，其餘時間不允許下單。"""
        from datetime import time as dtime
        now = datetime.now(_TZ).time()
        return dtime(9, 0) <= now <= dtime(13, 30) or dtime(13, 40) <= now <= dtime(14, 30)

    def _place_order(self, code: str, side: str, shares: int, price: float, reason: str = "", proba: float = 0.0) -> str | None:
        """
        Place a buy or sell order. Returns order_no if submitted, else None.
        側: "BUY" or "SELL"
        非市場時間（09:00~13:30 盤中、13:40~14:30 盤後）不下單。
        """
        if not self._is_market_open():
            now_str = datetime.now(_TZ).strftime("%H:%M")
            logger.warning(f"⛔ 非市場時間 ({now_str})，不下單: {side} {code}")
            return None

        from fubon_neo.sdk import Order
        from fubon_neo.constant import BSAction, MarketType, OrderType, PriceType, TimeInForce
        # 興櫃用 Emg / EmgOdd，一般股票用零股市場
        if code in EMERGING_TRADEABLE:
            market_type = MarketType.Emg
        else:
            market_type = self._odd_market_type()
        limit_price = self._ai_limit_price(code, side, price, proba)
        if side == "BUY":
            bs = BSAction.Buy
        else:
            bs = BSAction.Sell
        order = Order(
            buy_sell=bs,
            symbol=code,
            price=str(limit_price),
            quantity=shares,
            market_type=market_type,
            price_type=PriceType.Limit,
            time_in_force=TimeInForce.ROD,
            order_type=OrderType.Stock,
            user_def="AI_BOT",
        )
        result = self.sdk.stock.place_order(self.account, order)
        if result.is_success:
            # Fubon SDK uses seq_no as the unique order identifier (order_no is always None)
            seq_no = getattr(result.data, "seq_no", None) if result.data else None
            logger.info(f"委託送出: {side} {code} {shares}股 @ {limit_price} 序號={seq_no}")
            return seq_no
        else:
            logger.error(f"委託失敗: {side} {code} {result.message}")
            return None

    def _execute_buy(self, code: str, shares: int, price: float, proba: float = 0.0) -> int:
        """
        Place buy order and add to pending_orders. Returns shares for paper mode (immediate),
        or 0 for real mode (cash/holdings updated by _on_filled callback).
        Real mode with sdk=None: skip entirely — never simulate real orders.
        """
        if self.paper_trading or self._auto_paper:
            cost = buy_cost(shares, price)
            tag = "紙上交易" if self.paper_trading else "資金不足→紙上交易"
            logger.info(f"📄 [{tag}] 模擬買入 {code}: {shares}股 @ {price:.2f} "
                        f"成本={cost:,.0f}元 AI信心={proba:.0%}")
            return shares
        if self.sdk is None or self.account is None:
            logger.warning(f"SDK未連線，跳過買入 {code}（實盤不模擬）")
            return 0
        try:
            order_no = self._place_order(code, "BUY", shares, price, proba=proba)
            if not order_no:
                return 0
            with self._lock:
                self.pending_orders[order_no] = {
                    "side": "BUY", "code": code, "shares": shares,
                    "price": price, "placed_at": datetime.now(_TZ).isoformat(),
                    "retries": 0, "reason": "", "proba": proba,
                }
            logger.info(f"⏳ {code} 買單送出，等待 _on_filled 確認 (單號={order_no})")
            return 0  # cash/holdings 由 _on_filled 更新，不在此處雙重計算
        except Exception as e:
            logger.error(f"下單錯誤: {e}")
            return 0

    def _execute_sell(self, code: str, shares: int, price: float, reason: str, proba: float = 0.0) -> float:
        """
        Place sell order and add to pending_orders. Returns proceeds for paper mode (immediate),
        or 0.0 for real mode (cash/holdings updated by _on_filled callback).
        Real mode with sdk=None: skip entirely — never simulate real orders.
        """
        if self.paper_trading:
            proceeds = sell_proceeds(shares, price)
            logger.info(f"📄 [紙上交易] 模擬賣出 {code}: {shares}股 @ {price:.2f} "
                        f"回收={proceeds:,.0f}元 原因={reason}")
            return proceeds
        if self.sdk is None or self.account is None:
            logger.warning(f"SDK未連線，跳過賣出 {code}（實盤不模擬）")
            return 0.0
        try:
            order_no = self._place_order(code, "SELL", shares, price, reason, proba=proba)
            if not order_no:
                return 0.0
            with self._lock:
                self.pending_orders[order_no] = {
                    "side": "SELL", "code": code, "shares": shares,
                    "price": price, "placed_at": datetime.now(_TZ).isoformat(),
                    "retries": 0, "reason": reason, "proba": proba,
                }
            logger.info(f"⏳ {code} 賣單送出，等待 _on_filled 確認 (單號={order_no})")
            return 0.0  # cash/holdings 由 _on_filled 更新，不在此處雙重計算
        except Exception as e:
            logger.error(f"賣出錯誤: {e}")
            return 0.0

    def process_pending_orders(self):
        """
        每次監控週期呼叫。
        BUY 掛單：確認成交 → 記入持倉；超過 30 分鐘未成交 → 取消
        SELL 掛單：確認成交 → 更新現金；每次重試降價 1%，最多 3 次；超過則放棄
        """
        if not self.pending_orders or self.sdk is None:
            return
        now = datetime.now()
        to_remove = []

        for order_no, o in list(self.pending_orders.items()):
            # _on_filled 可能已在回呼執行緒中移除此單 → 跳過，避免雙重入帳
            with self._lock:
                if order_no not in self.pending_orders:
                    continue

            code = o["code"]
            placed_at = datetime.fromisoformat(o["placed_at"])
            elapsed_min = (now - placed_at).seconds // 60

            filled = self._check_filled(code, order_no)

            if filled == -1:
                # Order rejected by broker
                logger.warning(f"❌ 委託被拒 {code} {o['side']} 單號={order_no}，移除掛單")
                to_remove.append(order_no)
                continue

            if filled > 0:
                # ── 成交確認（_on_filled 未觸發時的備援路徑）─────────────
                with self._lock:
                    # 再次確認 _on_filled 尚未處理此單
                    if order_no not in self.pending_orders:
                        logger.info(f"⚡ {code} {order_no} 已由 _on_filled 處理，跳過重複入帳")
                        continue
                    if o["side"] == "BUY":
                        actual_cost = buy_cost(filled, o["price"])
                        self.cash -= actual_cost
                        self.holdings[code] = {
                            "shares": filled, "entry_price": o["price"],
                            "entry_date": now.strftime("%Y-%m-%d"),
                            "cost": actual_cost, "buy_proba": 0,
                        }
                        self.trade_log.append({
                            "date": now.strftime("%Y-%m-%d"), "action": "BUY",
                            "ticker": code, "shares": filled, "price": o["price"],
                        })
                        logger.info(f"✅ 掛單成交(BUY): {code} {filled}股 @ {o['price']}")
                    else:
                        proceeds = sell_proceeds(filled, o["price"])
                        self.cash += proceeds
                        if code in self.holdings:
                            del self.holdings[code]
                        self.trade_log.append({
                            "date": now.strftime("%Y-%m-%d"), "action": "SELL",
                            "ticker": code, "shares": filled, "price": o["price"],
                            "reason": o["reason"],
                        })
                        logger.info(f"✅ 掛單成交(SELL): {code} {filled}股 @ {o['price']}")
                to_remove.append(order_no)

            elif o["side"] == "BUY" and elapsed_min >= 30:
                # ── 買單逾時 → 取消，不買了 ───────────────────────────────
                self._cancel_order(order_no, code)
                logger.warning(f"❌ {code} 買單掛單逾30分鐘未成交，已取消")
                to_remove.append(order_no)

            elif o["side"] == "SELL" and elapsed_min >= 15:
                # ── 賣單逾時 → 重新問 AI，再決定要不要繼續賣 ────────────
                if o["retries"] >= 3:
                    logger.warning(f"❌ {code} 賣單重試3次仍未成交，暫停賣出，繼續持有")
                    to_remove.append(order_no)
                else:
                    self._cancel_order(order_no, code)
                    # 重新問 AI 當前訊號
                    signal = self._get_ai_signal(code)
                    current_price = self._get_latest_price(code)
                    ai_still_sell = (
                        signal and signal.get("signal") == "SELL"
                        and signal.get("sell_proba", 0) >= 0.50
                    )
                    is_stop_loss = "停損" in o["reason"]

                    if not ai_still_sell and not is_stop_loss:
                        logger.info(f"🤖 {code} AI已不建議賣出 (signal={signal.get('signal') if signal else 'N/A'})，取消賣出，繼續持有")
                        to_remove.append(order_no)
                    else:
                        # AI 仍建議賣，或是停損強制出場 → 用最新價格重掛
                        new_proba = signal.get("sell_proba", o.get("proba", 0.0)) if signal else o.get("proba", 0.0)
                        if is_stop_loss:
                            new_proba = 0.95  # 停損不猶豫
                        new_order_no = self._place_order(
                            code, "SELL", o["shares"], current_price, o["reason"], proba=new_proba
                        )
                        if new_order_no:
                            self.pending_orders[new_order_no] = {
                                **o, "price": current_price, "proba": new_proba,
                                "placed_at": now.isoformat(),
                                "retries": o["retries"] + 1,
                            }
                            logger.info(f"🔄 {code} AI確認賣出，重掛 @ {current_price} proba={new_proba:.0%} (第{o['retries']+1}次)")
                        to_remove.append(order_no)

        for k in to_remove:
            self.pending_orders.pop(k, None)
        if to_remove:
            self._save_state()

    def _cancel_order(self, order_no: str, code: str):
        """
        Cancel a pending order. cancel_order() requires an OrderResult object,
        so we fetch get_order_results and find the matching order first.
        """
        try:
            orders = self.sdk.stock.get_order_results(self.account)
            if not orders.is_success or not orders.data:
                logger.warning(f"取消委託: 無法取得委託列表 {code}")
                return
            target = next(
                (o for o in orders.data if getattr(o, "seq_no", None) == order_no),
                None,
            )
            if target is None:
                logger.warning(f"取消委託: 找不到委託 {code} 單號={order_no}")
                return
            result = self.sdk.stock.cancel_order(self.account, target)
            if result.is_success:
                logger.info(f"🗑️  取消委託: {code} 單號={order_no}")
            else:
                logger.warning(f"取消委託失敗: {code} {result.message}")
        except Exception as e:
            logger.warning(f"取消委託錯誤: {e}")

    def _check_filled(self, code: str, order_no: str | None) -> int:
        """
        Query get_order_results and return filled shares for this order (0 if none).
        status=90 means rejected; status=0 with filled_qty=0 means still pending.
        """
        try:
            result = self.sdk.stock.get_order_results(self.account)
            if not result.is_success or not result.data:
                return 0
            for o in result.data:
                o_no   = getattr(o, "seq_no", None)   # Fubon uses seq_no, not order_no
                o_code = getattr(o, "stock_no", "")
                o_qty  = int(getattr(o, "filled_qty", 0) or 0)
                o_status = getattr(o, "status", None)
                o_err  = getattr(o, "error_message", None)

                # rejected order → signal caller to remove from pending
                if o_status == 90 or (o_err and o_err.strip()):
                    if order_no and o_no == order_no:
                        logger.warning(f"委託被拒 {code} 序號={order_no}: {o_err}")
                        return -1   # signal: rejected
                    continue

                if order_no and o_no == order_no:
                    return o_qty
                if not order_no and o_code == code and o_qty > 0:
                    return o_qty
        except Exception as e:
            logger.warning(f"成交確認查詢失敗: {e}")
        return 0

    def _sync_holdings_from_fubon(self):
        """
        Reconcile local holdings with real Fubon inventory.
        Removes any local holding that doesn't exist in the real account.
        Also cancels pending orders that were already rejected.
        """
        if self.paper_trading or self.sdk is None or self.account is None:
            return  # paper mode 直接使用本地持倉記錄
        try:
            # ── 1. 雙向持倉同步 ───────────────────────────────────────────
            inv = self.sdk.accounting.inventories(self.account)
            real_holdings: dict[str, int] = {}   # code -> qty
            if inv.is_success and inv.data:
                for item in inv.data:
                    code = str(getattr(item, "stock_no", None) or getattr(item, "symbol", None) or "").strip()
                    # 零股數量在 odd 欄位
                    odd  = getattr(item, "odd", None)
                    if odd:
                        qty = int(getattr(odd, "today_qty", 0) or getattr(odd, "tradable_qty", 0) or 0)
                    else:
                        qty = int(getattr(item, "today_qty", 0) or getattr(item, "tradable_qty", 0) or 0)
                    if code and qty > 0:
                        real_holdings[code] = qty

            # 1a. 移除假持倉（本地有但帳戶沒有）
            stale = [c for c in self.holdings if c not in real_holdings]
            if stale:
                logger.warning(f"⚠️  移除虛假持倉: {stale}")
                for c in stale:
                    del self.holdings[c]

            # 1b. 取得真實成本價（從未實現損益 API）
            cost_map: dict[str, float] = {}
            try:
                unreal = self.sdk.accounting.unrealized_gains_and_loses(self.account)
                if unreal.is_success and unreal.data:
                    for u in unreal.data:
                        c = str(getattr(u, "stock_no", "") or "").strip()
                        cp = float(getattr(u, "cost_price", 0) or 0)
                        if c and cp > 0:
                            cost_map[c] = cp
            except Exception:
                pass

            # 1c. 新增未追蹤持倉（帳戶有但本地沒有）
            for code, qty in real_holdings.items():
                if code not in self.holdings:
                    entry_price = cost_map.get(code) or self._get_latest_price(code)
                    source = "實際成本" if code in cost_map else "現價估算"
                    # 優先從 trade_log 找回原始買入日期與信心，避免 pod 重啟後日期歸零
                    past_buy = next(
                        (t for t in reversed(self.trade_log)
                         if t.get("ticker") == code and t.get("action") == "BUY"),
                        None,
                    )
                    self.holdings[code] = {
                        "shares": qty,
                        "entry_price": entry_price,
                        "entry_date": past_buy["date"] if past_buy else datetime.now(_TZ).strftime("%Y-%m-%d"),
                        "cost": entry_price * qty,
                        "buy_proba": past_buy.get("buy_proba", 0.5) if past_buy else 0.5,
                        "peak_price": entry_price,
                    }
                    src2 = f"trade_log({past_buy['date']})" if past_buy else "今日"
                    logger.info(f"📥 新增未追蹤持倉: {code} {qty}股 成本={entry_price:.2f}（{source}）買入日={src2}")
                else:
                    # 更新股數（可能有部分成交）
                    if self.holdings[code]["shares"] != qty:
                        logger.info(f"📊 更新持倉股數: {code} {self.holdings[code]['shares']}→{qty}股")
                        self.holdings[code]["shares"] = qty

            # ── 2. 清除昨天（過期）的掛單 ─────────────────────────────────
            today = datetime.now().strftime("%Y-%m-%d")
            stale_pending = []
            expired = [
                no for no, o in list(self.pending_orders.items())
                if not o.get("placed_at", today).startswith(today)
            ]
            for no in expired:
                info = self.pending_orders.pop(no)
                logger.warning(f"⚠️  清除過期掛單（昨天）: {info['code']} {info['side']} 單號={no}")
                stale_pending.append(no)

            # ── 3. 清除已被拒絕的掛單 ─────────────────────────────────────
            result = self.sdk.stock.get_order_results(self.account)
            if result.is_success and result.data:
                rejected = {
                    getattr(o, "seq_no", None)
                    for o in result.data
                    if getattr(o, "status", None) == 90 or (getattr(o, "error_message", None) or "").strip()
                }
                newly_rejected = [no for no in list(self.pending_orders) if no in rejected]
                for no in newly_rejected:
                    info = self.pending_orders.pop(no)
                    logger.warning(f"⚠️  清除被拒掛單: {info['code']} {info['side']} 單號={no}")
                stale_pending.extend(newly_rejected)

            if stale or stale_pending:
                self._save_state()
            logger.info(f"✅ 持倉同步完成: 移除持倉={stale}, 移除掛單={stale_pending}")

        except Exception as e:
            logger.warning(f"持倉同步失敗: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_tradeable_from_db(self) -> set | None:
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("""
                SELECT code FROM tradeable_stocks
                WHERE updated_at > NOW() - INTERVAL '26 hours'
            """)
            rows = cur.fetchall()
            conn.close()
            if rows:
                codes = {r[0] for r in rows}
                logger.debug(f"從 DB 讀取可交易清單：{len(codes)} 支")
                return codes
        except Exception as e:
            logger.debug(f"DB 可交易清單讀取失敗: {e}")
        return None

    def _get_latest_price(self, code: str) -> float:
        try:
            sym = get_symbol(code)
            ticker = yf.Ticker(sym)
            df = ticker.history(period="2d")
            if df.empty:
                return 0.0
            return float(df["Close"].iloc[-1])
        except Exception:
            return 0.0

    def _get_ai_signal(self, code: str) -> dict | None:
        try:
            import pandas as pd
            row = None

            # ── 優先從 DB cache 讀取特徵（market-data service 預先計算）───────
            try:
                conn = get_conn()
                try:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT features, fetched_at FROM market_cache
                        WHERE code = %s
                          AND fetched_at > NOW() - INTERVAL '30 minutes'
                    """, (code,))
                    cached = cur.fetchone()
                    if cached:
                        import json as _json
                        feats = _json.loads(cached[0])
                        row = pd.DataFrame([feats])
                        logger.debug(f"{code} 使用 DB 快取特徵（{cached[1]}）")

                    # 情緒分數從 sentiment_cache 讀
                    sentiment_score = 0.0
                    cur.execute("SELECT score FROM sentiment_cache WHERE code = %s", (code,))
                    sent_row = cur.fetchone()
                    if sent_row:
                        sentiment_score = float(sent_row[0] or 0.0)
                finally:
                    conn.close()
            except Exception as db_err:
                logger.debug(f"{code} DB 讀取失敗，fallback 至 yfinance: {db_err}")

            # ── Fallback：DB 無快取時直接抓 yfinance ─────────────────────────
            if row is None:
                sym = get_symbol(code)
                df = yf.Ticker(sym).history(period="6mo")
                if len(df) < 70:
                    return None

                sentiment_score = 0.0
                try:
                    from utils.forum_sentiment import get_forum_sentiment
                    company = STOCK_DB.get(code, "").split()[0] if code in STOCK_DB else ""
                    sent = get_forum_sentiment(code, company_name=company, days=5)
                    sentiment_score = float(sent.get("score", 0.0))
                except Exception:
                    pass

                row = prepare_inference_row(df, code=code, sentiment_score=sentiment_score)
                if row.empty:
                    return None
            else:
                row["sentiment_score"] = sentiment_score

            lgbm_signal = self.model.predict(row)
            if "ATR" in row.columns:
                lgbm_signal["atr"] = float(row["ATR"].iloc[-1])

            # ── TradingAgents 第二意見（若已啟用）───────────────────────────
            try:
                from trader.ta_signal import get_ta_signal, combine_signals
                ta = get_ta_signal(code, mode="scan")
                if ta is not None:
                    combined = combine_signals(lgbm_signal, ta)
                    if combined is not None:
                        combined.setdefault("atr", lgbm_signal.get("atr", 0.0))
                        return combined
            except Exception as ta_err:
                logger.debug(f"TradingAgents 整合失敗: {ta_err}")

            return lgbm_signal
        except Exception:
            return None

    def _train_model(self):
        from ai.backtest import run_backtest
        result = run_backtest(self.model, period="2y", verbose=True)
        self.model.save()
        return result

    def get_portfolio_summary(self) -> dict:
        """Return current portfolio status for dashboard."""
        positions = []
        total_value = self.cash
        for code, h in self.holdings.items():
            price = self._get_latest_price(code)
            abs_pnl, pct_pnl = pnl(h["entry_price"], price, h["shares"])
            market_val = h["shares"] * price
            total_value += market_val
            positions.append({
                "ticker": code,
                "name": STOCK_DB.get(code, code),
                "shares": h["shares"],
                "entry_price": h["entry_price"],
                "current_price": price,
                "market_value": round(market_val, 0),
                "pnl": abs_pnl,
                "pnl_pct": pct_pnl,
                "entry_date": h["entry_date"],
            })
        return {
            "cash": round(self.cash, 0),
            "total_value": round(total_value, 0),
            "positions": positions,
            "trade_log": self.trade_log[-20:],
            "model_version": self.model.version,
        }

    # ── State Persistence ─────────────────────────────────────────────────────

    def _save_state(self):
        last = getattr(self, "_last_retrain", None)
        state = {
            "cash": self.cash,
            "holdings": self.holdings,
            "trade_log": self.trade_log,
            "pending_orders": self.pending_orders,
            "last_retrain": str(last) if last else None,
            "last_emergency_retrain": getattr(self, "_last_emergency_retrain", None),
            "daily_buy_date": self._daily_buy_date,
            "daily_buy_count": self._daily_buy_count,
            "daily_spend": self._daily_spend,
        }
        with open(self._state_file, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)

    def _load_state(self):
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    state = json.load(f)
                self.cash = state.get("cash", self.cash)
                self.holdings = state.get("holdings", {})
                self.trade_log = state.get("trade_log", [])
                self.pending_orders = state.get("pending_orders", {})
                lr = state.get("last_retrain")
                self._last_retrain = datetime.fromisoformat(lr) if lr else None
                self._last_emergency_retrain = state.get("last_emergency_retrain")
                self._daily_buy_date  = state.get("daily_buy_date", "")
                self._daily_buy_count = int(state.get("daily_buy_count", 0))
                self._daily_spend     = float(state.get("daily_spend", 0.0))
                pending_count = len(self.pending_orders)
                logger.info(f"載入狀態: 現金={self.cash:.0f}, 持倉={list(self.holdings.keys())}, 掛單={pending_count}筆")
            except Exception as e:
                logger.warning(f"載入狀態失敗: {e}")
