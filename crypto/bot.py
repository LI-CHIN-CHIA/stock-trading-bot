"""
CryptoBot — 加密貨幣交易機器人主類別
=====================================
協調 exchange / signals / scanner，執行完整的交易循環。

模式:
  stable  — 穩定幣種，每 15 分鐘 run_cycle()
  newcoin — 新幣策略，每 1 分鐘 run_newcoin_cycle()

Paper trading: CRYPTO_PAPER_TRADING=true（預設）
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from crypto.exchange import CryptoExchange
from crypto.risk import (
    CryptoRiskProfile,
    get_profile,
    pnl,
    position_size_usdt,
    should_atr_stop,
    should_stop_loss,
    should_take_profit,
    should_time_exit,
    sell_proceeds_usdt,
    buy_cost_usdt,
)
from crypto.scanner import STABLE_SYMBOLS, NewCoinScanner
from crypto.signals import get_signal_engine

ENABLE_TA = os.getenv("ENABLE_TRADING_AGENTS", "false").lower() == "true"

logger = logging.getLogger(__name__)

MAX_POSITIONS = int(os.getenv("CRYPTO_MAX_POSITIONS", "3"))
INITIAL_CAPITAL = float(os.getenv("CRYPTO_INITIAL_CAPITAL", "10000"))
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent.parent / "data"))


class CryptoBot:
    def __init__(self):
        self.mode         = os.getenv("CRYPTO_MODE", "stable").lower()
        self.paper_trading= os.getenv("CRYPTO_PAPER_TRADING", "true").lower() == "true"
        # stable 模式下的策略：dip = 多頭回檔買進（預設）；legacy = 舊的四指標投票（回測為負）
        self.strategy     = os.getenv("CRYPTO_STRATEGY", "dip").lower() if self.mode != "newcoin" else ""
        self.symbols      = ([s.strip() for s in os.getenv("CRYPTO_DIP_SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()]
                             if self.strategy == "dip" else STABLE_SYMBOLS)
        self.profile      : CryptoRiskProfile = get_profile(self.mode, self.strategy)
        self.max_positions = min(MAX_POSITIONS, len(self.symbols)) if self.strategy == "dip" else MAX_POSITIONS
        self.exchange     = CryptoExchange()
        self.signal_engine= get_signal_engine(self.mode, self.strategy)
        self.scanner      = NewCoinScanner()

        self.cash_usdt    : float = INITIAL_CAPITAL
        self.holdings     : dict  = {}   # symbol -> {amount, entry_price, entry_time, peak_price, atr}
        self.trade_log    : list  = []

        self._state_file  = DATA_DIR / (
            "crypto_paper_state.json" if self.paper_trading else "crypto_state.json"
        )
        self._load_state()

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        return self.exchange.connect()

    # ── 主循環 ────────────────────────────────────────────────────────────────

    def run_cycle(self):
        """穩定模式主循環（每 15 分鐘）。"""
        logger.info("=== [加密貨幣] 穩定模式循環開始 ===")
        self._check_holdings()
        self._scan_and_buy(self.symbols)
        self._save_state()

    def run_newcoin_cycle(self):
        """新幣模式主循環（每 1 分鐘）。"""
        self._check_holdings()
        candidates = self.scanner.scan(self.exchange, set(self.holdings.keys()))
        if candidates:
            self._scan_and_buy(candidates)
        self._save_state()

    # ── 持倉監控 ──────────────────────────────────────────────────────────────

    def _check_holdings(self):
        if not self.holdings:
            return
        for symbol in list(self.holdings.keys()):
            h = self.holdings[symbol]
            price = self.exchange.fetch_ticker(symbol)
            if price is None:
                continue

            h["peak_price"] = max(h.get("peak_price", h["entry_price"]), price)
            abs_pnl, pct_pnl = pnl(h["entry_price"], price, h["amount"])

            reason = None

            # 停損
            if should_stop_loss(h["entry_price"], price, self.profile):
                reason = f"停損 {pct_pnl:.1%}"
            # ATR 追蹤停損
            elif should_atr_stop(h["peak_price"], price, h.get("atr", 0), self.profile):
                reason = f"ATR追蹤停損 {pct_pnl:.1%}"
            # 停利
            elif should_take_profit(h["entry_price"], price, self.profile):
                reason = f"停利 {pct_pnl:.1%}"
            # 時間強制出場
            elif should_time_exit(h["entry_time"], self.profile):
                reason = f"超過持有時間 {pct_pnl:.1%}"
            # 回檔買進策略：日線收盤站回 5 日均線就出場
            elif self.strategy == "dip" and self._dip_exit(symbol, h):
                reason = f"站回5日線 {pct_pnl:.1%}"
            # TradingAgents SELL 建議（持倉監控，獲利 >5% 時才採納）
            elif ENABLE_TA and pct_pnl > 0.05:
                try:
                    from crypto.ta_signal import get_crypto_ta_signal
                    ta = get_crypto_ta_signal(symbol, mode="holding")
                    if ta and ta.get("signal") == "SELL" and ta.get("confidence", 0) >= 0.7:
                        reason = f"TA建議賣出 {pct_pnl:.1%} (信心={ta['confidence']:.0%})"
                except Exception as e:
                    logger.debug(f"[TA] 持倉監控查詢失敗 {symbol}: {e}")

            if reason:
                self._close_position(symbol, h, price, reason, abs_pnl, pct_pnl)

    def _dip_exit(self, symbol: str, h: dict) -> bool:
        try:
            import pandas as pd
            df = self.exchange.fetch_ohlcv(symbol, "1d", limit=300)
            return self.signal_engine.exit_signal(df, pd.Timestamp(h["entry_time"]))
        except Exception as e:
            logger.debug(f"[dip] {symbol} 出場判斷失敗: {e}")
            return False

    # ── 掃描買入 ─────────────────────────────────────────────────────────────

    def _scan_and_buy(self, symbols: list[str]):
        if len(self.holdings) >= self.max_positions:
            return

        total_capital = self.cash_usdt + sum(
            h["amount"] * (self.exchange.fetch_ticker(s) or h["entry_price"])
            for s, h in self.holdings.items()
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sold_today = {t["symbol"] for t in self.trade_log
                      if t.get("action") == "SELL" and t.get("date") == today}

        candidates = []
        for symbol in symbols:
            if symbol in self.holdings or symbol in sold_today:
                continue
            df = self.exchange.fetch_ohlcv(symbol, self.profile.candle_interval, limit=300)
            if df is None or df.empty:
                continue
            strength = self.signal_engine.score(df)
            if strength >= self.profile.min_signal_strength:
                price = self.exchange.fetch_ticker(symbol) or float(df["Close"].iloc[-1])
                atr = self._calc_atr(df)
                candidates.append((symbol, strength, price, atr, df))

        candidates.sort(key=lambda x: x[1], reverse=True)
        slots = self.max_positions - len(self.holdings)

        for symbol, strength, price, atr, df in candidates[:slots]:
            if self.cash_usdt < 10:
                break

            # TradingAgents 確認（若啟用）
            if ENABLE_TA:
                try:
                    from crypto.ta_signal import get_crypto_ta_signal
                    ta = get_crypto_ta_signal(symbol, mode="scan")
                    if ta and ta.get("signal") == "SELL":
                        logger.info(f"[TA] {symbol} TA 建議賣出，跳過買入 (信心={ta['confidence']:.0%})")
                        continue
                    if ta:
                        logger.info(f"[TA] {symbol} TA={ta['signal']} 信心={ta['confidence']:.0%}")
                except Exception as e:
                    logger.warning(f"[TA] {symbol} TA 查詢失敗（不影響交易）: {e}")

            amount_usdt = position_size_usdt(self.cash_usdt, total_capital, self.profile, slots)
            if amount_usdt < 10:
                continue
            result = self.exchange.place_market_buy(symbol, amount_usdt, price)
            if result:
                cost = buy_cost_usdt(amount_usdt)
                self.cash_usdt -= cost
                self.holdings[symbol] = {
                    "amount":      result["filled_amount"],
                    "entry_price": result["price"],
                    "entry_time":  datetime.now(timezone.utc).isoformat(),
                    "peak_price":  result["price"],
                    "atr":         0.0 if self.strategy == "dip" else atr,
                    "cost_usdt":   cost,
                }
                self.trade_log.append({
                    "date":     today,
                    "action":   "BUY",
                    "symbol":   symbol,
                    "amount":   result["filled_amount"],
                    "price":    result["price"],
                    "strength": round(strength, 3),
                    "mode":     self.mode,
                })
                mode_tag = "[紙上交易]" if self.paper_trading else ""
                logger.info(
                    f"{mode_tag} 買入 {symbol}: {result['filled_amount']:.6f} 枚 "
                    f"@ {result['price']:.4f} USDT，訊號強度 {strength:.2f}"
                )

    # ── 平倉 ─────────────────────────────────────────────────────────────────

    def _close_position(
        self, symbol: str, h: dict, price: float,
        reason: str, abs_pnl: float, pct_pnl: float,
    ):
        result = self.exchange.place_market_sell(symbol, h["amount"], price)
        if result:
            self.cash_usdt += result["proceeds_usdt"]
            del self.holdings[symbol]
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self.trade_log.append({
                "date":        today,
                "action":      "SELL",
                "symbol":      symbol,
                "amount":      h["amount"],
                "price":       result["price"],
                "pnl_usdt":    round(abs_pnl, 4),
                "pnl_pct":     round(pct_pnl, 4),
                "reason":      reason,
                "mode":        self.mode,
            })
            mode_tag = "[紙上交易]" if self.paper_trading else ""
            logger.info(
                f"{mode_tag} 賣出 {symbol}: {reason}，損益 {abs_pnl:+.2f} USDT ({pct_pnl:+.1%})"
            )
        else:
            logger.warning(f"⚠️  {symbol} 賣出失敗，繼續持有")

    # ── 輔助 ─────────────────────────────────────────────────────────────────

    def _calc_atr(self, df, period: int = 14) -> float:
        try:
            high  = df["High"]
            low   = df["Low"]
            close = df["Close"].shift(1)
            tr = pd.concat([
                high - low,
                (high - close).abs(),
                (low  - close).abs(),
            ], axis=1).max(axis=1)
            return float(tr.rolling(period).mean().iloc[-1] or 0)
        except Exception:
            return 0.0

    def daily_summary(self):
        """產出每小時／每日結算 log。"""
        total_value = self.cash_usdt
        positions   = []
        for symbol, h in self.holdings.items():
            price = self.exchange.fetch_ticker(symbol) or h["entry_price"]
            val   = h["amount"] * price
            total_value += val
            _, pct = pnl(h["entry_price"], price, h["amount"])
            positions.append(f"{symbol}: {pct:+.1%}")

        mode_tag = "📄[紙上交易]" if self.paper_trading else "📊"
        logger.info("=" * 55)
        logger.info(f"{mode_tag} 加密貨幣結算 ({self.profile.label}模式)")
        logger.info(f"   現金:     {self.cash_usdt:,.2f} USDT")
        logger.info(f"   總資產:   {total_value:,.2f} USDT")
        logger.info(f"   持倉數:   {len(self.holdings)} 支")
        for p in positions:
            logger.info(f"   {p}")
        logger.info("=" * 55)

    # ── 狀態儲存 ──────────────────────────────────────────────────────────────

    def _save_state(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "cash_usdt":   self.cash_usdt,
            "holdings":    self.holdings,
            "trade_log":   self.trade_log[-500:],   # 只保留最近 500 筆
            "mode":        self.mode,
            "saved_at":    datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"狀態儲存失敗: {e}")

    def _load_state(self):
        if not self._state_file.exists():
            logger.info(f"無既有狀態，初始資金 {INITIAL_CAPITAL} USDT")
            return
        try:
            with open(self._state_file, encoding="utf-8") as f:
                state = json.load(f)
            self.cash_usdt  = state.get("cash_usdt", INITIAL_CAPITAL)
            self.holdings   = state.get("holdings", {})
            self.trade_log  = state.get("trade_log", [])
            logger.info(
                f"載入加密貨幣狀態: 現金={self.cash_usdt:.2f} USDT, "
                f"持倉={list(self.holdings.keys())}"
            )
        except Exception as e:
            logger.warning(f"狀態載入失敗，重新開始: {e}")


# 避免在 signal 計算時 import error
try:
    import pandas as pd
except ImportError:
    pass
