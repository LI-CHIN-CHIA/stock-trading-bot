"""
US Stock Paper Trading Bot
===========================
LightGBM-based paper trading for US equities.
Data via yfinance; no broker API required (pure simulation).

Environment variables:
  US_INITIAL_CAPITAL=10000      USD starting capital
  US_MAX_POSITIONS=5            Max concurrent holdings
  US_MIN_BUY_PROBA=0.60         LightGBM buy confidence threshold
  US_STOP_LOSS_PCT=0.07         Hard stop-loss
  US_TAKE_PROFIT_PCT=0.08       Partial take-profit trigger
  US_TAKE_PROFIT_PCT2=0.15      Full take-profit trigger
  US_MAX_HOLD_DAYS=15           Max days to hold
  US_CASH_RESERVE_PCT=0.20      Minimum cash reserve fraction
  US_DRAWDOWN_BREAKER_PCT=0.20  Portfolio-level drawdown halt
  DATA_DIR=/data                State & model directory
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytz
import yfinance as yf

logger = logging.getLogger(__name__)

_TZ = pytz.timezone("US/Eastern")
_UTC = timezone.utc

INITIAL_CAPITAL      = float(os.getenv("US_INITIAL_CAPITAL",      "10000"))
MAX_POSITIONS        = int(os.getenv("US_MAX_POSITIONS",           "5"))
MIN_BUY_PROBA        = float(os.getenv("US_MIN_BUY_PROBA",         "0.60"))
STOP_LOSS_PCT        = float(os.getenv("US_STOP_LOSS_PCT",          "0.07"))
TAKE_PROFIT_PCT      = float(os.getenv("US_TAKE_PROFIT_PCT",        "0.08"))
TAKE_PROFIT_PCT2     = float(os.getenv("US_TAKE_PROFIT_PCT2",       "0.15"))
MAX_HOLD_DAYS        = int(os.getenv("US_MAX_HOLD_DAYS",            "15"))
CASH_RESERVE_PCT     = float(os.getenv("US_CASH_RESERVE_PCT",       "0.20"))
DRAWDOWN_BREAKER_PCT = float(os.getenv("US_DRAWDOWN_BREAKER_PCT",  "0.20"))
MAX_DAILY_BUYS       = int(os.getenv("US_MAX_DAILY_BUYS",           "3"))
MAX_PER_STOCK_PCT    = float(os.getenv("US_MAX_PER_STOCK_PCT",      "0.30"))
TRANSACTION_FEE      = 0.001   # 0.1% round-trip simulation

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent.parent / "data"))

# ── US Stock Universe ─────────────────────────────────────────────────────────
US_WATCHLIST = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    # Semiconductors
    "AMD", "AVGO", "QCOM", "INTC", "MU", "AMAT",
    # Finance
    "JPM", "BAC", "GS", "MS", "V", "MA",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "PFE",
    # Energy
    "XOM", "CVX",
    # ETFs (for regime detection)
    "SPY", "QQQ",
]
# SPY/QQQ used for regime, not traded directly
_NON_TRADEABLE = {"SPY", "QQQ"}


class USBot:
    def __init__(self):
        self.cash: float         = INITIAL_CAPITAL
        self.holdings: dict      = {}   # ticker -> {shares, entry_price, entry_date, cost, atr, peak_price}
        self.trade_log: list     = []
        self.pending_orders: dict= {}
        self._peak_capital: float= INITIAL_CAPITAL
        self._daily_buy_count: int = 0
        self._daily_buy_date: str  = ""

        self._state_file = DATA_DIR / "us_paper_state.json"
        self._model_dir  = DATA_DIR / "us_models"
        self._model_dir.mkdir(parents=True, exist_ok=True)

        from ai.model import TradingModel
        self.model = TradingModel(model_dir=self._model_dir)

        self._load_state()

    # ── State persistence ─────────────────────────────────────────────────────

    def _save_state(self):
        state = {
            "cash":            self.cash,
            "holdings":        self.holdings,
            "trade_log":       self.trade_log[-200:],
            "peak_capital":    self._peak_capital,
            "daily_buy_count": self._daily_buy_count,
            "daily_buy_date":  self._daily_buy_date,
        }
        tmp = self._state_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, self._state_file)

    def _load_state(self):
        if not self._state_file.exists():
            return
        try:
            with open(self._state_file) as f:
                s = json.load(f)
            self.cash            = float(s.get("cash", INITIAL_CAPITAL))
            self.holdings        = s.get("holdings", {})
            self.trade_log       = s.get("trade_log", [])
            if s.get("peak_capital") is not None:
                self._peak_capital = float(s["peak_capital"])
            self._daily_buy_count = int(s.get("daily_buy_count", 0))
            self._daily_buy_date  = s.get("daily_buy_date", "")
            logger.info(
                f"US Bot 載入狀態: 現金=${self.cash:,.0f} "
                f"持倉={list(self.holdings.keys())} "
                f"掛單={len(self.pending_orders)}筆"
            )
        except Exception as e:
            logger.warning(f"US Bot 載入狀態失敗: {e}")

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _fetch_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval="1d", auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.rename(columns=str.capitalize)
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception as e:
            logger.warning(f"yfinance 取得 {ticker} 失敗: {e}")
            return pd.DataFrame()

    def _get_latest_price(self, ticker: str) -> float:
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
            return float(price) if price else 0.0
        except Exception:
            return 0.0

    # ── Signal generation ─────────────────────────────────────────────────────

    def _get_ai_signal(self, ticker: str) -> dict | None:
        df = self._fetch_data(ticker, period="2y")
        if len(df) < 60:
            return None
        try:
            from analysis.indicators import TechnicalIndicators
            df = TechnicalIndicators.add_all(df.copy())
        except Exception as e:
            logger.warning(f"指標計算失敗 {ticker}: {e}")
            return None

        result = self.model.predict(df)
        if not result.get("trained"):
            return None

        atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not pd.isna(df["ATR"].iloc[-1]) else 0.0
        result["atr"] = atr
        return result

    # ── Market regime (SPY MA-based) ──────────────────────────────────────────

    def _get_market_regime(self) -> str:
        """Bull/neutral/bear based on SPY vs MA20/MA50/MA200."""
        try:
            df = self._fetch_data("SPY", period="2y")
            if len(df) < 210:
                return "neutral"
            close = df["Close"]
            ma20  = close.rolling(20).mean().iloc[-1]
            ma50  = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1]
            price = close.iloc[-1]
            score = sum([price > ma20, ma20 > ma50, ma50 > ma200])
            return ["bear", "neutral-", "neutral+", "bull"][score]
        except Exception:
            return "neutral"

    # ── Retraining ────────────────────────────────────────────────────────────

    def retrain(self, reason: str = "排程"):
        logger.info(f"=== US Bot 開始重訓 ({reason}) ===")
        try:
            from analysis.indicators import TechnicalIndicators
            all_dfs = []
            for ticker in US_WATCHLIST:
                df = self._fetch_data(ticker, period="2y")
                if len(df) < 80:
                    continue
                df = TechnicalIndicators.add_all(df.copy())
                df["ticker"] = ticker
                all_dfs.append(df)
            if not all_dfs:
                logger.warning("無資料可訓練")
                return
            combined = pd.concat(all_dfs, ignore_index=True)
            metrics = self.model.train(combined)
            logger.info(
                f"US 模型訓練完成  "
                f"Val F1={metrics.get('val_macro_f1', 0):.3f}  "
                f"Buy F1={metrics.get('buy_f1', 0):.3f}  "
                f"勝率={metrics.get('win_rate', 0):.1%}"
            )
        except Exception as e:
            logger.error(f"US 模型重訓失敗: {e}", exc_info=True)

    # ── Daily counter ─────────────────────────────────────────────────────────

    def _reset_daily_counters(self):
        today = datetime.now(_TZ).strftime("%Y-%m-%d")
        if self._daily_buy_date != today:
            self._daily_buy_count = 0
            self._daily_buy_date  = today

    # ── Check holdings (stop-loss / take-profit / time exit) ─────────────────

    def _check_holdings(self):
        if not self.holdings:
            return
        today = datetime.now(_TZ).strftime("%Y-%m-%d")
        to_sell = []

        for ticker, h in list(self.holdings.items()):
            price = self._get_latest_price(ticker)
            if price <= 0:
                continue

            entry   = h["entry_price"]
            atr     = h.get("atr", 0.0)
            peak    = h.get("peak_price", entry)
            shares  = h["shares"]
            pct_pnl = (price - entry) / entry
            hold_days = (datetime.now(_TZ).date() - datetime.fromisoformat(h["entry_date"]).date()).days

            # Update peak
            if price > peak:
                h["peak_price"] = price
                peak = price

            # ATR trailing stop (2× ATR below peak)
            atr_stop = peak - 2.0 * atr if atr > 0 else None
            if atr_stop and price < atr_stop and hold_days >= 3:
                to_sell.append((ticker, shares, price,
                    f"ATR追蹤停損 {pct_pnl:.1%} (peak={peak:.2f}, atr={atr:.2f})"))
                continue

            # Hard stop-loss
            if pct_pnl <= -STOP_LOSS_PCT:
                to_sell.append((ticker, shares, price,
                    f"停損 {pct_pnl:.1%}"))
                continue

            # Partial take-profit
            partial_done = h.get("partial_exit_done", False)
            if not partial_done and pct_pnl >= TAKE_PROFIT_PCT:
                sell_shares = max(1, shares // 2)
                to_sell.append((ticker, sell_shares, price,
                    f"停利(半出) {pct_pnl:.1%}"))
                h["partial_exit_done"] = True
                h["shares"] -= sell_shares
                continue

            # Full take-profit
            if partial_done and pct_pnl >= TAKE_PROFIT_PCT2:
                to_sell.append((ticker, shares, price,
                    f"停利(全出) {pct_pnl:.1%}"))
                continue

            # Time exit
            if hold_days >= MAX_HOLD_DAYS:
                to_sell.append((ticker, shares, price,
                    f"超過持有期 {hold_days}天"))
                continue

        for ticker, shares, price, reason in to_sell:
            self._execute_sell(ticker, shares, price, reason)

        if to_sell:
            self._save_state()

    # ── Scan and buy ─────────────────────────────────────────────────────────

    def _scan_and_buy(self):
        if len(self.holdings) >= MAX_POSITIONS:
            return

        self._reset_daily_counters()
        if self._daily_buy_count >= MAX_DAILY_BUYS:
            return

        regime = self._get_market_regime()
        regime_reserve = {
            "bull":     CASH_RESERVE_PCT,
            "neutral+": CASH_RESERVE_PCT + 0.05,
            "neutral-": CASH_RESERVE_PCT + 0.07,
            "neutral":  CASH_RESERVE_PCT + 0.05,
            "bear":     CASH_RESERVE_PCT + 0.10,
        }.get(regime, CASH_RESERVE_PCT)

        # Total capital estimate
        holding_value = sum(
            h["shares"] * (self._get_latest_price(t) or h["entry_price"])
            for t, h in self.holdings.items()
        )
        total_capital = self.cash + holding_value

        min_reserve = total_capital * regime_reserve
        if self.cash <= min_reserve:
            logger.info(f"💰 現金 ${self.cash:,.0f} ≤ 保留 ${min_reserve:,.0f} ({regime})，停止買入")
            return

        # Drawdown breaker
        self._peak_capital = max(self._peak_capital, total_capital)
        if self._peak_capital > 0:
            drawdown = (self._peak_capital - total_capital) / self._peak_capital
            if drawdown >= DRAWDOWN_BREAKER_PCT:
                logger.warning(f"🔴 Portfolio 回撤 {drawdown:.1%}，暫停新開倉")
                return

        today = datetime.now(_TZ).strftime("%Y-%m-%d")
        sold_today = {t["ticker"] for t in self.trade_log
                      if t.get("action") == "SELL" and t.get("date") == today}

        if regime in ("bear", "neutral-"):
            logger.info(f"⚖️  市場環境 {regime}，買入門檻提高")

        candidates = []
        for ticker in US_WATCHLIST:
            if ticker in _NON_TRADEABLE:
                continue
            if ticker in self.holdings or ticker in sold_today:
                continue
            if len(self.holdings) + len(candidates) >= MAX_POSITIONS:
                break

            signal = self._get_ai_signal(ticker)
            if not signal:
                continue
            boost = {"bear": 0.10, "neutral-": 0.07, "neutral+": 0.05}.get(regime, 0.0)
            if signal.get("signal") == "BUY" and signal.get("buy_proba", 0) >= MIN_BUY_PROBA + boost:
                price = self._get_latest_price(ticker)
                if price > 0:
                    candidates.append((ticker, signal["buy_proba"], price, signal.get("atr", 0.0)))

        candidates.sort(key=lambda x: x[1], reverse=True)

        for ticker, buy_proba, price, atr in candidates:
            if len(self.holdings) >= MAX_POSITIONS:
                break
            if self._daily_buy_count >= MAX_DAILY_BUYS:
                break

            # Position sizing: min(MAX_PER_STOCK_PCT of total, available cash - reserve)
            max_spend = min(
                total_capital * MAX_PER_STOCK_PCT,
                self.cash - min_reserve,
            )
            if max_spend < price:
                continue

            shares = int(max_spend / (price * (1 + TRANSACTION_FEE)))
            if shares < 1:
                continue

            self._execute_buy(ticker, shares, price, buy_proba, atr)

        if candidates:
            self._save_state()

    # ── Order execution (paper) ───────────────────────────────────────────────

    def _execute_buy(self, ticker: str, shares: int, price: float, buy_proba: float, atr: float):
        cost = shares * price * (1 + TRANSACTION_FEE)
        if cost > self.cash:
            return
        self.cash -= cost
        today = datetime.now(_TZ).strftime("%Y-%m-%d")
        self.holdings[ticker] = {
            "shares":     shares,
            "entry_price": price,
            "entry_date":  today,
            "cost":        cost,
            "atr":         atr,
            "peak_price":  price,
            "buy_proba":   buy_proba,
        }
        self.trade_log.append({
            "date":      today,
            "action":    "BUY",
            "ticker":    ticker,
            "shares":    shares,
            "price":     price,
            "buy_proba": buy_proba,
        })
        self._daily_buy_count += 1
        logger.info(
            f"📈 [紙上交易] BUY {ticker} {shares}股 @${price:.2f} "
            f"信心={buy_proba:.0%} 花費=${cost:,.0f}"
        )

    def _execute_sell(self, ticker: str, shares: int, price: float, reason: str):
        if ticker not in self.holdings:
            return
        h = self.holdings[ticker]
        proceeds = shares * price * (1 - TRANSACTION_FEE)
        cost_basis = h["cost"] * (shares / h["shares"]) if h["shares"] > 0 else 0
        pnl_usd = proceeds - cost_basis
        pnl_pct = pnl_usd / cost_basis if cost_basis > 0 else 0

        self.cash += proceeds
        today = datetime.now(_TZ).strftime("%Y-%m-%d")

        if shares >= h["shares"]:
            del self.holdings[ticker]
        else:
            h["shares"] -= shares
            h["cost"]   -= cost_basis

        self.trade_log.append({
            "date":    today,
            "action":  "SELL",
            "ticker":  ticker,
            "shares":  shares,
            "price":   price,
            "pnl":     round(pnl_usd, 2),
            "pnl_pct": round(pnl_pct, 4),
            "reason":  reason,
        })
        emoji = "✅" if pnl_usd > 0 else "🔴"
        logger.info(
            f"{emoji} [紙上交易] SELL {ticker} {shares}股 @${price:.2f} "
            f"損益={pnl_usd:+,.0f} ({pnl_pct:+.1%}) [{reason}]"
        )

    # ── Settlement report ─────────────────────────────────────────────────────

    def settle(self):
        holding_value = sum(
            h["shares"] * (self._get_latest_price(t) or h["entry_price"])
            for t, h in self.holdings.items()
        )
        total = self.cash + holding_value
        sells = [t for t in self.trade_log if t["action"] == "SELL"]
        realized = sum(t.get("pnl", 0) for t in sells)

        logger.info("=" * 55)
        logger.info(f"📊 US 每日結算 {datetime.now(_TZ).strftime('%Y-%m-%d')}")
        logger.info(f"   現金:       ${self.cash:>10,.2f}")
        logger.info(f"   持倉市值:   ${holding_value:>10,.2f}")
        logger.info(f"   總資產:     ${total:>10,.2f}")
        logger.info(f"   已實現損益: ${realized:>+10,.2f}")
        logger.info(f"   大盤環境:   {self._get_market_regime()}")
        for ticker, h in self.holdings.items():
            price = self._get_latest_price(ticker) or h["entry_price"]
            pct   = (price - h["entry_price"]) / h["entry_price"]
            logger.info(f"   持倉 {ticker}: {h['shares']}股 損益 {pct:+.1%}")
        logger.info("=" * 55)
        self._save_state()

    # ── Main cycle ────────────────────────────────────────────────────────────

    def run_cycle(self):
        logger.info("=== US Bot 交易循環開始 ===")
        if not self.model.load():
            logger.info("無 US 模型，先訓練...")
            self.retrain("首次啟動")
        self._check_holdings()
        self._scan_and_buy()
        self._save_state()
        logger.info("=== US Bot 交易循環結束 ===")
