"""
Main trading engine: scan → predict → order → monitor.
Orchestrates all components for automated trading.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import pytz

from ai.features import build_feature_matrix, build_target, prepare_inference_row
from ai.model import TradingModel
from analysis.indicators import TechnicalIndicators
from analysis.signals import SignalEngine
from data.fetcher import StockDataFetcher
from data.state import StateStore
from trading.fugle_client import FugleClient
from trading.risk_manager import RiskManager
from utils.constants import MODERATE_BUY_THRESHOLD, STOCK_DB

logger = logging.getLogger(__name__)

TZ = pytz.timezone("Asia/Taipei")
MAX_SCAN_WORKERS = 5  # parallel yfinance fetches


class TradingEngine:

    def __init__(
        self,
        state: StateStore,
        fugle: FugleClient,
        risk: RiskManager,
        model: TradingModel,
    ):
        self.state = state
        self.fugle = fugle
        self.risk = risk
        self.model = model
        self.fetcher = StockDataFetcher()
        self.signal_engine = SignalEngine()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fetch_and_analyze(self, ticker_code: str) -> dict | None:
        """Fetch data, compute indicators, run rule engine and AI model."""
        try:
            df = self.fetcher.fetch_historical(ticker_code, period="6mo")
            df = TechnicalIndicators.add_all(df)
            rule_result = self.signal_engine.analyze(df, ticker_code)
            price = float(df["Close"].iloc[-1])

            if self.model.is_trained():
                # Try loading if not already loaded
                if not self.model._booster:
                    self.model.load("latest")
                X_row = prepare_inference_row(df)
                ai_pred = self.model.predict(X_row) if not X_row.empty else {
                    "signal": "HOLD", "buy_proba": 0.0, "trained": False
                }
            else:
                ai_pred = {"signal": "HOLD", "buy_proba": 0.0, "trained": False}

            return {
                "ticker": ticker_code,
                "name": STOCK_DB.get(ticker_code, ticker_code),
                "price": price,
                "rule_score": rule_result.total_score,
                "rule_direction": rule_result.direction,
                "ai_signal": ai_pred["signal"],
                "ai_buy_proba": ai_pred.get("buy_proba", 0.0),
                "ai_trained": ai_pred.get("trained", False),
                "df": df,
            }
        except Exception as e:
            logger.debug(f"Skipping {ticker_code}: {e}")
            return None

    # ── Scan & Trade ──────────────────────────────────────────────────────────

    def scan_and_trade(self) -> list[dict]:
        """
        Morning scan: evaluate all stocks and place buy orders for top candidates.
        Called at 09:10 AM on trading days.
        """
        logger.info("=== SCAN & TRADE started ===")
        scan_results = []
        actions = []

        # Parallel fetch for all stocks
        ticker_codes = list(STOCK_DB.keys())
        with ThreadPoolExecutor(max_workers=MAX_SCAN_WORKERS) as executor:
            futures = {executor.submit(self._fetch_and_analyze, t): t for t in ticker_codes}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    scan_results.append(result)
                time.sleep(0.1)  # light throttle

        # Sort by combined score: rule_score + AI buy probability
        scan_results.sort(
            key=lambda r: r["rule_score"] + r["ai_buy_proba"] * 3,
            reverse=True
        )

        # Save scan results to state for dashboard display
        summary = [
            {
                "ticker": r["ticker"],
                "name": r["name"],
                "price": r["price"],
                "rule_score": r["rule_score"],
                "ai_buy_proba": r["ai_buy_proba"],
                "ai_signal": r["ai_signal"],
            }
            for r in scan_results[:20]
        ]
        self.state.update_meta(
            last_scan=datetime.now().isoformat(),
            last_scan_results=summary,
        )

        # Try to open positions for top candidates
        for r in scan_results:
            ticker = r["ticker"]
            price = r["price"]
            rule_score = r["rule_score"]
            ai_signal = r["ai_signal"]
            ai_buy_proba = r["ai_buy_proba"]
            ai_trained = r["ai_trained"]

            # Entry gate: rule score + AI both bullish (or rule-only if AI not trained)
            rule_is_buy = rule_score >= MODERATE_BUY_THRESHOLD
            if ai_trained:
                gate = rule_is_buy and (ai_signal == "BUY")
            else:
                # Cold start: rule-only mode with stricter threshold
                gate = rule_score >= 4  # Strong buy only

            if not gate:
                continue

            # Risk check
            ok, reason = self.risk.can_open_position(ticker, price, self.state)
            if not ok:
                logger.debug(f"Skip {ticker}: {reason}")
                continue

            # Calculate shares
            available = self.risk.get_available_budget(self.state)
            shares = self.risk.calculate_position_size(price, available)
            if shares == 0:
                continue

            # Place order
            try:
                order = self.fugle.place_odd_lot_buy(ticker, shares, price, session="intraday")
            except Exception as e:
                logger.error(f"Order failed for {ticker}: {e}")
                continue

            entry_price = price
            position = {
                "ticker": ticker,
                "name": STOCK_DB.get(ticker, ticker),
                "shares": shares,
                "entry_price": entry_price,
                "entry_date": datetime.now().isoformat(),
                "cost_basis": round(shares * entry_price, 2),
                "stop_loss": self.risk.calculate_stop_loss(entry_price),
                "take_profit": self.risk.calculate_take_profit(entry_price),
                "order_id": order["order_id"],
                "order_type": "odd_lot_intraday",
                "rule_score": rule_score,
                "ai_buy_proba": ai_buy_proba,
                "dry_run": self.fugle.dry_run,
            }
            self.state.add_position(ticker, position)
            self.state.add_trade({
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker,
                "name": STOCK_DB.get(ticker, ticker),
                "action": "BUY",
                "shares": shares,
                "price": entry_price,
                "amount": round(shares * entry_price, 2),
                "reason": f"規則分數={rule_score:+d}, AI買入機率={ai_buy_proba:.0%}",
                "order_id": order["order_id"],
                "dry_run": self.fugle.dry_run,
                "pnl": None,
            })
            actions.append(position)
            logger.info(
                f"BUY: {ticker} {shares}股 @ {entry_price:.2f} "
                f"(成本={shares*entry_price:.0f} TWD, rule={rule_score:+d}, AI={ai_buy_proba:.0%})"
            )

            # Stop scanning once max positions reached
            if len(self.state.get_positions()) >= self.risk.max_positions:
                break

        logger.info(f"=== SCAN done. Bought {len(actions)} positions ===")
        return actions

    # ── Monitor Positions ─────────────────────────────────────────────────────

    def monitor_positions(self) -> list[dict]:
        """
        Check stop-loss / take-profit for all open positions.
        Called every 3 minutes during trading hours.
        """
        positions = self.state.get_positions()
        if not positions:
            return []

        actions = []
        for ticker, pos in positions.items():
            try:
                quote = self.fetcher.fetch_realtime_quote(ticker)
                current_price = quote.get("price", 0)
                if not current_price:
                    continue

                exit_signal, reason = self.risk.check_exit_conditions(
                    ticker, current_price, self.state
                )
                if exit_signal == "HOLD":
                    continue

                # Sell
                shares = pos["shares"]
                try:
                    order = self.fugle.place_odd_lot_sell(
                        ticker, shares, current_price, session="intraday"
                    )
                except Exception as e:
                    logger.error(f"Sell order failed for {ticker}: {e}")
                    continue

                pnl = round((current_price - pos["entry_price"]) * shares, 2)
                self.state.remove_position(ticker)
                self.state.add_trade({
                    "timestamp": datetime.now().isoformat(),
                    "ticker": ticker,
                    "name": pos.get("name", ticker),
                    "action": "SELL",
                    "shares": shares,
                    "price": current_price,
                    "amount": round(shares * current_price, 2),
                    "reason": f"{exit_signal}: {reason}",
                    "order_id": order["order_id"],
                    "dry_run": self.fugle.dry_run,
                    "pnl": pnl,
                })
                actions.append({
                    "ticker": ticker,
                    "action": "SELL",
                    "exit_signal": exit_signal,
                    "price": current_price,
                    "pnl": pnl,
                })
                logger.info(f"SELL: {ticker} @ {current_price:.2f} | {exit_signal} | P&L {pnl:+.0f} TWD")

            except Exception as e:
                logger.error(f"Monitor error for {ticker}: {e}")

        return actions

    # ── Model Retraining ──────────────────────────────────────────────────────

    def retrain_model(self) -> dict:
        """
        Retrain LightGBM on 2 years of data for all stocks.
        Called at 14:35 after market close.
        """
        logger.info("=== MODEL RETRAIN started ===")
        all_X = []
        all_y = []
        skipped = []

        for ticker_code in STOCK_DB.keys():
            try:
                df = self.fetcher.fetch_historical(ticker_code, period="2y")
                df = TechnicalIndicators.add_all(df)
                X = build_feature_matrix(df)
                y = build_target(df)
                common = X.index.intersection(y.index)
                if len(common) < 60:
                    skipped.append(ticker_code)
                    continue
                all_X.append(X.loc[common])
                all_y.append(y.loc[common])
                time.sleep(0.3)  # avoid rate-limiting yfinance
            except Exception as e:
                logger.warning(f"Skipping {ticker_code} for training: {e}")
                skipped.append(ticker_code)

        if not all_X:
            logger.error("No training data collected, retrain aborted.")
            return {"error": "no data"}

        X_combined = pd.concat(all_X)
        y_combined = pd.concat(all_y)

        try:
            metrics = self.model.train(X_combined, y_combined)
            version = datetime.now().strftime("%Y%m%d_%H%M")
            self.model.save(version=version)
            metrics["version"] = version
            metrics["stocks_used"] = len(all_X)
            metrics["stocks_skipped"] = skipped

            self.state.update_meta(
                last_retrain=datetime.now().isoformat(),
                model_version=version,
                training_metrics=metrics,
            )
            logger.info(f"=== MODEL RETRAIN done. Version={version}, F1={metrics.get('val_macro_f1')} ===")
            return metrics
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"error": str(e)}
