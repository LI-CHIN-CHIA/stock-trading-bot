"""
Fugle Trade API wrapper.
Supports LIVE mode (requires config.ini + .p12 cert) and DRY_RUN mode.

Config.ini format:
  [Core]
  Entry = https://api.fugle.tw/trade/v1.0

  [User]
  Account = YOUR_ACCOUNT_NUMBER

  [Cert]
  Path = /path/to/cert.p12

  [Api]
  Key = YOUR_API_KEY
  Secret = YOUR_API_SECRET
"""

import configparser
import logging
import random
import string
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config.ini"


def _random_order_id() -> str:
    return "DRY_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


class FugleAuthError(Exception):
    pass


class FugleClient:
    """
    Wraps the Fugle Trade SDK for odd-lot buy/sell operations.
    Falls back to DRY_RUN mode automatically when config is missing/invalid.
    """

    def __init__(self, config_path: str | Path = CONFIG_PATH, dry_run: bool = False):
        self._config_path = Path(config_path)
        self._sdk = None
        self._account = "N/A"

        if dry_run:
            self.dry_run = True
            logger.info("FugleClient: DRY_RUN mode (forced)")
            return

        self.dry_run = self._try_init_sdk()

    def _try_init_sdk(self) -> bool:
        """Attempt to initialise the Fugle SDK. Returns True if dry_run."""
        if not self._config_path.exists():
            logger.warning(f"config.ini not found at {self._config_path}. Running in DRY_RUN mode.")
            return True

        config = configparser.ConfigParser()
        config.read(self._config_path)

        # Check for placeholder values
        api_key = config.get("Api", "Key", fallback="YOUR_API_KEY_HERE")
        account = config.get("User", "Account", fallback="")
        cert_path = config.get("Cert", "Path", fallback="")

        if "YOUR_API_KEY" in api_key or not account or not cert_path:
            logger.warning("config.ini contains placeholder values. Running in DRY_RUN mode.")
            return True

        if not Path(cert_path).exists():
            logger.warning(f"Certificate file not found: {cert_path}. Running in DRY_RUN mode.")
            return True

        try:
            from fugle_trade.sdk import SDK
            self._sdk = SDK(config)
            self._sdk.login()
            self._account = account
            logger.info(f"FugleClient: LIVE mode. Account={account}")
            return False
        except Exception as e:
            logger.error(f"Fugle SDK init failed: {e}. Running in DRY_RUN mode.")
            return True

    # ── Order Placement ───────────────────────────────────────────────────────

    def place_odd_lot_buy(
        self,
        ticker_code: str,
        shares: int,
        price: float,
        session: str = "intraday",
    ) -> dict:
        """
        Place odd-lot buy order.
        session: 'intraday' (盤中零股 09:10-13:30) or 'after_close' (盤後零股 14:00-14:30)
        """
        if self.dry_run:
            return self._dry_run_order("BUY", ticker_code, shares, price, session)

        return self._live_order("BUY", ticker_code, shares, price, session)

    def place_odd_lot_sell(
        self,
        ticker_code: str,
        shares: int,
        price: float,
        session: str = "intraday",
    ) -> dict:
        """Place odd-lot sell order."""
        if self.dry_run:
            return self._dry_run_order("SELL", ticker_code, shares, price, session)

        return self._live_order("SELL", ticker_code, shares, price, session)

    def _live_order(
        self, action: str, ticker_code: str, shares: int, price: float, session: str
    ) -> dict:
        from fugle_trade.constant import Action, APCode, BSFlag, PriceFlag, Trade
        from fugle_trade.order import OrderObject

        ap_code = APCode.IntradayOdd if session == "intraday" else APCode.Odd
        buy_sell = Action.Buy if action == "BUY" else Action.Sell

        order = OrderObject(
            buy_sell=buy_sell,
            price=round(price, 2),
            stock_no=ticker_code,
            quantity=int(shares),
            ap_code=ap_code,
            bs_flag=BSFlag.ROD,
            price_flag=PriceFlag.Limit,
            trade=Trade.Cash,
        )

        for attempt in range(2):
            try:
                result = self._sdk.place_order(order)
                order_id = result.get("ord_no") or result.get("ordno") or "UNKNOWN"
                logger.info(f"Order placed: {action} {shares}x{ticker_code} @ {price} | order_id={order_id}")
                return {
                    "order_id": str(order_id),
                    "status": "submitted",
                    "ticker": ticker_code,
                    "action": action,
                    "shares": shares,
                    "price": price,
                    "session": session,
                    "dry_run": False,
                    "raw": result,
                }
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Order attempt 1 failed: {e}, retrying...")
                    time.sleep(1)
                else:
                    logger.error(f"Order failed after retry: {e}")
                    raise

    def _dry_run_order(
        self, action: str, ticker_code: str, shares: int, price: float, session: str
    ) -> dict:
        order_id = _random_order_id()
        logger.info(
            f"[DRY_RUN] {action} {shares}x{ticker_code} @ {price:.2f} "
            f"(session={session}, amount={shares * price:.0f} TWD) order_id={order_id}"
        )
        return {
            "order_id": order_id,
            "status": "simulated",
            "ticker": ticker_code,
            "action": action,
            "shares": shares,
            "price": price,
            "session": session,
            "dry_run": True,
        }

    # ── Account Info ──────────────────────────────────────────────────────────

    def get_account_balance(self) -> dict:
        if self.dry_run:
            return {"available_cash": 20000.0, "note": "DRY_RUN"}

        try:
            data = self._sdk.get_balance()
            return {
                "available_cash": float(data.get("available_balance", 0)),
                "raw": data,
            }
        except Exception as e:
            logger.error(f"get_balance failed: {e}")
            return {"available_cash": 0.0, "error": str(e)}

    def get_brokerage_positions(self) -> list[dict]:
        """Fetch open positions from Fugle (for reconciliation on restart)."""
        if self.dry_run:
            return []

        try:
            inventories = self._sdk.get_inventories()
            positions = []
            for item in inventories:
                positions.append({
                    "ticker": item.get("stock_no", ""),
                    "shares": int(item.get("qty_l", 0)),
                    "cost_price": float(item.get("cost_r", 0)),
                })
            return positions
        except Exception as e:
            logger.error(f"get_inventories failed: {e}")
            return []

    def cancel_order(self, order_result: dict) -> dict:
        if self.dry_run:
            logger.info(f"[DRY_RUN] Cancel order {order_result.get('order_id')}")
            return {"status": "simulated_cancel"}

        try:
            result = self._sdk.cancel_order(order_result)
            return result
        except Exception as e:
            logger.error(f"cancel_order failed: {e}")
            return {"error": str(e)}

    @property
    def account(self) -> str:
        return self._account

    @property
    def mode(self) -> str:
        return "DRY_RUN" if self.dry_run else "LIVE"
