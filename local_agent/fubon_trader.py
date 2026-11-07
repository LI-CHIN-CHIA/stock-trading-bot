"""
富邦 SDK 下單介面。
若本機沒有安裝 fubon_neo SDK，自動退回 paper 模式。
"""
import os
import logging

logger = logging.getLogger("agent.fubon")

try:
    from fubon_neo.sdk import FubonSDK, Mode
    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False
    logger.warning("[Fubon] fubon_neo SDK 未安裝，強制 paper 模式")


class FubonTrader:
    def __init__(self, cert_path: str, cert_pass: str, account: str):
        self._available = _HAS_SDK and bool(cert_path) and bool(cert_pass)
        self._sdk = None
        self._account = account

        if self._available:
            try:
                sdk = FubonSDK()
                sdk.login(cert_path, cert_pass, Mode.Normal)
                accounts = sdk.stock.get_account()
                if accounts.is_success:
                    matched = next((a for a in accounts.data if a.account == account), None)
                    self._account_obj = matched or accounts.data[0]
                    self._sdk = sdk
                    logger.info(f"[Fubon] 登入成功 account={self._account_obj.account}")
                else:
                    logger.error(f"[Fubon] 取得帳號失敗: {accounts.message}")
                    self._available = False
            except Exception as e:
                logger.error(f"[Fubon] 登入失敗: {e}")
                self._available = False

    @property
    def is_real(self) -> bool:
        return self._available and self._sdk is not None

    def place_order(self, code: str, side: str, shares: int, price: float) -> str | None:
        if not self.is_real:
            return None
        try:
            from fubon_neo.constant import BSAction, TradeType, OrderType
            action = BSAction.Buy if side == "BUY" else BSAction.Sell
            result = self._sdk.stock.place_order(
                self._account_obj,
                code=code,
                bs_action=action,
                price=price,
                qty=shares // 1000,
                trade_type=TradeType.DayTrade,
                order_type=OrderType.Limit,
            )
            if result.is_success:
                order_no = getattr(result.data, "seq_no", "")
                logger.info(f"[Fubon] 下單成功 {side} {code} {shares}股 @{price} 單號={order_no}")
                return order_no
            else:
                logger.warning(f"[Fubon] 下單失敗: {result.message}")
                return None
        except Exception as e:
            logger.error(f"[Fubon] place_order error: {e}")
            return None

    def check_filled(self, code: str, order_no: str) -> tuple[int, float]:
        """回傳 (filled_qty, filled_price)。0=未成交，-1=被拒。"""
        if not self.is_real:
            return 0, 0.0
        try:
            result = self._sdk.stock.get_order_results(self._account_obj)
            if not result.is_success:
                return 0, 0.0
            for o in (result.data or []):
                if getattr(o, "seq_no", "") == order_no:
                    qty   = int(getattr(o, "filled_qty", 0)) * 1000
                    price = float(getattr(o, "filled_price", 0.0))
                    status= getattr(o, "status", "")
                    if "reject" in str(status).lower():
                        return -1, 0.0
                    return qty, price
            return 0, 0.0
        except Exception as e:
            logger.error(f"[Fubon] check_filled error: {e}")
            return 0, 0.0
