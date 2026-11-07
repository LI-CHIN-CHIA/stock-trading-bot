"""
Trading Platform — Local Agent
User 本機常駐服務：接收 5090 AI 信號 → 依 config 決定是否下單 → 回報結果
"""
import asyncio
import json
import logging
import os
import time
from datetime import date

import aiohttp

from .config import AgentConfig, load as load_config
from .fubon_trader import FubonTrader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agent")

RECONNECT_DELAY = 5    # 斷線後等幾秒重連
HEARTBEAT_INTERVAL = 30


class LocalAgent:
    def __init__(self, cfg: AgentConfig | None = None):
        self.cfg = cfg or load_config()
        self._trader: FubonTrader | None = None
        self._user_config: dict = {}
        self._pending: dict[str, dict] = {}   # order_no → order info
        self._session: aiohttp.ClientSession | None = None

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.cfg.token}"}

    async def _get(self, path: str) -> dict:
        async with self._session.get(
            f"{self.cfg.server_url}{path}", headers=self._headers()
        ) as r:
            return await r.json()

    async def _post(self, path: str, data: dict) -> dict:
        async with self._session.post(
            f"{self.cfg.server_url}{path}",
            json=data, headers=self._headers()
        ) as r:
            return await r.json()

    # ── Startup ───────────────────────────────────────────────────────────────

    async def _init_fubon(self):
        if self.cfg.paper_mode or not self.cfg.cert_path:
            logger.info("[Agent] paper 模式（不下真實單）")
            return
        cert_pass = os.getenv(self.cfg.cert_pass_env, "")
        self._trader = FubonTrader(self.cfg.cert_path, cert_pass, self.cfg.fubon_account)
        if self._trader.is_real:
            logger.info("[Agent] 富邦 SDK 就緒，真實下單模式")
        else:
            logger.warning("[Agent] 富邦 SDK 初始化失敗，退回 paper 模式")

    async def _sync_config(self):
        try:
            self._user_config = await self._get("/api/config")
            logger.info(f"[Agent] 設定同步: auto_trade={self._user_config.get('auto_trade')} "
                        f"budget={self._user_config.get('day_budget')} "
                        f"day={self._user_config.get('enable_day')} "
                        f"short={self._user_config.get('enable_short')}")
        except Exception as e:
            logger.warning(f"[Agent] 設定同步失敗: {e}")

    # ── Signal handling ───────────────────────────────────────────────────────

    async def _handle_signal(self, ws, data: dict):
        code   = data.get("code", "")
        action = data.get("action", "")
        score  = data.get("score", 0)
        price  = data.get("price", 0.0)
        reason = data.get("reason", "")

        logger.info(f"[Agent] 📡 信號 {action} {code} score={score} @{price:.2f} ({reason})")

        cfg = self._user_config
        if not cfg.get("auto_trade"):
            logger.info(f"[Agent] ⏸  auto_trade=false，略過 {code}")
            return

        # 模式過濾
        if action in ("BUY", "SELL") and not cfg.get("enable_day"):
            logger.info(f"[Agent] 當沖關閉，略過 {code}"); return
        if action in ("SHORT", "COVER") and not cfg.get("enable_short"):
            logger.info(f"[Agent] 反沖關閉，略過 {code}"); return

        # paper 模式：模擬成交，直接回報
        if self.cfg.paper_mode or not (self._trader and self._trader.is_real):
            await self._report_trade(ws, code, action, 1000, price, 0.0, reason, "", is_real=False)
            return

        # 真實下單
        order_no = self._trader.place_order(code, action, 1000, price)
        if order_no:
            self._pending[order_no] = {
                "code": code, "action": action, "shares": 1000,
                "price": price, "reason": reason, "placed_at": time.time(),
            }
            logger.info(f"[Agent] ⏳ 委託送出 {code} 單號={order_no}")
        else:
            logger.warning(f"[Agent] ❌ 下單失敗 {code}")

    async def _report_trade(self, ws, code, action, shares, price, pnl, reason, order_no, is_real=True):
        payload = {
            "code": code, "action": action, "shares": shares,
            "price": price, "pnl": pnl, "reason": reason,
            "order_no": order_no, "is_real": is_real,
            "trade_date": str(date.today()),
        }
        try:
            result = await self._post("/api/trades", payload)
            # 也透過 WS 通知（iOS app 即時更新）
            await ws.send_json({"type": "trade_result", "data": {**payload, "id": result.get("id")}})
            logger.info(f"[Agent] ✅ 回報成功 {action} {code} pnl={pnl:+.0f}")
        except Exception as e:
            logger.warning(f"[Agent] 回報失敗: {e}")

    # ── Pending order confirmation ─────────────────────────────────────────────

    async def _confirm_pending(self, ws):
        done = []
        for order_no, o in list(self._pending.items()):
            if not self._trader:
                done.append(order_no); continue
            filled, fill_price = self._trader.check_filled(o["code"], order_no)
            if filled > 0:
                p = fill_price if fill_price > 0 else o["price"]
                await self._report_trade(ws, o["code"], o["action"], filled, p, 0.0, o["reason"], order_no)
                done.append(order_no)
            elif filled == -1:
                logger.warning(f"[Agent] 委託被拒 {o['code']} {order_no}")
                done.append(order_no)
            elif time.time() - o["placed_at"] > 120:
                logger.warning(f"[Agent] 委託超時 {o['code']} {order_no}")
                done.append(order_no)
        for k in done:
            self._pending.pop(k, None)

    # ── WebSocket main loop ───────────────────────────────────────────────────

    async def _ws_loop(self):
        ws_url = (self.cfg.server_url
                  .replace("https://", "wss://")
                  .replace("http://", "ws://"))
        ws_url += f"/ws/{self._user_id}?token={self.cfg.token}"

        async with self._session.ws_connect(ws_url) as ws:
            logger.info(f"[Agent] WebSocket 連線成功")
            await self._sync_config()

            last_hb = time.time()
            last_confirm = time.time()

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    t = data.get("type")

                    if t == "connected":
                        logger.info(f"[Agent] 伺服器確認連線 user_id={data.get('user_id')}")

                    elif t == "signal":
                        await self._handle_signal(ws, data.get("data", {}))

                    elif t == "config_update":
                        self._user_config.update(data.get("data", {}))
                        logger.info(f"[Agent] 設定更新: {data.get('data')}")

                    elif t == "ping":
                        await ws.send_json({"type": "pong"})

                now = time.time()
                if now - last_hb > HEARTBEAT_INTERVAL:
                    await ws.send_json({"type": "pong"})
                    last_hb = now

                if now - last_confirm > 10 and self._pending:
                    await self._confirm_pending(ws)
                    last_confirm = now

                if msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                    break

    # ── Entry point ───────────────────────────────────────────────────────────

    async def _get_user_id(self) -> int:
        try:
            import base64, json as _j
            payload = self.cfg.token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            return int(_j.loads(base64.urlsafe_b64decode(payload))["sub"])
        except Exception:
            return 0

    async def run(self):
        self._user_id = await self._get_user_id()
        if not self._user_id:
            logger.error("[Agent] token 無效，請重新登入")
            return

        await self._init_fubon()

        async with aiohttp.ClientSession() as session:
            self._session = session
            while True:
                try:
                    await self._ws_loop()
                except Exception as e:
                    logger.warning(f"[Agent] 連線中斷: {e}，{RECONNECT_DELAY}s 後重連")
                await asyncio.sleep(RECONNECT_DELAY)


def main():
    """Entry point: python -m local_agent"""
    import sys
    cfg = load_config()
    if not cfg.token:
        print("請先設定 token，執行: python -m local_agent login")
        sys.exit(1)
    asyncio.run(LocalAgent(cfg).run())
