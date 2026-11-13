"""
Trading Platform — Local Agent
User 本機常駐服務：接收 5090 AI 信號 → 依 config 決定是否下單 → 回報結果
"""
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import aiohttp

from .config import AgentConfig, load as load_config
from .fubon_trader import FubonTrader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agent")

RECONNECT_DELAY   = 5
HEARTBEAT_INTERVAL = 30
CONFIRM_INTERVAL  = 10

_TZ_TAIPEI = timezone(timedelta(hours=8))
_PENDING_FILE = Path.home() / ".trading-agent" / "pending.json"


def _today_taipei() -> str:
    return datetime.now(_TZ_TAIPEI).strftime("%Y-%m-%d")


class LocalAgent:
    def __init__(self, cfg: AgentConfig | None = None):
        self.cfg = cfg or load_config()
        self._trader: FubonTrader | None = None
        self._user_config: dict = {}
        self._pending: dict[str, dict] = self._load_pending()
        self._session: aiohttp.ClientSession | None = None
        self._user_id: int = 0

        # {code: {"shares": int, "entry_price": float, "direction": "LONG"|"SHORT"}}
        self._holdings: dict[str, dict] = {}

    # ── Pending persistence ───────────────────────────────────────────────────

    def _load_pending(self) -> dict:
        try:
            if _PENDING_FILE.exists():
                return json.loads(_PENDING_FILE.read_text())
        except Exception:
            pass
        return {}

    def _save_pending(self):
        try:
            _PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
            _PENDING_FILE.write_text(json.dumps(self._pending))
        except Exception as e:
            logger.warning(f"[Agent] 無法儲存 pending: {e}")

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.cfg.token}"}

    async def _get(self, path: str) -> dict | list:
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
        cert_pass  = os.getenv(self.cfg.cert_pass_env, "")
        fubon_pass = os.getenv(self.cfg.fubon_pass_env, "")
        self._trader = FubonTrader(self.cfg.cert_path, cert_pass, self.cfg.fubon_account, fubon_pass)
        if self._trader.is_real:
            logger.info("[Agent] 富邦 SDK 就緒，真實下單模式")
        else:
            logger.warning("[Agent] 富邦 SDK 初始化失敗，退回 paper 模式")

    async def _sync_config(self):
        try:
            self._user_config = await self._get("/api/config")
            logger.info(f"[Agent] 設定同步: auto_trade={self._user_config.get('auto_trade')} "
                        f"budget={self._user_config.get('day_budget')} "
                        f"max_pos={self._user_config.get('max_positions')} "
                        f"day={self._user_config.get('enable_day')} short={self._user_config.get('enable_short')}")
        except Exception as e:
            logger.warning(f"[Agent] 設定同步失敗: {e}")

    async def _sync_positions(self):
        """Load server-side positions on startup so local holdings match."""
        try:
            rows = await self._get("/api/positions")
            self._holdings.clear()
            for p in (rows if isinstance(rows, list) else []):
                self._holdings[p["code"]] = {
                    "shares":      p["shares"],
                    "entry_price": p["entry_price"],
                    "direction":   p.get("direction", "LONG"),
                }
            logger.info(f"[Agent] 持倉同步: {list(self._holdings.keys())}")
        except Exception as e:
            logger.warning(f"[Agent] 持倉同步失敗: {e}")

    # ── Position helpers ──────────────────────────────────────────────────────

    def _shares_for(self, price: float) -> int:
        """Compute lot size (1 lot = 1000 shares) from day_budget."""
        budget = float(self._user_config.get("day_budget", 50000))
        lots = max(1, int(budget / (price * 1000)))
        return lots * 1000

    def _can_open(self) -> bool:
        """Check max_positions limit."""
        max_pos = int(self._user_config.get("max_positions", 3))
        return len(self._holdings) < max_pos

    async def _update_position(self, code: str, direction: str, shares: int, entry_price: float, is_real: bool):
        try:
            await self._post("/api/positions/upsert", {
                "code": code,
                "direction": direction,
                "shares": shares,
                "entry_price": entry_price,
                "is_real": is_real,
            })
        except Exception as e:
            logger.warning(f"[Agent] 持倉更新失敗: {e}")

    # ── Signal handling ───────────────────────────────────────────────────────

    async def _handle_signal(self, ws, data: dict):
        code      = data.get("code", "")
        action    = data.get("action", "")
        score     = data.get("score", 0)
        price     = data.get("price", 0.0)
        reason    = data.get("reason", "")

        logger.info(f"[Agent] 📡 信號 {action} {code} score={score} @{price:.2f} ({reason})")

        cfg = self._user_config
        if not cfg.get("auto_trade"):
            logger.info(f"[Agent] ⏸  auto_trade=false，略過 {code}")
            return

        if action in ("BUY", "SELL") and not cfg.get("enable_day"):
            logger.info(f"[Agent] 當沖關閉，略過 {code}"); return
        if action in ("SHORT", "COVER") and not cfg.get("enable_short"):
            logger.info(f"[Agent] 反沖關閉，略過 {code}"); return

        # Determine direction
        direction = "SHORT" if action == "SHORT" else "LONG"

        # Position gate
        if action in ("BUY", "SHORT"):
            if code in self._holdings:
                logger.info(f"[Agent] 已持有 {code}，略過重複 {action}"); return
            if not self._can_open():
                logger.info(f"[Agent] 持倉已達上限 {cfg.get('max_positions')}，略過 {code}"); return

        elif action in ("SELL", "COVER"):
            if code not in self._holdings:
                logger.info(f"[Agent] 未持有 {code}，略過 {action}"); return

        shares = self._shares_for(price) if action in ("BUY", "SHORT") else \
                 self._holdings.get(code, {}).get("shares", 1000)

        # 空單前檢查借券額度（真實模式才有意義）
        if action == "SHORT" and self._trader is not None and self._trader.is_real:
            quota = self._trader.check_short_quota(code)
            needed = shares // 1000  # lots
            if quota < needed:
                logger.warning(f"[Agent] {code} 借券額度不足（需要 {needed} 張，剩餘 {quota} 張），略過 SHORT")
                return

        is_real = self._trader is not None and self._trader.is_real

        if not is_real:
            # Paper mode: simulate fill immediately
            await self._on_fill(ws, code, action, direction, shares, price, reason, "", is_real=False)
            return

        # Real order
        order_no = self._trader.place_order(code, action, shares, price)
        if order_no:
            self._pending[order_no] = {
                "code": code, "action": action, "direction": direction,
                "shares": shares, "price": price, "reason": reason,
                "placed_at": time.time(),
            }
            self._save_pending()
            logger.info(f"[Agent] ⏳ 委託送出 {code} 單號={order_no}")
        else:
            logger.warning(f"[Agent] ❌ 下單失敗 {code}")

    async def _on_fill(self, ws, code, action, direction, shares, price, reason, order_no, is_real=True):
        """Called on confirmed fill. Update holdings, compute PnL, report."""
        pnl = 0.0

        if action in ("BUY", "SHORT"):
            self._holdings[code] = {"shares": shares, "entry_price": price, "direction": direction}
            await self._update_position(code, direction, shares, price, is_real)

        elif action in ("SELL", "COVER"):
            pos = self._holdings.pop(code, {})
            entry = pos.get("entry_price", price)
            if action == "SELL":
                pnl = (price - entry) * shares
            else:  # COVER (short)
                pnl = (entry - price) * shares
            await self._update_position(code, direction, 0, 0.0, is_real)

        await self._report_trade(ws, code, action, shares, price, round(pnl, 0), reason, order_no, is_real)

    async def _report_trade(self, ws, code, action, shares, price, pnl, reason, order_no, is_real=True):
        payload = {
            "code": code, "action": action, "shares": shares,
            "price": price, "pnl": pnl, "reason": reason,
            "order_no": order_no, "is_real": is_real,
            "trade_date": _today_taipei(),
        }
        try:
            result = await self._post("/api/trades", payload)
            await ws.send_json({"type": "trade_result", "data": {**payload, "id": result.get("id")}})
            logger.info(f"[Agent] ✅ 回報成功 {action} {code} pnl={pnl:+.0f}")
        except Exception as e:
            logger.warning(f"[Agent] 回報失敗: {e}")

    # ── Pending order confirmation (runs as background task) ──────────────────

    async def _confirm_loop(self, ws):
        """Periodically check pending orders; runs independently of WS messages."""
        while True:
            await asyncio.sleep(CONFIRM_INTERVAL)
            if not self._pending or not self._trader:
                continue
            done = []
            for order_no, o in list(self._pending.items()):
                filled, fill_price = self._trader.check_filled(o["code"], order_no)
                if filled > 0:
                    p = fill_price if fill_price > 0 else o["price"]
                    await self._on_fill(ws, o["code"], o["action"], o["direction"],
                                        filled, p, o["reason"], order_no)
                    done.append(order_no)
                elif filled == -1:
                    logger.warning(f"[Agent] 委託被拒 {o['code']} {order_no}")
                    done.append(order_no)
                elif time.time() - o["placed_at"] > 120:
                    logger.warning(f"[Agent] 委託超時 {o['code']} {order_no}")
                    done.append(order_no)
            if done:
                for k in done:
                    self._pending.pop(k, None)
                self._save_pending()

    # ── WebSocket main loop ───────────────────────────────────────────────────

    async def _ws_loop(self):
        ws_url = (self.cfg.server_url
                  .replace("https://", "wss://")
                  .replace("http://", "ws://"))
        ws_url += f"/ws/{self._user_id}?token={self.cfg.token}"

        async with self._session.ws_connect(ws_url) as ws:
            logger.info("[Agent] WebSocket 連線成功")
            await self._sync_config()
            await self._sync_positions()

            confirm_task = asyncio.create_task(self._confirm_loop(ws))
            last_hb = time.time()

            try:
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

                    if msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                        break
            finally:
                confirm_task.cancel()
                try:
                    await confirm_task
                except asyncio.CancelledError:
                    pass

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
