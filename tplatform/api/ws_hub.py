import asyncio
import json
import logging
from fastapi import WebSocket

logger = logging.getLogger("platform.ws")


class ConnectionHub:
    """每個 user_id 可有多個 WebSocket 連線（同一 user 多裝置）。"""

    def __init__(self):
        self._connections: dict[int, set[WebSocket]] = {}

    async def connect(self, user_id: int, ws: WebSocket):
        await ws.accept()
        self._connections.setdefault(user_id, set()).add(ws)
        logger.info(f"[WS] user={user_id} connected  total={self._count()}")

    def disconnect(self, user_id: int, ws: WebSocket):
        conns = self._connections.get(user_id, set())
        conns.discard(ws)
        if not conns:
            self._connections.pop(user_id, None)
        logger.info(f"[WS] user={user_id} disconnected  total={self._count()}")

    async def send(self, user_id: int, msg: dict):
        dead = set()
        for ws in list(self._connections.get(user_id, set())):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.disconnect(user_id, ws)

    async def broadcast(self, msg: dict):
        """把同一訊息推給所有連線 user（用於廣播 AI 信號）。"""
        tasks = [self.send(uid, msg) for uid in list(self._connections)]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def online_users(self) -> list[int]:
        return list(self._connections.keys())

    def _count(self) -> int:
        return sum(len(v) for v in self._connections.values())


hub = ConnectionHub()
