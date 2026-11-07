import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from ..auth import decode_token
from ..ws_hub import hub

router = APIRouter(tags=["websocket"])
logger = logging.getLogger("platform.ws")


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(
    ws: WebSocket,
    user_id: int,
    token: str = Query(...),
):
    # Token 驗證
    tid = decode_token(token)
    if tid != user_id:
        await ws.close(code=4001)
        return

    await hub.connect(user_id, ws)
    # 連線後立即推送目前 config（讓 Agent 同步）
    await hub.send(user_id, {"type": "connected", "user_id": user_id})

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "pong":
                pass

            elif msg_type == "trade_result":
                # Local Agent 回報成交 → log，iOS 訂閱端同步
                logger.info(f"[WS] trade_result user={user_id} data={data.get('data')}")
                # 廣播給同一 user 的其他裝置（iOS app）
                await hub.send(user_id, {"type": "trade_update", "data": data.get("data")})

            elif msg_type == "position_update":
                await hub.send(user_id, {"type": "position_update", "data": data.get("data")})

    except WebSocketDisconnect:
        hub.disconnect(user_id, ws)
    except Exception as e:
        logger.warning(f"[WS] user={user_id} error: {e}")
        hub.disconnect(user_id, ws)
