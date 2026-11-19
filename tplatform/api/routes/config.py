from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..database import get_db
from ..models import UserConfig
from ..auth import get_current_user_id
from ..schemas import ConfigResponse, ConfigUpdate
from ..ws_hub import hub
from ..accounts import parse_budgets

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config", response_model=ConfigResponse)
async def get_config(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    cfg = await db.scalar(select(UserConfig).where(UserConfig.user_id == user_id))
    if not cfg:
        raise HTTPException(status_code=404, detail="Config not found")
    return cfg


@router.get("/bot-status")
async def get_bot_status(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """伺服器端實際的真實交易狀態（由 bot 每 5 分鐘回報）。"""
    import json
    from sqlalchemy import text
    row = (await db.execute(
        text("SELECT data, updated_at FROM platform_bot_status WHERE user_id = :u"), {"u": user_id})).first()
    if not row:
        return {"available": False}
    data = json.loads(row[0])
    data["available"] = True
    data["reported_at"] = row[1].isoformat()
    return data


@router.put("/config", response_model=ConfigResponse)
async def update_config(
    body: ConfigUpdate,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    cfg = await db.scalar(select(UserConfig).where(UserConfig.user_id == user_id))
    if not cfg:
        raise HTTPException(status_code=404, detail="Config not found")

    changes = body.model_dump(exclude_none=True)
    if "paper_budgets" in changes:
        changes["paper_budgets"] = parse_budgets(changes["paper_budgets"])
    for field, value in changes.items():
        if field == "paper_budgets":
            value = ",".join(str(v) for v in value)
        setattr(cfg, field, value)

    await db.commit()
    await db.refresh(cfg)

    # 即時推給該 user 的 Local Agent
    await hub.send(user_id, {"type": "config_update", "data": changes})

    return cfg
