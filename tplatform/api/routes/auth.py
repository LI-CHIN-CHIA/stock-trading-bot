from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from ..database import get_db
from ..models import User, UserConfig
from ..auth import hash_password, verify_password, create_token
from ..schemas import RegisterRequest, LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.scalar(select(User).where(User.email == body.email))
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # First user to register becomes admin
    user_count = await db.scalar(select(func.count()).select_from(User))
    is_admin = (user_count == 0)

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        display_name=body.display_name or body.email.split("@")[0],
        is_admin=is_admin,
    )
    db.add(user)
    await db.flush()

    db.add(UserConfig(user_id=user.id))
    await db.commit()
    await db.refresh(user)

    return TokenResponse(
        access_token=create_token(user.id),
        user_id=user.id,
        display_name=user.display_name,
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(select(User).where(User.email == body.email))
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    return TokenResponse(
        access_token=create_token(user.id),
        user_id=user.id,
        display_name=user.display_name,
    )
