import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

def _build_url() -> str:
    if url := os.getenv("PLATFORM_DATABASE_URL"):
        return url
    user = os.getenv("PLATFORM_DB_USER", "trading")
    pw   = os.getenv("PLATFORM_DB_PASS", "")
    host = os.getenv("PLATFORM_DB_HOST", "postgres")
    port = os.getenv("PLATFORM_DB_PORT", "5432")
    name = os.getenv("PLATFORM_DB_NAME", "trading")
    return f"postgresql+asyncpg://{user}:{pw}@{host}:{port}/{name}"

DATABASE_URL = _build_url()

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    from sqlalchemy import text
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # create_all 不會替既有資料表加欄位 → 這裡補（可重複執行）
        for col in ("real_enable_day", "real_enable_short"):
            await conn.execute(text(
                f"ALTER TABLE platform_configs ADD COLUMN IF NOT EXISTS {col} boolean NOT NULL DEFAULT false"))
        await conn.execute(text(
            "CREATE TABLE IF NOT EXISTS platform_bot_status ("
            " user_id integer PRIMARY KEY, data text NOT NULL,"
            " updated_at timestamptz NOT NULL DEFAULT now())"))
