import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .database import init_db
from .routes import auth, config, trades, signals, positions, ws

_STATIC = Path(__file__).parent.parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logging.getLogger("platform").info("Platform API started — DB tables ready")
    yield


app = FastAPI(title="Trading Platform API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # using Bearer tokens, not cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(config.router)
app.include_router(trades.router)
app.include_router(signals.router)
app.include_router(positions.router)
app.include_router(ws.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/app")
async def pwa():
    return FileResponse(_STATIC / "app.html", media_type="text/html")
