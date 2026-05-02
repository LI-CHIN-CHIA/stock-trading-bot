# ── Stage 1: Builder ─────────────────────────────────────────────────────────
# 安裝所有套件（含 build tools），只有這層需要 gcc
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY fubon_neo-2.2.8-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /tmp/
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install \
    /tmp/fubon_neo-2.2.8-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
# 乾淨的 runtime，不含 gcc 或 build artifacts
FROM python:3.12-slim AS runtime

# LightGBM 需要 libgomp1 (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 從 builder 複製已安裝的套件
COPY --from=builder /install /usr/local

WORKDIR /app

# 複製程式碼（不含 venv、__pycache__、.env 等 — 見 .dockerignore）
COPY analysis/      ./analysis/
COPY ai/            ./ai/
COPY trader/        ./trader/
COPY charts/        ./charts/
COPY data/          ./data/
COPY utils/         ./utils/
COPY app.py dashboard.py trading_bot.py run_backtest.py ./

ENV DATA_DIR=/data
RUN mkdir -p /data /certs

CMD ["python", "trading_bot.py"]
