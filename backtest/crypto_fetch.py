"""抓 Binance 公開現貨 K 線（不需金鑰）存成 pickle，供加密貨幣回測使用。

用法: venv/bin/python backtest/crypto_fetch.py <cache_dir> [1h] [BTCUSDT,ETHUSDT]
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://data-api.binance.vision/api/v3/klines"


def fetch(symbol: str, interval: str, start_ms: int) -> pd.DataFrame:
    rows, cur = [], start_ms
    while True:
        r = requests.get(URL, params={"symbol": symbol, "interval": interval, "startTime": cur, "limit": 1000}, timeout=30)
        r.raise_for_status()
        d = r.json()
        if not d:
            break
        rows += d
        cur = d[-1][0] + 1
        if len(d) < 1000:
            break
        time.sleep(0.15)
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["ts", "Open", "High", "Low", "Close", "Volume"]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts").astype(float)


if __name__ == "__main__":
    cache = Path(sys.argv[1]); cache.mkdir(parents=True, exist_ok=True)
    iv = sys.argv[2] if len(sys.argv) > 2 else "1h"
    syms = (sys.argv[3] if len(sys.argv) > 3 else "BTCUSDT,ETHUSDT").split(",")
    start = int(pd.Timestamp("2018-01-01", tz="UTC").timestamp() * 1000)
    out = {}
    for s in syms:
        out[s] = fetch(s, iv, start)
        print(s, iv, len(out[s]), out[s].index[0], out[s].index[-1], flush=True)
    pickle.dump(out, open(cache / f"crypto_{iv}.pkl", "wb"))
