"""抓取重播回測用的分K快取（yfinance）。

用法: venv/bin/python backtest/replay_fetch.py <cache_dir>
產出: bars_1m.pkl (近30日1分K)、bars_2m.pkl (近60日2分K)、twii_5m.pkl、daily.pkl
"""
from __future__ import annotations

import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.constants import OTC_SYMBOLS, TRADEABLE  # noqa: E402


def sym(code: str) -> str:
    return f"{code}.TWO" if code in OTC_SYMBOLS else f"{code}.TW"


def fetch_1m(symbol: str) -> pd.DataFrame:
    """yfinance 1m 只保留近30日，且每次請求最多8日 → 分段抓。"""
    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=29)
    parts = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=7), end)
        try:
            df = yf.Ticker(symbol).history(start=cur, end=nxt, interval="1m")
            if not df.empty:
                parts.append(df)
        except Exception as e:
            print(f"  {symbol} {cur:%m-%d} 失敗: {e}")
        cur = nxt
        time.sleep(0.3)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts)
    return out[~out.index.duplicated()].sort_index()


def main(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    codes = sorted(TRADEABLE)

    bars_1m, bars_2m, daily = {}, {}, {}
    for i, code in enumerate(codes):
        s = sym(code)
        bars_1m[code] = fetch_1m(s)
        try:
            bars_2m[code] = yf.Ticker(s).history(period="60d", interval="2m")
        except Exception:
            bars_2m[code] = pd.DataFrame()
        daily[code] = yf.Ticker(s).history(period="1y", interval="1d")
        print(f"[{i+1}/{len(codes)}] {code} 1m={len(bars_1m[code])} 2m={len(bars_2m[code])} d={len(daily[code])}", flush=True)
        time.sleep(0.5)

    daily["^TWII"] = yf.Ticker("^TWII").history(period="1y", interval="1d")
    twii_5m = yf.Ticker("^TWII").history(period="60d", interval="5m")
    print(f"TWII 5m={len(twii_5m)} d={len(daily['^TWII'])}")

    for name, obj in (("bars_1m", bars_1m), ("bars_2m", bars_2m), ("daily", daily), ("twii_5m", twii_5m)):
        with open(cache / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
    print("done")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
