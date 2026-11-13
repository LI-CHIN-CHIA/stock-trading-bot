"""用富邦行情 API 抓 1 分 K，輸出成 replay_live 可讀的 bars_1m.pkl。

需先在自己的 shell 設好環境變數（不要貼到聊天室）:
  FUBON_ACCOUNT / FUBON_PASS / FUBON_CERT_PATH / FUBON_CERT_PASS

用法: venv/bin/python backtest/fetch_fubon_candles.py <cache_dir>
富邦歷史行情限制 60 req/min，本腳本每檔間隔 1.1 秒。
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.constants import TRADEABLE  # noqa: E402


def login():
    from fubon_neo.sdk import FubonSDK
    need = ["FUBON_ACCOUNT", "FUBON_PASS", "FUBON_CERT_PATH", "FUBON_CERT_PASS"]
    missing = [k for k in need if not os.getenv(k)]
    if missing:
        sys.exit(f"缺少環境變數: {missing}")
    sdk = FubonSDK()
    r = sdk.login(*(os.environ[k] for k in need))
    if not getattr(r, "is_success", False):
        sys.exit(f"登入失敗: {getattr(r, 'message', r)}")
    sdk.init_realtime()
    return sdk


def to_df(data: list[dict]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Asia/Taipei")
    df = df.set_index("date").sort_index()
    return df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                              "close": "Close", "volume": "Volume"})[["Open", "High", "Low", "Close", "Volume"]]


def main(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    sdk = login()
    hist = sdk.marketdata.rest_client.stock.historical
    out = {}
    for i, code in enumerate(sorted(TRADEABLE)):
        try:
            # 分 K 不接受 from/to，固定回傳近 30 日
            r = hist.candles(**{"symbol": code, "timeframe": "1"})
            out[code] = to_df(r.get("data", []) if isinstance(r, dict) else [])
        except Exception as e:
            print(f"{code} 失敗: {e}")
            out[code] = pd.DataFrame()
        df = out[code]
        span = f"{df.index[0]:%m-%d} ~ {df.index[-1]:%m-%d}" if not df.empty else "-"
        print(f"[{i+1}] {code} bars={len(df)} {span}", flush=True)
        time.sleep(1.1)
    with open(cache / "bars_fubon_1m.pkl", "wb") as f:
        pickle.dump(out, f)
    print("saved", cache / "bars_fubon_1m.pkl")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
