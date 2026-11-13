"""每日收盤後把當日 1 分 K 存檔（yfinance 只保留 30 天，不存就永遠只有一個月樣本）。

輸出: DATA_DIR/bars_1m/<YYYY-MM-DD>.csv.gz，欄位 ts,code,Open,High,Low,Close,Volume
（code 含 ^TWII 與 0050）。重跑同一天會覆蓋。
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
import yfinance as yf

from utils.constants import OTC_SYMBOLS, TRADEABLE

logger = logging.getLogger(__name__)
TZ = pytz.timezone("Asia/Taipei")
_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _symbol(code: str) -> str:
    if code.startswith("^"):
        return code
    return f"{code}.TWO" if code in OTC_SYMBOLS else f"{code}.TW"


def archive_today(codes: list[str] | None = None) -> Path | None:
    today = datetime.now(TZ).date()
    codes = codes or (sorted(TRADEABLE) + ["0050", "^TWII"])
    parts = []
    for code in codes:
        try:
            df = yf.Ticker(_symbol(code)).history(period="1d", interval="1m")
            if df.empty:
                continue
            df = df.tz_convert(TZ)
            df = df[df.index.date == today][_COLS]
            if df.empty:
                continue
            df.insert(0, "code", code)
            parts.append(df)
        except Exception as e:
            logger.warning(f"[BarArchive] {code} 失敗: {e}")
        time.sleep(0.2)
    if not parts:
        logger.warning("[BarArchive] 今日無資料（休市？）")
        return None
    out = pd.concat(parts)
    out.index.name = "ts"
    d = Path(os.getenv("DATA_DIR", ".")) / "bars_1m"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{today}.csv.gz"
    out.to_csv(path, compression="gzip")
    logger.info(f"[BarArchive] 已存 {path}  {out.code.nunique()} 檔 {len(out)} 根")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    archive_today()
