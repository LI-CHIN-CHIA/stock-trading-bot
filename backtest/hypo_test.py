"""通用進場假說測試器（建立在 replay_live 的特徵表上）。

每個假說 = 一個布林遮罩（每檔每日只取第一次觸發）+ 方向；
出場統一為 停損 / 停利 / 13:20 平倉，扣來回成本，並回報前半、後半期間是否一致。

用法: venv/bin/python backtest/hypo_test.py <cache_dir> [2m|1m]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
COST = 0.001425 * 2 + 0.0015      # 來回手續費 + 當沖證交稅
MAX_PRICE = 165                    # 50 萬模擬帳戶單筆約 16.5 萬，只能做一張 ≤165 元


def prep(f: pd.DataFrame) -> pd.DataFrame:
    f = f[f.close <= MAX_PRICE].sort_values(["code", "day", "ts"]).reset_index(drop=True)
    g = f.groupby(["code", "day"], sort=False)
    f["from_open"] = f.close / f.day_open - 1
    f["gap_open"] = f.day_open / f.prev_close - 1
    f["mkt_from_open"] = f.mkt_cur / f.mkt_open - 1
    med = f.groupby(["code", "hm"]).cum_vol.transform("median")
    f["rvol"] = f.cum_vol / med
    # 開盤區間（09:00~09:15）高低點
    first = f[f.hm <= 915].groupby(["code", "day"]).agg(or_hi=("high", "max"), or_lo=("low", "min"))
    f = f.join(first, on=["code", "day"])
    f["prev_close_bar"] = g.close.shift(1)
    f["ret5bar"] = f.close / g.close.shift(5) - 1
    f["off_high"] = f.close / f.day_high - 1        # 距當日高點
    f["off_low"] = f.close / f.day_low - 1
    return f


def run(f: pd.DataFrame, mask: pd.Series, side: str, sl: float, tp: float) -> pd.DataFrame:
    """每檔每日第一次觸發進場（收盤價），逐根檢查 SL/TP，13:20 平倉。"""
    out = []
    sig = f[mask & (f.hm >= 905) & (f.hm < 1230)]
    firsts = sig.groupby(["code", "day"], sort=False).head(1)
    grp = {k: v for k, v in f.groupby(["code", "day"], sort=False)}
    for r in firsts.itertuples():
        d = grp[(r.code, r.day)]
        after = d[(d.ts > r.ts)]
        e = r.close
        ret = None
        for b in after.itertuples():
            if side == "L":
                if b.low <= e * (1 - sl):
                    ret = min(e * (1 - sl), b.open) / e - 1; break
                if b.high >= e * (1 + tp):
                    ret = max(e * (1 + tp), b.open) / e - 1; break
            else:
                if b.high >= e * (1 + sl):
                    ret = 1 - max(e * (1 + sl), b.open) / e; break
                if b.low <= e * (1 - tp):
                    ret = 1 - min(e * (1 - tp), b.open) / e; break
            if b.hm >= 1320:
                ret = (b.close / e - 1) if side == "L" else (1 - b.close / e); break
        if ret is None:
            if after.empty:
                continue
            last = after.iloc[-1].close
            ret = (last / e - 1) if side == "L" else (1 - last / e)
        out.append((r.day, r.code, r.hm, ret - COST))
    return pd.DataFrame(out, columns=["day", "code", "hm", "ret"])


def stat(t: pd.DataFrame, days: list) -> str:
    if t.empty:
        return "n=0"
    half = set(days[: len(days) // 2])
    a, b = t[t.day.isin(half)], t[~t.day.isin(half)]
    dm = t.groupby("day").ret.mean()
    tval = dm.mean() / dm.std() * np.sqrt(len(dm)) if len(dm) > 2 and dm.std() > 0 else 0
    def s(x):
        return f"n={len(x):4d} win={100 * (x.ret > 0).mean() if len(x) else 0:4.1f}% {1e4 * x.ret.mean() if len(x) else 0:+6.1f}bp"
    return f"{s(t)} t={tval:+.1f} | 前半 {s(a)} | 後半 {s(b)}"


HYPOS = {
    # 名稱: (方向, 遮罩函式)
    "H1 爆量急漲回落(空): 漲≥3% rvol≥2 跌破VWAP":
        ("S", lambda f: (f.stk_pct >= 0.03) & (f.rvol >= 2) & (~f.above_vwap) & (f.hm >= 930)),
    "H1b 爆量急漲回落(空): 漲≥3% rvol≥2 距高點≤-1.5%":
        ("S", lambda f: (f.stk_pct >= 0.03) & (f.rvol >= 2) & (f.off_high <= -0.015) & (f.hm >= 930)),
    "H1c 急漲回落(空): 漲≥5% 跌破VWAP":
        ("S", lambda f: (f.stk_pct >= 0.05) & (~f.above_vwap) & (f.hm >= 930)),
    "H2 爆量急跌反彈(多): 跌≥3% rvol≥2 站上VWAP":
        ("L", lambda f: (f.stk_pct <= -0.03) & (f.rvol >= 2) & (f.above_vwap) & (f.hm >= 930)),
    "H2b 急跌反彈(多): 跌≥5% 距低點≥1.5%":
        ("L", lambda f: (f.stk_pct <= -0.05) & (f.off_low >= 0.015) & (f.hm >= 930)),
    "H3 開盤區間突破(多): 破15分高 rvol≥1.5 大盤不弱":
        ("L", lambda f: (f.hm > 915) & (f.close > f.or_hi) & (f.rvol >= 1.5) & (f.mkt_from_open >= 0)),
    "H3b 開盤區間跌破(空): 破15分低 rvol≥1.5 大盤不強":
        ("S", lambda f: (f.hm > 915) & (f.close < f.or_lo) & (f.rvol >= 1.5) & (f.mkt_from_open <= 0)),
    "H4 跳空高開走弱(空): gap≥2% 跌破開盤價與VWAP":
        ("S", lambda f: (f.gap_open >= 0.02) & (f.from_open < 0) & (~f.above_vwap) & (f.hm >= 915)),
    "H4b 跳空低開走強(多): gap≤-2% 站上開盤價與VWAP":
        ("L", lambda f: (f.gap_open <= -0.02) & (f.from_open > 0) & (f.above_vwap) & (f.hm >= 915)),
    "H5 趨勢日順勢(多): 10:00後 個股自開盤≥+1.5% 大盤自開盤≥+0.4%":
        ("L", lambda f: (f.hm >= 1000) & (f.from_open >= 0.015) & (f.above_vwap) & (f.mkt_from_open >= 0.004)),
    "H5b 趨勢日順勢(空): 10:00後 個股自開盤≤-1.5% 大盤自開盤≤-0.4%":
        ("S", lambda f: (f.hm >= 1000) & (f.from_open <= -0.015) & (~f.above_vwap) & (f.mkt_from_open <= -0.004)),
    "H6 多日弱勢續跌(空): 3日跌≥5% 今日再跌破VWAP 且弱於大盤":
        ("S", lambda f: (f.ret3 <= -0.05) & (~f.above_vwap) & (f.rel <= -0.01) & (f.hm >= 930)),
    "H6b 多日強勢續漲(多): 3日漲≥5% 今日站上VWAP 且強於大盤":
        ("L", lambda f: (f.ret3 >= 0.05) & (f.above_vwap) & (f.rel >= 0.01) & (f.hm >= 930)),
    "B0 對照組: 09:30 隨機全買(多)": ("L", lambda f: f.hm.between(930, 931)),
    "B0s 對照組: 09:30 隨機全空(空)": ("S", lambda f: f.hm.between(930, 931)),
}

if __name__ == "__main__":
    cache = Path(sys.argv[1]); iv = sys.argv[2] if len(sys.argv) > 2 else "2m"
    f = prep(pd.read_pickle(cache / f"feat_{iv}.pkl"))
    days = sorted(f.day.unique())
    print(f"{iv}: {len(days)} 日 {days[0]}~{days[-1]}  {f.code.nunique()} 檔(≤{MAX_PRICE}元)  成本 {COST*1e4:.1f}bp")
    for exits in ((0.02, 0.04), (0.015, 0.025)):
        print(f"\n--- 出場 SL {exits[0]*100:.1f}% / TP {exits[1]*100:.1f}% / 13:20 ---")
        for name, (side, fn) in HYPOS.items():
            print(f"{name[:44]:46s} {stat(run(f, fn(f), side, *exits), days)}")
