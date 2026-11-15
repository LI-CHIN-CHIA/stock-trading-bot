"""台股小波段（持有 1~10 個交易日）回測：收盤出訊號、隔日開盤進場；可設停利/停損/時間出場。

- 成本：手續費 0.1425%×2×折數 + 證交稅 0.3%（當日進出算當沖 0.15%）。
- 漲跌停：隔日開盤跳空 ≥ +9.5% 視為買不到（放棄）；出場日一字跌停（開=高=低 且 ≤ 昨收×0.905）順延一天。
- 停利以限價成交（開盤已超過則以開盤價）；同一根同時碰停損與停利時先算停損。
- 每檔同時只持有一筆。資料為還原權息日線（daily_edge.py fetch 的 pickle）。

用法: venv/bin/python backtest/swing_research.py <cache_dir> [daily6y_400.pkl] [fee_discount]
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
FEE, TAX, TAX_DAY, LIMIT = 0.001425, 0.003, 0.0015, 0.095
PRICE_MAX, PRICE_MIN = 330, 10


def rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def features(raw: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for code, d in raw.items():
        d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
        d = d[d.Volume > 0]
        if len(d) < 260:
            continue
        d = d.copy()
        d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
        c = d.Close
        for n in (5, 10, 20, 60, 200):
            d[f"ma{n}"] = c.rolling(n).mean()
        d["rsi2"], d["rsi3"] = rsi(c, 2), rsi(c, 3)
        d["ret1"], d["ret3"], d["ret5"] = c.pct_change(), c.pct_change(3), c.pct_change(5)
        d["volx"] = d.Volume / d.Volume.rolling(20).mean().shift(1)
        d["hi20"] = d.High.rolling(20).max().shift(1)
        d["lo5"] = d.Low.rolling(5).min().shift(1)
        d["dn3"] = (d.ret1 < 0) & (d.ret1.shift(1) < 0) & (d.ret1.shift(2) < 0)
        d["ok"] = (c <= PRICE_MAX) & (c >= PRICE_MIN) & d.ma200.notna()
        out[code] = d
    return out


def simulate(F: dict, entry, exit_=None, *, tp=0.0, sl=0.0, max_days=5, disc=1.0, rank=None,
             at_close=False, tax=TAX) -> pd.DataFrame:
    """entry/exit_: f(df) -> bool Series（以當日收盤判斷）。rank: f(df) -> Series，資金模擬時排序用（小者優先）。"""
    rows = []
    for code, d in F.items():
        en = (entry(d) & d.ok).values
        ex = exit_(d).values if exit_ is not None else np.zeros(len(d), bool)
        rk = rank(d).values if rank is not None else np.zeros(len(d))
        o, h, l, c = d.Open.values, d.High.values, d.Low.values, d.Close.values
        idx, n, i = d.index, len(d), 0
        while i < n - 1:
            if not en[i]:
                i += 1
                continue
            if at_close:                             # 收盤前進場（以收盤價近似），隔日起才檢查出場
                j, px, k = i, c[i], i + 1
            else:
                j = i + 1
                if o[j] >= c[i] * (1 + LIMIT):      # 開盤漲停，買不到
                    i += 1
                    continue
                px, k = o[j], j
            out, why = None, ""
            while k < n:
                stop, target = px * (1 - sl), px * (1 + tp)
                if k > j and (ex[k - 1] or k - j >= max_days):
                    locked = o[k] <= c[k - 1] * (1 - LIMIT) and h[k] == l[k]
                    if not locked or k == n - 1:
                        out, why = o[k], "訊號" if ex[k - 1] else "時間"
                        break
                elif sl and l[k] <= stop:
                    out, why = min(stop, o[k]) if k > j else stop, "停損"
                    break
                elif tp and h[k] >= target:
                    out, why = max(target, o[k]) if k > j else target, "停利"
                    break
                k += 1
            if out is None:
                break
            cost = FEE * 2 * disc + (min(TAX_DAY, tax) if k == j else tax)
            rows.append((code, idx[j], idx[k], px, out / px - 1 - cost, why, k - j, rk[i]))
            i = k if why in ("訊號", "時間") else k + 1    # 開盤出場當天不再進場判斷（訊號在前一日收盤）
    t = pd.DataFrame(rows, columns=["code", "d_in", "d_out", "px", "ret", "why", "days", "rank"])
    t["year"] = t.d_in.dt.year
    return t


def stat(t: pd.DataFrame) -> dict:
    if t.empty:
        return dict(n=0)
    d = t.ret.groupby(t.d_in).mean()
    tt = d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 2 and d.std() > 0 else np.nan
    w, l = t.ret[t.ret > 0], t.ret[t.ret <= 0]
    return dict(n=len(t), win=100 * (t.ret > 0).mean(), bp=1e4 * t.ret.mean(), med=1e4 * t.ret.median(),
                aw=100 * w.mean() if len(w) else 0, al=100 * l.mean() if len(l) else 0,
                pf=w.sum() / -l.sum() if l.sum() < 0 else np.inf, t=tt, days=t.days.mean(),
                worst=100 * t.ret.min())


def line(name: str, t: pd.DataFrame) -> str:
    s = stat(t)
    if not s["n"]:
        return f"{name:<44} 無交易"
    return (f"{name:<44} n={s['n']:>6} 勝率{s['win']:5.1f}% 每筆{s['bp']:+7.1f}bp 中位{s['med']:+6.0f} "
            f"均賺{s['aw']:+5.2f}% 均賠{s['al']:+6.2f}% PF{s['pf']:5.2f} t={s['t']:+5.2f} "
            f"持有{s['days']:.1f}日 最差{s['worst']:+.0f}%")


def by_year(t: pd.DataFrame) -> str:
    g = t.groupby("year").ret
    return "  ".join(f"{y}:{m*1e4:+.0f}bp/{100*w:.0f}%({n})"
                     for y, m, w, n in zip(g.mean().index, g.mean(), g.apply(lambda r: (r > 0).mean()), g.size()))


def portfolio(t: pd.DataFrame, all_days: pd.DatetimeIndex, capital=1_000_000, maxpos=3) -> dict:
    """同時最多 maxpos 檔、每檔 capital/maxpos、整張；同日多檔訊號取 rank 最小者。損益記在出場日。"""
    alloc = capital / maxpos
    t = t.sort_values(["d_in", "rank"])
    open_until: list[pd.Timestamp] = []
    pnl = pd.Series(0.0, index=all_days)
    taken = 0
    for r in t.itertuples():
        open_until = [x for x in open_until if x >= r.d_in]     # 出場日當天資金尚未回來
        lots = np.floor(alloc / (r.px * 1000))
        if len(open_until) >= maxpos or lots < 1:
            continue
        open_until.append(r.d_out)
        pnl[r.d_out] += lots * 1000 * r.px * r.ret
        taken += 1
    eq = pnl.cumsum()
    yrs = (all_days[-1] - all_days[0]).days / 365.25
    y = pnl.groupby(pnl.index.year).sum()
    return dict(taken=taken, total=eq.iloc[-1], per_year=eq.iloc[-1] / yrs, ann_pct=100 * eq.iloc[-1] / yrs / capital,
                dd=(eq - eq.cummax()).min(), per_month=taken / yrs / 12,
                years="  ".join(f"{k}:{v/1e4:+.1f}萬" for k, v in y.items()))


def main(cache: Path, name: str, disc: float):
    F = features(pickle.load(open(cache / name, "rb")))
    all_days = pd.DatetimeIndex(sorted(set().union(*[set(d.index[200:]) for d in F.values()])))
    print(f"{len(F)} 檔  {all_days[0].date()}~{all_days[-1].date()}  手續費折數 {disc}  "
          f"隔日沖來回成本 {(FEE*2*disc+TAX)*1e4:.1f}bp  價格 {PRICE_MIN}~{PRICE_MAX}")
    up200 = lambda d: d.Close > d.ma200
    above5 = lambda d: d.Close > d.ma5
    S = lambda *a, **k: simulate(F, *a, disc=disc, **k)
    R2 = lambda d: d.rsi2

    print("\n== 1. 各類小波段規則（隔日開盤進場）==")
    cands = {
        "A1 多頭回檔 RSI2<10 → 站回5日線 (≤5日)": S(lambda d: up200(d) & (d.rsi2 < 10), above5, rank=R2),
        "A2 同上 RSI2<5": S(lambda d: up200(d) & (d.rsi2 < 5), above5, rank=R2),
        "A3 同上 趨勢改 60 日線": S(lambda d: (d.Close > d.ma60) & (d.rsi2 < 10), above5, rank=R2),
        "A4 A1 + 停利 3%": S(lambda d: up200(d) & (d.rsi2 < 10), above5, tp=0.03, rank=R2),
        "A5 A1 + 停利 5%": S(lambda d: up200(d) & (d.rsi2 < 10), above5, tp=0.05, rank=R2),
        "A6 A1 + 停損 5%": S(lambda d: up200(d) & (d.rsi2 < 10), above5, sl=0.05, rank=R2),
        "A7 A1 最長 10 日": S(lambda d: up200(d) & (d.rsi2 < 10), above5, max_days=10, rank=R2),
        "A8 無趨勢濾網 RSI2<10": S(lambda d: d.rsi2 < 10, above5, rank=R2),
        "B1 多頭連跌 3 日 → 站回5日線": S(lambda d: up200(d) & d.dn3, above5, rank=R2),
        "B2 多頭創 5 日新低 → 站回5日線": S(lambda d: up200(d) & (d.Close < d.lo5), above5, rank=R2),
        "C1 5日跌≥8% → 停利3% (≤3日)": S(lambda d: d.ret5 <= -0.08, tp=0.03, max_days=3, rank=lambda d: d.ret5),
        "C2 5日跌≥8% 且多頭 → 停利3% (≤3日)": S(lambda d: up200(d) & (d.ret5 <= -0.08), tp=0.03, max_days=3,
                                          rank=lambda d: d.ret5),
        "C3 5日跌≥12% → 停利5% (≤5日)": S(lambda d: d.ret5 <= -0.12, tp=0.05, max_days=5, rank=lambda d: d.ret5),
        "D1 帶量突破 20 日高 → 抱 3 日": S(lambda d: (d.Close > d.hi20) & (d.volx >= 2) & (d.Close > d.ma60),
                                    max_days=3, rank=lambda d: -d.volx),
        "D2 同上 停利5%/停損3% (≤5日)": S(lambda d: (d.Close > d.hi20) & (d.volx >= 2) & (d.Close > d.ma60),
                                     tp=0.05, sl=0.03, max_days=5, rank=lambda d: -d.volx),
        "D3 長紅(≥5%)帶量 → 抱 1 日": S(lambda d: (d.ret1 >= 0.05) & (d.volx >= 2), max_days=1,
                                   rank=lambda d: -d.ret1),
        "E1 強勢股回測 20 日線收回 → 5%/3% (≤5日)": S(
            lambda d: (d.Close > d.ma60) & (d.ma60 > d.ma200) & (d.Low <= d.ma20) & (d.Close > d.ma20),
            tp=0.05, sl=0.03, max_days=5, rank=R2),
    }
    for k, t in cands.items():
        print(line(k, t))

    print("\n== 2. 逐年（每筆bp/勝率(筆數)）==")
    for k in ("A1", "A4", "A5", "B1", "C1", "D1"):
        key = next(x for x in cands if x.startswith(k))
        print(f"{k}: {by_year(cands[key])}")

    print("\n== 3. A 類參數鄰近值（穩健性）==")
    for th in (5, 10, 15, 20):
        for tr in (60, 200):
            for xm in (5, 10):
                t = S(lambda d, th=th, tr=tr: (d.Close > d[f"ma{tr}"]) & (d.rsi2 < th),
                      lambda d, xm=xm: d.Close > d[f"ma{xm}"], rank=R2)
                print(line(f"RSI2<{th} 趨勢{tr}日 出場>{xm}日線", t))

    print("\n== 4. 訊號擁擠度：同日觸發檔數（A1）==")
    a1 = cands[next(x for x in cands if x.startswith("A1"))]
    cnt = a1.groupby("d_in").code.transform("size")
    for lo, hi in ((1, 5), (6, 20), (21, 60), (61, 999)):
        print(line(f"同日 {lo}~{hi} 檔", a1[(cnt >= lo) & (cnt <= hi)]))
    top = a1.groupby("d_in").ret.sum().sort_values(ascending=False)
    print(f"獲利最高 10 天佔總獲利 {100*top.head(10).sum()/a1.ret.sum():.0f}%（共 {len(top)} 個進場日）")

    print("\n== 5. 資金模擬（100 萬、整張、同日取 RSI2 最低者）==")
    for k in ("A1", "A4", "A5", "A7", "B1", "C1", "C3"):
        key = next(x for x in cands if x.startswith(k))
        for mp in (3, 5):
            p = portfolio(cands[key], all_days, maxpos=mp)
            print(f"{k} 最多{mp}檔: 成交{p['taken']:>5}筆(每月{p['per_month']:.1f}) 年均{p['per_year']/1e4:+6.1f}萬 "
                  f"({p['ann_pct']:+.1f}%/年) 最大回撤{p['dd']/1e4:.1f}萬  {p['years']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "daily6y_400.pkl",
         float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
