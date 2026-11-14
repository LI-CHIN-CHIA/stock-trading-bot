"""以「勝率高且期望值為正」為目標的檢驗：超跌做多加停利、隔夜持有（收盤買→隔日開盤賣）。

用法: venv/bin/python backtest/winrate_research.py <cache_dir> [daily6y.pkl] [fee_discount]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fade_research import FEE, load  # noqa: E402


def line(r: pd.Series, dates: pd.Series, years: pd.Series) -> str:
    if len(r) < 30:
        return f"n={len(r)} 樣本不足"
    d = r.groupby(dates).mean()
    t = d.mean() / d.std() * np.sqrt(len(d))
    w, l = r[r > 0], r[r <= 0]
    y = r.groupby(years).agg(["mean", lambda x: (x > 0).mean()])
    return (f"n={len(r):>6} 勝率 {100*(r>0).mean():.0f}% 每筆 {r.mean()*1e4:+.0f}bp 賺均 {w.mean()*1e4:+.0f} 賠均 {l.mean()*1e4:+.0f} "
            f"t={t:.1f} | " + " ".join(f"{k}:{m*1e4:+.0f}/{100*wr:.0f}%" for k, (m, wr) in y.iterrows()))


def main(cache: Path, name: str, disc: float):
    p = load(cache, name)
    dt_cost = FEE * 2 * disc + 0.0015      # 當沖
    on_cost = FEE * 2 * disc + 0.003       # 隔夜（非當沖，證交稅 0.3%）
    g = p.groupby("code")
    p["next_close"] = g.close.shift(-1)
    p["day_ret"] = p.close / p.prev_close - 1
    p["pos"] = (p.close - p.low) / (p.high - p.low).replace(0, np.nan)
    p["on"] = p.next_open / p.close - 1            # 隔夜報酬
    p["lock_up"] = p.day_ret >= 0.095
    p["lock_dn"] = p.day_ret <= -0.095
    liq = (p.turnover >= 1e8) & (p.close <= 330)
    print(f"{p.code.nunique()} 檔 {len(p)} 股日  當沖成本 {dt_cost*1e4:.1f}bp  隔夜成本 {on_cost*1e4:.1f}bp")

    print("\n== A. 超跌做多（5日跌≥8%、跳空-3~-9%）加停利：盤中漲到 開盤×(1+x) 就賣，否則收盤賣（鎖跌停隔日賣）==")
    b = p[(p.ret5 <= -0.08) & (p.gap <= -0.03) & (p.gap >= -0.09) & (p.open <= 330)].copy()
    lsq = np.where((b.close / b.prev_close - 1 <= -0.095) & b.next_open.notna(), b.next_open / b.open - 1, b.l_ret)
    for x in (None, 0.01, 0.015, 0.02, 0.03, 0.05):
        r = pd.Series(lsq if x is None else np.where(b.hi_o >= x, x, lsq), index=b.index) - dt_cost
        print(f" 停利 {'無' if x is None else f'{x:.1%}':>5}: " + line(r, b.date, b.year))
    calm = b[b.groupby("date").code.transform("size") <= 15]
    lc = pd.Series(np.where(calm.hi_o >= 0.02, 0.02, np.where((calm.close / calm.prev_close - 1 <= -0.095) & calm.next_open.notna(),
                   calm.next_open / calm.open - 1, calm.l_ret)), index=calm.index) - dt_cost
    print(" 非崩盤日+停利2%: " + line(lc, calm.date, calm.year))

    print("\n== B. 隔夜持有：收盤買 → 隔日開盤賣（排除收漲停買不到者）==")
    q = p[liq & ~p.lock_up & p.on.notna()].copy()
    r = q.on - on_cost
    print(" 全部           : " + line(r, q.date, q.year))
    tests = {
        "當日漲 3~9.5%": (q.day_ret >= 0.03),
        "當日漲 5~9.5%": (q.day_ret >= 0.05),
        "當日漲 7~9.5%": (q.day_ret >= 0.07),
        "漲5%+收在最高附近(pos≥0.9)": (q.day_ret >= 0.05) & (q.pos >= 0.9),
        "漲7%+收在最高附近": (q.day_ret >= 0.07) & (q.pos >= 0.9),
        "當日跌 ≥5%": (q.day_ret <= -0.05),
        "當日跌 ≥7%(未跌停)": (q.day_ret <= -0.07) & ~q.lock_dn,
        "5日跌≥15%+當日收紅": (q.close / q.groupby('code').close.shift(5) - 1 <= -0.15) & (q.day_ret > 0),
        "大盤當日跌≥2%": (q.mkt_oc + q.mkt_gap <= -0.02),
        "大盤當日漲≥1.5%": (q.mkt_oc + q.mkt_gap >= 0.015),
    }
    for k, m in tests.items():
        s = q[m]
        print(f" {k:<22}: " + line(s.on - on_cost, s.date, s.year))

    print("\n== C. 隔夜強勢股：每日取『當日漲幅最大（未漲停）』前 N 檔，收盤買隔日開盤賣 ==")
    for lo in (0.03, 0.05):
        x = q[(q.day_ret >= lo) & (q.pos >= 0.8)].copy()
        x["rk"] = x.groupby("date").day_ret.rank(ascending=False, method="first")
        for n in (1, 3):
            s = x[x.rk <= n]
            print(f" 漲≥{lo:.0%} 取{n}檔: " + line(s.on - on_cost, s.date, s.year))
    s = x[x.rk <= 3].copy()
    s["lots"] = np.floor(330_000 / (s.close * 1000)); s = s[s.lots >= 1]
    s["pnl"] = s.lots * 1000 * s.close * (s.on - on_cost)
    days = pd.DatetimeIndex(sorted(p.date.unique()))
    d = s.groupby("date").pnl.sum().reindex(days, fill_value=0.0); eq = d.cumsum()
    print(f" 資金模擬(100萬,3檔,漲≥5%): 日均 {d.mean():+.0f} 元 有交易日 {100*(d!=0).mean():.0f}% 勝日 {100*(d[d!=0]>0).mean():.0f}% "
          f"最大回撤 {(eq-eq.cummax()).min()/1e4:.1f} 萬 最差日 {d.min()/1e4:.1f} 萬 | "
          + " ".join(f"{y}:{v/1e4:+.0f}" for y, v in d.groupby(d.index.year).sum().items()))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "daily6y.pkl",
         float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
