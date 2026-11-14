"""跳空反轉：每日放空跳空最大 N 檔 / 做多跳空最小 N 檔（開盤進、收盤出），含鎖漲跌停情境。

用法: venv/bin/python backtest/gap_research.py <cache_dir> [daily6y.pkl] [fee_discount]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fade_research import FEE, TAX, by_year, load, stat  # noqa: E402


def main(cache: Path, name: str, disc: float):
    p = load(cache, name)
    cost = FEE * 2 * disc + TAX
    p["lockd"] = p.close / p.prev_close - 1 <= -0.095
    p["l_sq"] = np.where(p.lockd & p.next_open.notna(), p.next_open / p.open - 1, p.l_ret)
    lvl = p.prev_close * 1.08
    p["s_stop8"] = np.where((p.high >= lvl) & (lvl > p.open), 1 - lvl / p.open, p.s_ret_sq)
    all_days = pd.DatetimeIndex(sorted(p.date.unique()))
    u = p[(p.open <= 330) & (p.gap.abs() <= 0.09) & (p.turnover >= 1e8)]
    print(f"{p.code.nunique()} 檔  可交易股日 {len(u)}  成本 {cost*1e4:.1f}bp（排除開盤即漲跌停、價>330、成交值<1億）")

    for side, asc, cols in (("空 跳空最大", False, [("收盤補(樂觀)", "s_ret"), ("鎖漲停隔日補", "s_ret_sq"), ("昨收+8%停損", "s_stop8")]),
                            ("多 跳空最小", True, [("收盤賣(樂觀)", "l_ret"), ("鎖跌停隔日賣", "l_sq")])):
        for n in (1, 3, 5, 10):
            for gmin in (0.0, 0.02, 0.04):
                x = u[(u.gap >= gmin) if not asc else (u.gap <= -gmin)].copy()
                x["rk"] = x.groupby("date").gap.rank(ascending=asc, method="first")
                s = x[x.rk <= n]
                out = "  ".join(f"{lab}: {(st:=stat(s[c], s.date, cost))['bp']:+.0f}bp t={st['t']}" for lab, c in cols)
                print(f" {side}{n}檔 |gap|≥{gmin:.0%} n={len(s):>5} 交易日 {100*s.date.nunique()/len(all_days):.0f}% | {out}")
        x = u[(u.gap >= 0.02) if not asc else (u.gap <= -0.02)].copy()
        x["rk"] = x.groupby("date").gap.rank(ascending=asc, method="first")
        s = x[x.rk <= 3]
        for lab, c in cols:
            print(f"   逐年 3檔|gap|≥2% {lab}: {by_year(s, c, cost)}")
        c = cols[1][1]
        s = s.copy()
        s["lots"] = np.floor(330_000 / (s.open * 1000)); s = s[s.lots >= 1]
        s["pnl"] = s.lots * 1000 * s.open * (s[c] - cost)
        d = s.groupby("date").pnl.sum().reindex(all_days, fill_value=0.0)
        eq = d.cumsum(); yr = d.groupby(d.index.year).sum()
        print(f"   資金模擬(100萬,3檔,{cols[1][0]}): 日均 {d.mean():+.0f} 元 ({d.mean()/1e4:+.3f}%/日) 勝日 {100*(d[d!=0]>0).mean():.0f}% "
              f"最大回撤 {(eq-eq.cummax()).min()/1e4:.1f} 萬 最差日 {d.min()/1e4:.1f} 萬 | " + " ".join(f"{y}:{v/1e4:+.0f}" for y, v in yr.items()))
        print()


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "daily6y.pkl",
         float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
