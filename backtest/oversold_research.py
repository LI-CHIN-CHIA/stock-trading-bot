"""超跌股開盤做多（鏡像 fade）的驗證：跌停鎖死情境、參數格點、逐年前推、集中度、資金模擬。

用法: venv/bin/python backtest/oversold_research.py <cache_dir> [daily6y.pkl] [fee_discount] [core_codes_pickle]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fade_research import FEE, TAX, by_year, load, stat  # noqa: E402


def mask(p, r5=0.08, g=0.03, pmax=330):
    return (p.ret5 <= -r5) & (p.gap <= -g) & (p.gap >= -0.09) & (p.open <= pmax)


def main(cache: Path, name: str, disc: float):
    pd.set_option("display.width", 250)
    p = load(cache, name)
    cost = FEE * 2 * disc + TAX
    # 收在跌停（≤-9.5%）視為賣不掉 → 隔日開盤賣
    p["lockd"] = p.close / p.prev_close - 1 <= -0.095
    p["l_sq"] = np.where(p.lockd & p.next_open.notna(), p.next_open / p.open - 1, p.l_ret)
    all_days = pd.DatetimeIndex(sorted(p.date.unique()))
    print(f"{p.code.nunique()} 檔 {len(p)} 股日 成本 {cost*1e4:.1f}bp")

    b = p[mask(p)]
    print("\n== 1. 基準（5日跌≥8%、向下跳空 3~9%、價≤330）==")
    print(" 收盤賣       ", stat(b.l_ret, b.date, cost), "|", by_year(b, "l_ret", cost))
    print(" 跌停隔日賣   ", stat(b.l_sq, b.date, cost), "|", by_year(b, "l_sq", cost))
    print(f" 收跌停比例 {100*b.lockd.mean():.1f}%  其隔日賣平均 {b[b.lockd].l_sq.mean()*1e4:+.0f}bp")
    d = (b.l_sq - cost).groupby(b.date).agg(["sum", "size"])
    tot = d["sum"].sum()
    top = d.sort_values("sum", ascending=False).head(10)
    print(f" 交易日數 {len(d)}（佔 {100*len(d)/len(all_days):.0f}%）  前 10 大獲利日佔總獲利 {100*top['sum'].sum()/tot:.0f}%")
    print("  " + "  ".join(f"{i.date()}:{r['sum']*100:+.0f}%/{int(r['size'])}筆" for i, r in top.head(6).iterrows()))
    dm = (b.l_sq - cost).groupby(b.date).mean()
    print(f" 去掉最好 10 天後 每日等權平均 {dm.drop(top.index).mean()*1e4:+.0f}bp（全部 {dm.mean()*1e4:+.0f}bp） 勝日 {100*(dm>0).mean():.0f}%")
    cap = b[b.groupby("date").code.transform("size") <= 15]
    print(" 排除單日 >15 檔同時觸發的崩盤日:", stat(cap.l_sq, cap.date, cost), "|", by_year(cap, "l_sq", cost))

    print("\n== 2. 參數格點（跌停隔日賣；淨bp / 筆數 / t）==")
    rows = []
    for r5 in (0.0, 0.05, 0.08, 0.10, 0.15):
        row = [f"ret5≤-{r5:.0%}"]
        for g in (0.01, 0.02, 0.03, 0.04, 0.05, 0.07):
            s = p[mask(p, r5, g)]
            st = stat(s.l_sq, s.date, cost)
            row.append(f"{st['bp']:+.0f}/{st['n']}/{st['t']}")
        rows.append(row)
    print(pd.DataFrame(rows, columns=["", "gap≤-1%", "-2%", "-3%", "-4%", "-5%", "-7%"]).to_string(index=False))

    print("\n== 3. 逐年前推 ==")
    grid = [(a, g) for a in (0.0, 0.05, 0.08, 0.10, 0.15) for g in (0.01, 0.02, 0.03, 0.04, 0.05)]
    oos = []
    for y in sorted(p.year.unique())[2:]:
        past = p[p.year < y]
        def score(gp):
            s = past[mask(past, *gp)]
            t = stat(s.l_sq, s.date, cost)["t"] if len(s) >= 60 else np.nan
            return -9 if pd.isna(t) else t
        best = max(grid, key=score)
        cur = p[p.year == y]; cur = cur[mask(cur, *best)]
        oos.append(cur)
        print(f" {y}: ret5≤-{best[0]:.0%} gap≤-{best[1]:.0%} →", stat(cur.l_sq, cur.date, cost))
    o = pd.concat(oos)
    print(" 樣本外合計:", stat(o.l_sq, o.date, cost))

    print("\n== 4. 條件切分（基準；淨bp/筆數/t）==")
    cuts = {
        "大盤跳空": pd.cut(b.mkt_gap, [-1, -0.02, -0.01, -0.003, 0.003, 1], labels=["<-2%", "-2~-1", "-1~-0.3", "平", "漲"]),
        "相對大盤跳空": pd.cut(b.rel_gap, [-1, -0.05, -0.03, -0.015, 1], labels=["<-5", "-5~-3", "-3~-1.5", ">-1.5"]),
        "股價": pd.cut(b.open, [0, 30, 60, 120, 330], labels=["<30", "30~60", "60~120", "120~330"]),
        "20日均成交值": pd.cut(b.turnover, [0, 1e8, 5e8, 2e9, 1e15], labels=["<1億", "1~5億", "5~20億", ">20億"]),
        "前日漲跌": pd.cut(b.prev_ret, [-1, -0.095, -0.05, 0, 1], labels=["跌停", "-9.5~-5", "-5~0", "漲"]),
        "20日漲跌": pd.cut(b.ret20, [-1, -0.2, -0.1, 0, 9], labels=["<-20", "-20~-10", "-10~0", ">0"]),
    }
    for k, c in cuts.items():
        print(f" {k:<8}", " | ".join(f"{lv}: {(st:=stat(s.l_sq, s.date, cost))['bp']:+.0f}/{st['n']}/{st['t']}"
                                     for lv, s in b.groupby(c, observed=True)))

    print("\n== 5. 資金模擬（100 萬、每檔 33%、最多 3 檔、整張；同日多檔時取 5 日跌最深者）==")
    for label, m in (("基準", mask(p)), ("gap≤-2%", mask(p, 0.08, 0.02)), ("gap≤-4%", mask(p, 0.08, 0.04)),
                     ("排除大盤跳空<-2%", mask(p) & (p.mkt_gap > -0.02))):
        s = p[m].copy()
        s["lots"] = np.floor(330_000 / (s.open * 1000))
        s = s[s.lots >= 1]
        s["rk"] = s.groupby("date").ret5.rank(method="first")
        s = s[s.rk <= 3]
        s["pnl"] = s.lots * 1000 * s.open * (s.l_sq - cost)
        d = s.groupby("date").pnl.sum().reindex(all_days, fill_value=0.0)
        eq = d.cumsum(); dd = (eq - eq.cummax()).min(); yr = d.groupby(d.index.year).sum()
        print(f" [{label}] 筆數 {len(s)} 有交易日 {100*(d!=0).mean():.0f}% 日均 {d.mean():+.0f} 元 ({d.mean()/1e4:+.3f}%/日) "
              f"勝日 {100*(d[d!=0]>0).mean():.0f}%  最大回撤 {dd/1e4:.1f} 萬 最差日 {d.min()/1e4:.1f} 萬")
        print("   年損益(萬): " + "  ".join(f"{y}:{v/1e4:+.1f}" for y, v in yr.items()))

    if len(sys.argv) > 4:
        core = set(pickle.load(open(sys.argv[4], "rb")))
        for lab, s in (("原 50 檔", b[b.code.isin(core)]), ("其餘", b[~b.code.isin(core)])):
            print(f"\n [{lab}]", stat(s.l_sq, s.date, cost), "|", by_year(s, "l_sq", cost))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "daily6y.pkl",
         float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
