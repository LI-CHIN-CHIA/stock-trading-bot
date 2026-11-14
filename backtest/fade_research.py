"""過熱股放空（fade）的延伸回測：參數穩健性、逐年前推、條件切分、鎖漲停情境、資金模擬。

用法: venv/bin/python backtest/fade_research.py <cache_dir> [daily6y.pkl] [fee_discount]
日線資料由 daily_edge.py fetch（或同格式的 pickle）提供；進場=開盤、出場=收盤。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from daily_edge import panel  # noqa: E402

TAX = 0.0015            # 當沖證交稅
FEE = 0.001425
RET = "s_stop8"       # 第 2 節起使用的可執行報酬：昨收+8% 停損、鎖漲停隔日補
LIMIT = 0.095           # 收盤漲幅 ≥9.5% 視為鎖漲停（空單可能無法回補）


def load(cache: Path, name: str) -> pd.DataFrame:
    p = panel(cache, name).sort_values(["code", "date"]).reset_index(drop=True)
    p["prev_close"] = p.open / (1 + p.gap)
    p["next_open"] = p.groupby("code").open.shift(-1)
    p["dow"] = p.date.dt.dayofweek
    p["locked"] = p.close / p.prev_close - 1 >= LIMIT
    # 放空報酬（開盤空、收盤補）；鎖漲停者改用隔日開盤回補（隔日沖，另加一次來回成本的差額不計）
    p["s_ret"] = 1 - p.close / p.open
    p["s_ret_sq"] = np.where(p.locked & p.next_open.notna(), 1 - p.next_open / p.open, p.s_ret)
    p["l_ret"] = p.close / p.open - 1
    return p


def stat(r: pd.Series, dates: pd.Series, cost: float) -> dict:
    if len(r) == 0:
        return dict(n=0, bp=np.nan, win=np.nan, t=np.nan)
    net = r - cost
    d = net.groupby(dates).mean()
    t = d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 2 and d.std() > 0 else np.nan
    return dict(n=len(r), bp=round(net.mean() * 1e4, 1), win=round(100 * (net > 0).mean(), 1), t=round(t, 2))


def by_year(sel: pd.DataFrame, col: str, cost: float) -> str:
    g = (sel[col] - cost).groupby(sel.year)
    return "  ".join(f"{y}:{m*1e4:+.0f}({n})" for y, m, n in zip(g.mean().index, g.mean(), g.size()))


def fade_mask(p, ret5=0.10, gmin=0.015, gmax=0.09, pmax=330):
    return (p.ret5 >= ret5) & (p.gap >= gmin) & (p.gap <= gmax) & (p.open <= pmax)


def portfolio(sel: pd.DataFrame, col: str, cost: float, capital=1_000_000, pct=0.33, maxpos=3, all_days=None):
    """每日最多 maxpos 檔（依 ret5 由高到低），每檔 capital*pct，只能整張。回傳每日損益（元）。"""
    alloc = capital * pct
    s = sel.copy()
    s["lots"] = np.floor(alloc / (s.open * 1000))
    s = s[s.lots >= 1]
    s["rk"] = s.groupby("date").ret5.rank(ascending=False, method="first")
    s = s[s.rk <= maxpos]
    s["pnl"] = s.lots * 1000 * s.open * (s[col] - cost)
    d = s.groupby("date").pnl.sum()
    if all_days is not None:
        d = d.reindex(all_days, fill_value=0.0)
    return d, s


def main(cache: Path, name: str, disc: float):
    pd.set_option("display.width", 250)
    p = load(cache, name)
    cost = FEE * 2 * disc + TAX
    all_days = pd.DatetimeIndex(sorted(p.date.unique()))
    print(f"{p.code.nunique()} 檔  {p.date.min().date()}~{p.date.max().date()}  {len(p)} 股日  "
          f"手續費折數 {disc}  來回成本 {cost*1e4:.1f}bp")
    print(f"全體 open→close {p.l_ret.mean()*1e4:+.1f}bp  隔夜(gap) {p.gap.mean()*1e4:+.1f}bp")

    base = p[fade_mask(p)]
    print("\n== 1. 基準 fade（5日漲≥10%、跳空1.5~9%、價≤330）==")
    print(" 收盤回補      ", stat(base.s_ret, base.date, cost), "|", by_year(base, "s_ret", cost))
    print(" 鎖漲停隔日補  ", stat(base.s_ret_sq, base.date, cost), "|", by_year(base, "s_ret_sq", cost))
    print(f" 鎖漲停比例 {100*base.locked.mean():.1f}%  鎖漲停者隔日回補平均 {base[base.locked].s_ret_sq.mean()*1e4:+.0f}bp")

    print("\n== 1b. 接近漲停前先回補（盤中觸及 昨收×(1+x) 即以該價回補；沒觸及者收盤補、仍鎖漲停者隔日補）==")
    for x in (0.05, 0.06, 0.07, 0.08, 0.09):
        lvl = base.prev_close * (1 + x)
        hit = (base.high >= lvl) & (lvl > base.open)
        r = np.where(hit, 1 - lvl / base.open, base.s_ret_sq)
        r = pd.Series(r, index=base.index)
        print(f" x={x:.0%}: 觸發 {100*hit.mean():.0f}%  ", stat(r, base.date, cost), "|",
              by_year(base.assign(_r=r), "_r", cost))
        base[f"s_stop{int(x*100)}"] = r
    p = p.copy()
    for x in (7, 8):
        lvl = p.prev_close * (1 + x / 100)
        hit = (p.high >= lvl) & (lvl > p.open)
        p[f"s_stop{x}"] = np.where(hit, 1 - lvl / p.open, p.s_ret_sq)

    print("\n== 2. 參數格點（淨bp / 筆數 / t；看是否整片為正而非單點）==")
    rows = []
    for r5 in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20):
        row = [f"ret5≥{r5:.0%}"]
        for g in (0.0, 0.01, 0.015, 0.02, 0.03, 0.05):
            s = p[fade_mask(p, r5, g)]
            st = stat(s[RET], s.date, cost)
            row.append(f"{st['bp']:+.0f}/{st['n']}/{st['t']}")
        rows.append(row)
    print(pd.DataFrame(rows, columns=["", "gap≥0", "≥1%", "≥1.5%", "≥2%", "≥3%", "≥5%"]).to_string(index=False))

    print("\n== 3. 逐年前推（用過去年份挑 t 最高的參數，套到下一年；真正的樣本外）==")
    grid = [(a, b) for a in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20) for b in (0.0, 0.01, 0.015, 0.02, 0.03)]
    years = sorted(p.year.unique())
    oos = []
    for y in years[2:]:
        past = p[p.year < y]
        best = max(grid, key=lambda g: (lambda s: (stat(s[RET], s.date, cost)["t"] if len(s) >= 60 else -9) or -9)(past[fade_mask(past, *g)]))
        cur = p[(p.year == y)]
        cur = cur[fade_mask(cur, *best)]
        oos.append(cur)
        st = stat(cur[RET], cur.date, cost)
        print(f" {y}: 參數 ret5≥{best[0]:.0%} gap≥{best[1]:.1%}  → {st}")
    o = pd.concat(oos)
    print(" 樣本外合計:", stat(o[RET], o.date, cost))

    print("\n== 4. 鏡像：超跌股開盤做多（5日跌≥X、向下跳空≥Y）==")
    for r5, g in ((0.10, 0.015), (0.10, 0.0), (0.15, 0.015), (0.08, 0.03)):
        s = p[(p.ret5 <= -r5) & (p.gap <= -g) & (p.gap >= -0.09) & (p.open <= 330)]
        print(f" ret5≤-{r5:.0%} gap≤-{g:.1%}:", stat(s.l_ret, s.date, cost), "|", by_year(s, "l_ret", cost))

    print("\n== 5. 基準 fade 的條件切分（淨bp / 筆數 / t）==")
    b = p[fade_mask(p)].copy()
    cuts = {
        "大盤跳空": pd.cut(b.mkt_gap, [-1, -0.005, 0.005, 1], labels=["跌>0.5%", "平", "漲>0.5%"]),
        "大盤5日": pd.cut(b.mkt_ret5, [-1, -0.02, 0.02, 1], labels=["弱", "中", "強"]),
        "前日量比": pd.cut(b.prev_volx, [0, 1, 2, 4, 1e9], labels=["<1", "1~2", "2~4", ">4"]),
        "前日收盤位置": pd.cut(b.prev_pos, [-0.01, 0.33, 0.66, 1.01], labels=["低", "中", "高"]),
        "前日漲跌": pd.cut(b.prev_ret, [-1, 0, 0.05, 0.095, 1], labels=["跌", "0~5%", "5~9.5%", "漲停"]),
        "跳空幅度": pd.cut(b.gap, [0, 0.025, 0.04, 0.06, 0.1], labels=["1.5~2.5", "2.5~4", "4~6", "6~9"]),
        "5日漲幅": pd.cut(b.ret5, [0, 0.15, 0.2, 0.3, 9], labels=["10~15", "15~20", "20~30", ">30"]),
        "股價": pd.cut(b.open, [0, 30, 60, 120, 330], labels=["<30", "30~60", "60~120", "120~330"]),
        "星期": b.dow.map({0: "一", 1: "二", 2: "三", 3: "四", 4: "五"}),
        "20日均成交值": pd.cut(b.turnover, [0, 1e8, 5e8, 2e9, 1e15], labels=["<1億", "1~5億", "5~20億", ">20億"]),
    }
    for k, c in cuts.items():
        parts = []
        for lv, s in b.groupby(c, observed=True):
            st = stat(s[RET], s.date, cost)
            parts.append(f"{lv}: {st['bp']:+.0f}/{st['n']}/{st['t']}")
        print(f" {k:<8}", " | ".join(parts))

    print("\n== 6. 資金模擬（100 萬、每檔 33%、最多 3 檔、整張、漲停前停損）==")
    for label, m in (("基準", fade_mask(p)), ("放寬 ret5≥8% gap≥1%", fade_mask(p, 0.08, 0.01)),
                     ("嚴格 ret5≥15% gap≥2%", fade_mask(p, 0.15, 0.02))):
        d, s = portfolio(p[m], RET, cost, all_days=all_days)
        eq = d.cumsum()
        dd = (eq - eq.cummax()).min()
        yr = d.groupby(d.index.year).sum()
        print(f" [{label}] 筆數 {len(s)}  有交易日 {100*(d!=0).mean():.0f}%  日均 {d.mean():+.0f} 元 "
              f"({d.mean()/1e6*100:+.3f}%/日)  有交易日均 {d[d!=0].mean():+.0f}  勝日 {100*(d[d!=0]>0).mean():.0f}%")
        print(f"   年損益(萬): " + "  ".join(f"{y}:{v/1e4:+.1f}" for y, v in yr.items())
              + f"   最大回撤 {dd/1e4:.1f} 萬  最差單日 {d.min()/1e4:.1f} 萬  最佳單日 {d.max()/1e4:+.1f} 萬")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "daily6y.pkl",
         float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
