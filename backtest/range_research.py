"""台股個股「日內振幅」的規律摸底：振幅有多大、能否事先預期、方向是否可預測、扣成本後抓得到多少。

用法: venv/bin/python backtest/range_research.py <cache_dir> [daily6y_400.pkl]
定義（皆以昨收為分母）：振幅 = (高−低)/昨收；跳空 = 開/昨收−1；日內 = 收/開−1。
當沖來回成本 43.5bp（手續費 0.1425%×2 + 當沖稅 0.15%）。
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
COST = 0.001425 * 2 + 0.0015


def build(raw: dict) -> pd.DataFrame:
    rows = []
    for code, d in raw.items():
        d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
        d = d[(d.Volume > 0) & (d.Open > 0)]
        if len(d) < 260:
            continue
        x = pd.DataFrame(index=pd.DatetimeIndex(d.index).tz_localize(None).normalize())
        o, h, l, c, v = (d[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
        pc = np.r_[np.nan, c[:-1]]
        x["code"], x["o"], x["h"], x["l"], x["c"], x["pc"] = code, o, h, l, c, pc
        x["rng"], x["gap"], x["oc"], x["cc"] = (h - l) / pc, o / pc - 1, c / o - 1, c / pc - 1
        x["up"], x["dn"] = h / o - 1, 1 - l / o                    # 開盤後最大上漲 / 最大下跌
        s = pd.Series(x.rng.values)
        x["atr10"] = s.rolling(10).mean().shift(1).values           # 前 10 日平均振幅（事前可知）
        x["rng1"] = s.shift(1).values
        x["cc1"] = pd.Series(x.cc.values).shift(1).values
        x["pos1"] = pd.Series(((c - l) / np.where(h > l, h - l, np.nan))).shift(1).values
        x["ret5"] = pd.Series(c).pct_change(5).shift(1).values
        x["tv60"] = pd.Series(c * v).rolling(60).mean().shift(1).values
        x["volx"] = (pd.Series(v) / pd.Series(v).rolling(20).mean().shift(1)).shift(1).values
        rows.append(x.iloc[60:])
    p = pd.concat(rows).rename_axis("date").reset_index()
    p["year"] = p.date.dt.year
    p["mgap"] = p.groupby("date").gap.transform("mean")           # 當日全體平均跳空（大盤開盤氣氛）
    p["lim_up"], p["lim_dn"] = p.cc >= 0.095, p.cc <= -0.095
    return p.dropna(subset=["atr10", "ret5", "tv60"])


def tstat(r: pd.Series, dates: pd.Series) -> float:
    d = r.groupby(dates).mean()
    return d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 2 and d.std() > 0 else np.nan


def table(p: pd.DataFrame, col: str, bins, labels=None, title=""):
    print(f"\n-- {title or col}")
    g = pd.cut(p[col], bins, labels=labels)
    for k, s in p.groupby(g, observed=True):
        print(f"  {str(k):<16} n={len(s):>7} 振幅{100*s.rng.mean():5.2f}%  日內{1e4*s.oc.mean():+6.1f}bp (t={tstat(s.oc, s.date):+5.1f}) "
              f"上漲日{100*(s.oc>0).mean():4.1f}%  多扣成本{1e4*(s.oc.mean()-COST):+6.1f}  空扣成本{1e4*(-s.oc.mean()-COST):+6.1f}  "
              f"收漲停{100*s.lim_up.mean():4.1f}% 收跌停{100*s.lim_dn.mean():4.1f}%")


def main(cache: Path, name: str):
    p = build(pickle.load(open(cache / name, "rb")))
    print(f"{p.code.nunique()} 檔  {p.date.min().date()}~{p.date.max().date()}  {len(p)} 股日")

    print("\n== 1. 振幅到底多大（逐年；全部個股日）==")
    for y, s in p.groupby("year"):
        print(f"  {y}: 平均振幅{100*s.rng.mean():.2f}% 中位{100*s.rng.median():.2f}%  ≥3%佔{100*(s.rng>=.03).mean():.0f}% ≥5%佔{100*(s.rng>=.05).mean():.0f}% "
              f"≥7%佔{100*(s.rng>=.07).mean():.0f}% | |日內|平均{100*s.oc.abs().mean():.2f}% |跳空|{100*s.gap.abs().mean():.2f}% |收盤漲跌|{100*s.cc.abs().mean():.2f}%")
    last = p[p.date >= p.date.max() - pd.Timedelta(days=90)]
    per = last.groupby("code").rng.mean().sort_values(ascending=False)
    print(f"  近 90 天：每檔平均振幅 中位 {100*per.median():.2f}%；≥5% 的有 {int((per>=.05).sum())} 檔、≥4% 的 {int((per>=.04).sum())} 檔、"
          f"≥3% 的 {int((per>=.03).sum())} 檔（共 {len(per)} 檔）")
    pr = last.groupby("code").c.last()
    for lo, hi in ((0, 50), (50, 150), (150, 330), (330, 1e9)):
        m = per[(pr >= lo) & (pr < hi)]
        print(f"    股價 {lo}~{hi if hi<1e8 else '以上'}: {len(m)} 檔 平均振幅 {100*m.mean():.2f}%")

    print("\n== 2. 振幅裡有多少是「方向」（近 90 天，振幅≥5% 的股日）==")
    b = last[last.rng >= 0.05]
    print(f"  n={len(b)}  振幅{100*b.rng.mean():.2f}%  |日內|{100*b.oc.abs().mean():.2f}%  開盤後最大上漲{100*b.up.mean():.2f}% 最大下跌{100*b.dn.mean():.2f}%  "
          f"日內淨值{1e4*b.oc.mean():+.0f}bp  上漲日{100*(b.oc>0).mean():.0f}%")

    print("\n== 3. 振幅能否事先知道？（依前 10 日平均振幅分組 → 今日實際）==")
    p["q"] = p.groupby("date").atr10.transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    for q, s in p.groupby("q"):
        print(f"  第{int(q)+1}組 前10日{100*s.atr10.mean():.2f}% → 今日振幅{100*s.rng.mean():.2f}% ≥5%機率{100*(s.rng>=.05).mean():.0f}%  "
              f"日內{1e4*s.oc.mean():+.1f}bp  跳空{1e4*s.gap.mean():+.1f}bp  |日內|{100*s.oc.abs().mean():.2f}%")
    print(f"  相關係數（前10日振幅 vs 今日振幅）: {p.atr10.corr(p.rng):.2f}")

    hv = p[p.q == 4]
    print(f"\n== 4. 高振幅股（每日前 20%，{len(hv)} 股日）方向有沒有規律：各條件下「開盤→收盤」==")
    table(hv, "gap", [-1, -.05, -.03, -.01, .01, .03, .05, 1], title="今日跳空")
    table(hv, "cc1", [-1, -.07, -.03, 0, .03, .07, 1], title="昨日漲跌")
    table(hv, "pos1", [-.01, .2, .4, .6, .8, 1.01], title="昨日收盤位置（0=收最低 1=收最高）")
    table(hv, "ret5", [-1, -.15, -.08, 0, .08, .15, 9], title="前 5 日漲跌")
    table(hv, "volx", [0, .7, 1.5, 3, 99], title="昨日量 / 20 日均量")
    table(hv, "mgap", [-1, -.01, -.003, .003, .01, 1], title="全市場平均跳空（大盤開盤氣氛）")
    print("  逐年（高振幅股整體 開→收）: " + "  ".join(f"{y}:{1e4*s.oc.mean():+.0f}bp" for y, s in hv.groupby("year")))

    print("\n== 5. 不猜方向、只靠振幅行不行：開盤進場、碰到 ±k% 停利、否則收盤出（高振幅股，扣 43.5bp）==")
    for k in (0.01, 0.02, 0.03, 0.04):
        lg = np.where(hv.up >= k, k, hv.oc) - COST
        sh = np.where(hv.dn >= k, k, -hv.oc) - COST
        print(f"  停利{100*k:.0f}%: 做多 碰到{100*(hv.up>=k).mean():.0f}% 勝率{100*(lg>0).mean():.0f}% 每筆{1e4*lg.mean():+.0f}bp | "
              f"做空 碰到{100*(hv.dn>=k).mean():.0f}% 勝率{100*(sh>0).mean():.0f}% 每筆{1e4*sh.mean():+.0f}bp")
    print("  （未計漲跌停鎖死；先前回測顯示計入後放空更差）")

    print("\n== 6. 掛低買：開盤後跌 k% 掛買、收盤賣（高振幅股；只看成交的那些日子）==")
    for k in (0.02, 0.03, 0.04, 0.05):
        s = hv[hv.dn >= k]
        r = s.c / (s.o * (1 - k)) - 1 - COST
        r2 = np.where(s.h >= s.o * (1 - k) * 1.02, np.nan, 0)   # 佔位：日線無法判斷先後，不做停利版
        print(f"  開盤價−{100*k:.0f}% 掛買: 成交{100*len(s)/len(hv):.0f}%的日子 勝率{100*(r>0).mean():.0f}% 每筆{1e4*r.mean():+.0f}bp "
              f"t={tstat(r, s.date):+.1f} 收跌停{100*s.lim_dn.mean():.0f}%  逐年 "
              + " ".join(f"{str(y)[2:]}:{1e4*v:+.0f}" for y, v in r.groupby(s.year).mean().items()))
    print("\n== 6b. 掛高空：開盤後漲 k% 掛空、收盤補 ==")
    for k in (0.02, 0.03, 0.04, 0.05):
        s = hv[hv.up >= k]
        r = 1 - s.c / (s.o * (1 + k)) - COST
        print(f"  開盤價+{100*k:.0f}% 掛空: 成交{100*len(s)/len(hv):.0f}%的日子 勝率{100*(r>0).mean():.0f}% 每筆{1e4*r.mean():+.0f}bp "
              f"t={tstat(r, s.date):+.1f} 收漲停{100*s.lim_up.mean():.0f}%  逐年 "
              + " ".join(f"{str(y)[2:]}:{1e4*v:+.0f}" for y, v in r.groupby(s.year).mean().items()))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "daily6y_400.pkl")
