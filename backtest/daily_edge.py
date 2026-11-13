"""日線層級的當沖優勢檢驗：開盤進、收盤出（open→close），5 年、逐年驗證。

所有特徵都只用「開盤當下已知」的資訊；選股用每日排名（前/後 N 名），不設個股門檻。

用法:
  venv/bin/python backtest/daily_edge.py fetch <cache_dir>
  venv/bin/python backtest/daily_edge.py test  <cache_dir>
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
COST = 0.001425 * 2 + 0.0015


def fetch(cache: Path):
    import yfinance as yf
    from utils.constants import OTC_SYMBOLS, TRADEABLE
    out = {}
    for i, c in enumerate(sorted(TRADEABLE)):
        s = f"{c}.TWO" if c in OTC_SYMBOLS else f"{c}.TW"
        out[c] = yf.Ticker(s).history(period="6y", interval="1d", auto_adjust=False)
        print(i + 1, c, len(out[c]), flush=True)
        time.sleep(0.3)
    for name in ("^TWII", "^GSPC", "^SOX"):
        out[name] = yf.Ticker(name).history(period="6y", interval="1d")
    pickle.dump(out, open(cache / "daily6y.pkl", "wb"))


def panel(cache: Path) -> pd.DataFrame:
    raw = pickle.load(open(cache / "daily6y.pkl", "rb"))

    def norm(df):
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df

    tw = norm(raw["^TWII"])
    mk = pd.DataFrame({
        "mkt_gap": tw.Open / tw.Close.shift(1) - 1,
        "mkt_prev": tw.Close.shift(1) / tw.Close.shift(2) - 1,
        "mkt_ret5": tw.Close.shift(1) / tw.Close.shift(6) - 1,
        "mkt_oc": tw.Close / tw.Open - 1,
    })
    # 美股前一交易日（台股開盤前已收盤）
    for name, col in (("^GSPC", "spx_prev"), ("^SOX", "sox_prev")):
        u = norm(raw[name]).Close.pct_change()
        mk[col] = u.reindex(mk.index.union(u.index)).ffill().shift(1).reindex(mk.index)

    rows = []
    for c, df in raw.items():
        if c.startswith("^") or len(df) < 300:
            continue
        d = norm(df)
        d = d[(d.Volume > 0) & (d.Open > 0)]
        pc = d.Close.shift(1)
        x = pd.DataFrame({
            "code": c, "open": d.Open, "close": d.Close, "high": d.High, "low": d.Low,
            "oc": d.Close / d.Open - 1,
            "gap": d.Open / pc - 1,
            "prev_ret": pc / d.Close.shift(2) - 1,
            "ret3": pc / d.Close.shift(4) - 1,
            "ret5": pc / d.Close.shift(6) - 1,
            "ret20": pc / d.Close.shift(21) - 1,
            "ma20_dist": pc / d.Close.shift(1).rolling(20).mean() - 1,
            "prev_volx": d.Volume.shift(1) / d.Volume.shift(2).rolling(20).mean(),
            "prev_pos": ((d.Close - d.Low) / (d.High - d.Low).replace(0, np.nan)).shift(1),
            "prev_oc": (d.Close / d.Open - 1).shift(1),
            "atr": ((d.High - d.Low) / d.Close).shift(1).rolling(10).mean(),
            "turnover": (d.Close * d.Volume).shift(1).rolling(20).mean(),
            "hi_o": d.High / d.Open - 1, "lo_o": d.Low / d.Open - 1,
        })
        rows.append(x)
    p = pd.concat(rows).join(mk).dropna(subset=["gap", "ret20", "atr", "mkt_gap"])
    p["rel_gap"] = p.gap - p.mkt_gap
    p["gap_atr"] = p.gap / p.atr
    p["year"] = p.index.year
    p.index.name = "date"
    return p.reset_index()


def trade_ret(p: pd.DataFrame, side: str, sl: float | None) -> pd.Series:
    """開盤進、收盤出；若日內觸及停損（保守：先碰停損）以停損價出場。"""
    if side == "L":
        r = p.oc.copy()
        if sl:
            r = r.where(p.lo_o > -sl, -sl)
    else:
        r = -p.oc / 1.0
        r = 1 - p.close / p.open
        if sl:
            r = r.where(p.hi_o < sl, -sl)
    return r - COST


def test(cache: Path):
    p = panel(cache)
    pd.set_option("display.width", 250)
    years = sorted(p.year.unique())
    print(f"{p.code.nunique()} 檔  {p.date.min().date()} ~ {p.date.max().date()}  {len(p)} 股日  成本 {COST*1e4:.1f}bp")
    print(f"全體 open→close 平均 {p.oc.mean()*1e4:+.1f}bp  |oc| 平均 {p.oc.abs().mean()*1e4:.0f}bp  大盤 oc {p.groupby('date').mkt_oc.first().mean()*1e4:+.1f}bp")

    feats = ["gap", "rel_gap", "gap_atr", "prev_ret", "prev_oc", "ret3", "ret5", "ret20", "ma20_dist",
             "prev_volx", "prev_pos", "atr", "mkt_gap", "mkt_prev", "spx_prev", "sox_prev"]
    print("\n== 各特徵對 open→close 的橫斷面 rank IC（逐年；|IC|≥0.02 且各年同號才有意義）==")
    rows = []
    for ft in feats:
        if ft in ("mkt_gap", "mkt_prev", "spx_prev", "sox_prev"):
            d = p.groupby("date").agg(x=(ft, "first"), y=("oc", "mean"), year=("year", "first"))
            ic = d.groupby("year").apply(lambda g: g.x.rank().corr(g.y.rank()))
            rows.append([ft + "(時序)"] + [round(ic.get(y, np.nan), 3) for y in years])
        else:
            ic = p.groupby("date").apply(lambda g: g[ft].rank().corr(g.oc.rank()) if len(g) > 10 else np.nan)
            icy = ic.groupby(ic.index.year).mean()
            rows.append([ft] + [round(icy.get(y, np.nan), 3) for y in years])
    print(pd.DataFrame(rows, columns=["feat"] + years).to_string(index=False))

    print("\n== 排名選股策略：每日取 N=3 檔，開盤進收盤出，扣成本後『每筆平均 bp』（逐年）==")
    strategies = {
        "空 跳空最大3檔":            ("S", "gap", False),
        "多 跳空最小3檔":            ("L", "gap", True),
        "空 相對大盤跳空最大3檔":     ("S", "rel_gap", False),
        "多 相對大盤跳空最小3檔":     ("L", "rel_gap", True),
        "空 gap/ATR 最大3檔":        ("S", "gap_atr", False),
        "多 gap/ATR 最小3檔":        ("L", "gap_atr", True),
        "空 前日漲最多3檔":           ("S", "prev_ret", False),
        "多 前日跌最多3檔":           ("L", "prev_ret", True),
        "多 前日漲最多3檔(動能)":     ("L", "prev_ret", False),
        "空 前日跌最多3檔(動能)":     ("S", "prev_ret", True),
        "空 5日漲最多3檔":            ("S", "ret5", False),
        "多 5日跌最多3檔":            ("L", "ret5", True),
        "多 5日漲最多3檔(動能)":      ("L", "ret5", False),
        "空 前日爆量最大3檔":         ("S", "prev_volx", False),
        "多 前日爆量最大3檔":         ("L", "prev_volx", False),
        "多 波動最大3檔":             ("L", "atr", False),
        "空 波動最大3檔":             ("S", "atr", False),
    }
    for sl in (None, 0.02):
        print(f"\n-- 停損: {'無' if not sl else f'{sl*100:.0f}%（保守假設先碰停損）'} --")
        out = []
        for name, (side, ft, asc) in strategies.items():
            pp = p.copy()
            pp["rk"] = pp.groupby("date")[ft].rank(ascending=asc, method="first")
            sel = pp[pp.rk <= 3].copy()
            sel["r"] = trade_ret(sel, side, sl)
            by = sel.groupby("year").r.mean() * 1e4
            daily = sel.groupby("date").r.mean()
            t = daily.mean() / daily.std() * np.sqrt(len(daily))
            out.append([name] + [round(by.get(y, np.nan), 0) for y in years]
                       + [round(sel.r.mean() * 1e4, 1), round(100 * (sel.r > 0).mean(), 1), round(t, 1),
                          round(100 * (daily > 0).mean(), 0)])
        print(pd.DataFrame(out, columns=["策略"] + years + ["全期bp", "勝率%", "t", "勝日%"]).to_string(index=False))

    p.to_pickle(cache / "daily_panel.pkl")


if __name__ == "__main__":
    {"fetch": fetch, "test": test}[sys.argv[1]](Path(sys.argv[2]))
