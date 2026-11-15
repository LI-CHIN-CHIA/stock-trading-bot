"""加密貨幣小波段（持有數小時~數天）回測：BTC/ETH 現貨、只做多、含手續費。

沿用 crypto_research.py 的小時線模擬器（已收完的 K 棒出訊號、下一根開盤成交）。
用法: venv/bin/python backtest/crypto_swing.py <cache_dir> [BTCUSDT,ETHUSDT] [fee_per_side=0.0015]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from crypto_research import resample, rsi, simulate, stats, to_hourly  # noqa: E402

SPLIT = "2022-07-01"


def rules(h: pd.DataFrame, fee: float) -> dict[str, pd.DataFrame]:
    d, f4 = resample(h, "1D"), resample(h, "4h")
    H = lambda s: to_hourly(s, h)
    S = lambda *a, **k: simulate(h, *a, fee=fee, **k)
    up200, up50 = d.Close > d.Close.rolling(200).mean(), d.Close > d.Close.rolling(50).mean()
    r2d, x5d = rsi(d.Close, 2), d.Close > d.Close.rolling(5).mean()
    r2f, x5f = rsi(f4.Close, 2), f4.Close > f4.Close.rolling(5).mean()
    r2h, x5h = rsi(h.Close, 2), h.Close > h.Close.rolling(5).mean()
    U200, U50 = H(up200).fillna(False).astype(bool), H(up50).fillna(False).astype(bool)
    ret24, ret72 = h.Close.pct_change(24), h.Close.pct_change(72)
    hi5d = f4.High.rolling(30).max().shift(1)
    o = {}
    o["D1 日線RSI2<10 >200日 →>5日線 (現行)"] = S(H(up200 & (r2d < 10)), H(x5d), max_hours=240)
    o["D2 日線RSI2<10 >50日 →>5日線"] = S(H(up50 & (r2d < 10)), H(x5d), max_hours=240)
    o["D3 日線RSI2<10 無趨勢 →>5日線"] = S(H(r2d < 10), H(x5d), max_hours=240)
    o["D4 日線RSI2<20 >200日 →>5日線"] = S(H(up200 & (r2d < 20)), H(x5d), max_hours=240)
    o["D5 D1 + 停利3% (≤3日)"] = S(H(up200 & (r2d < 10)), tp=0.03, max_hours=72)
    o["D6 D1 + 停利5% (≤5日)"] = S(H(up200 & (r2d < 10)), tp=0.05, max_hours=120)
    o["D7 D1 →日線RSI2>60"] = S(H(up200 & (r2d < 10)), H(r2d > 60), max_hours=240)
    o["F1 4H RSI2<10 >200日 →>4H 5均 (≤48h)"] = S(H(r2f < 10) & U200, H(x5f), max_hours=48)
    o["F2 4H RSI2<5 >200日 →>4H 5均 (≤48h)"] = S(H(r2f < 5) & U200, H(x5f), max_hours=48)
    o["F3 4H RSI2<10 >50日 →>4H 5均 (≤48h)"] = S(H(r2f < 10) & U50, H(x5f), max_hours=48)
    o["F4 4H RSI2<5 >200日 停利3% (≤72h)"] = S(H(r2f < 5) & U200, tp=0.03, max_hours=72)
    o["H1 1H RSI2<5 >200日 →>1H 5均 (≤24h)"] = S((r2h < 5) & U200, x5h, max_hours=24)
    o["P1 24h跌≥5% >200日 停利3% (≤72h)"] = S((ret24 <= -0.05) & U200, tp=0.03, max_hours=72)
    o["P2 24h跌≥7% >200日 停利4% (≤72h)"] = S((ret24 <= -0.07) & U200, tp=0.04, max_hours=72)
    o["P3 24h跌≥7% 無趨勢 停利4% (≤72h)"] = S(ret24 <= -0.07, tp=0.04, max_hours=72)
    o["P4 72h跌≥10% >200日 停利5% (≤120h)"] = S((ret72 <= -0.10) & U200, tp=0.05, max_hours=120)
    o["P5 72h跌≥10% >200日 →>5日線 (≤240h)"] = S((ret72 <= -0.10) & U200, H(x5d), max_hours=240)
    o["B1 4H突破5日高 >200日 5%/3% (≤72h)"] = S(H(f4.Close > hi5d) & U200, tp=0.05, sl=0.03, max_hours=72)
    o["B2 4H突破5日高 >200日 抱72h"] = S(H(f4.Close > hi5d) & U200, max_hours=72)
    o["B3 4H突破5日高 >200日 →跌破4H 20均"] = S(H(f4.Close > hi5d) & U200,
                                          H(f4.Close < f4.Close.rolling(20).mean()), max_hours=240)
    return o


def row(name: str, t: pd.DataFrame, h: pd.DataFrame) -> str:
    s = stats(t, h)
    if not s["n"]:
        return f"{name:<40} 無交易"
    a, b = t[t.t_in < SPLIT], t[t.t_in >= SPLIT]
    half = lambda x: f"{100*(x.ret>0).mean():.0f}%/{100*x.ret.mean():+.2f}%({len(x)})" if len(x) else "–"
    yrs = (h.index[-1] - h.index[0]).days / 365.25
    return (f"{name:<40} n={s['n']:>4} ({s['n']/yrs:4.1f}/年) 勝率{s['win']:5.1f}% 每筆{s['avg']:+5.2f}% "
            f"PF{s['pf']:5.2f} 年化{s['cagr']:+6.1f}% 回撤{s['dd']:6.1f}% 持有{t.hours.mean():5.1f}h "
            f"負年{s['neg_years']}/{s['yrs']} 前段{half(a)} 後段{half(b)}")


if __name__ == "__main__":
    cache = Path(sys.argv[1])
    syms = (sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT,ETHUSDT").split(",")
    fee = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0015
    data = pickle.load(open(cache / "crypto_1h.pkl", "rb"))
    for s in syms:
        h = data[s]
        print(f"\n===== {s}  {h.index[0].date()}~{h.index[-1].date()}  手續費 {fee*100:.2f}%/邊 =====")
        for k, t in rules(h, fee).items():
            print(row(k, t, h))
