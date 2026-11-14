"""加密貨幣策略比較回測（現貨、只做多、含手續費），資料來自 crypto_fetch.py 的小時線。

所有訊號只用「已收完的 K 棒」，下一根小時線開盤價成交；停損/停利用小時線高低價檢查（同一根先算停損）。
用法: venv/bin/python backtest/crypto_research.py <cache_dir> [BTCUSDT,ETHUSDT] [fee_per_side=0.0015]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── 指標 ──────────────────────────────────────────────────────────────────────
def rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df.Close.shift(1)
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def resample(h: pd.DataFrame, rule: str) -> pd.DataFrame:
    return h.resample(rule, label="right", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()


def to_hourly(sig: pd.Series, h: pd.DataFrame) -> pd.Series:
    """高週期訊號（index = 該棒收盤時間）對齊到小時線：收盤後才可用。"""
    return sig.reindex(h.index, method="ffill")


# ── 模擬 ──────────────────────────────────────────────────────────────────────
def simulate(h: pd.DataFrame, entry: pd.Series, exit_: pd.Series | None = None, *, fee: float,
             sl: float = 0, tp: float = 0, trail_atr: pd.Series | None = None, trail_mult: float = 0,
             max_hours: int = 0) -> pd.DataFrame:
    o, hi, lo = h.Open.values, h.High.values, h.Low.values
    en = entry.fillna(False).values.astype(bool)
    ex = exit_.fillna(False).values.astype(bool) if exit_ is not None else np.zeros(len(h), bool)
    ta = trail_atr.values if trail_atr is not None else None
    idx = h.index
    trades, pos = [], None
    for i in range(1, len(h)):
        if pos is None:
            if en[i - 1]:
                pos = dict(i=i, px=o[i], peak=o[i], atr=(ta[i - 1] if ta is not None else 0.0))
            continue
        px_out, why = None, ""
        stop = pos["px"] * (1 - sl) if sl else -1.0
        if ta is not None and pos["atr"] > 0:
            stop = max(stop, pos["peak"] - trail_mult * pos["atr"])
        if ex[i - 1]:
            px_out, why = o[i], "訊號"
        elif max_hours and i - pos["i"] >= max_hours:
            px_out, why = o[i], "時間"
        elif stop > 0 and lo[i] <= stop:
            px_out, why = min(stop, o[i]), "停損"
        elif tp and hi[i] >= pos["px"] * (1 + tp):
            px_out, why = max(pos["px"] * (1 + tp), o[i]), "停利"
        if px_out is not None:
            r = px_out * (1 - fee) / (pos["px"] * (1 + fee)) - 1
            trades.append((idx[pos["i"]], idx[i], r, why, i - pos["i"]))
            pos = None
        else:
            pos["peak"] = max(pos["peak"], hi[i])
    return pd.DataFrame(trades, columns=["t_in", "t_out", "ret", "why", "hours"])


def stats(t: pd.DataFrame, h: pd.DataFrame) -> dict:
    if t.empty:
        return dict(n=0)
    eq = (1 + t.ret).cumprod()
    years = (h.index[-1] - h.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min()
    if len(t) == 1 and t.hours.iloc[0] == len(h):     # 買進持有：回撤用價格序列算
        dd = (h.Close / h.Close.cummax() - 1).min()
    w, l = t.ret[t.ret > 0], t.ret[t.ret <= 0]
    yr = (1 + t.ret).groupby(t.t_out.dt.year).prod() - 1
    return dict(n=len(t), win=100 * (t.ret > 0).mean(), avg=100 * t.ret.mean(),
                aw=100 * w.mean() if len(w) else 0, al=100 * l.mean() if len(l) else 0,
                pf=w.sum() / -l.sum() if len(l) and l.sum() < 0 else np.inf,
                tot=100 * (eq.iloc[-1] - 1), cagr=100 * (eq.iloc[-1] ** (1 / years) - 1), dd=100 * dd,
                expo=100 * t.hours.sum() / len(h), neg_years=int((yr < 0).sum()), yrs=len(yr), yr=yr)


# ── 策略 ──────────────────────────────────────────────────────────────────────
def strategies(h: pd.DataFrame, fee: float) -> dict[str, pd.DataFrame]:
    d, f4 = resample(h, "1D"), resample(h, "4h")
    out = {}
    H = lambda s: to_hourly(s, h)
    S = lambda *a, **k: simulate(h, *a, fee=fee, **k)

    # 0. 現行邏輯（MACD/布林因欄位名大小寫不符而從未生效 → 實際只有 RSI + 均線兩票）
    r, ma20, ma60 = rsi(f4.Close, 14), f4.Close.rolling(20).mean(), f4.Close.rolling(60).mean()
    v_rsi = np.select([r < 30, r < 45, r > 70, r > 60], [1.0, 0.5, -1.0, -0.5], 0.0)
    v_ma = np.select([(f4.Close > ma20) & (ma20 > ma60), (f4.Close < ma20) & (ma20 < ma60)], [0.6, -0.6], 0.0)
    live = pd.Series((v_rsi + v_ma) / 2, index=f4.index)
    a4 = atr(f4)
    live_kw = dict(sl=0.15, tp=0.30, trail_atr=H(a4), trail_mult=3.0, max_hours=336)
    out["0 現行(實際: RSI+均線)"] = S(H(live >= 0.3), **live_kw)
    # 0b. 原本想寫的四票版本
    macd = f4.Close.ewm(span=12, adjust=False).mean() - f4.Close.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    v_macd = np.select([(hist > 0) & (hist > hist.shift()), hist > 0, (hist < 0) & (hist < hist.shift()), hist < 0],
                       [0.8, 0.3, -0.8, -0.3], 0.0)
    mid, sd = f4.Close.rolling(20).mean(), f4.Close.rolling(20).std(ddof=0)
    pos = (f4.Close - (mid - 2 * sd)) / (4 * sd)
    v_bb = np.select([pos < 0.2, pos < 0.4, pos > 0.8, pos > 0.6], [0.8, 0.3, -0.8, -0.3], 0.0)
    full = pd.Series((v_rsi + v_ma + v_macd + v_bb) / 4, index=f4.index)
    out["0b 原設計(四票)"] = S(H(full >= 0.3), **live_kw)

    # 1. 買進持有（單筆，期末結算）
    out["1 買進持有"] = pd.DataFrame([(h.index[0], h.index[-1], h.Close.iloc[-1] * (1 - fee) / (h.Open.iloc[0] * (1 + fee)) - 1,
                                      "期末", len(h))], columns=["t_in", "t_out", "ret", "why", "hours"])

    # 2. 趨勢：Donchian 突破（海龜）
    for n_in, n_out in ((20, 10), (55, 20)):
        en = d.Close > d.High.rolling(n_in).max().shift()
        ex = d.Close < d.Low.rolling(n_out).min().shift()
        out[f"2 日線突破{n_in}日高/跌破{n_out}日低"] = S(H(en), H(ex))
    # 3. 趨勢：站上 200 日均線持有
    ma200 = d.Close.rolling(200).mean()
    out["3 站上200日線持有"] = S(H(d.Close > ma200), H(d.Close < ma200))
    out["3b 站上100日線持有"] = S(H(d.Close > d.Close.rolling(100).mean()), H(d.Close < d.Close.rolling(100).mean()))
    # 4. 4H Donchian 15/7 + EMA50>EMA200
    e50, e200 = f4.Close.ewm(span=50, adjust=False).mean(), f4.Close.ewm(span=200, adjust=False).mean()
    en = (f4.Close > f4.High.rolling(15).max().shift()) & (e50 > e200)
    ex = f4.Close < f4.Low.rolling(7).min().shift()
    out["4 4H突破15/7+EMA50>200"] = S(H(en), H(ex))
    # 5. 時間序列動能：30 日報酬 > 0
    m = d.Close / d.Close.shift(30) - 1
    out["5 30日動能>0持有"] = S(H(m > 0), H(m < 0))

    up = d.Close > ma200
    # 6. 多頭中的短線超賣（Connors RSI2）
    r2 = rsi(d.Close, 2)
    ma5 = d.Close.rolling(5).mean()
    for th in (10, 20):
        out[f"6 多頭+日RSI2<{th}→站回5日線"] = S(H((r2 < th) & up), H(d.Close > ma5), max_hours=240)
    out["6c 不看趨勢 日RSI2<10"] = S(H(r2 < 10), H(d.Close > ma5), max_hours=240)
    # 7. 多頭中的 4H RSI14 超賣
    up4 = H(up).reindex(f4.index, method="ffill").fillna(False).astype(bool)
    for th in (30, 35):
        out[f"7 多頭+4H RSI<{th}→RSI>55"] = S(H((r < th) & up4), H(r > 55), max_hours=336)
    # 8. 多頭中的 4H 布林下軌 → 回中軌
    out["8 多頭+4H破布林下軌→回中軌"] = S(H((f4.Close < mid - 2 * sd) & up4), H(f4.Close > mid), max_hours=336)
    # 9. 多頭中單日急跌 → 固定停利/停損
    dr = d.Close / d.Close.shift() - 1
    for drop, tp, sl in ((0.05, 0.04, 0.08), (0.05, 0.06, 0.08), (0.07, 0.05, 0.10)):
        out[f"9 多頭+單日跌≥{drop:.0%} 停利{tp:.0%}/停損{sl:.0%}"] = S(H((dr <= -drop) & up), tp=tp, sl=sl, max_hours=240)
    return out


def main(cache: Path, syms: list[str], fee: float):
    data = pickle.load(open(cache / "crypto_1h.pkl", "rb"))
    pd.set_option("display.width", 250)
    print(f"手續費 單邊 {fee*100:.2f}%（來回 {fee*200:.2f}%）")
    for s in syms:
        h = data[s]
        print(f"\n===== {s}  {h.index[0].date()} ~ {h.index[-1].date()} =====")
        rows = []
        for name, t in strategies(h, fee).items():
            st = stats(t, h)
            if not st["n"]:
                rows.append([name, 0] + [np.nan] * 10); continue
            half = t[t.t_out >= "2023-01-01"]
            r2 = 100 * ((1 + half.ret).prod() - 1) if len(half) else np.nan
            rows.append([name, st["n"], round(st["win"]), round(st["avg"], 2), round(st["aw"], 1), round(st["al"], 1),
                         round(st["pf"], 2), round(st["cagr"]), round(st["dd"]), round(st["expo"]),
                         f"{st['neg_years']}/{st['yrs']}", round(r2)])
        print(pd.DataFrame(rows, columns=["策略", "筆數", "勝率%", "每筆%", "賺均%", "賠均%", "獲利因子", "年化%", "最大回撤%",
                                          "持倉時間%", "虧損年", "2023起報酬%"]).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT,ETHUSDT").split(","),
         float(sys.argv[3]) if len(sys.argv) > 3 else 0.0015)
