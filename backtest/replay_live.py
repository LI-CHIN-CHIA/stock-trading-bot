"""線上當沖/反沖邏輯重播回測。

直接呼叫 trader.day_trader 的 _analyze_bars / _mkt_score，逐分鐘重播
_scan_entries + _check_holdings，並在每筆進場記錄通用特徵（大盤、相對強弱、
多日趨勢、beta、量能、VWAP 距離、時段）供規則驗證。

用法:
  venv/bin/python backtest/replay_live.py features <cache_dir> [1m|2m]   # 建特徵表（慢，一次）
  之後由分析腳本 import simulate()。
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from trader.day_trader import _analyze_bars, _mkt_score, _vwap  # noqa: E402

TZ = "Asia/Taipei"
COMM, STT, MIN_FEE = 0.001425, 0.0015, 20


# ── 特徵表 ────────────────────────────────────────────────────────────────────

_MODEL = None


def _model_pred(g: pd.DataFrame, a: dict, model_path: str | None) -> float:
    """線上 DayModel 的預測值（僅多方候選才算，省時間）。"""
    global _MODEL
    if not model_path:
        return np.nan
    if _MODEL is None:
        from ai.day_model import DayModel
        _MODEL = DayModel.load(model_path)
    return float(_MODEL.predict_gain(g, a))


def _features_one(args) -> pd.DataFrame:
    code, df, daily, model_path, min_vol = args
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.tz_convert(TZ) if df.index.tz is not None else df.tz_localize(TZ)
    df = df.dropna(subset=["Close"])
    df = df[df["Volume"] > 0]
    dclose = daily["Close"].copy()
    dclose.index = pd.to_datetime(dclose.index).tz_localize(None).normalize()
    rows = []
    for day, g in df.groupby(df.index.date):
        g = g.between_time("09:00", "13:30")
        if len(g) < 30:
            continue
        hist = dclose[dclose.index < pd.Timestamp(day)]
        if len(hist) < 6:
            continue
        prev_close = float(hist.iloc[-1])
        ret3 = prev_close / float(hist.iloc[-4]) - 1
        ret5 = prev_close / float(hist.iloc[-6]) - 1
        ma20 = float(hist.iloc[-20:].mean())
        vwap = _vwap(g).values
        o, h, l, c, v = (g[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
        day_open = float(o[0])
        for i in range(4, len(g)):
            a = _analyze_bars(g.iloc[: i + 1])
            if not a:
                continue
            hm = g.index[i].hour * 100 + g.index[i].minute
            pred = np.nan
            if a["score"] >= 0 and 905 <= hm < 1230 and a["avg_volume"] >= min_vol:
                pred = _model_pred(g.iloc[: i + 1], a, model_path)
            rows.append({
                "model_pred": pred,
                "code": code, "day": day, "ts": g.index[i], "i": i,
                "open": o[i], "high": h[i], "low": l[i], "close": c[i], "vwap": vwap[i],
                "score": a["score"], "above_vwap": a["above_vwap"],
                "delta": a["vol_delta_signal"], "avg_vol": a["avg_volume"],
                "last_vol": a["last_volume"], "vol_surge": a["vol_surge"],
                "breakout": a["breakout"], "pattern": a["pattern"],
                "prev_close": prev_close, "day_open": day_open,
                "ret3": ret3, "ret5": ret5, "above_ma20": prev_close > ma20,
                "day_high": float(h[: i + 1].max()), "day_low": float(l[: i + 1].min()),
                "cum_vol": float(v[: i + 1].sum()),
            })
    return pd.DataFrame(rows)


def build_features(cache: Path, interval: str = "1m") -> pd.DataFrame:
    bars = pickle.load(open(cache / f"bars_{interval}.pkl", "rb"))
    daily = pickle.load(open(cache / "daily.pkl", "rb"))
    mp = cache / "day_model_latest.pkl"
    min_vol = 15_000 * (2 if interval == "2m" else 1)
    jobs = [(c, bars[c], daily[c], str(mp) if mp.exists() else None, min_vol)
            for c in sorted(bars) if c in daily and not daily[c].empty]
    with Pool(12) as p:
        parts = p.map(_features_one, jobs)
    feat = pd.concat([x for x in parts if not x.empty], ignore_index=True)

    # beta（近60日日報酬 vs TWII，僅用當日之前資料 → 簡化為全期間，僅作分群用）
    tw = daily["^TWII"]["Close"].pct_change().dropna()
    tw.index = pd.to_datetime(tw.index).tz_localize(None).normalize()
    betas = {}
    for c in feat["code"].unique():
        r = daily[c]["Close"].pct_change().dropna()
        r.index = pd.to_datetime(r.index).tz_localize(None).normalize()
        j = pd.concat([r, tw], axis=1, join="inner").dropna().iloc[-120:]
        betas[c] = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / np.var(j.iloc[:, 1])) if len(j) > 30 else 1.0
    feat["beta"] = feat["code"].map(betas)

    # 大盤：5m K → 每個 ts 對應「已收完」的最後一根
    tw5 = pickle.load(open(cache / "twii_5m.pkl", "rb")).tz_convert(TZ)
    twd = daily["^TWII"]["Close"].copy()
    twd.index = pd.to_datetime(twd.index).tz_localize(None).normalize()
    mk = []
    for day, g in tw5.groupby(tw5.index.date):
        hist = twd[twd.index < pd.Timestamp(day)]
        if hist.empty:
            continue
        pc = float(hist.iloc[-1])
        g = g.copy()
        g["mkt_gap"] = g["Close"] / pc - 1
        g["mkt_high"] = g["High"].cummax()
        g["mkt_open"] = float(g["Open"].iloc[0])
        g["mkt_pull"] = g["Close"] / g["mkt_high"] - 1
        g["mkt_ret3"] = pc / float(hist.iloc[-4]) - 1 if len(hist) >= 4 else 0.0
        g["mkt_avail"] = g.index + pd.Timedelta(minutes=5)   # 收完才可見
        mk.append(g[["mkt_gap", "mkt_pull", "mkt_open", "Close", "mkt_ret3", "mkt_avail"]].rename(columns={"Close": "mkt_cur"}))
    mk = pd.concat(mk).sort_values("mkt_avail")
    mk["mkt_avail"] = mk["mkt_avail"].astype("datetime64[ns, Asia/Taipei]")
    feat["ts"] = pd.to_datetime(feat["ts"]).astype("datetime64[ns, Asia/Taipei]")
    feat = feat.sort_values("ts")
    feat = pd.merge_asof(feat, mk, left_on="ts", right_on="mkt_avail", direction="backward",
                         tolerance=pd.Timedelta(minutes=30))
    feat = feat.dropna(subset=["mkt_gap"])
    feat["intraday_sig"] = np.where(feat["mkt_pull"] <= -0.015, "FADE",
                                    np.where(feat["mkt_cur"] > feat["mkt_open"] * 1.005, "PUSH", "NONE"))
    feat["stk_pct"] = feat["close"] / feat["prev_close"] - 1
    feat["rel"] = feat["stk_pct"] - feat["mkt_gap"]            # >0 = 強於大盤
    feat["vwap_dist"] = feat["close"] / feat["vwap"] - 1
    feat["vol_ratio"] = feat["last_vol"] / feat["avg_vol"].replace(0, np.nan)
    feat["hm"] = feat["ts"].dt.hour * 100 + feat["ts"].dt.minute
    feat["range_pos"] = (feat["close"] - feat["day_low"]) / (feat["day_high"] - feat["day_low"]).replace(0, np.nan)
    return feat.sort_values(["ts", "code"]).reset_index(drop=True)


# ── 模擬 ──────────────────────────────────────────────────────────────────────

@dataclass
class Cfg:
    budget: float = 50_000          # 單筆上限
    max_pos: int = 3
    sl: float = 0.015
    tp: float = 0.025
    trail_act: float = 0.015
    trail_dist: float = 0.010
    decay_min: int = 30
    decay_band: float = 0.005
    min_avg_vol: float = 15_000     # 以 1m K 為準；2m 資料請 ×2
    min_entry_vol_pct: float = 0.8
    surge_x: float = 3.0
    buy_start: int = 905
    buy_end: int = 1230
    enable_long: bool = True
    enable_short: bool = True
    short_cooldown_min: int = 60
    mkt_breaker: bool = True        # gap>1.5% 且非 FADE 停空
    vwap_exit: bool = True
    vwap_exit_long: bool | None = None     # None=沿用 vwap_exit
    vwap_exit_short: bool | None = None
    decay_long: bool = True
    decay_short: bool = True
    bar_min: int = 1
    model_thr: float | None = 0.008        # 線上 DAY_MODEL_THRESHOLD；None=不套模型
    long_filter: Callable | None = None    # f(row)->bool，True=允許
    short_filter: Callable | None = None
    unlimited: bool = False         # True=不限倉位數（產生候選全集供規則研究）
    us_dir: dict = field(default_factory=dict)   # {day: 'BULL'|'BEAR'|'NEUTRAL'}


def _fee(v: float) -> float:
    return max(MIN_FEE, round(v * COMM))


def simulate(feat: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    trades = []
    cols = list(feat.columns)
    for day, fd in feat.groupby("day", sort=True):
        us = cfg.us_dir.get(day, "NEUTRAL")
        hold: dict[str, dict] = {}
        cooldown: dict[str, pd.Timestamp] = {}

        def close(code, price, reason, ts):
            h = hold.pop(code)
            sh = h["shares"]
            if h["side"] == "L":
                pnl = sh * price - _fee(sh * price) - round(sh * price * STT) - (sh * h["entry"] + _fee(sh * h["entry"]))
            else:
                pnl = (sh * h["entry"] - _fee(sh * h["entry"]) - round(sh * h["entry"] * STT)) - (sh * price + _fee(sh * price))
                cooldown[code] = ts
            t = dict(h["feat"])
            t.update(side=h["side"], entry=h["entry"], exit=price, shares=sh, pnl=pnl,
                     ret=pnl / (sh * h["entry"]), reason=reason, exit_ts=ts,
                     hold_min=(ts - h["ts"]).total_seconds() / 60)
            trades.append(t)

        for ts, g in fd.groupby("ts", sort=True):
            hm = ts.hour * 100 + ts.minute
            recs = g.to_dict("records")
            byc = {r["code"]: r for r in recs}

            # 1) 出場
            for code in list(hold):
                r = byc.get(code)
                if r is None:
                    continue
                h = hold[code]
                held = (ts - h["ts"]).total_seconds() / 60
                if h["side"] == "L":
                    if r["low"] <= h["sl"]:
                        close(code, min(h["sl"], r["open"]), "停損", ts); continue
                    if r["high"] >= h["tp"]:
                        close(code, max(h["tp"], r["open"]), "停利", ts); continue
                    pct = r["close"] / h["entry"] - 1
                    if pct >= cfg.trail_act:
                        h["sl"] = max(h["sl"], round(r["close"] * (1 - cfg.trail_dist), 2))
                    if hm >= 1320:
                        close(code, r["close"], "強制平倉", ts); continue
                    if cfg.decay_long and held >= cfg.decay_min and abs(pct) < cfg.decay_band:
                        close(code, r["close"], "時間衰退", ts); continue
                    vx = cfg.vwap_exit if cfg.vwap_exit_long is None else cfg.vwap_exit_long
                    if vx and held > 5 and r["close"] < r["vwap"] and pct > -cfg.sl:
                        close(code, r["close"], "跌破VWAP", ts)
                else:
                    if r["high"] >= h["sl"]:
                        close(code, max(h["sl"], r["open"]), "停損", ts); continue
                    if r["low"] <= h["tp"]:
                        close(code, min(h["tp"], r["open"]), "停利", ts); continue
                    pct = 1 - r["close"] / h["entry"]
                    if hm >= 1320:
                        close(code, r["close"], "強制平倉", ts); continue
                    if cfg.decay_short and held >= cfg.decay_min and abs(pct) < cfg.decay_band:
                        close(code, r["close"], "時間衰退", ts); continue
                    vx = cfg.vwap_exit if cfg.vwap_exit_short is None else cfg.vwap_exit_short
                    if vx and held > 5 and r["close"] > r["vwap"] and pct > -cfg.sl:
                        close(code, r["close"], "突破VWAP", ts)

            # 2) 進場
            if not (cfg.buy_start <= hm < cfg.buy_end):
                continue
            if not cfg.unlimited and len(hold) >= cfg.max_pos:
                continue
            longs, shorts = [], []
            for r in recs:
                code = r["code"]
                if code in hold:
                    continue
                if r["avg_vol"] < cfg.min_avg_vol or r["last_vol"] < r["avg_vol"] * cfg.min_entry_vol_pct:
                    continue
                shares = int(cfg.budget // (r["close"] * 1000)) * 1000
                if shares <= 0:
                    continue
                lb, sb = _mkt_score(r["mkt_gap"], us, r["intraday_sig"])
                surge = r["vol_ratio"] >= cfg.surge_x
                thr = 2 if surge else 3
                eff_l, eff_s = r["score"] + lb, r["score"] + sb
                model_ok = cfg.model_thr is None or not (r["model_pred"] < cfg.model_thr)
                if (cfg.enable_long and eff_l >= thr and model_ok
                        and (cfg.long_filter is None or cfg.long_filter(r))):
                    longs.append((eff_l, r, shares))
                blocked = cfg.mkt_breaker and r["mkt_gap"] > 0.015 and r["intraday_sig"] != "FADE"
                cd = cooldown.get(code)
                in_cd = cd is not None and (ts - cd).total_seconds() < cfg.short_cooldown_min * 60
                if (cfg.enable_short and not blocked and not in_cd and eff_s <= -thr
                        and not r["above_vwap"] and r["delta"] == "sell"
                        and (cfg.short_filter is None or cfg.short_filter(r))):
                    shorts.append((eff_s, r, shares))
            longs.sort(key=lambda x: -x[0])
            shorts.sort(key=lambda x: x[0])
            for side, lst in (("L", longs), ("S", shorts)):
                for eff, r, shares in lst:
                    if not cfg.unlimited and len(hold) >= cfg.max_pos:
                        break
                    if r["code"] in hold:
                        continue
                    p = r["close"]
                    f = {k: r[k] for k in cols}
                    f["eff"] = eff
                    hold[r["code"]] = {
                        "side": side, "entry": p, "ts": ts, "shares": shares, "feat": f,
                        "sl": round(p * (1 - cfg.sl), 2) if side == "L" else round(p * (1 + cfg.sl), 2),
                        "tp": round(p * (1 + cfg.tp), 2) if side == "L" else round(p * (1 - cfg.tp), 2),
                    }
        # 收盤仍未平（資料缺 13:20 後的 K）→ 以最後一根收盤平倉
        last = fd.sort_values("ts").groupby("code").tail(1).set_index("code")
        for code in list(hold):
            close(code, float(last.loc[code, "close"]), "強制平倉", last.loc[code, "ts"])
    return pd.DataFrame(trades)


def summarize(tr: pd.DataFrame, by=None, label: str = "") -> pd.DataFrame:
    if tr.empty:
        return pd.DataFrame()
    g = tr.groupby(by) if by is not None else tr.groupby(lambda _: label or "all")
    return pd.DataFrame({
        "n": g["pnl"].size(), "win%": (g["pnl"].apply(lambda s: (s > 0).mean()) * 100).round(1),
        "avg_ret_bp": (g["ret"].mean() * 1e4).round(1), "pnl": g["pnl"].sum().round(0),
        "avg_pnl": g["pnl"].mean().round(0),
    })


if __name__ == "__main__":
    cache = Path(sys.argv[2])
    interval = sys.argv[3] if len(sys.argv) > 3 else "1m"
    feat = build_features(cache, interval)
    feat.to_pickle(cache / f"feat_{interval}.pkl")
    print(feat.shape, feat["day"].nunique(), "days", feat["code"].nunique(), "codes")
