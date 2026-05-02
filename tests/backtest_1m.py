"""
1 個月整合回測 — 驗證 feature/trading-agents 分支的所有新功能

測試內容:
  A. 一般股票 + 舊策略（固定停損 -7%）
  B. 一般股票 + 新策略（ATR追蹤 + Strategy B + RiskProfile）
  C. 興櫃模擬 + EMERGING RiskProfile（在一般股票上套用興櫃參數模擬）
  D. combine_signals 影響（模擬 TA 訊號，測試合併邏輯）

最後生成 HTML 比較報告。

用法:
  venv/bin/python tests/backtest_1m.py
  venv/bin/python tests/backtest_1m.py --period 3mo  # 拉長至3個月
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.features import build_feature_matrix, build_target
from ai.model import TradingModel
from trader.risk import (
    NORMAL, EMERGING,
    buy_cost, sell_proceeds, should_atr_stop, should_stop_loss, should_take_profit,
)
from utils.constants import OTC_SYMBOLS, STOCK_DB, TRADEABLE

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

CAPITAL = 20_000
MAX_POSITIONS = 5
TRANSACTION_FEE = 0.001425
TAX_RATE = 0.003


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_symbol(code: str) -> str:
    return f"{code}.TWO" if code in OTC_SYMBOLS else f"{code}.TW"


def _get_price(df: pd.DataFrame | None, date) -> float:
    if df is None or df.empty:
        return 0.0
    try:
        return float(df.loc[:date]["Close"].iloc[-1])
    except Exception:
        return 0.0


@dataclass
class Trade:
    ticker: str
    entry_date: str
    entry_price: float
    shares: int
    cost: float
    exit_date: str = ""
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""
    strategy: str = ""


@dataclass
class BacktestResult:
    strategy_name: str
    total_return_pct: float
    annualised_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    final_capital: float
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trades: list = field(default_factory=list)
    reason_counts: dict = field(default_factory=dict)


# ── Core Simulation ───────────────────────────────────────────────────────────

def simulate(
    strategy_name: str,
    model: TradingModel,
    ticker_dfs: dict,
    raw_data: dict,
    test_dates: list,
    *,
    profile=NORMAL,
    use_combine_signals: bool = False,
    ta_signal_override: dict | None = None,  # code -> "BUY"|"SELL"|"HOLD"
    capital: float = CAPITAL,
    verbose: bool = False,
) -> BacktestResult:
    """
    Walk-forward simulation on test_dates.

    profile: RiskProfile — controls stop-loss, take-profit, ATR multiplier
    use_combine_signals: simulate combine_signals() logic with ta_signal_override
    ta_signal_override: per-code TA signal (static, for testing combine logic)
    """
    cash = capital
    holdings: dict = {}
    equity_curve: dict = {}
    completed_trades: list[Trade] = []
    reason_counts: dict = {}

    # Benchmark (0050) for market filter
    benchmark_df = raw_data.get("0050")

    def _market_bullish(date) -> bool:
        if benchmark_df is None or benchmark_df.empty:
            return True
        try:
            sub = benchmark_df.loc[:date]["Close"]
            return len(sub) >= 60 and float(sub.iloc[-1]) > float(sub.iloc[-60:].mean())
        except Exception:
            return True

    def _effective_signal(code, row, base_pred):
        """Apply combine_signals if enabled."""
        if not use_combine_signals or ta_signal_override is None:
            return base_pred
        ta_sig = ta_signal_override.get(code, "HOLD")
        ta_conf = 0.75 if ta_sig in ("BUY", "SELL") else 0.55
        ta = {"signal": ta_sig, "confidence": ta_conf}

        from trader.ta_signal import combine_signals
        combined = combine_signals(base_pred, ta)
        return combined if combined else base_pred

    for date in test_dates:
        # ── 持倉管理 ─────────────────────────────────────────────────────────
        for code in list(holdings.keys()):
            h = holdings[code]
            price = _get_price(raw_data.get(code), date)
            if price <= 0:
                continue

            pnl_pct = (price - h["entry_price"]) / h["entry_price"]
            h["hold_days"] = h.get("hold_days", 0) + 1
            h["peak_price"] = max(h.get("peak_price", h["entry_price"]), price)

            feat, _ = ticker_dfs.get(code, (None, None))
            atr, pred = 0.0, None
            if feat is not None and date in feat.index:
                row = feat.loc[[date]]
                atr = float(row["ATR"].iloc[0]) if "ATR" in row.columns else 0.0
                base_pred = model.predict(row)
                pred = _effective_signal(code, row, base_pred)

            reason = None

            # 1. ATR trailing stop
            if should_atr_stop(h["peak_price"], price, atr, profile):
                reason = f"ATR追蹤停損[{profile.label}] {pnl_pct:.1%}"
            # 2. Fixed stop-loss fallback
            elif atr <= 0 and should_stop_loss(h["entry_price"], price, profile):
                reason = f"停損[{profile.label}] {pnl_pct:.1%}"
            # 3. Take-profit (興櫃直接出場；一般 Strategy B)
            elif should_take_profit(h["entry_price"], price, profile):
                if profile.is_emerging:
                    reason = f"停利[興櫃] {pnl_pct:.1%}"
                else:
                    buy_proba = pred.get("buy_proba", 0) if pred else 0
                    if pred and pred.get("signal") == "BUY" and buy_proba >= 0.70:
                        pass  # Strategy B: AI still bullish, hold
                    else:
                        reason = f"停利 {pnl_pct:.1%}"
            # 4. Max hold days
            elif h["hold_days"] >= profile.max_hold_days:
                reason = f"超過持有期 {pnl_pct:.1%}"
            # 5. AI sell signal
            elif pred and pred.get("signal") == "SELL" and pred.get("sell_proba", 0) >= 0.55:
                reason = f"AI賣出 {pred['sell_proba']:.0%}"

            if reason:
                sell_value = h["shares"] * price * (1 - TAX_RATE - TRANSACTION_FEE)
                trade_pnl = sell_value - h["cost"]
                completed_trades.append(Trade(
                    ticker=code, entry_date=h["entry_date"],
                    entry_price=h["entry_price"], shares=h["shares"],
                    cost=h["cost"], exit_date=str(date)[:10],
                    exit_price=price, pnl=round(trade_pnl, 2),
                    pnl_pct=round(pnl_pct, 4), reason=reason,
                    strategy=strategy_name,
                ))
                reason_key = next(
                    (k for k in ["停損", "停利", "AI賣出", "持有期", "追蹤"] if k in reason),
                    "其他"
                )
                reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
                cash += sell_value
                del holdings[code]

        # ── 買入掃描 ─────────────────────────────────────────────────────────
        if len(holdings) < MAX_POSITIONS and cash > 500 and _market_bullish(date):
            candidates = []
            for code, (feat, _) in ticker_dfs.items():
                if code in holdings or date not in feat.index:
                    continue
                row = feat.loc[[date]]
                base_pred = model.predict(row)
                pred = _effective_signal(code, row, base_pred)
                min_proba = profile.min_buy_proba
                if pred["signal"] == "BUY" and pred["buy_proba"] >= min_proba:
                    entry_atr = float(row["ATR"].iloc[0]) if "ATR" in row.columns else 0.0
                    candidates.append((code, pred["buy_proba"], entry_atr))

            candidates.sort(key=lambda x: x[1], reverse=True)
            slots = MAX_POSITIONS - len(holdings)
            budget = cash / max(slots, 1)

            for code, buy_proba, entry_atr in candidates[:slots]:
                price = _get_price(raw_data.get(code), date)
                if price <= 0:
                    continue
                # 使用 profile 控制單支上限
                max_budget = cash * profile.max_per_stock_pct
                actual_budget = min(budget, max_budget)
                shares = int(actual_budget / (price * (1 + TRANSACTION_FEE)))
                if shares < 1:
                    continue
                cost = shares * price * (1 + TRANSACTION_FEE)
                if cost > cash:
                    continue
                holdings[code] = {
                    "shares": shares, "entry_price": price,
                    "entry_date": str(date)[:10], "cost": cost,
                    "hold_days": 0, "peak_price": price, "atr": entry_atr,
                }
                cash -= cost

        # ── 記錄資產淨值 ─────────────────────────────────────────────────────
        holding_value = sum(
            h["shares"] * _get_price(raw_data.get(c), date)
            for c, h in holdings.items()
        )
        equity_curve[date] = cash + holding_value

    # ── 計算指標 ─────────────────────────────────────────────────────────────
    equity = pd.Series(equity_curve).sort_index()
    if equity.empty:
        equity = pd.Series([capital])

    final_cap = equity.iloc[-1]
    total_return = (final_cap - capital) / capital
    days = max((equity.index[-1] - equity.index[0]).days, 1) if len(equity) > 1 else 1
    ann_return = (1 + total_return) ** (365 / days) - 1

    daily_ret = equity.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    roll_max = equity.cummax()
    max_dd = ((equity - roll_max) / roll_max).min()

    wins   = [t for t in completed_trades if t.pnl > 0]
    losses = [t for t in completed_trades if t.pnl <= 0]
    win_rate = len(wins) / max(len(completed_trades), 1)

    if verbose:
        print(f"  [{strategy_name}] 報酬={total_return:+.2%} 夏普={sharpe:.2f} "
              f"勝率={win_rate:.1%} 交易={len(completed_trades)} 期末=${final_cap:,.0f}")

    return BacktestResult(
        strategy_name=strategy_name,
        total_return_pct=round(total_return * 100, 2),
        annualised_return_pct=round(ann_return * 100, 2),
        sharpe_ratio=round(float(sharpe), 3),
        max_drawdown_pct=round(float(max_dd) * 100, 2),
        win_rate_pct=round(win_rate * 100, 1),
        total_trades=len(completed_trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        final_capital=round(final_cap, 0),
        equity_curve=equity,
        trades=completed_trades,
        reason_counts=reason_counts,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all(period: str = "1mo", verbose: bool = True) -> list[BacktestResult]:
    print("=" * 60)
    print(f"  整合回測 — feature/trading-agents 分支")
    print(f"  期間: {period}  初始資金: ${CAPITAL:,}")
    print("=" * 60)

    # 1. 下載資料
    # 技術指標需要至少 80 天（MA60+buffer），因此下載 6mo 但只測試最後 period
    codes = [c for c in TRADEABLE if c not in {"0050", "0056", "00878", "006208", "00713"}]
    download_period = "6mo"  # 固定下載 6 個月以確保特徵可計算
    print(f"\n📥 下載 {len(codes)} 支股票資料 (download={download_period}, test={period})…")

    raw_data: dict = {}
    for code in codes:
        sym = get_symbol(code)
        try:
            df = yf.Ticker(sym).history(period=download_period)
            if len(df) >= 60:
                raw_data[code] = df
        except Exception:
            pass
    # 加入 0050 用於大盤過濾
    try:
        raw_data["0050"] = yf.Ticker("0050.TW").history(period=download_period)
    except Exception:
        pass
    print(f"✅ 取得 {len(raw_data)} 支")

    # 2. 建立特徵
    print("🔧 建立特徵矩陣…")
    all_feat, all_target = [], []
    ticker_dfs: dict = {}
    for code, df in raw_data.items():
        if code == "0050":
            continue
        try:
            feat = build_feature_matrix(df, code=code)
            target = build_target(df)
            common = feat.index.intersection(target.index)
            feat, target = feat.loc[common], target.loc[common]
            if len(feat) >= 10:
                ticker_dfs[code] = (feat, target)
                all_feat.append(feat)
                all_target.append(target)
        except Exception as e:
            logger.debug(f"{code}: {e}")

    if not all_feat:
        raise RuntimeError("特徵資料不足")

    X_all = pd.concat(all_feat).sort_index()
    y_all = pd.concat(all_target).sort_index()
    common = X_all.index.intersection(y_all.index)
    X_all, y_all = X_all.loc[common], y_all.loc[common]

    # 3. 以日期切分：最後 period 對應天數作為測試期
    period_to_days = {"1mo": 22, "3mo": 65, "6mo": 130, "1y": 252, "2y": 504}
    test_days = period_to_days.get(period, 22)

    all_dates = sorted(X_all.index.unique())
    if len(all_dates) <= test_days + 10:
        raise RuntimeError(f"日期不足: {len(all_dates)} 天（需要至少 {test_days + 10} 天）")

    # test_start 為最後 test_days 個交易日的起始
    test_start = all_dates[-(test_days)]
    train_end  = all_dates[-(test_days + 1)]

    X_train = X_all.loc[X_all.index <= train_end]
    y_train = y_all.loc[y_all.index <= train_end]
    if len(X_train) < 200:
        raise RuntimeError(f"訓練資料不足: {len(X_train)} rows")

    print(f"🤖 訓練模型 (train={len(X_train)} rows, test≈{test_days}天)…")
    model = TradingModel()
    metrics = model.train(X_train, y_train)
    print(f"   Val F1={metrics['val_macro_f1']:.3f}, Buy F1={metrics['val_buy_f1']:.3f}")

    test_dates = sorted(set(
        idx for code, (feat, _) in ticker_dfs.items()
        for idx in feat.index if idx >= test_start
    ))
    print(f"📊 測試期間: {str(test_start)[:10]} ～ {str(test_dates[-1])[:10]} ({len(test_dates)} 天)")

    # ── 模擬 TA 訊號（測試 combine_signals：50%看多 / 50%看空 隨機）─────────
    import random
    random.seed(42)
    ta_bullish  = {c: "BUY"  for c in list(ticker_dfs.keys())[:len(ticker_dfs)//2]}
    ta_bearish  = {c: "SELL" for c in list(ticker_dfs.keys())[len(ticker_dfs)//2:]}
    ta_all_hold = {c: "HOLD" for c in ticker_dfs}

    print("\n▶ 執行各策略模擬…")

    results = []

    # A. 舊策略：固定停損（用 NORMAL profile 但不用 ATR，即 ATR=0 時用固定停損）
    r_old = simulate(
        "A. 舊策略 (固定停損-7%)",
        model, ticker_dfs, raw_data, test_dates,
        profile=NORMAL, use_combine_signals=False,
        ta_signal_override=ta_all_hold,
        verbose=verbose,
    )
    results.append(r_old)

    # B. 新策略：ATR追蹤 + Strategy B + RiskProfile
    r_new = simulate(
        "B. 新策略 (ATR追蹤 + Strategy B)",
        model, ticker_dfs, raw_data, test_dates,
        profile=NORMAL, use_combine_signals=False,
        verbose=verbose,
    )
    results.append(r_new)

    # C. 新策略 + combine_signals（TA 50%看多）
    r_combined_bull = simulate(
        "C. 新策略 + TA看多",
        model, ticker_dfs, raw_data, test_dates,
        profile=NORMAL, use_combine_signals=True,
        ta_signal_override=ta_bullish,
        verbose=verbose,
    )
    results.append(r_combined_bull)

    # D. 新策略 + combine_signals（TA 50%看空）
    r_combined_bear = simulate(
        "D. 新策略 + TA看空",
        model, ticker_dfs, raw_data, test_dates,
        profile=NORMAL, use_combine_signals=True,
        ta_signal_override=ta_bearish,
        verbose=verbose,
    )
    results.append(r_combined_bear)

    # E. 興櫃風險設定（用一般股票套用興櫃參數，驗證保護機制更嚴格）
    r_emg = simulate(
        "E. 興櫃 RiskProfile (停損-5% ATR×1.5 持倉5天)",
        model, ticker_dfs, raw_data, test_dates,
        profile=EMERGING, use_combine_signals=False,
        verbose=verbose,
    )
    results.append(r_emg)

    return results


# ── HTML Report ───────────────────────────────────────────────────────────────

def generate_report(results: list[BacktestResult], period: str) -> Path:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly 未安裝，跳過圖表")
        return Path(".")

    colors = ["#00d4aa", "#7c83fd", "#ffa500", "#ff6b6b", "#b8f0e6"]
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 策略比較摘要表 ────────────────────────────────────────────────────────
    rows = ""
    for i, r in enumerate(results):
        color = "#00d4aa" if r.total_return_pct >= 0 else "#ff4d4d"
        rows += f"""<tr>
            <td style="color:{colors[i % len(colors)]};font-weight:bold">{r.strategy_name}</td>
            <td style="color:{color}">{r.total_return_pct:+.2f}%</td>
            <td style="color:{color}">{r.annualised_return_pct:+.2f}%</td>
            <td>{r.sharpe_ratio:.2f}</td>
            <td>{r.max_drawdown_pct:.2f}%</td>
            <td>{r.win_rate_pct:.1f}%</td>
            <td>{r.total_trades}</td>
            <td style="color:{color}">${r.final_capital:,.0f}</td>
        </tr>"""

    # ── Chart 1: Equity curves ───────────────────────────────────────────────
    fig_eq = go.Figure()
    for i, r in enumerate(results):
        eq = r.equity_curve
        if not eq.empty:
            fig_eq.add_trace(go.Scatter(
                x=eq.index, y=eq.values,
                mode="lines", name=r.strategy_name,
                line=dict(color=colors[i % len(colors)], width=2),
            ))
    fig_eq.add_hline(y=CAPITAL, line_dash="dash", line_color="gray",
                     annotation_text=f"初始資金 ${CAPITAL:,}")
    fig_eq.update_layout(
        title="📈 各策略資產淨值比較",
        xaxis_title="日期", yaxis_title="資產 (TWD)",
        template="plotly_dark", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # ── Chart 2: Returns bar ─────────────────────────────────────────────────
    fig_ret = go.Figure(go.Bar(
        x=[r.strategy_name for r in results],
        y=[r.total_return_pct for r in results],
        marker_color=[colors[i % len(colors)] for i in range(len(results))],
        text=[f"{r.total_return_pct:+.2f}%" for r in results],
        textposition="outside",
    ))
    fig_ret.update_layout(
        title="📊 各策略總報酬率",
        yaxis_title="報酬率 (%)",
        template="plotly_dark", height=350,
        margin=dict(l=40, r=20, t=60, b=120),
        xaxis=dict(tickangle=-20),
    )

    # ── Chart 3: Exit reasons (grouped) ──────────────────────────────────────
    all_reasons = sorted(set(k for r in results for k in r.reason_counts))
    fig_reason = go.Figure()
    for i, r in enumerate(results):
        fig_reason.add_trace(go.Bar(
            name=r.strategy_name,
            x=all_reasons,
            y=[r.reason_counts.get(k, 0) for k in all_reasons],
            marker_color=colors[i % len(colors)],
        ))
    fig_reason.update_layout(
        title="🚪 出場原因比較",
        barmode="group",
        template="plotly_dark", height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=80, b=40),
    )

    # ── Chart 4: Sharpe / win-rate scatter ───────────────────────────────────
    fig_scatter = go.Figure()
    for i, r in enumerate(results):
        fig_scatter.add_trace(go.Scatter(
            x=[r.win_rate_pct], y=[r.sharpe_ratio],
            mode="markers+text",
            name=r.strategy_name,
            text=[r.strategy_name.split(".")[0]],
            textposition="top center",
            marker=dict(size=15, color=colors[i % len(colors)]),
        ))
    fig_scatter.update_layout(
        title="🎯 夏普比率 vs 勝率",
        xaxis_title="勝率 (%)", yaxis_title="夏普比率",
        template="plotly_dark", height=350,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # ── Chart 5: Drawdown comparison ─────────────────────────────────────────
    fig_dd = go.Figure()
    for i, r in enumerate(results):
        eq = r.equity_curve
        if not eq.empty:
            dd = (eq - eq.cummax()) / eq.cummax() * 100
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values,
                mode="lines", name=r.strategy_name,
                line=dict(color=colors[i % len(colors)], width=1.5),
            ))
    fig_dd.update_layout(
        title="📉 各策略回撤比較",
        xaxis_title="日期", yaxis_title="回撤 (%)",
        template="plotly_dark", height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=80, b=40),
    )

    # ── HTML ─────────────────────────────────────────────────────────────────
    test_dates_str = ""
    for r in results:
        if not r.equity_curve.empty:
            start = str(r.equity_curve.index[0])[:10]
            end   = str(r.equity_curve.index[-1])[:10]
            test_dates_str = f"{start} ～ {end}"
            break

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>回測比較報告 — feature/trading-agents</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:#0f0f1a; color:#e0e0e0; padding:20px; }}
  h1 {{ text-align:center; color:#00d4aa; font-size:1.8em; margin:20px 0 5px; }}
  .subtitle {{ text-align:center; color:#888; margin-bottom:25px; font-size:0.9em; }}
  .chart-box {{ background:#1a1a2e; border-radius:12px; padding:15px; margin-bottom:20px; border:1px solid #2a2a4a; }}
  .charts-2col {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px; }}
  @media(max-width:900px) {{ .charts-2col {{ grid-template-columns:1fr; }} }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85em; }}
  th {{ background:#2a2a4a; padding:10px 12px; text-align:left; color:#00d4aa; }}
  td {{ padding:8px 12px; border-bottom:1px solid #2a2a4a; }}
  tr:hover td {{ background:#252540; }}
  .section-title {{ color:#00d4aa; font-size:1.1em; margin-bottom:12px; padding-bottom:6px; border-bottom:1px solid #2a2a4a; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.75em; margin:2px; }}
  .badge-new {{ background:#1a3a2a; color:#00d4aa; border:1px solid #00d4aa44; }}
  .badge-risk {{ background:#3a1a1a; color:#ff6b6b; border:1px solid #ff6b6b44; }}
  footer {{ text-align:center; color:#444; font-size:0.8em; margin-top:30px; }}
</style>
</head>
<body>
<h1>📊 回測比較報告 — feature/trading-agents</h1>
<p class="subtitle">
  測試期間: {test_dates_str} &nbsp;|&nbsp; 初始資金: ${CAPITAL:,} &nbsp;|&nbsp;
  資料期: {period} &nbsp;|&nbsp; 生成: {generated_at}
</p>

<!-- 新功能說明 -->
<div class="chart-box">
  <div class="section-title">✅ 本分支新功能驗證項目</div>
  <p style="margin-bottom:10px;color:#aaa">以下功能均在回測中驗證:</p>
  <span class="badge badge-new">ATR 追蹤停損 (×2.0)</span>
  <span class="badge badge-new">Strategy B 停利確認</span>
  <span class="badge badge-new">RiskProfile 雙軌參數</span>
  <span class="badge badge-risk">興櫃 EMERGING 設定 (停損-5% ATR×1.5 持倉5天)</span>
  <span class="badge badge-new">combine_signals 合併邏輯</span>
  <span class="badge badge-new">TA看多 vs TA看空影響分析</span>
  <span class="badge badge-new">最大持有天數管理</span>
</div>

<!-- 策略比較表 -->
<div class="chart-box">
  <div class="section-title">🏆 策略績效比較</div>
  <div style="overflow-x:auto">
  <table>
    <thead><tr>
      <th>策略</th><th>總報酬</th><th>年化報酬</th><th>夏普</th>
      <th>最大回撤</th><th>勝率</th><th>交易次數</th><th>期末資金</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
</div>

<!-- Equity curves -->
<div class="chart-box">
  {fig_eq.to_html(full_html=False, include_plotlyjs=True)}
</div>

<!-- Returns + Scatter -->
<div class="charts-2col">
  <div class="chart-box">
    {fig_ret.to_html(full_html=False, include_plotlyjs=False)}
  </div>
  <div class="chart-box">
    {fig_scatter.to_html(full_html=False, include_plotlyjs=False)}
  </div>
</div>

<!-- Drawdown + Exit reasons -->
<div class="charts-2col">
  <div class="chart-box">
    {fig_dd.to_html(full_html=False, include_plotlyjs=False)}
  </div>
  <div class="chart-box">
    {fig_reason.to_html(full_html=False, include_plotlyjs=False)}
  </div>
</div>

<footer>
  ⚠️ 回測僅供驗證策略邏輯，不代表未來實盤績效。
  模擬 TA 訊號（非真實 Ollama 推論）用於測試 combine_signals 合併邏輯。
  生成: {generated_at}
</footer>
</body>
</html>"""

    report_path = Path(__file__).parent.parent / "data" / "backtest_1m_report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="1mo",
                        help="yfinance period: 1mo, 3mo, 6mo (預設 1mo)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    results = run_all(period=args.period, verbose=not args.quiet)

    print("\n" + "=" * 60)
    print("  策略績效摘要")
    print("=" * 60)
    for r in results:
        sign = "+" if r.total_return_pct >= 0 else ""
        print(f"  {r.strategy_name}")
        print(f"    報酬={sign}{r.total_return_pct:.2f}%  夏普={r.sharpe_ratio:.2f}  "
              f"勝率={r.win_rate_pct:.1f}%  交易={r.total_trades}  "
              f"回撤={r.max_drawdown_pct:.2f}%")

    report = generate_report(results, args.period)
    print(f"\n📂 報告已生成: {report}")
    print(f"   瀏覽器開啟: file://{report}")
