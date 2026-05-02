"""
Run backtest and generate comprehensive HTML report.
Usage: python run_backtest.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

from ai.backtest import CAPITAL, Trade, _get_price, get_symbol
from ai.features import build_feature_matrix, build_target
from ai.model import TradingModel
from ai.backtest import run_backtest
from utils.constants import STOCK_DB, OTC_SYMBOLS

DATA_DIR    = Path(os.getenv("DATA_DIR", Path(__file__).parent))
REPORT_PATH = DATA_DIR / "backtest_report.html"


# ── Extra metrics ──────────────────────────────────────────────────────────────

def sortino_ratio(equity: pd.Series, risk_free: float = 0.02) -> float:
    daily_ret = equity.pct_change().dropna()
    excess = daily_ret - risk_free / 252
    downside = excess[excess < 0].std() * np.sqrt(252)
    ann_ret = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
    return (ann_ret - risk_free) / downside if downside > 0 else 0.0


def calmar_ratio(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    max_dd = abs(dd.min())
    ann_ret = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
    return ann_ret / max_dd if max_dd > 0 else 0.0


def fetch_benchmark(start: str, end: str) -> pd.Series:
    try:
        df = yf.Ticker("0050.TW").history(start=start, end=end)
        if df.empty:
            return pd.Series(dtype=float)
        return df["Close"]
    except Exception:
        return pd.Series(dtype=float)


def run_1y_backtest():
    model = TradingModel()
    print("=" * 55)
    print("  AI 交易策略回測（測試期：1 年）")
    print("=" * 55)
    result = run_backtest(model, period="3y", capital=CAPITAL, verbose=True)
    model.save()
    return result, model


def generate_report(result, model):
    print("\n📝 生成回測報告...")

    equity  = result.equity_curve
    trades  = result.trades

    # ── Derived metrics ───────────────────────────────────────────────────────
    sortino = sortino_ratio(equity)
    calmar  = calmar_ratio(equity)

    avg_win  = np.mean([t.pnl for t in trades if t.pnl > 0]) if result.winning_trades else 0
    avg_loss = np.mean([t.pnl for t in trades if t.pnl <= 0]) if result.losing_trades else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    hold_days = []
    for t in trades:
        try:
            d = (pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days
            hold_days.append(d)
        except Exception:
            pass
    avg_hold = np.mean(hold_days) if hold_days else 0

    # Exit reason breakdown
    reason_counts = {}
    for t in trades:
        key = "停損" if "停損" in t.reason else "停利" if "停利" in t.reason else "AI賣出" if "AI" in t.reason else "到期"
        reason_counts[key] = reason_counts.get(key, 0) + 1

    # Per-stock stats
    stock_stats = {}
    for t in trades:
        if t.ticker not in stock_stats:
            stock_stats[t.ticker] = {"trades": 0, "wins": 0, "total_pnl": 0.0}
        stock_stats[t.ticker]["trades"] += 1
        stock_stats[t.ticker]["wins"] += 1 if t.pnl > 0 else 0
        stock_stats[t.ticker]["total_pnl"] += t.pnl

    # Rolling 20-trade win rate
    pnl_series = pd.Series([1 if t.pnl > 0 else 0 for t in trades])
    rolling_wr = pnl_series.rolling(20, min_periods=5).mean() * 100

    # ── Benchmark 0050 ────────────────────────────────────────────────────────
    test_start = str(equity.index[0])[:10] if not equity.empty else ""
    test_end   = str(equity.index[-1])[:10] if not equity.empty else ""
    bench = fetch_benchmark(test_start, test_end)
    bench_norm = pd.Series(dtype=float)
    if not bench.empty:
        bench_norm = bench / bench.iloc[0] * CAPITAL

    # ── Chart 1: Equity vs Benchmark ─────────────────────────────────────────
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode="lines", name="AI策略",
        line=dict(color="#00d4aa", width=2),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.07)",
    ))
    if not bench_norm.empty:
        fig_eq.add_trace(go.Scatter(
            x=bench_norm.index, y=bench_norm.values,
            mode="lines", name="0050 大盤",
            line=dict(color="#ffa500", width=1.5, dash="dot"),
        ))
    fig_eq.add_hline(y=CAPITAL, line_dash="dash", line_color="gray",
                     annotation_text=f"初始資金 ${CAPITAL:,}")
    fig_eq.update_layout(
        title="📈 資產淨值 vs 大盤（0050）",
        xaxis_title="日期", yaxis_title="資產 (TWD)",
        template="plotly_dark", height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # ── Chart 2: Drawdown ─────────────────────────────────────────────────────
    roll_max  = equity.cummax()
    drawdown  = (equity - roll_max) / roll_max * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        mode="lines", name="回撤 %",
        line=dict(color="#ff4d4d", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,77,77,0.15)",
    ))
    if not bench_norm.empty:
        bench_dd = (bench_norm - bench_norm.cummax()) / bench_norm.cummax() * 100
        fig_dd.add_trace(go.Scatter(
            x=bench_dd.index, y=bench_dd.values,
            mode="lines", name="0050回撤",
            line=dict(color="#ffa500", width=1, dash="dot"),
        ))
    fig_dd.update_layout(
        title="📉 資金回撤", xaxis_title="日期", yaxis_title="回撤 (%)",
        template="plotly_dark", height=250,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    # ── Chart 3: Monthly Heatmap ──────────────────────────────────────────────
    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    months = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]
    years = sorted(monthly_df["year"].unique())
    z = []
    for yr in years:
        row = []
        for mo in range(1, 13):
            val = monthly_df[(monthly_df.year == yr) & (monthly_df.month == mo)]["return"]
            row.append(float(val.iloc[0]) if len(val) > 0 else None)
        z.append(row)
    fig_heat = go.Figure(go.Heatmap(
        z=z, x=months, y=[str(y) for y in years],
        colorscale=[[0,"#ff4d4d"],[0.5,"#1a1a2e"],[1,"#00d4aa"]],
        zmid=0, text=[[f"{v:.1f}%" if v is not None else "" for v in row] for row in z],
        texttemplate="%{text}", colorbar=dict(title="月報酬%"),
    ))
    fig_heat.update_layout(
        title="🗓️ 月報酬熱力圖",
        template="plotly_dark", height=220,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    # ── Chart 4: PnL Bar ──────────────────────────────────────────────────────
    if trades:
        colors = ["#00d4aa" if t.pnl > 0 else "#ff4d4d" for t in trades]
        fig_pnl = go.Figure(go.Bar(
            x=[f"{t.ticker} {t.exit_date}" for t in trades],
            y=[t.pnl for t in trades],
            marker_color=colors,
            text=[f"${t.pnl:+.0f}" for t in trades],
            textposition="outside",
        ))
        fig_pnl.update_layout(
            title="💰 每筆交易損益",
            xaxis_title="交易", yaxis_title="損益 (TWD)",
            template="plotly_dark", height=320,
            margin=dict(l=40, r=20, t=50, b=80),
        )
    else:
        fig_pnl = go.Figure()
        fig_pnl.update_layout(title="無交易記錄", template="plotly_dark", height=320)

    # ── Chart 5: Exit reason pie ───────────────────────────────────────────────
    fig_reason = go.Figure(go.Pie(
        labels=list(reason_counts.keys()),
        values=list(reason_counts.values()),
        hole=0.45,
        marker_colors=["#ff4d4d", "#00d4aa", "#7c83fd", "#ffa500"],
    ))
    fig_reason.update_layout(
        title="🚪 出場原因分佈",
        template="plotly_dark", height=280,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    # ── Chart 6: Holding days histogram ──────────────────────────────────────
    fig_hold = go.Figure(go.Histogram(
        x=hold_days, nbinsx=15,
        marker_color="#7c83fd",
    ))
    fig_hold.update_layout(
        title="📅 持有天數分佈",
        xaxis_title="持有天數", yaxis_title="次數",
        template="plotly_dark", height=280,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    # ── Chart 7: Rolling win rate ─────────────────────────────────────────────
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=list(range(len(rolling_wr))), y=rolling_wr.values,
        mode="lines", name="滾動勝率(20筆)",
        line=dict(color="#7c83fd", width=2),
    ))
    fig_roll.add_hline(y=50, line_dash="dash", line_color="gray",
                       annotation_text="50%基準")
    fig_roll.update_layout(
        title="📊 滾動勝率（每20筆）",
        xaxis_title="交易序號", yaxis_title="勝率 (%)",
        template="plotly_dark", height=280,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    # ── Chart 8: Feature importance ───────────────────────────────────────────
    feat_imp = model.feature_importance().head(20)
    if not feat_imp.empty:
        # 標記三大法人特徵
        colors_imp = ["#ffa500" if any(k in f for k in ["foreign","trust","institution"])
                      else "#7c83fd" for f in feat_imp.index[::-1]]
        fig_imp = go.Figure(go.Bar(
            x=feat_imp.values[::-1], y=feat_imp.index[::-1],
            orientation="h", marker_color=colors_imp,
        ))
        fig_imp.update_layout(
            title="🔍 AI 特徵重要性 Top20（橘色=三大法人）",
            xaxis_title="重要性分數", yaxis_title="特徵",
            template="plotly_dark", height=500,
            margin=dict(l=180, r=20, t=50, b=40),
        )
    else:
        fig_imp = go.Figure()

    # ── Chart 9: Per-stock bar ────────────────────────────────────────────────
    stock_df = pd.DataFrame(stock_stats).T.sort_values("total_pnl", ascending=False)
    stock_df["win_rate"] = stock_df["wins"] / stock_df["trades"] * 100
    fig_stock = go.Figure()
    fig_stock.add_trace(go.Bar(
        name="總損益",
        x=stock_df.index,
        y=stock_df["total_pnl"],
        marker_color=["#00d4aa" if v >= 0 else "#ff4d4d" for v in stock_df["total_pnl"]],
    ))
    fig_stock.update_layout(
        title="🏆 個股損益排行",
        xaxis_title="股票", yaxis_title="損益 (TWD)",
        template="plotly_dark", height=320,
        margin=dict(l=40, r=20, t=50, b=60),
    )

    # ── Trade table ───────────────────────────────────────────────────────────
    if trades:
        trade_rows = "".join([
            f"""<tr class="{'win' if t.pnl > 0 else 'loss'}">
                <td>{t.entry_date}</td>
                <td>{t.exit_date}</td>
                <td><b>{t.ticker}</b><br><small>{STOCK_DB.get(t.ticker,'')}</small></td>
                <td>{t.shares}</td>
                <td>${t.entry_price:.2f}</td>
                <td>${t.exit_price:.2f}</td>
                <td class="{'pos' if t.pnl > 0 else 'neg'}">${t.pnl:+,.0f}</td>
                <td class="{'pos' if t.pnl_pct > 0 else 'neg'}">{t.pnl_pct:.1%}</td>
                <td>{t.reason}</td>
            </tr>"""
            for t in trades
        ])
    else:
        trade_rows = '<tr><td colspan="9" style="text-align:center">無交易記錄</td></tr>'

    # ── Per-stock table ───────────────────────────────────────────────────────
    stock_rows = "".join([
        f"""<tr>
            <td><b>{code}</b><br><small>{STOCK_DB.get(code,'')}</small></td>
            <td>{int(row['trades'])}</td>
            <td>{int(row['wins'])}</td>
            <td>{row['win_rate']:.0f}%</td>
            <td class="{'pos' if row['total_pnl'] >= 0 else 'neg'}">${row['total_pnl']:+,.0f}</td>
        </tr>"""
        for code, row in stock_df.iterrows()
    ])

    # ── Benchmark comparison ──────────────────────────────────────────────────
    bench_return = ""
    if not bench_norm.empty:
        b_ret = (bench_norm.iloc[-1] / bench_norm.iloc[0] - 1) * 100
        alpha = result.annualised_return_pct - b_ret
        bench_return = f"""
        <tr><td><b>0050 大盤報酬</b></td><td style="color:#ffa500">{b_ret:+.2f}%</td></tr>
        <tr><td><b>Alpha（超額報酬）</b></td><td class="{'pos' if alpha >= 0 else 'neg'}">{alpha:+.2f}%</td></tr>
        """

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pnl_color    = "#00d4aa" if result.total_return_pct >= 0 else "#ff4d4d"
    sharpe_color = "#00d4aa" if result.sharpe_ratio >= 1 else "#ffa500" if result.sharpe_ratio >= 0 else "#ff4d4d"

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 交易策略回測報告</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f1a; color: #e0e0e0; padding: 20px; }}
  h1 {{ text-align: center; color: #00d4aa; font-size: 2em; margin: 20px 0 5px; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; font-size: 0.9em; }}
  .metrics {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 25px; }}
  @media(max-width:900px) {{ .metrics {{ grid-template-columns: repeat(3,1fr); }} }}
  @media(max-width:600px) {{ .metrics {{ grid-template-columns: repeat(2,1fr); }} }}
  .card {{ background: #1a1a2e; border-radius: 12px; padding: 16px; text-align: center; border: 1px solid #2a2a4a; }}
  .card .label {{ font-size: 0.75em; color: #888; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; }}
  .card .value {{ font-size: 1.6em; font-weight: bold; }}
  .card .sub {{ font-size: 0.75em; color: #666; margin-top: 4px; }}
  .chart-box {{ background: #1a1a2e; border-radius: 12px; padding: 15px; margin-bottom: 20px; border: 1px solid #2a2a4a; }}
  .charts-2col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .charts-3col {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  @media(max-width:900px) {{ .charts-2col, .charts-3col {{ grid-template-columns: 1fr; }} }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
  th {{ background: #2a2a4a; padding: 10px 12px; text-align: left; color: #00d4aa; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #2a2a4a; vertical-align: middle; }}
  tr.win td {{ border-left: 3px solid #00d4aa; }}
  tr.loss td {{ border-left: 3px solid #ff4d4d; }}
  tr:hover td {{ background: #252540; }}
  .pos {{ color: #00d4aa; font-weight: bold; }}
  .neg {{ color: #ff4d4d; font-weight: bold; }}
  .section-title {{ color: #00d4aa; font-size: 1.1em; margin-bottom: 12px; padding-bottom: 6px; border-bottom: 1px solid #2a2a4a; }}
  footer {{ text-align: center; color: #444; font-size: 0.8em; margin-top: 30px; }}
  small {{ color: #888; }}
</style>
</head>
<body>

<h1>📈 AI 自動交易策略回測報告</h1>
<p class="subtitle">測試期間: {test_start} ～ {test_end} &nbsp;|&nbsp; 生成時間: {generated_at}</p>

<!-- Key Metrics Row 1 -->
<div class="metrics">
  <div class="card">
    <div class="label">總報酬率</div>
    <div class="value" style="color:{pnl_color}">{result.total_return_pct:+.2f}%</div>
    <div class="sub">初始 ${CAPITAL:,}</div>
  </div>
  <div class="card">
    <div class="label">年化報酬率</div>
    <div class="value" style="color:{pnl_color}">{result.annualised_return_pct:+.2f}%</div>
    <div class="sub">年化計算</div>
  </div>
  <div class="card">
    <div class="label">夏普比率</div>
    <div class="value" style="color:{sharpe_color}">{result.sharpe_ratio:.2f}</div>
    <div class="sub">&gt;1 為佳</div>
  </div>
  <div class="card">
    <div class="label">Sortino 比率</div>
    <div class="value" style="color:{'#00d4aa' if sortino >= 1 else '#ffa500'}">{sortino:.2f}</div>
    <div class="sub">下行風險調整</div>
  </div>
  <div class="card">
    <div class="label">Calmar 比率</div>
    <div class="value" style="color:{'#00d4aa' if calmar >= 1 else '#ffa500'}">{calmar:.2f}</div>
    <div class="sub">報酬/最大回撤</div>
  </div>
  <div class="card">
    <div class="label">最大回撤</div>
    <div class="value" style="color:#ffa500">{result.max_drawdown_pct:.2f}%</div>
    <div class="sub">最大虧損幅度</div>
  </div>
  <div class="card">
    <div class="label">勝率</div>
    <div class="value" style="color:#7c83fd">{result.win_rate_pct:.1f}%</div>
    <div class="sub">{result.winning_trades}勝 / {result.losing_trades}負</div>
  </div>
  <div class="card">
    <div class="label">獲利因子</div>
    <div class="value" style="color:{'#00d4aa' if profit_factor >= 1.5 else '#ffa500'}">{profit_factor:.2f}</div>
    <div class="sub">平均盈/平均虧</div>
  </div>
  <div class="card">
    <div class="label">總交易次數</div>
    <div class="value">{result.total_trades}</div>
    <div class="sub">平均持有 {avg_hold:.1f} 天</div>
  </div>
  <div class="card">
    <div class="label">期末資金</div>
    <div class="value" style="color:{pnl_color}">${result.final_capital:,.0f}</div>
    <div class="sub">損益 ${result.final_capital - CAPITAL:+,.0f}</div>
  </div>
</div>

<!-- Equity Curve -->
<div class="chart-box">
  {fig_eq.to_html(full_html=False, include_plotlyjs=True)}
</div>

<!-- Drawdown -->
<div class="chart-box">
  {fig_dd.to_html(full_html=False, include_plotlyjs=False)}
</div>

<!-- Monthly + PnL -->
<div class="charts-2col">
  <div class="chart-box">
    {fig_heat.to_html(full_html=False, include_plotlyjs=False)}
  </div>
  <div class="chart-box">
    {fig_pnl.to_html(full_html=False, include_plotlyjs=False)}
  </div>
</div>

<!-- Exit reason + Hold days + Rolling WR -->
<div class="charts-3col">
  <div class="chart-box">
    {fig_reason.to_html(full_html=False, include_plotlyjs=False)}
  </div>
  <div class="chart-box">
    {fig_hold.to_html(full_html=False, include_plotlyjs=False)}
  </div>
  <div class="chart-box">
    {fig_roll.to_html(full_html=False, include_plotlyjs=False)}
  </div>
</div>

<!-- Per-stock bar + Feature importance -->
<div class="charts-2col">
  <div class="chart-box">
    {fig_stock.to_html(full_html=False, include_plotlyjs=False)}
  </div>
  <div class="chart-box">
    {fig_imp.to_html(full_html=False, include_plotlyjs=False)}
  </div>
</div>

<!-- Per-stock table + Strategy params -->
<div class="charts-2col">
  <div class="chart-box">
    <div class="section-title">🏆 個股績效明細</div>
    <table>
      <thead><tr><th>股票</th><th>交易次數</th><th>獲勝</th><th>勝率</th><th>總損益</th></tr></thead>
      <tbody>{stock_rows}</tbody>
    </table>
  </div>
  <div class="chart-box">
    <div class="section-title">⚙️ 策略摘要</div>
    <table>
      <tr><td><b>AI 模型</b></td><td>LightGBM 三分類（買/持/賣）</td></tr>
      <tr><td><b>特徵數量</b></td><td>57 個（技術指標 + 三大法人）</td></tr>
      <tr><td><b>買入條件</b></td><td>AI 買入機率 ≥ 52%</td></tr>
      <tr><td><b>停損</b></td><td>-7%</td></tr>
      <tr><td><b>停利</b></td><td>+10%</td></tr>
      <tr><td><b>最大持有</b></td><td>10 個交易日</td></tr>
      <tr><td><b>最大持倉</b></td><td>5 支</td></tr>
      <tr><td><b>交易成本</b></td><td>0.1425% + 賣出 0.3%</td></tr>
      {bench_return}
    </table>
  </div>
</div>

<!-- Full trade log -->
<div class="chart-box">
  <div class="section-title">📋 交易明細（共 {result.total_trades} 筆）</div>
  <div style="overflow-x:auto">
  <table>
    <thead>
      <tr>
        <th>買入日</th><th>賣出日</th><th>股票</th><th>股數</th>
        <th>買入價</th><th>賣出價</th><th>損益(元)</th><th>損益%</th><th>出場原因</th>
      </tr>
    </thead>
    <tbody>{trade_rows}</tbody>
  </table>
  </div>
</div>

<footer>
  ⚠️ 本報告為歷史回測結果，過去績效不代表未來表現。投資有風險，請謹慎評估。<br>
  生成時間: {generated_at}
</footer>

</body>
</html>"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ 報告已生成: {REPORT_PATH}")
    return REPORT_PATH


if __name__ == "__main__":
    result, model = run_1y_backtest()
    report_path = generate_report(result, model)
    print(f"\n📂 用瀏覽器開啟: file://{report_path}")
