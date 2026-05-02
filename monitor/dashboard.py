"""
Remote monitoring dashboard for the trading bot.
Run on port 8502: streamlit run monitor/dashboard.py --server.port 8502
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="交易機器人監控",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp { background-color: #131722; color: #d1d4dc; }
.metric-card {
    background: #1e2030;
    border-radius: 8px;
    padding: 12px 16px;
    border: 1px solid #2a2e39;
}
.badge-live   { background: #1b5e20; color: #a5d6a7; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }
.badge-dry    { background: #e65100; color: #ffe0b2; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }
.badge-buy    { color: #00e676; font-weight: bold; }
.badge-sell   { color: #ff1744; font-weight: bold; }
.badge-hold   { color: #90a4ae; }
div[data-testid="stSidebarContent"] { background-color: #1e2030; }
</style>
""", unsafe_allow_html=True)

# ── Imports (after sys.path setup) ───────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import StockDataFetcher
from data.state import StateStore
from trading.risk_manager import TOTAL_BUDGET

_state = StateStore()
_fetcher = StockDataFetcher()


@st.cache_data(ttl=25, show_spinner=False)
def _quote(ticker: str) -> dict:
    return _fetcher.fetch_realtime_quote(ticker)


def _fmt_pnl(val: float) -> str:
    color = "#00e676" if val >= 0 else "#ff1744"
    sign = "+" if val >= 0 else ""
    return f'<span style="color:{color};font-weight:bold">{sign}{val:.0f}</span>'


def _fmt_pct(val: float) -> str:
    color = "#00e676" if val >= 0 else "#ff1744"
    sign = "+" if val >= 0 else ""
    return f'<span style="color:{color}">{sign}{val:.1f}%</span>'


# ── Main dashboard ────────────────────────────────────────────────────────────

def main():
    # Header
    meta = _state.get_meta()
    positions = _state.get_positions()
    trades = _state.get_trade_history(200)

    # Mode badge
    any_live = any(not t.get("dry_run", True) for t in trades[-5:]) if trades else False
    mode_html = '<span class="badge-live">● LIVE</span>' if any_live else '<span class="badge-dry">● DRY RUN</span>'

    st.markdown(f"## 🤖 交易機器人監控 {mode_html}", unsafe_allow_html=True)
    st.markdown(f"上次更新: {datetime.now().strftime('%H:%M:%S')} | 自動每30秒重整")
    st.divider()

    # ── Section 1: Portfolio summary ─────────────────────────────────────────
    total_cost = _state.get_total_cost()
    available = max(0.0, TOTAL_BUDGET - total_cost)
    win_rate = _state.get_win_rate()

    # Compute unrealized P&L
    total_unrealized = 0.0
    pos_with_prices = {}
    for ticker, pos in positions.items():
        q = _quote(ticker)
        price = q.get("price", pos["entry_price"])
        pnl_data = {
            "current_price": price,
            "unrealized_pnl": round((price - pos["entry_price"]) * pos["shares"], 2),
            "unrealized_pct": round((price / pos["entry_price"] - 1) * 100, 2) if pos["entry_price"] else 0,
        }
        pos_with_prices[ticker] = {**pos, **pnl_data}
        total_unrealized += pnl_data["unrealized_pnl"]

    realized_pnl = sum(t.get("pnl", 0) or 0 for t in trades if t.get("action") == "SELL")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("總預算", f"NT${TOTAL_BUDGET:,.0f}")
    with c2:
        st.metric("已投入", f"NT${total_cost:,.0f}", delta=f"-{total_cost:,.0f}" if total_cost else None)
    with c3:
        st.metric("未實現損益", f"NT${total_unrealized:+,.0f}",
                  delta=f"{total_unrealized/total_cost*100:+.1f}%" if total_cost else None,
                  delta_color="normal")
    with c4:
        st.metric("已實現損益", f"NT${realized_pnl:+,.0f}", delta_color="normal")
    with c5:
        st.metric("勝率", f"{win_rate:.0f}%" if win_rate else "—",
                  help="已平倉交易的獲利比例")

    st.divider()

    # ── Section 2: Open positions ─────────────────────────────────────────────
    st.subheader(f"持倉 ({len(positions)})")

    if pos_with_prices:
        rows = []
        for ticker, pos in pos_with_prices.items():
            rows.append({
                "股票": f"{ticker} {pos.get('name', '')}",
                "股數": pos["shares"],
                "買入價": f"{pos['entry_price']:.2f}",
                "現價": f"{pos['current_price']:.2f}",
                "未實現損益": pos["unrealized_pnl"],
                "損益%": pos["unrealized_pct"],
                "停損": f"{pos['stop_loss']:.2f}",
                "停利": f"{pos['take_profit']:.2f}",
                "AI機率": f"{pos.get('ai_buy_proba', 0):.0%}",
                "規則分數": f"{pos.get('rule_score', 0):+d}",
                "買入日期": pos.get("entry_date", "")[:10],
            })

        df_pos = pd.DataFrame(rows)

        # Style P&L columns
        def color_pnl(val):
            color = "#00e676" if val > 0 else "#ff1744" if val < 0 else "#d1d4dc"
            return f"color: {color}; font-weight: bold"

        styled = df_pos.style.applymap(color_pnl, subset=["未實現損益", "損益%"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Horizontal P&L bar chart
        fig = go.Figure(go.Bar(
            x=[r["未實現損益"] for r in rows],
            y=[r["股票"] for r in rows],
            orientation="h",
            marker_color=["#00e676" if r["未實現損益"] >= 0 else "#ff1744" for r in rows],
            text=[f"{r['損益%']:+.1f}%" for r in rows],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="#131722",
            plot_bgcolor="#1e2030",
            font_color="#d1d4dc",
            height=max(200, len(rows) * 45),
            margin=dict(l=0, r=60, t=10, b=10),
            xaxis_title="未實現損益 (TWD)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("目前無持倉")

    st.divider()

    # ── Section 3: Trade history ──────────────────────────────────────────────
    st.subheader("交易紀錄")

    if trades:
        recent = list(reversed(trades[-50:]))
        df_trades = pd.DataFrame(recent)

        def action_color(val):
            if val == "BUY":
                return "color: #00e676; font-weight: bold"
            elif val == "SELL":
                return "color: #ff1744; font-weight: bold"
            return ""

        display_cols = ["timestamp", "ticker", "name", "action", "shares",
                        "price", "amount", "pnl", "reason", "dry_run"]
        existing = [c for c in display_cols if c in df_trades.columns]
        df_show = df_trades[existing].copy()

        if "timestamp" in df_show:
            df_show["timestamp"] = df_show["timestamp"].astype(str).str[:16]
        if "pnl" in df_show:
            df_show["pnl"] = df_show["pnl"].apply(
                lambda x: f"{x:+.0f}" if x is not None and x == x else "—"
            )

        styled = df_show.style.applymap(action_color, subset=["action"] if "action" in df_show.columns else [])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Cumulative P&L chart
        sell_trades = [t for t in trades if t.get("action") == "SELL" and t.get("pnl") is not None]
        if sell_trades:
            dates = [t["timestamp"][:10] for t in sell_trades]
            pnls = [t.get("pnl", 0) or 0 for t in sell_trades]
            cum_pnl = pd.Series(pnls, index=dates).cumsum()

            fig2 = go.Figure(go.Scatter(
                x=cum_pnl.index,
                y=cum_pnl.values,
                mode="lines+markers",
                fill="tozeroy",
                line_color="#00e676" if cum_pnl.iloc[-1] >= 0 else "#ff1744",
                fillcolor="rgba(0,230,118,0.1)" if cum_pnl.iloc[-1] >= 0 else "rgba(255,23,68,0.1)",
                name="累積損益",
            ))
            fig2.update_layout(
                paper_bgcolor="#131722",
                plot_bgcolor="#1e2030",
                font_color="#d1d4dc",
                height=250,
                margin=dict(l=0, r=0, t=10, b=10),
                yaxis_title="累積損益 (TWD)",
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("尚無交易紀錄")

    st.divider()

    # ── Section 4: AI model info ──────────────────────────────────────────────
    st.subheader("AI 模型狀態")

    col1, col2 = st.columns([1, 2])
    with col1:
        last_retrain = meta.get("last_retrain", "未訓練")
        model_version = meta.get("model_version", "—")
        training_metrics = meta.get("training_metrics", {})

        st.markdown(f"**模型版本:** {model_version}")
        st.markdown(f"**上次訓練:** {str(last_retrain)[:16] if last_retrain else '未訓練'}")
        if training_metrics:
            st.markdown(f"**驗證 F1:** {training_metrics.get('val_macro_f1', '—')}")
            st.markdown(f"**買入 F1:** {training_metrics.get('val_buy_f1', '—')}")
            st.markdown(f"**訓練樣本:** {training_metrics.get('train_samples', '—')}")
            dist = training_metrics.get("class_distribution", {})
            if dist:
                st.markdown(f"**資料分布:** BUY={dist.get('BUY',0)}, HOLD={dist.get('HOLD',0)}, SELL={dist.get('SELL',0)}")

    with col2:
        # Feature importance from saved model
        try:
            from ai.model import TradingModel
            m = TradingModel()
            if m.load("latest"):
                imp = m.feature_importance().head(15)
                if not imp.empty:
                    fig3 = go.Figure(go.Bar(
                        x=imp.values,
                        y=imp.index,
                        orientation="h",
                        marker_color="#42a5f5",
                    ))
                    fig3.update_layout(
                        paper_bgcolor="#131722",
                        plot_bgcolor="#1e2030",
                        font_color="#d1d4dc",
                        height=350,
                        margin=dict(l=0, r=0, t=10, b=10),
                        xaxis_title="Gain",
                        title="Top 15 特徵重要性",
                    )
                    st.plotly_chart(fig3, use_container_width=True)
        except Exception:
            st.info("模型尚未訓練，今日14:35後將進行首次訓練")

    st.divider()

    # ── Section 5: Last scan results ─────────────────────────────────────────
    st.subheader("最新掃描結果")

    scan_results = meta.get("last_scan_results", [])
    last_scan = meta.get("last_scan", "")
    if last_scan:
        st.caption(f"掃描時間: {str(last_scan)[:16]}")

    if scan_results:
        df_scan = pd.DataFrame(scan_results)
        if "ai_buy_proba" in df_scan.columns:
            df_scan["ai_buy_proba"] = df_scan["ai_buy_proba"].apply(lambda x: f"{x:.0%}")
        if "rule_score" in df_scan.columns:
            df_scan["rule_score"] = df_scan["rule_score"].apply(lambda x: f"{x:+d}")

        def highlight_buy(row):
            ai_sig = row.get("ai_signal", "")
            if ai_sig == "BUY":
                return ["background-color: #1b5e20"] * len(row)
            return [""] * len(row)

        styled_scan = df_scan.style.apply(highlight_buy, axis=1)
        st.dataframe(styled_scan, use_container_width=True, hide_index=True)
    else:
        st.info("尚無掃描資料 (每天 09:10 進行掃描)")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    time.sleep(30)
    st.rerun()


if __name__ == "__main__":
    main()
