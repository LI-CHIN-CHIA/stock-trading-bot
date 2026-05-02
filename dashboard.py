"""
Remote monitoring dashboard for the AI trading bot.
Run with: streamlit run dashboard.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import os
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent))
STATE_FILE = DATA_DIR / "trader_state.json"
LOG_FILE   = DATA_DIR / "trading.log"

st.set_page_config(
    page_title="AI 交易監控",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📈 AI 自動交易監控")
st.caption(f"更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── Load State ────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if not STATE_FILE.exists():
        return {"cash": 20000, "total_value": 20000, "positions": [], "trade_log": [], "model_version": ""}
    with open(STATE_FILE) as f:
        raw = json.load(f)
    # Compute total value
    total = raw.get("cash", 20000)
    positions = []
    for code, h in raw.get("holdings", {}).items():
        positions.append({
            "ticker": code,
            "shares": h.get("shares", 0),
            "entry_price": h.get("entry_price", 0),
            "entry_date": h.get("entry_date", ""),
            "cost": h.get("cost", 0),
        })
        total += h.get("shares", 0) * h.get("entry_price", 0)  # approximate
    return {
        "cash": raw.get("cash", 20000),
        "total_value": total,
        "positions": positions,
        "trade_log": raw.get("trade_log", []),
        "model_version": "",
    }

state = load_state()
INITIAL_CAPITAL = 20_000

# ── Top Metrics ───────────────────────────────────────────────────────────────

total_value = state["total_value"]
cash = state["cash"]
pnl_abs = total_value - INITIAL_CAPITAL
pnl_pct = pnl_abs / INITIAL_CAPITAL * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 總資產", f"${total_value:,.0f}", f"{pnl_pct:+.2f}%")
col2.metric("💵 現金", f"${cash:,.0f}")
col3.metric("📊 持倉數", f"{len(state['positions'])} / 3")
col4.metric("📉 損益", f"${pnl_abs:+,.0f}", f"{pnl_pct:+.2f}%",
            delta_color="normal" if pnl_abs >= 0 else "inverse")

st.divider()

# ── Current Positions ─────────────────────────────────────────────────────────

st.subheader("📋 目前持倉")

if state["positions"]:
    pos_df = pd.DataFrame(state["positions"])
    pos_df = pos_df.rename(columns={
        "ticker": "股票代號",
        "shares": "持有股數",
        "entry_price": "買入價格",
        "entry_date": "買入日期",
        "cost": "買入成本",
    })
    st.dataframe(pos_df, use_container_width=True, hide_index=True)
else:
    st.info("目前無持倉")

st.divider()

# ── Trade Log ─────────────────────────────────────────────────────────────────

st.subheader("📜 交易記錄")

trade_log = state.get("trade_log", [])
if trade_log:
    log_df = pd.DataFrame(reversed(trade_log))
    if "pnl" in log_df.columns:
        log_df["pnl"] = log_df["pnl"].apply(lambda x: f"${x:+,.0f}" if pd.notna(x) else "-")
    if "pnl_pct" in log_df.columns:
        log_df["pnl_pct"] = log_df["pnl_pct"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    if "buy_proba" in log_df.columns:
        log_df["buy_proba"] = log_df["buy_proba"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "-")

    col_rename = {
        "date": "日期", "action": "操作", "ticker": "股票",
        "shares": "股數", "price": "價格",
        "pnl": "損益", "pnl_pct": "損益%",
        "buy_proba": "AI信心", "reason": "原因",
    }
    log_df = log_df.rename(columns={k: v for k, v in col_rename.items() if k in log_df.columns})
    st.dataframe(log_df, use_container_width=True, hide_index=True)
else:
    st.info("尚無交易記錄")

st.divider()

# ── Manual Controls ───────────────────────────────────────────────────────────

st.subheader("⚙️ 手動控制")

col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("🔄 立即執行交易循環", type="primary", use_container_width=True):
        with st.spinner("執行中..."):
            try:
                from trader.bot import TradingBot
                b = TradingBot()
                b.login()
                b.run_cycle()
                st.success("執行完成")
                st.rerun()
            except Exception as e:
                st.error(f"錯誤: {e}")

with col_b:
    if st.button("🤖 重新訓練模型", use_container_width=True):
        with st.spinner("訓練中（約5分鐘）..."):
            try:
                from trader.bot import TradingBot
                b = TradingBot()
                b.retrain()
                st.success("訓練完成")
                st.rerun()
            except Exception as e:
                st.error(f"錯誤: {e}")

with col_c:
    if st.button("🔃 重新整理", use_container_width=True):
        st.rerun()

st.divider()

# ── System Log ────────────────────────────────────────────────────────────────

st.subheader("📡 系統日誌")
if LOG_FILE.exists():
    with open(LOG_FILE) as f:
        lines = f.readlines()
    st.code("".join(lines[-30:]), language=None)
else:
    st.info("日誌檔案不存在")

# Auto-refresh every 60 seconds
st.markdown(
    "<meta http-equiv='refresh' content='60'>",
    unsafe_allow_html=True,
)
