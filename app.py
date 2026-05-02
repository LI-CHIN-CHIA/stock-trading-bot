import logging

import pandas as pd
import streamlit as st

from analysis.indicators import TechnicalIndicators
from analysis.signals import AnalysisResult, SignalEngine
from charts.candlestick import KLineChart
from data.fetcher import StockDataFetcher
from utils.constants import INTRADAY_PERIODS, PERIOD_OPTIONS, STOCK_DB

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="台股K線分析 AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #131722; color: #d1d4dc; }
    .metric-card {
        background: #1e2030;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #2a2e39;
    }
    .signal-buy  { color: #00e676; font-weight: bold; }
    .signal-sell { color: #ff1744; font-weight: bold; }
    .signal-neutral { color: #90a4ae; font-weight: bold; }
    .badge-strong  { background: #1b5e20; color: #a5d6a7; padding: 2px 8px; border-radius: 4px; }
    .badge-moderate { background: #1565c0; color: #90caf9; padding: 2px 8px; border-radius: 4px; }
    .badge-weak    { background: #4a148c; color: #ce93d8; padding: 2px 8px; border-radius: 4px; }
    .badge-neutral { background: #37474f; color: #90a4ae; padding: 2px 8px; border-radius: 4px; }
    div[data-testid="stSidebarContent"] { background-color: #1e2030; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)   # 5-min TTL for intraday; 15-min otherwise
def load_data(ticker_code: str, period: str):
    """
    Cached data loading.
    Returns (df_raw, df_indicators, signal_df, quote, analysis_result, stock_name, is_intraday).
    """
    fetcher = StockDataFetcher()
    engine = SignalEngine()
    is_intraday = period in INTRADAY_PERIODS

    df_raw = fetcher.fetch_historical(ticker_code, period=period)
    df_indicators = TechnicalIndicators.add_all(df_raw)

    # Historical signals need at least 62 bars (MA60 warmup); skip for intraday
    if not is_intraday and len(df_indicators) >= 62:
        signal_df = engine.generate_historical_signals(df_indicators, ticker_code)
    else:
        signal_df = pd.DataFrame()

    quote = fetcher.fetch_realtime_quote(ticker_code)
    result = engine.analyze(df_indicators, ticker_code)
    stock_name = fetcher.get_stock_name(ticker_code)

    return df_raw, df_indicators, signal_df, quote, result, stock_name, is_intraday


# ─────────────────────────────────────────────────────────────────────────────
# STOCK SEARCH
# ─────────────────────────────────────────────────────────────────────────────
def search_stocks(query: str) -> list[tuple[str, str]]:
    """
    Search STOCK_DB by code or name (Chinese/English).
    Returns list of (code, name) tuples, max 10 results.
    """
    q = query.strip().lower()
    if not q:
        return []
    results = []
    for code, name in STOCK_DB.items():
        if q in code.lower() or q in name.lower():
            results.append((code, name))
    return results[:10]


def resolve_ticker_from_input(raw: str) -> tuple[str, str]:
    """
    Given user's raw input (code or name), return (ticker_code, display_name).
    - If raw matches a code exactly → return that code
    - If raw matches exactly one search result → use it
    - Otherwise → treat raw as a bare code and let fetcher resolve it
    """
    raw = raw.strip()
    if raw in STOCK_DB:
        return raw, STOCK_DB[raw]
    matches = search_stocks(raw)
    if len(matches) == 1:
        return matches[0][0], matches[0][1]
    return raw.upper(), raw.upper()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    st.sidebar.title("📈 台股分析設定")
    st.sidebar.markdown("---")

    st.sidebar.subheader("股票搜尋")

    search_input = st.sidebar.text_input(
        "輸入代號或公司名稱",
        value="",
        placeholder="例如: 2330、台積電、TSMC、0050",
        key="stock_search",
    )

    # Show matched results as selectable options
    ticker_code = ""
    if search_input.strip():
        matches = search_stocks(search_input)
        if matches:
            options = [f"{code}  {name}" for code, name in matches]
            chosen = st.sidebar.selectbox(
                f"搜尋結果（{len(matches)} 筆）",
                options,
                key="search_results",
            )
            ticker_code = chosen.split()[0]
        else:
            # Treat raw input as a stock code directly
            ticker_code = search_input.strip().upper()
            st.sidebar.caption(f"未在資料庫找到，將直接查詢代號: **{ticker_code}**")
    else:
        # Default to 2330 on first load
        ticker_code = st.session_state.get("loaded_ticker") or "2330"
        st.sidebar.caption(f"目前查詢: **{ticker_code}**")

    analyze_btn = st.sidebar.button("🔍 開始分析", type="primary", use_container_width=True)

    st.sidebar.markdown("---")

    # Period selection
    st.sidebar.subheader("資料期間")
    selected_period_label = st.sidebar.radio(
        "選擇期間",
        list(PERIOD_OPTIONS.keys()),
        index=2,  # default: 6個月
        horizontal=False,
    )
    period = PERIOD_OPTIONS[selected_period_label]

    st.sidebar.markdown("---")

    # Display options
    st.sidebar.subheader("顯示設定")

    show_ma_options = st.sidebar.multiselect(
        "顯示均線 (MA)",
        options=[5, 10, 20, 60],
        default=[5, 20, 60],
        format_func=lambda x: f"MA{x}",
    )

    show_bb = st.sidebar.checkbox("布林通道 (Bollinger Bands)", value=True)
    show_signals = st.sidebar.checkbox("買賣訊號標記 (Signal Markers)", value=True)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "📡 資料來源: Yahoo Finance\n\n"
        "⏱ 快取時間: 15分鐘\n\n"
        "⚠️ 僅供參考，不構成投資建議"
    )

    return {
        "ticker": ticker_code,
        "period": period,
        "period_label": selected_period_label,
        "show_ma": show_ma_options,
        "show_bb": show_bb,
        "show_signals": show_signals,
        "analyze_clicked": analyze_btn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HEADER METRICS
# ─────────────────────────────────────────────────────────────────────────────
def render_header_metrics(
    stock_name: str,
    ticker: str,
    quote: dict,
    result: AnalysisResult,
) -> None:
    price = quote.get("price", 0)
    change = quote.get("change", 0)
    pct = quote.get("pct_change", 0)
    volume = quote.get("volume", 0)

    direction = result.direction
    confidence = result.confidence

    # Color coding
    if "BUY" in direction:
        dir_color = "#00e676"
        dir_class = "signal-buy"
    elif "SELL" in direction:
        dir_color = "#ff1744"
        dir_class = "signal-sell"
    else:
        dir_color = "#90a4ae"
        dir_class = "signal-neutral"

    conf_badge_class = {
        "強力": "badge-strong",
        "中度": "badge-moderate",
        "弱": "badge-weak",
        "觀望": "badge-neutral",
    }.get(confidence, "badge-neutral")

    st.markdown(f"## {stock_name} ({ticker})")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "現價 Price",
            f"NT$ {price:,.2f}" if price else "N/A",
            delta=f"{change:+.2f}" if change else None,
        )
    with col2:
        pct_color = "normal" if pct == 0 else ("inverse" if pct < 0 else "normal")
        st.metric("漲跌幅 Change%", f"{pct:+.2f}%" if pct else "N/A")
    with col3:
        vol_display = f"{volume / 1000:.0f}K" if volume > 1000 else str(volume)
        st.metric("成交量 Volume", vol_display if volume else "N/A")
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <small>AI建議</small><br>
            <span class="{dir_class}" style="font-size:1.1em">{direction}</span>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <small>信心度 | 分數</small><br>
            <span class="{conf_badge_class}">{confidence}</span>
            <span style="color:#d1d4dc; margin-left:8px; font-size:1.1em">
                {result.total_score:+d}
            </span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PANEL
# ─────────────────────────────────────────────────────────────────────────────
def render_signal_panel(result: AnalysisResult) -> None:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📊 觸發訊號 (Active Signals)")
        if not result.signals:
            st.info("目前無明確買賣訊號")
        else:
            for sig in sorted(result.signals, key=lambda x: abs(x.score), reverse=True):
                if sig.score > 0:
                    icon = "🟢"
                    color = "#00e676"
                elif sig.score < 0:
                    icon = "🔴"
                    color = "#ff1744"
                else:
                    icon = "⚪"
                    color = "#90a4ae"

                with st.expander(
                    f"{icon} [{sig.score:+d}] {sig.source}: {sig.reason[:50]}...",
                    expanded=False,
                ):
                    st.markdown(
                        f"<span style='color:{color}'>{sig.reason}</span>",
                        unsafe_allow_html=True,
                    )

        # Score bar
        st.markdown("**信號強度 Signal Strength**")
        max_score = 16
        normalized = (result.total_score + max_score) / (2 * max_score)
        st.progress(max(0.0, min(1.0, normalized)))
        st.caption(f"總分: {result.total_score:+d} (範圍: -16 ~ +16)")

    with col_right:
        st.subheader("💡 綜合建議 (Recommendation)")

        direction = result.direction
        confidence = result.confidence

        if "BUY" in direction:
            box_color = "#1b5e20"
            text_color = "#a5d6a7"
            emoji = "📈"
        elif "SELL" in direction:
            box_color = "#b71c1c"
            text_color = "#ef9a9a"
            emoji = "📉"
        else:
            box_color = "#37474f"
            text_color = "#90a4ae"
            emoji = "⏸️"

        st.markdown(
            f"""
            <div style="
                background: {box_color};
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                margin-bottom: 16px;
            ">
                <div style="font-size: 2.5em">{emoji}</div>
                <div style="color: {text_color}; font-size: 1.4em; font-weight: bold; margin: 8px 0">
                    {direction}
                </div>
                <div style="color: {text_color}; font-size: 1em">
                    信心度: {confidence}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Indicator summary table
        ind = result.indicators
        rows = []
        if not _isnan(ind.get("RSI")):
            rows.append({"指標": "RSI", "數值": f"{ind['RSI']:.1f}",
                         "狀態": "超賣" if ind["RSI"] < 30 else "超買" if ind["RSI"] > 70 else "正常"})
        if not _isnan(ind.get("MACD")):
            rows.append({"指標": "MACD", "數值": f"{ind['MACD']:.4f}",
                         "狀態": "上升" if ind["MACD"] > 0 else "下降"})
        if not _isnan(ind.get("MA20")) and not _isnan(ind.get("BB_Pct")):
            rows.append({"指標": "BB %B", "數值": f"{ind['BB_Pct']:.2f}",
                         "狀態": "超賣" if ind["BB_Pct"] < 0 else "超買" if ind["BB_Pct"] > 1 else "正常"})

        if rows:
            df_ind = pd.DataFrame(rows)
            st.dataframe(df_ind, use_container_width=True, hide_index=True)

        st.warning("⚠️ 本分析僅供參考，不構成任何投資建議。投資有風險，請自行判斷。")


def _isnan(val) -> bool:
    import math
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return True


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR TABLE TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_indicator_tab(df: pd.DataFrame) -> None:
    st.subheader("最新技術指標數值")

    # Show last 30 rows with key indicator columns
    cols_to_show = [
        c for c in ["Close", "RSI", "MACD", "MACD_Signal", "MACD_Hist",
                    "BB_Upper", "BB_Mid", "BB_Lower", "BB_Pct",
                    "MA5", "MA10", "MA20", "MA60", "Volume_Ratio"]
        if c in df.columns
    ]

    display_df = df[cols_to_show].tail(30).copy()

    # Format index: keep HH:MM for intraday (avoids duplicate date strings)
    has_time = df.index.normalize().nunique() < len(df.index)  # True if intraday
    if has_time:
        display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M")
    else:
        display_df.index = display_df.index.strftime("%Y-%m-%d")

    display_df = display_df.round(2)

    # Reset index to avoid Styler non-unique index error
    # The index name may be "Date", "Datetime", or None → rename to "時間"
    idx_name = display_df.index.name or "index"
    display_df = display_df.reset_index()
    display_df = display_df.rename(columns={idx_name: "時間"})

    def _color_rsi(val):
        try:
            v = float(val)
            if v > 70:
                return "color: #ff1744"
            elif v < 30:
                return "color: #00e676"
        except (TypeError, ValueError):
            pass
        return ""

    styled = display_df.style
    if "RSI" in cols_to_show:
        styled = styled.applymap(_color_rsi, subset=["RSI"])

    st.dataframe(styled, use_container_width=True, height=500, hide_index=True)

    # Summary statistics
    st.subheader("統計摘要")
    summary_cols = [c for c in ["Close", "RSI", "MACD", "BB_Pct", "Volume_Ratio"] if c in df.columns]
    st.dataframe(df[summary_cols].describe().round(2), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL HISTORY TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_signal_history_tab(signal_df: pd.DataFrame) -> None:
    st.subheader("歷史買賣訊號 (Signal History)")

    if signal_df.empty:
        st.info("資料不足，無法產生歷史訊號（需要至少 65 個交易日）")
        return

    # Filter meaningful signals
    active = signal_df[signal_df["total_score"].abs() >= 2].copy()
    active.index = active.index.strftime("%Y-%m-%d")
    active = active.sort_index(ascending=False)

    if active.empty:
        st.info("歷史期間內無明確買賣訊號")
        return

    # Color-coded display
    def color_direction(val: str) -> str:
        if "BUY" in val:
            return "color: #00e676"
        elif "SELL" in val:
            return "color: #ff1744"
        return "color: #90a4ae"

    def color_score(val: int) -> str:
        if val > 0:
            return "color: #00e676"
        elif val < 0:
            return "color: #ff1744"
        return ""

    display = active[["Close", "total_score", "direction", "confidence"]].copy()
    idx_name = display.index.name or "index"
    display = display.reset_index()
    display = display.rename(columns={idx_name: "日期"})

    st.dataframe(
        display.style
        .applymap(color_direction, subset=["方向"])
        .applymap(color_score, subset=["訊號分數"]),
        use_container_width=True,
        height=500,
        hide_index=True,
    )
    st.caption(f"共 {len(active)} 個有效訊號 (分數 |score| >= 2)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    config = render_sidebar()

    # Initialize session state
    if "loaded_ticker" not in st.session_state:
        st.session_state.loaded_ticker = None
    if "loaded_period" not in st.session_state:
        st.session_state.loaded_period = None
    if "data" not in st.session_state:
        st.session_state.data = None

    ticker = config["ticker"]
    period = config["period"]

    # Auto-load on first visit or load when button clicked
    should_load = (
        config["analyze_clicked"]
        or st.session_state.loaded_ticker is None
        or st.session_state.loaded_ticker != ticker
        or st.session_state.loaded_period != period
    )

    if should_load and ticker:
        with st.spinner(f"📡 正在載入 {ticker} 資料... Loading data"):
            try:
                data = load_data(ticker, period)
                st.session_state.data = data
                st.session_state.loaded_ticker = ticker
                st.session_state.loaded_period = period
            except ValueError as e:
                st.error(f"❌ {e}")
                st.info("請確認:\n- 股票代號是否正確\n- 台股代號如: 2330, 0050, 2317\n- OTC股票: 6547, 3711")
                st.session_state.data = None
            except Exception as e:
                st.error(f"❌ 載入失敗: {e}")
                st.info("可能是網路問題或 Yahoo Finance 暫時無法連線，請稍後再試。")
                st.session_state.data = None

    # Render content if data is available
    if st.session_state.data is not None:
        df_raw, df_indicators, signal_df, quote, result, stock_name, is_intraday = st.session_state.data

        render_header_metrics(stock_name, ticker, quote, result)
        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 K線圖 Chart",
            "📈 技術指標 Indicators",
            "🔔 訊號歷史 Signal History",
            "💡 買賣說明 Analysis",
        ])

        chart = KLineChart()

        with tab1:
            if is_intraday:
                interval_label = "5分鐘" if period == "1d" else "1小時"
                st.info(
                    f"⚡ 盤中模式（{interval_label}K棒）｜"
                    f"MA60 及歷史訊號僅在日線（1個月以上）可用｜"
                    f"資料每 5 分鐘更新"
                )

            fig = chart.build_full_chart(
                df=df_indicators,
                ticker=f"{stock_name} ({ticker})",
                show_ma=config["show_ma"],
                show_bb=config["show_bb"],
                show_signals=config["show_signals"] and not is_intraday,
                signal_df=signal_df if not signal_df.empty else None,
                is_intraday=is_intraday,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Data period info
            if not df_indicators.empty:
                if is_intraday:
                    start = df_indicators.index[0].strftime("%Y-%m-%d %H:%M")
                    end = df_indicators.index[-1].strftime("%Y-%m-%d %H:%M")
                    st.caption(f"資料時間: {start} ~ {end} | 共 {len(df_indicators)} 根K棒")
                else:
                    start = df_indicators.index[0].strftime("%Y-%m-%d")
                    end = df_indicators.index[-1].strftime("%Y-%m-%d")
                    st.caption(f"資料期間: {start} ~ {end} | 共 {len(df_indicators)} 個交易日")

        with tab2:
            render_indicator_tab(df_indicators)

        with tab3:
            render_signal_history_tab(signal_df)

        with tab4:
            render_signal_panel(result)

    else:
        # Welcome screen
        st.markdown(
            """
            <div style="text-align: center; padding: 60px 20px;">
                <h1>📈 台股 K 線圖 AI 分析</h1>
                <h3 style="color: #90a4ae;">Taiwan Stock Technical Analysis AI</h3>
                <br>
                <p style="color: #d1d4dc; font-size: 1.1em;">
                    在左側輸入股票代號，點擊「開始分析」即可查看<br>
                    K 線圖、技術指標分析，以及 AI 買賣建議
                </p>
                <br>
                <p style="color: #546e7a;">
                    支援台灣上市 (TWSE) 及上櫃 (OTC) 股票<br>
                    技術指標: RSI、MACD、布林通道、移動均線
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick start examples
        st.markdown("### 快速開始 Quick Start")
        cols = st.columns(4)
        examples = [("2330", "台積電"), ("0050", "台灣50"), ("2454", "聯發科"), ("2317", "鴻海")]
        for col, (code, name) in zip(cols, examples):
            with col:
                if st.button(f"{code}\n{name}", use_container_width=True):
                    st.session_state.loaded_ticker = None  # Force reload
                    st.rerun()


if __name__ == "__main__":
    main()
