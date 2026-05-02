import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class KLineChart:
    """
    Builds interactive Plotly candlestick charts with indicator overlays.

    Chart layout (4 rows with shared x-axis):
      Row 1 (55%): Candlestick + BB Bands + MA Lines + Buy/Sell markers
      Row 2 (15%): Volume bars (colored by price direction) + Volume MA
      Row 3 (18%): MACD histogram + MACD/Signal lines
      Row 4 (12%): RSI line with overbought/oversold zones
    """

    COLORS = {
        "candle_up": "#26a69a",
        "candle_down": "#ef5350",
        "ma5": "#ff9800",
        "ma10": "#2196f3",
        "ma20": "#9c27b0",
        "ma60": "#a1887f",
        "bb_upper": "#546e7a",
        "bb_lower": "#546e7a",
        "bb_fill": "rgba(84, 110, 122, 0.1)",
        "macd_line": "#2196f3",
        "macd_signal": "#ff9800",
        "macd_hist_pos": "#26a69a",
        "macd_hist_neg": "#ef5350",
        "rsi_line": "#ce93d8",
        "volume_up": "rgba(38, 166, 154, 0.7)",
        "volume_down": "rgba(239, 83, 80, 0.7)",
        "volume_ma": "rgba(255, 193, 7, 0.8)",
        "buy_marker": "#00e676",
        "sell_marker": "#ff1744",
        "grid": "#2a2e39",
        "bg": "#131722",
        "text": "#d1d4dc",
    }

    def build_full_chart(
        self,
        df: pd.DataFrame,
        ticker: str,
        show_ma: list = None,
        show_bb: bool = True,
        show_signals: bool = True,
        signal_df: pd.DataFrame = None,
        is_intraday: bool = False,
    ) -> go.Figure:
        """Build the complete 4-panel interactive K-line chart.

        is_intraday=True: formats x-axis as HH:MM and skips MA60/signals.
        """
        if show_ma is None:
            show_ma = [5, 20, 60]

        # Intraday: drop MA60 (not meaningful for <1 day or 5 days)
        if is_intraday:
            show_ma = [m for m in show_ma if m != 60]

        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.55, 0.15, 0.18, 0.12],
            subplot_titles=("K線圖 Candlestick", "成交量 Volume", "MACD", "RSI"),
        )

        self._add_candlestick(fig, df, row=1)

        if show_ma:
            self._add_moving_averages(fig, df, show_ma, row=1)

        if show_bb:
            self._add_bollinger_bands(fig, df, row=1)

        if show_signals and signal_df is not None and not signal_df.empty:
            self._add_buy_sell_markers(fig, signal_df, df, row=1)

        self._add_volume_bars(fig, df, row=2)
        self._add_macd(fig, df, row=3)
        self._add_rsi(fig, df, row=4)
        self._apply_layout(fig, ticker, is_intraday=is_intraday)

        return fig

    def _add_candlestick(self, fig: go.Figure, df: pd.DataFrame, row: int = 1) -> None:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="K線",
                increasing_fillcolor=self.COLORS["candle_up"],
                increasing_line_color=self.COLORS["candle_up"],
                decreasing_fillcolor=self.COLORS["candle_down"],
                decreasing_line_color=self.COLORS["candle_down"],
                showlegend=False,
                hovertext=[
                    f"開: {o:.2f} 高: {h:.2f} 低: {l:.2f} 收: {c:.2f}"
                    for o, h, l, c in zip(df["Open"], df["High"], df["Low"], df["Close"])
                ],
                hoverinfo="x+text",
            ),
            row=row,
            col=1,
        )

    def _add_moving_averages(
        self, fig: go.Figure, df: pd.DataFrame, periods: list, row: int = 1
    ) -> None:
        ma_colors = {5: self.COLORS["ma5"], 10: self.COLORS["ma10"],
                     20: self.COLORS["ma20"], 60: self.COLORS["ma60"]}
        for period in periods:
            col_name = f"MA{period}"
            if col_name not in df.columns:
                continue
            color = ma_colors.get(period, "#ffffff")
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col_name],
                    name=f"MA{period}",
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    hovertemplate=f"MA{period}: %{{y:.2f}}<extra></extra>",
                ),
                row=row,
                col=1,
            )

    def _add_bollinger_bands(self, fig: go.Figure, df: pd.DataFrame, row: int = 1) -> None:
        if "BB_Upper" not in df.columns:
            return

        # Upper band
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                name="BB上軌",
                mode="lines",
                line=dict(color=self.COLORS["bb_upper"], width=1, dash="dot"),
                hovertemplate="BB上軌: %{y:.2f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        # Lower band with fill
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                name="BB下軌",
                mode="lines",
                line=dict(color=self.COLORS["bb_lower"], width=1, dash="dot"),
                fill="tonexty",
                fillcolor=self.COLORS["bb_fill"],
                hovertemplate="BB下軌: %{y:.2f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        # Midline
        if "BB_Mid" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["BB_Mid"],
                    name="BB中軌",
                    mode="lines",
                    line=dict(color=self.COLORS["bb_upper"], width=1),
                    hovertemplate="BB中軌: %{y:.2f}<extra></extra>",
                ),
                row=row,
                col=1,
            )

    def _add_volume_bars(self, fig: go.Figure, df: pd.DataFrame, row: int = 2) -> None:
        colors = [
            self.COLORS["volume_up"] if c >= o else self.COLORS["volume_down"]
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="成交量",
                marker_color=colors,
                showlegend=False,
                hovertemplate="成交量: %{y:,.0f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        if "Volume_MA20" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Volume_MA20"],
                    name="Vol MA20",
                    mode="lines",
                    line=dict(color=self.COLORS["volume_ma"], width=1.5),
                    hovertemplate="Vol MA20: %{y:,.0f}<extra></extra>",
                ),
                row=row,
                col=1,
            )

    def _add_macd(self, fig: go.Figure, df: pd.DataFrame, row: int = 3) -> None:
        if "MACD_Hist" not in df.columns:
            return

        hist_colors = [
            self.COLORS["macd_hist_pos"] if v >= 0 else self.COLORS["macd_hist_neg"]
            for v in df["MACD_Hist"].fillna(0)
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_Hist"],
                name="MACD柱",
                marker_color=hist_colors,
                showlegend=False,
                hovertemplate="MACD柱: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                name="MACD",
                mode="lines",
                line=dict(color=self.COLORS["macd_line"], width=1.5),
                hovertemplate="MACD: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                name="Signal",
                mode="lines",
                line=dict(color=self.COLORS["macd_signal"], width=1.5),
                hovertemplate="Signal: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color=self.COLORS["grid"],
                      line_width=1, row=row, col=1)

    def _add_rsi(self, fig: go.Figure, df: pd.DataFrame, row: int = 4) -> None:
        if "RSI" not in df.columns:
            return

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                name="RSI",
                mode="lines",
                line=dict(color=self.COLORS["rsi_line"], width=1.5),
                hovertemplate="RSI: %{y:.1f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        # Overbought/Oversold reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="#ef5350",
                      line_width=1, row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#26a69a",
                      line_width=1, row=row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color=self.COLORS["grid"],
                      line_width=1, row=row, col=1)

    def _add_buy_sell_markers(
        self,
        fig: go.Figure,
        signal_df: pd.DataFrame,
        price_df: pd.DataFrame,
        row: int = 1,
    ) -> None:
        """Add triangle markers for significant buy/sell signals."""
        buy_signals = signal_df[signal_df["total_score"] >= 2]
        sell_signals = signal_df[signal_df["total_score"] <= -2]

        if not buy_signals.empty:
            # Buy markers below the candle low
            buy_prices = []
            for date in buy_signals.index:
                if date in price_df.index:
                    low = price_df.loc[date, "Low"]
                    buy_prices.append(low * 0.995)
                else:
                    buy_prices.append(buy_signals.loc[date, "Low"])

            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_prices,
                    mode="markers",
                    name="買進信號",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color=self.COLORS["buy_marker"],
                        line=dict(color="#ffffff", width=1),
                    ),
                    hovertemplate="買進信號 Buy Signal<br>分數: %{customdata:+d}<extra></extra>",
                    customdata=buy_signals["total_score"],
                ),
                row=row,
                col=1,
            )

        if not sell_signals.empty:
            # Sell markers above the candle high
            sell_prices = []
            for date in sell_signals.index:
                if date in price_df.index:
                    high = price_df.loc[date, "High"]
                    sell_prices.append(high * 1.005)
                else:
                    sell_prices.append(sell_signals.loc[date, "High"])

            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_prices,
                    mode="markers",
                    name="賣出信號",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color=self.COLORS["sell_marker"],
                        line=dict(color="#ffffff", width=1),
                    ),
                    hovertemplate="賣出信號 Sell Signal<br>分數: %{customdata:+d}<extra></extra>",
                    customdata=sell_signals["total_score"],
                ),
                row=row,
                col=1,
            )

    def _apply_layout(self, fig: go.Figure, ticker: str, is_intraday: bool = False) -> None:
        """Apply TradingView-inspired dark theme layout."""
        fig.update_layout(
            title=dict(
                text=f"📈 {ticker} K線圖技術分析",
                font=dict(size=16, color=self.COLORS["text"]),
                x=0.02,
            ),
            template="plotly_dark",
            height=820,
            paper_bgcolor=self.COLORS["bg"],
            plot_bgcolor=self.COLORS["bg"],
            legend=dict(
                orientation="h",
                x=0,
                y=1.02,
                font=dict(size=11, color=self.COLORS["text"]),
                bgcolor="rgba(19,23,34,0.8)",
            ),
            margin=dict(l=60, r=20, t=60, b=20),
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
        )

        # X-axis: intraday shows time, daily shows date
        if is_intraday:
            # Show HH:MM for intraday; hide non-trading gaps
            xaxis_extra = dict(
                tickformat="%H:%M",
                dtick=3600000,          # 1-hour ticks (ms)
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),              # skip weekends
                    dict(bounds=[13.5, 9], pattern="hour"),   # skip 13:30-09:00 (market closed)
                ],
            )
            # No rangeselector for intraday
            fig.update_xaxes(**xaxis_extra, row=1, col=1)
        else:
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="全部"),
                    ],
                    bgcolor=self.COLORS["bg"],
                    activecolor="#2196f3",
                    font=dict(color=self.COLORS["text"]),
                ),
                row=1,
                col=1,
            )

        # Grid styling for all axes
        axis_style = dict(
            gridcolor=self.COLORS["grid"],
            gridwidth=1,
            zerolinecolor=self.COLORS["grid"],
            tickfont=dict(color=self.COLORS["text"]),
        )
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)

        # RSI y-axis fixed range
        fig.update_yaxes(range=[0, 100], row=4, col=1)

    def build_mini_chart(self, df: pd.DataFrame, ticker: str) -> go.Figure:
        """Lightweight single-panel chart for comparison grid view."""
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=ticker,
                increasing_fillcolor=self.COLORS["candle_up"],
                increasing_line_color=self.COLORS["candle_up"],
                decreasing_fillcolor=self.COLORS["candle_down"],
                decreasing_line_color=self.COLORS["candle_down"],
                showlegend=False,
            )
        )
        if "MA20" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MA20"],
                    name="MA20",
                    mode="lines",
                    line=dict(color=self.COLORS["ma20"], width=1),
                )
            )
        fig.update_layout(
            title=ticker,
            height=250,
            template="plotly_dark",
            paper_bgcolor=self.COLORS["bg"],
            plot_bgcolor=self.COLORS["bg"],
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            showlegend=False,
        )
        return fig
