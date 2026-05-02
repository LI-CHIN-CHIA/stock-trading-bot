# Stock Project Memory

## Project: 台股 K 線圖 AI 分析系統

### Stack
- Python + Streamlit frontend
- yfinance 1.2.0 for Taiwan stock data (.TW / .TWO suffix)
- pandas-ta for technical indicators
- Plotly for interactive K-line charts

### Key Files
- [app.py](app.py) - Streamlit entry point
- [data/fetcher.py](data/fetcher.py) - yfinance data fetching (NO custom session)
- [analysis/indicators.py](analysis/indicators.py) - RSI, MACD, BB, MA, ATR
- [analysis/signals.py](analysis/signals.py) - Rule-based buy/sell signal engine
- [charts/candlestick.py](charts/candlestick.py) - 4-panel Plotly chart
- [utils/constants.py](utils/constants.py) - Constants, popular stocks list

### Critical: yfinance 1.2.0 Breaking Change
- yfinance 1.2.0 uses curl_cffi internally (impersonates browser)
- Custom sessions (requests-cache CachedLimiterSession) are NOT supported
- Must use `yf.Ticker(symbol)` WITHOUT passing session parameter
- Caching is handled at app level via `@st.cache_data(ttl=900)`

### Virtual Environment
- Located at `/home/ljj/Documents/stock/venv/`
- Start app: `/home/ljj/Documents/stock/venv/bin/streamlit run app.py`
- Python 3.12.3

### Taiwan Stock Symbols
- TWSE (listed): {code}.TW (e.g., 2330.TW for TSMC)
- OTC/TPEX: {code}.TWO (e.g., 6547.TWO)
- OTC codes stored in `OTC_SYMBOLS` set in constants.py

### Signal Scoring System
- Each rule contributes -2 to +2 points
- Total score >= +4 → Strong Buy, +2/+3 → Moderate Buy, +1 → Weak Buy
- 0 → Neutral, -1 → Weak Sell, -2/-3 → Moderate Sell, <=-4 → Strong Sell
- Indicators used: RSI, MACD crossover, MACD zero line, Bollinger Bands, MA crossovers, MA alignment, Volume

### Running the App
```bash
cd /home/ljj/Documents/stock
source venv/bin/activate
streamlit run app.py
# or
venv/bin/streamlit run app.py --server.port 8501
```

## Automated Trading Bot (added 2026-03-21)
See [memory/project_trading_bot.md](memory/project_trading_bot.md) for full details.

### Run Commands
```bash
# Trading bot
source venv/bin/activate && python trading_bot.py

# Monitoring dashboard (port 8502)
source venv/bin/activate && streamlit run monitor/dashboard.py --server.port 8502
```
