---
name: trading_bot_architecture
description: Automated Taiwan stock trading bot added to the project - architecture, files, and setup requirements
type: project
---

# Automated Trading Bot (added 2026-03-21)

## New Files Created
- `data/state.py` - Persistent JSON state store (positions, trades, meta)
- `ai/features.py` - LightGBM feature engineering (52 features, 7 groups)
- `ai/model.py` - LightGBM 3-class classifier (BUY/HOLD/SELL), versioned save/load
- `ai/models/` - Directory for saved model .pkl files
- `trading/fugle_client.py` - Fugle Trade SDK wrapper (DRY_RUN auto-fallback)
- `trading/risk_manager.py` - Position sizing, stop-loss, take-profit
- `trading/engine.py` - Main trading engine (scan/trade/monitor/retrain)
- `monitor/dashboard.py` - Streamlit monitoring dashboard (port 8502)
- `trading_bot.py` - APScheduler entry point
- `config.ini.example` - Fugle config template

## Trading Parameters
- Budget: 20,000 TWD total, max 5 positions (~4,000 TWD each)
- Stop-loss: -10%, Take-profit: +15%, Max hold: 15 days
- Odd-lot (零股) trading via Fugle Trade API

## Schedule (Asia/Taipei)
- 09:10 → scan all stocks, open positions
- every 3 min → monitor stop-loss/take-profit
- 14:35 → retrain LightGBM model

## Entry Gate (AND logic)
- Rule score ≥ +2 (MODERATE_BUY) from existing signals.py
- AI buy probability ≥ 50% (cold start: rule-only with score ≥ +4)

## Setup Requirements (for LIVE mode)
User needs to set up `config.ini` (from config.ini.example) with:
- Fugle account number
- .p12 certificate file (申請富邦程式交易)
- API Key + Secret from developer.fugle.tw
Without config.ini, runs in DRY_RUN mode automatically.

## Run Commands
```bash
# Trading bot
source venv/bin/activate && python trading_bot.py

# Monitoring dashboard (port 8502)
source venv/bin/activate && streamlit run monitor/dashboard.py --server.port 8502
```

**Why:** User wants automated Taiwan stock trading with remote monitoring using Fugle (富邦) API.
**How to apply:** When user asks about the trading bot, refer to this architecture. The bot runs independently from the original app.py.
