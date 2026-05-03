#!/bin/bash
# 啟動 feature/trading-agents 紙上交易機器人
# 獨立的資料目錄，不與 production bot 衝突

cd /home/ljj/Documents/stock

export DATA_DIR=/home/ljj/Documents/stock/data/paper
export PAPER_TRADING=true
export ENABLE_TRADING_AGENTS=false   # 先關閉，待觀察穩定後再開

nohup venv/bin/python trading_bot.py \
  > data/paper/trading_bot.log 2>&1 &

echo "✅ 紙上交易機器人已啟動 PID=$!"
echo "   Log: data/paper/trading_bot.log"
echo "   State: data/paper/paper_trader_state.json"
echo "   Report: data/paper/paper_daily_report.json"
