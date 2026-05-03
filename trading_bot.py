"""
自動交易排程主程式
==================
時間表 (Asia/Taipei, UTC+8):
  09:10  週一~週五  → 掃描選股、開新倉位
  每15分 09:10~13:25 → 監控停損/停利
  13:30               → 收盤掃描（最後一次）
  14:35               → AI 模型重新訓練 + 儲存

啟動方式:
  source venv/bin/activate
  python trading_bot.py

背景執行 (nohup):
  nohup venv/bin/python trading_bot.py > trading_bot.log 2>&1 &
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
import os
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent))
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = DATA_DIR / "trading.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("bot")

TZ = pytz.timezone("Asia/Taipei")

# ── Trading Bot Instance ──────────────────────────────────────────────────────
from trader.bot import TradingBot

bot = TradingBot()


# ── Market Day Check ──────────────────────────────────────────────────────────

def is_trading_day() -> bool:
    """台股是否開盤（排除週末與國定假日）。"""
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar("XTAI")
        today = datetime.now(TZ).date()
        schedule = cal.schedule(start_date=str(today), end_date=str(today))
        return not schedule.empty
    except Exception:
        # fallback: 週一~週五
        return datetime.now(TZ).weekday() < 5


def within_market_hours() -> bool:
    """是否在盤中時間 09:00 ~ 13:30。"""
    now = datetime.now(TZ)
    open_time  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    close_time = now.replace(hour=13, minute=30, second=0, microsecond=0)
    return open_time <= now <= close_time


# ── Job Definitions ───────────────────────────────────────────────────────────

def job_open_scan():
    """09:10 — 登入、掃描全市場、開新倉位。"""
    if not is_trading_day():
        logger.info("非交易日，跳過開盤掃描")
        return
    if not within_market_hours():
        logger.warning("開盤掃描: 已超出市場時間，跳過（可能是服務重啟補跑）")
        return
    logger.info("=== 開盤掃描開始 ===")
    try:
        if not bot.login():
            logger.error("富邦登入失敗，跳過掃描")
            return
        bot.run_cycle()
    except Exception as e:
        logger.error(f"開盤掃描失敗: {e}", exc_info=True)


def job_monitor():
    """每15分鐘盤中 — 監控持倉 + 檢查緊急重訓條件。"""
    if not is_trading_day() or not within_market_hours():
        return
    logger.info("--- 持倉監控 ---")
    try:
        bot._check_holdings()
        bot._save_state()
    except Exception as e:
        logger.error(f"持倉監控失敗: {e}", exc_info=True)

    # ── 檢查是否需要緊急重訓 ─────────────────────────────────────────────────
    try:
        needed, reason = bot.should_retrain()
        if needed:
            logger.warning(f"⚠️  觸發緊急重訓: {reason}")
            job_retrain(reason=reason)
    except Exception as e:
        logger.error(f"重訓條件檢查失敗: {e}", exc_info=True)


def job_close_scan():
    """13:30 — 收盤前最後一次掃描與監控。"""
    if not is_trading_day():
        return
    if not within_market_hours():
        logger.warning("收盤掃描: 已超出市場時間，跳過（可能是服務重啟補跑）")
        return
    logger.info("=== 收盤掃描 ===")
    try:
        bot._check_holdings()
        bot._scan_and_buy()
        bot._save_state()
    except Exception as e:
        logger.error(f"收盤掃描失敗: {e}", exc_info=True)


def job_daily_summary():
    """13:35 — 收盤後記錄當天交易成果到 daily_report.json。"""
    if not is_trading_day():
        return
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    today_trades = [t for t in bot.trade_log if t.get("date") == today]
    buys  = [t for t in today_trades if t["action"] == "BUY"]
    sells = [t for t in today_trades if t["action"] == "SELL"]

    realized_pnl = sum(t.get("pnl", 0) for t in sells)

    # 持倉市值
    holding_value = 0.0
    positions = []
    for code, h in bot.holdings.items():
        price = bot._get_latest_price(code)
        val = h["shares"] * price
        holding_value += val
        _, pct = __import__("trader.risk", fromlist=["pnl"]).pnl(h["entry_price"], price, h["shares"])
        positions.append({"ticker": code, "shares": h["shares"],
                          "entry": h["entry_price"], "close": price, "pnl_pct": round(pct, 4)})

    total_value = bot.cash + holding_value
    paper = bot.paper_trading
    report = {
        "date": today,
        "paper_trading": paper,
        "cash": round(bot.cash, 0),
        "holding_value": round(holding_value, 0),
        "total_value": round(total_value, 0),
        "realized_pnl": round(realized_pnl, 0),
        "buys": buys,
        "sells": sells,
        "positions": positions,
        "model_version": bot.model.version,
    }

    import os as _os, json as _json
    # paper trading 存到獨立檔案，避免與實盤記錄混淆
    filename = "paper_daily_report.json" if paper else "daily_report.json"
    report_path = Path(_os.getenv("DATA_DIR", Path(__file__).parent)) / filename
    history = []
    if report_path.exists():
        try:
            with open(report_path) as f:
                history = _json.load(f)
        except Exception:
            history = []
    history = [r for r in history if r.get("date") != today]
    history.append(report)
    with open(report_path, "w") as f:
        _json.dump(history, f, ensure_ascii=False, indent=2, default=str)

    mode_tag = "📄[紙上交易]" if paper else "📊"
    logger.info("=" * 55)
    logger.info(f"{mode_tag} 每日結算 {today}")
    logger.info(f"   現金:         {bot.cash:,.0f} 元")
    logger.info(f"   持倉市值:     {holding_value:,.0f} 元")
    logger.info(f"   總資產:       {total_value:,.0f} 元")
    logger.info(f"   今日實現損益: {realized_pnl:+.0f} 元")
    logger.info(f"   今日買入:     {len(buys)} 筆  賣出: {len(sells)} 筆")
    for p in positions:
        logger.info(f"   持倉 {p['ticker']}: {p['shares']}股 損益 {p['pnl_pct']:+.1%}")
    if paper:
        logger.info("   ⚠️  以上為模擬交易，未產生真實損益")
    logger.info("=" * 55)


def job_retrain(reason: str = "每日排程"):
    """14:35 或條件觸發 — 重新訓練 AI 模型。"""
    if not is_trading_day() and reason == "每日排程":
        logger.info("非交易日，跳過重訓")
        return
    logger.info(f"=== 開始重新訓練 AI 模型 (原因: {reason}) ===")
    try:
        bot.retrain(reason=reason)
        # 記錄緊急重訓日期，避免同日重複觸發
        if reason != "每日排程":
            bot._last_emergency_retrain = datetime.now(TZ).strftime("%Y-%m-%d")
        logger.info("=== 模型重訓完成 ===")
    except Exception as e:
        logger.error(f"模型重訓失敗: {e}", exc_info=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    paper = bot.paper_trading
    mode_label = "📄 紙上交易（模擬）" if paper else "🔴 實盤交易"
    logger.info("=" * 55)
    logger.info(f"  AI 自動交易機器人 啟動  [{mode_label}]")
    logger.info(f"  時區: Asia/Taipei (UTC+8)")
    logger.info(f"  資金: NT${bot.cash:,.0f} | 最大持倉: 5 支")
    if paper:
        logger.info("  ⚠️  所有買賣均為模擬，不會送出真實委託")
        logger.info("  ⚠️  切換實盤請在 .env 設定 PAPER_TRADING=false")
    logger.info("=" * 55)

    # 啟動時先載入模型
    if bot.model.is_trained():
        bot.model.load()
        logger.info(f"已載入模型版本: {bot.model.version}")
    else:
        logger.warning("尚無訓練好的模型，建議先執行 run_backtest.py")

    scheduler = BlockingScheduler(timezone=TZ)

    # misfire_grace_time=60: 若 job 超過 60 秒才啟動則直接跳過，避免重啟後補跑過期 job
    GRACE = 60

    # 09:10 — 開盤掃描
    scheduler.add_job(
        job_open_scan,
        CronTrigger(hour=9, minute=10, day_of_week="mon-fri", timezone=TZ),
        id="open_scan", name="開盤掃描",
        max_instances=1, coalesce=True, misfire_grace_time=GRACE,
    )

    # 每15分鐘 09:10~13:25 — 持倉監控
    scheduler.add_job(
        job_monitor,
        CronTrigger(
            minute="10,25,40,55",
            hour="9,10,11,12",
            day_of_week="mon-fri",
            timezone=TZ,
        ),
        id="monitor", name="持倉監控",
        max_instances=1, coalesce=True, misfire_grace_time=GRACE,
    )
    # 13:00 和 13:20 額外監控
    scheduler.add_job(
        job_monitor,
        CronTrigger(hour=13, minute="0,20", day_of_week="mon-fri", timezone=TZ),
        id="monitor_close", name="收盤前監控",
        max_instances=1, coalesce=True, misfire_grace_time=GRACE,
    )

    # 13:20 — 收盤掃描（提前10分鐘，確保盤中零股來得及成交）
    scheduler.add_job(
        job_close_scan,
        CronTrigger(hour=13, minute=20, day_of_week="mon-fri", timezone=TZ),
        id="close_scan", name="收盤掃描",
        max_instances=1, coalesce=True, misfire_grace_time=GRACE,
    )

    # 13:35 — 每日結算報告
    scheduler.add_job(
        job_daily_summary,
        CronTrigger(hour=13, minute=35, day_of_week="mon-fri", timezone=TZ),
        id="daily_summary", name="每日結算",
        max_instances=1, coalesce=True, misfire_grace_time=GRACE,
    )

    # 14:35 — 重新訓練
    scheduler.add_job(
        job_retrain,
        CronTrigger(hour=14, minute=35, day_of_week="mon-fri", timezone=TZ),
        id="retrain", name="AI重訓",
        max_instances=1, coalesce=True, misfire_grace_time=GRACE,
    )

    logger.info("排程設定完成:")
    logger.info("  09:10        → 開盤掃描選股")
    logger.info("  每15分鐘     → 監控停損/停利 (09:10~13:20)")
    logger.info("  13:30        → 收盤最後掃描")
    logger.info("  14:35        → AI 模型重新訓練")
    logger.info("按 Ctrl+C 停止")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("機器人已手動停止")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
