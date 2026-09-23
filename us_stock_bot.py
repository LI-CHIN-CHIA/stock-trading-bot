"""
US Stock Paper Trading Bot — Entry Point
========================================
Schedule (US/Eastern):
  09:25  Mon-Fri  → 開盤前掃股（市場準備）
  09:35  Mon-Fri  → 開盤買入掃描
  每30分 09:35~15:30 → 停損/停利監控
  15:45           → 收盤掃描
  16:15           → 每日結算
  06:00 UTC       → AI 模型重訓（台灣時間 14:00）

啟動方式:
  python us_stock_bot.py
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data" / "us"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DATA_DIR"] = str(DATA_DIR)

LOG_FILE = DATA_DIR / "us_trading.log"
_handlers = [logging.FileHandler(LOG_FILE, encoding="utf-8")]
if sys.stdout.isatty():
    _handlers.append(logging.StreamHandler(sys.stdout))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=_handlers,
)
logger = logging.getLogger("us_bot")

TZ_ET = pytz.timezone("US/Eastern")

from trader.us_bot import USBot
bot = USBot()


# ── Market calendar ───────────────────────────────────────────────────────────

def is_us_trading_day() -> bool:
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar("XNYS")
        today = datetime.now(TZ_ET).date()
        schedule = cal.schedule(start_date=str(today), end_date=str(today))
        return not schedule.empty
    except Exception:
        return datetime.now(TZ_ET).weekday() < 5


# ── Jobs ─────────────────────────────────────────────────────────────────────

def job_open_scan():
    if not is_us_trading_day():
        return
    logger.info("=== US 開盤掃描 ===")
    try:
        bot.run_cycle()
    except Exception as e:
        logger.error(f"開盤掃描失敗: {e}", exc_info=True)


def job_monitor():
    if not is_us_trading_day():
        return
    logger.info("--- US 持倉監控 ---")
    try:
        bot._check_holdings()
        bot._save_state()
    except Exception as e:
        logger.error(f"持倉監控失敗: {e}", exc_info=True)


def job_close_scan():
    if not is_us_trading_day():
        return
    logger.info("=== US 收盤掃描 ===")
    try:
        bot._check_holdings()
        bot._save_state()
    except Exception as e:
        logger.error(f"收盤掃描失敗: {e}", exc_info=True)


def job_settle():
    if not is_us_trading_day():
        return
    try:
        bot.settle()
    except Exception as e:
        logger.error(f"結算失敗: {e}", exc_info=True)


def job_retrain():
    try:
        bot.retrain("每日排程")
    except Exception as e:
        logger.error(f"重訓失敗: {e}", exc_info=True)


# ── Scheduler ────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 55)
    logger.info(f"  US Stock Paper Trading Bot  啟動")
    logger.info(f"  資金: ${bot.cash:,.0f} USD | 最大持倉: {bot.MAX_POSITIONS if hasattr(bot, 'MAX_POSITIONS') else 5} 支")
    logger.info("=" * 55)

    scheduler = BlockingScheduler(timezone=TZ_ET)

    # 開盤買入：ET 09:35 (Mon-Fri)
    scheduler.add_job(job_open_scan, CronTrigger(
        day_of_week="mon-fri", hour=9, minute=35, timezone=TZ_ET), id="開盤掃描")

    # 持倉監控：每30分鐘 09:35~15:30 ET
    scheduler.add_job(job_monitor, CronTrigger(
        day_of_week="mon-fri", hour="9-15", minute="5,35", timezone=TZ_ET), id="持倉監控")

    # 收盤掃描：ET 15:45
    scheduler.add_job(job_close_scan, CronTrigger(
        day_of_week="mon-fri", hour=15, minute=45, timezone=TZ_ET), id="收盤掃描")

    # 每日結算：ET 16:15
    scheduler.add_job(job_settle, CronTrigger(
        day_of_week="mon-fri", hour=16, minute=15, timezone=TZ_ET), id="每日結算")

    # 模型重訓：UTC 06:00 (ET 01:00/02:00，每日盤後)
    scheduler.add_job(job_retrain, CronTrigger(
        day_of_week="tue-sat", hour=1, minute=0, timezone=TZ_ET), id="AI重訓")

    logger.info("排程設定完成:")
    logger.info("  09:35 ET  → 開盤掃描選股")
    logger.info("  每30分    → 監控停損/停利 (09:35~15:30 ET)")
    logger.info("  15:45 ET  → 收盤掃描")
    logger.info("  16:15 ET  → 每日結算")
    logger.info("  01:00 ET  → AI 模型重訓")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("US Bot 停止")


if __name__ == "__main__":
    main()
