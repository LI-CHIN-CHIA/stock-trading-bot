"""
自動交易排程主程式
==================
時間表 (Asia/Taipei, UTC+8):
  08:30  週一~週五  → 開盤前選股發現（PTT/三大法人掃描，動態更新清單）
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
_handlers = [logging.FileHandler(LOG_FILE, encoding="utf-8")]
# 只在互動終端（非 systemd/nohup）加控制台輸出，避免 systemd append: 重複寫入
if sys.stdout.isatty():
    _handlers.append(logging.StreamHandler(sys.stdout))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=_handlers,
)
logger = logging.getLogger("bot")

TZ = pytz.timezone("Asia/Taipei")

# ── Trading Bot Instance ──────────────────────────────────────────────────────
from trader.bot import TradingBot
from trader.day_trader import DayTrader
from trader.overnight_trader import OvernightTrader

bot           = TradingBot()
day_bot       = DayTrader()
overnight_bot = OvernightTrader()


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

def job_day_trader():
    """每 2 分鐘盤中 — 當沖盯盤 / 紙上交易。"""
    if not is_trading_day():
        return
    try:
        day_bot.run_cycle()
    except Exception as e:
        logger.error(f"當沖週期失敗: {e}", exc_info=True)


def job_overnight_trader():
    """每 2 分鐘 — 隔日沖尾盤佈局 + 隔日早盤出場。"""
    if not is_trading_day():
        return
    try:
        overnight_bot.run_cycle()
    except Exception as e:
        logger.error(f"隔日沖週期失敗: {e}", exc_info=True)


def job_stock_discovery():
    """
    08:30 — 開盤前選股發現任務：
    掃描 PTT/三大法人，自動新增或暫停科技股交易清單。
    """
    if not is_trading_day():
        logger.info("非交易日，跳過選股發現")
        return
    logger.info("=== 開盤前選股發現任務 ===")
    try:
        from utils.stock_discovery import run_discovery, print_dynamic_status
        summary = run_discovery()
        if summary["added"]:
            for item in summary["added"]:
                logger.info(f"  ✅ 新增交易標的: {item['code']} {item['name']} — {item['reason']}")
        if summary["suspended"]:
            for item in summary["suspended"]:
                logger.warning(f"  ⏸️  暫停交易標的: {item['code']} {item['name']} — {item['reason']}")
        logger.info(
            f"  目前有效清單: {summary['total_effective']} 支 "
            f"(新增 {len(summary['added'])}，暫停 {len(summary['suspended'])})"
        )
    except Exception as e:
        logger.error(f"選股發現任務失敗: {e}", exc_info=True)


def job_open_scan():
    """09:10 — 夜盤門控 → 登入 → 掃描全市場、開新倉位。"""
    if not is_trading_day():
        logger.info("非交易日，跳過開盤掃描")
        return
    if not within_market_hours():
        logger.warning("開盤掃描: 已超出市場時間，跳過（可能是服務重啟補跑）")
        return

    # ── 夜盤方向門控 ──────────────────────────────────────────────────────
    _thresh_orig: float | None = None
    try:
        from data.night_futures import night_gate_ok
        import ai.model as _ai_model
        gate_open, extra_thresh = night_gate_ok()
        if not gate_open:
            logger.warning("🚫 夜盤門控關閉，今日跳過開盤買入掃描（持倉監控仍正常運行）")
            bot._check_holdings()
            bot._save_state()
            return
        if extra_thresh > 0:
            _thresh_orig = _ai_model.BUY_THRESHOLD
            _ai_model.BUY_THRESHOLD = min(_thresh_orig + extra_thresh, 0.90)
            logger.info(f"⚠️  夜盤下跌：買入門檻暫提至 {_ai_model.BUY_THRESHOLD:.2f}（原 {_thresh_orig:.2f}）")
    except Exception as e:
        logger.warning(f"夜盤門控判斷失敗（放行）: {e}")

    logger.info("=== 開盤掃描開始 ===")
    try:
        # 已在啟動時登入 → 不重複登入，避免覆蓋 SDK 實例
        if bot.sdk is None and not bot.login():
            logger.error("富邦登入失敗，跳過掃描")
            return
        bot.run_cycle()
    except Exception as e:
        logger.error(f"開盤掃描失敗: {e}", exc_info=True)
    finally:
        # 恢復原始買入門檻（若有暫時調整）
        if _thresh_orig is not None:
            try:
                import ai.model as _ai_model
                _ai_model.BUY_THRESHOLD = _thresh_orig
            except Exception:
                pass


def job_monitor():
    """每3分鐘盤中 — 監控持倉 + 檢查緊急重訓條件。"""
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


def job_intraday_scan():
    """每15分鐘盤中 — 日間補掃，抓住盤中新機會。"""
    if not is_trading_day() or not within_market_hours():
        return
    logger.info("--- 日間補掃 ---")
    try:
        bot._sync_account_balance()
        bot._sync_holdings_from_fubon()
        bot._scan_and_buy()
        bot._save_state()
    except Exception as e:
        logger.error(f"日間補掃失敗: {e}", exc_info=True)


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

    # ── Telegram 每日戰報 ────────────────────────────────────────────────────
    try:
        from utils.telegram_notify import send
        from utils.constants import STOCK_DB  # 提前 import，避免 day_holdings 段 NameError

        # 實盤部分
        real_mode = "🔴 實盤" if not paper else "📄 紙上"
        pos_lines = ""
        for p in positions:
            name = STOCK_DB.get(p["ticker"], p["ticker"]).split()[0]
            pos_lines += f"  {p['ticker']} {name} {p['pnl_pct']:+.1%}\n"

        real_msg = (
            f"📊 <b>台股每日戰報 {today}</b>  {real_mode}\n"
            f"{'─'*28}\n"
            f"💰 現金: {bot.cash:,.0f}\n"
            f"📦 持倉市值: {holding_value:,.0f}\n"
            f"🏦 總資產: {total_value:,.0f}\n"
            f"{'─'*28}\n"
            f"今日實現損益: <b>{realized_pnl:+,.0f}</b> 元\n"
            f"買入 {len(buys)} 筆  賣出 {len(sells)} 筆"
            + (f"\n持倉:\n{pos_lines}" if pos_lines else "\n（無持倉）")
        )

        # 當沖紙上部分
        day_summary = day_bot.get_summary()
        day_holdings = day_summary.get("holdings", {})
        day_pnl      = day_summary.get("daily_pnl", 0.0)
        day_cash     = day_summary.get("cash", 0.0)
        day_trades   = day_summary.get("trades_today", 0)
        day_pos_lines = ""
        for c, h in day_holdings.items():
            name = STOCK_DB.get(c, c).split()[0]
            day_pos_lines += f"  {c} {name} 未平倉\n"

        day_msg = (
            f"\n📄 <b>當沖紙上</b>\n"
            f"{'─'*28}\n"
            f"💵 現金: {day_cash:,.0f}\n"
            f"今日損益: <b>{day_pnl:+,.0f}</b> 元\n"
            f"成交: {day_trades} 筆"
            + (f"\n持倉:\n{day_pos_lines}" if day_pos_lines else "\n（全部平倉）")
        )

        send(real_msg + day_msg, key=f"daily:{today}", cooldown=0)
    except Exception as e:
        logger.warning(f"Telegram 每日戰報發送失敗: {e}")

    # 自動 commit 績效報告至 git，供 cloud agent 每週分析
    try:
        import subprocess as _sp
        repo_root = Path(__file__).parent
        # 只 commit 報告檔，不動其他未追蹤的東西
        _sp.run(["git", "add", str(report_path)], cwd=repo_root, check=True)
        result = _sp.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_root
        )
        if result.returncode != 0:  # 有變更才 commit
            _sp.run([
                "git", "commit", "-m",
                f"perf: 每日績效報告 {today} (總資產={total_value:,.0f}, 損益={realized_pnl:+.0f})"
            ], cwd=repo_root, check=True)
            _sp.run(["git", "push"], cwd=repo_root, check=True)
            logger.info(f"📤 績效報告已推送至 GitHub: {today}")
    except Exception as e:
        logger.warning(f"績效報告 git push 失敗（不影響交易）: {e}")


def job_retrain(reason: str = "每日排程"):
    """14:35 或條件觸發 — 重新訓練 AI 模型。"""
    if not is_trading_day() and reason == "每日排程":
        logger.info("非交易日，跳過重訓")
        return
    logger.info(f"=== 開始重新訓練 AI 模型 (原因: {reason}) ===")
    try:
        bot.retrain(reason=reason)
        if reason != "每日排程":
            now_tz = datetime.now(TZ)
            bot._last_emergency_retrain = now_tz.strftime("%Y-%m-%d")
            bot._last_emergency_retrain_dt = now_tz  # 2小時冷卻用
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
        # 隔日沖共用同一個日線 model
        overnight_bot.set_model(bot.model)
    else:
        logger.warning("尚無訓練好的模型，建議先執行 run_backtest.py")

    # 實盤模式：啟動時立即登入，不等 09:10 排程
    if not paper:
        if bot.login():
            logger.info("✅ 富邦實盤登入成功（啟動時）")
        else:
            logger.error("❌ 富邦實盤登入失敗，監控仍會執行但委託需等 09:10 重新登入")

    scheduler = BlockingScheduler(timezone=TZ)

    # misfire_grace_time=60: 若 job 超過 60 秒才啟動則直接跳過，避免重啟後補跑過期 job
    GRACE = 60

    # 每 1 分鐘 09:00~13:30 — 當沖盯盤 / 紙上交易（1min K棒，更敏銳）
    scheduler.add_job(
        job_day_trader,
        CronTrigger(
            minute="*",
            hour="9,10,11,12,13",
            day_of_week="mon-fri",
            timezone=TZ,
        ),
        id="day_trader", name="當沖盯盤",
        max_instances=1, coalesce=True, misfire_grace_time=30,
    )

    # 每 2 分鐘 09:00~14:30 — 隔日沖（早盤出場 + 尾盤佈局）
    scheduler.add_job(
        job_overnight_trader,
        CronTrigger(
            minute="*/2",
            hour="9,10,11,13,14",
            day_of_week="mon-fri",
            timezone=TZ,
        ),
        id="overnight_trader", name="隔日沖",
        max_instances=1, coalesce=True, misfire_grace_time=30,
    )

    # 08:30 — 開盤前選股發現（社群掃描 + 三大法人）
    scheduler.add_job(
        job_stock_discovery,
        CronTrigger(hour=8, minute=30, day_of_week="mon-fri", timezone=TZ),
        id="stock_discovery", name="開盤前選股發現",
        max_instances=1, coalesce=True, misfire_grace_time=GRACE,
    )

    # 09:10 — 開盤掃描
    scheduler.add_job(
        job_open_scan,
        CronTrigger(hour=9, minute=10, day_of_week="mon-fri", timezone=TZ),
        id="open_scan", name="開盤掃描",
        max_instances=1, coalesce=True, misfire_grace_time=GRACE,
    )

    # 每3分鐘 09:00~13:30 — 持倉監控
    scheduler.add_job(
        job_monitor,
        CronTrigger(
            minute="*/3",
            hour="9,10,11,12,13",
            day_of_week="mon-fri",
            timezone=TZ,
        ),
        id="monitor", name="持倉監控",
        max_instances=1, coalesce=True, misfire_grace_time=GRACE,
    )

    # 每15分鐘 09:25~13:05 — 日間補掃（抓盤中新機會；TA 有 8h 快取不會重複呼叫 LLM）
    scheduler.add_job(
        job_intraday_scan,
        CronTrigger(
            minute="*/15",
            hour="9,10,11,12,13",
            day_of_week="mon-fri",
            timezone=TZ,
        ),
        id="intraday_scan", name="日間補掃",
        max_instances=1, coalesce=True, misfire_grace_time=GRACE,
    )

    # 13:20 — 收盤掃描（提前10分鐘，確保盤中零股來得及成交）
    scheduler.add_job(
        job_close_scan,
        CronTrigger(hour=13, minute=20, day_of_week="mon-fri", timezone=TZ),
        id="close_scan", name="收盤掃描",
        max_instances=1, coalesce=True, misfire_grace_time=GRACE,
    )

    # 13:32 — 每日結算報告（收盤後立即 Telegram 戰報）
    scheduler.add_job(
        job_daily_summary,
        CronTrigger(hour=13, minute=32, day_of_week="mon-fri", timezone=TZ),
        id="daily_summary", name="每日結算",
        max_instances=1, coalesce=True, misfire_grace_time=GRACE,
    )

    # 14:35 週一 — 週度走步重訓（每週一次，使用最近 2 年資料）
    scheduler.add_job(
        job_retrain,
        CronTrigger(hour=14, minute=35, day_of_week="mon", timezone=TZ),
        id="retrain", name="AI週度重訓",
        max_instances=1, coalesce=True, misfire_grace_time=GRACE,
    )

    logger.info("排程設定完成:")
    logger.info("  08:30        → 開盤前選股發現（社群掃描 + 三大法人）")
    logger.info("  09:10        → 開盤掃描選股")
    logger.info("  每15分鐘     → 日間補掃（09:00~13:30，抓盤中新機會）")
    logger.info("  每1分鐘      → 當沖盯盤 / 大量告警 / 紙上交易 (09:00~13:30)")
    logger.info("  每2分鐘      → 隔日沖 早盤出場+尾盤佈局 (09:00~14:30)")
    logger.info("  每3分鐘      → 監控停損/停利 (09:00~13:30)")
    logger.info("  13:30        → 收盤最後掃描")
    logger.info("  14:35 (週一) → AI 週度重訓（walk-forward）")
    logger.info("按 Ctrl+C 停止")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("機器人已手動停止")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
